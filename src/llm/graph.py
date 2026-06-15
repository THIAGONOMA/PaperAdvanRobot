"""Grafo LangGraph: monta prompt -> chama LLM -> valida -> retry/fallback.

A saída estruturada é obtida pedindo JSON no prompt e parseando/validando com
Pydantic do nosso lado (em vez do guided decoding do vLLM, que neste servidor
causa geração lenta/instável).
"""
from __future__ import annotations

import json
import re
import time

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from ..config import CFG
from .prompts import build_messages
from .schema import EmotionPrediction, ValencePrediction
from .state import GraphState

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _llm_for(task: str) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=CFG.llm_base_url,
        model=CFG.llm_model,
        api_key=CFG.llm.api_key,
        temperature=CFG.llm.temperature,
        timeout=CFG.llm.timeout,
        max_retries=0,
        max_tokens=512,
    )


def _parse_prediction(text: str, task: str) -> dict:
    """Extrai o JSON da resposta, valida com Pydantic e normaliza para [-1,1]/[0,1].

    O modelo responde em escala inteira (-100..100 e 0-100); aqui dividimos por
    100 para manter a interface downstream em [-1,1] (valência/arousal) e
    [0,1] (emoções/confiança).
    """
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError(f"Sem JSON na resposta: {text[:200]!r}")
    data = json.loads(m.group(0))
    schema = ValencePrediction if task == "valence" else EmotionPrediction
    d = schema(**data).model_dump()
    for key in ("valence", "arousal", "confidence"):
        if isinstance(d.get(key), (int, float)):
            d[key] = d[key] / 100.0
    if isinstance(d.get("ekman"), dict):
        d["ekman"] = {k: v / 100.0 for k, v in d["ekman"].items()}
    return d


def build_prompt(state: GraphState) -> dict:
    return {"messages": build_messages(state)}


def call_llm(state: GraphState) -> dict:
    t0 = time.perf_counter()
    try:
        resp = _llm_for(state["task"]).invoke(state["messages"])
        pred = _parse_prediction(resp.content, state["task"])
        return {
            "prediction": pred,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "error": None,
        }
    except Exception as exc:  # formato inválido / timeout
        return {
            "error": str(exc),
            "attempts": state.get("attempts", 0) + 1,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }


def fallback(state: GraphState) -> dict:
    pred = {
        "emotion_label": "neutral",
        "valence": 0.0,
        "arousal": 0.0,
        "confidence": 0.0,
        "fallback": True,
    }
    return {"prediction": pred}


def route_after_llm(state: GraphState) -> str:
    if state.get("prediction") is not None:
        return "ok"
    if state.get("attempts", 0) < state.get("max_attempts", CFG.run.max_attempts):
        return "retry"
    return "fallback"


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("build_prompt", build_prompt)
    g.add_node("call_llm", call_llm)
    g.add_node("fallback", fallback)

    g.set_entry_point("build_prompt")
    g.add_edge("build_prompt", "call_llm")
    g.add_conditional_edges(
        "call_llm",
        route_after_llm,
        {"ok": END, "retry": "call_llm", "fallback": "fallback"},
    )
    g.add_edge("fallback", END)
    return g.compile()
