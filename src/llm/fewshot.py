"""Construção de exemplos few-shot (in-context) a partir do split de treino.

Para o OMG/C2, cada exemplo é (descrição dos blendshapes -> JSON de valência),
usando a valência real (ground truth) do treino e o arousal estimado pela
baseline de regras como referência de formato. Sem vazamento: exemplos vêm
exclusivamente do split de treino.
"""
from __future__ import annotations

import json
import random

from ..baseline.rule_engine import BaselineRuleEngine
from ..config import CFG
from ..data import mosei_loader, omg_loader
from ..data.types import dominant_emotion
from ..features.extract import blendshapes_sequence_for
from ..features.serialize import blendshapes_sequence_to_text, facet_to_text

EKMAN = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]


def _example_input_text(blendshapes_text: str) -> str:
    return (
        "Facial blendshape coefficients for the target person. These INPUT values "
        "are on a 0-1 scale (higher = stronger activation); your OUTPUT scores must "
        "follow the scales specified above (0-100 and -100..100):\n"
        f"{blendshapes_text}\n\n"
        "Infer the emotional state from these facial signals."
    )


def build_omg_c2_fewshot(k: int = 3, seed: int = 42) -> list[dict]:
    """Gera k exemplos (blendshapes->valência) do split de treino do OMG."""
    rng = random.Random(seed)
    train = list(omg_loader.iter_samples("train", window_s=CFG.data.window_s))
    rng.shuffle(train)

    engine = BaselineRuleEngine()
    examples: list[dict] = []
    for s in train:
        if len(examples) >= k:
            break
        seq = blendshapes_sequence_for(s)
        if not seq:
            continue
        text = blendshapes_sequence_to_text(seq, CFG.data.top_n_blendshapes)
        arousal = engine.predict_sequence(seq)["arousal"]
        output = json.dumps({
            "valence": int(round(s.ground_truth.valence * 100)),
            "arousal": int(round(arousal * 100)),
            "confidence": 85,
        })
        examples.append({"input": _example_input_text(text), "output": output})
    return examples


# --- MOSEI (classificação de emoção) ---------------------------------------

def _mosei_example_output(gt) -> str:
    """JSON de exemplo no formato EmotionPrediction (escalas inteiras)."""
    es = gt.emotion_scores  # intensidades 0-3
    ekman = {e: int(round(min(es.get(e, 0.0) / 3.0, 1.0) * 100)) for e in EKMAN}
    valence = int(round(max(-1.0, min(1.0, (gt.sentiment or 0.0) / 3.0)) * 100))
    return json.dumps({
        "ekman": ekman,
        "emotion_label": dominant_emotion(gt),
        "valence": valence,
        "arousal": 0,
        "confidence": 85,
    })


def _mosei_c1_input(transcript: str) -> str:
    return (f'Transcript of what the person said:\n"{transcript}"\n\n'
            "Infer the emotional state from the text.")


def _mosei_c2_input(facet_text: str) -> str:
    return (
        "Facial blendshape coefficients for the target person. These INPUT values "
        "are on a 0-1 scale (higher = stronger activation); your OUTPUT scores must "
        "follow the scales specified above (0-100 and -100..100):\n"
        f"{facet_text}\n\n"
        "Infer the emotional state from these facial signals."
    )


def build_mosei_fewshot(condition: str, k: int = 5, seed: int = 42) -> list[dict]:
    """Gera k exemplos do split de treino do MOSEI (C1=texto, C2=FACET)."""
    rng = random.Random(seed)
    train = list(mosei_loader.iter_samples("train"))
    rng.shuffle(train)

    examples: list[dict] = []
    for s in train:
        if len(examples) >= k:
            break
        if condition == "C1":
            if not s.transcript:
                continue
            inp = _mosei_c1_input(s.transcript)
        elif condition == "C2":
            if not s.facet:
                continue
            inp = _mosei_c2_input(facet_to_text(s.facet, CFG.data.top_n_blendshapes))
        else:
            continue
        examples.append({"input": inp, "output": _mosei_example_output(s.ground_truth)})
    return examples


def build_fewshot(dataset: str, condition: str, k: int, seed: int) -> list[dict]:
    """Dispatcher: monta exemplos few-shot conforme dataset/condição."""
    if k <= 0:
        return []
    if dataset == "omg" and condition == "C2":
        return build_omg_c2_fewshot(k, seed)
    if dataset == "mosei" and condition in ("C1", "C2"):
        return build_mosei_fewshot(condition, k, seed)
    return []
