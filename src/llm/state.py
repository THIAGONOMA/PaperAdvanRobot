"""Estado do grafo LangGraph."""
from __future__ import annotations

from typing import Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    sample_id: str
    dataset: Literal["omg", "mosei"]
    task: Literal["valence", "emotion"]
    condition: Literal["C1", "C2", "C3"]
    k_shots: int
    features: dict
    messages: list[BaseMessage]
    prediction: Optional[dict]
    attempts: int
    max_attempts: int
    error: Optional[str]
    latency_ms: Optional[float]
