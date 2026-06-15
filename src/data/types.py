"""Tipos compartilhados da camada de dados."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Dataset = Literal["omg", "mosei"]
Task = Literal["valence", "emotion"]


@dataclass
class GroundTruth:
    valence: Optional[float] = None          # OMG (contínuo)
    emotions: list[str] = field(default_factory=list)  # MOSEI (multi-rótulo Ekman)
    sentiment: Optional[float] = None
    emotion_scores: dict = field(default_factory=dict)  # MOSEI: intensidade por emoção Ekman


def dominant_emotion(gt: "GroundTruth", min_intensity: float = 1e-9) -> str:
    """Emoção dominante (maior intensidade) ou 'neutral' se todas ~0."""
    scores = gt.emotion_scores
    if not scores:
        return "neutral"
    top = max(scores, key=scores.get)
    return top if scores[top] > min_intensity else "neutral"


@dataclass
class Sample:
    sample_id: str
    dataset: Dataset
    split: str
    ground_truth: GroundTruth
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    transcript: Optional[str] = None
    blendshapes: Optional[dict[str, float]] = None   # apenas OMG
    facet: Optional[list[float]] = None              # apenas MOSEI
    image_path: Optional[str] = None                 # keyframe p/ condição C3
    video_path: Optional[str] = None                 # vídeo de origem (OMG)
    frame_time: Optional[float] = None               # instante representativo da janela (s)
