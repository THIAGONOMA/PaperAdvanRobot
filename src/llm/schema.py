"""Schemas Pydantic para a saída estruturada do LLM (guided decoding)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]


class EkmanScores(BaseModel):
    """Intensidade (0-100) de cada uma das 6 emoções básicas de Ekman."""

    happiness: int = Field(ge=0, le=100)
    sadness: int = Field(ge=0, le=100)
    anger: int = Field(ge=0, le=100)
    fear: int = Field(ge=0, le=100)
    disgust: int = Field(ge=0, le=100)
    surprise: int = Field(ge=0, le=100)


class EmotionPrediction(BaseModel):
    """Saída estruturada do LLM para classificação (MOSEI).

    Escalas inteiras (0-100 e -100..100) para granularidade fina; normalizadas
    para [-1,1] / [0,1] no parsing.
    """

    ekman: EkmanScores
    emotion_label: Literal[
        "happiness", "sadness", "anger", "fear", "disgust", "surprise", "neutral"
    ]
    valence: int = Field(ge=-100, le=100)
    arousal: int = Field(ge=-100, le=100)
    confidence: int = Field(ge=0, le=100)


class ValencePrediction(BaseModel):
    """Saída estruturada do LLM para regressão de valência (OMG).

    Foco apenas na valência/arousal (a tarefa do OMG) — sem decomposição de
    Ekman, que não é avaliada neste dataset. Escala inteira -100..100
    (granularidade fina), normalizada para [-1,1] no parsing.
    """

    valence: int = Field(ge=-100, le=100)
    arousal: int = Field(ge=-100, le=100)
    confidence: int = Field(ge=0, le=100)
