"""Testes de fumaça que rodam sem dependências pesadas (sem LLM/MediaPipe)."""
from __future__ import annotations

from src.baseline.rule_engine import BaselineRuleEngine
from src.eval.metrics import accuracy, ccc, weighted_f1
from src.features.serialize import blendshapes_to_text


def test_ccc_perfect_correlation():
    y = [0.1, 0.2, 0.3, 0.4]
    assert abs(ccc(y, y) - 1.0) < 1e-9


def test_baseline_smile_is_positive_valence():
    engine = BaselineRuleEngine()
    out = engine.predict({"mouthSmileLeft": 0.9, "mouthSmileRight": 0.9})
    assert out["valence"] > 0
    assert out["emotion_label"] in {"happiness", "surprise", "neutral"}


def test_blendshapes_to_text_topn():
    text = blendshapes_to_text({"a": 0.1, "b": 0.9, "c": 0.5}, top_n=2)
    assert "b: 0.90" in text and "a:" not in text


def test_classification_metrics():
    yt = ["happiness", "sadness", "anger"]
    yp = ["happiness", "sadness", "neutral"]
    assert 0.0 <= weighted_f1(yt, yp) <= 1.0
    assert abs(accuracy(yt, yp) - 2 / 3) < 1e-9
