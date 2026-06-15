"""Motor baseline baseado em regras (replica a lógica do `face_blendshape`).

Mapeia os 52 blendshapes faciais para o espaço dimensional valência-arousal e,
a partir daí, para um rótulo categórico de emoção. Implementação inicial
aproximada — deve ser calibrada contra o módulo original da Léa.
"""
from __future__ import annotations

import math

# Mapa emoção -> (valência, arousal) conforme o README do face_blendshape.
EMOTION_VA: dict[str, tuple[float, float]] = {
    "happiness": (0.8, 0.6),
    "surprise": (0.1, 0.8),
    "anger": (-0.6, 0.7),
    "fear": (-0.7, 0.8),
    "sadness": (-0.7, -0.4),
    "disgust": (-0.6, 0.2),
    "contempt": (-0.5, 0.3),
    "neutral": (0.0, 0.0),
}

# Blendshapes que empurram cada dimensão (peso inicial = 1.0; calibrar depois).
_VALENCE_POS = ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft", "cheekSquintRight"]
_VALENCE_NEG = ["mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight"]
_AROUSAL_POS = ["eyeWideLeft", "eyeWideRight", "jawOpen", "browInnerUp"]
_AROUSAL_NEG = ["eyeSquintLeft", "eyeSquintRight", "mouthClose"]


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class BaselineRuleEngine:
    name = "baseline_rules"

    def predict(self, coefs: dict[str, float]) -> dict:
        valence, arousal = self.blendshape_to_va(coefs)
        return {
            "predictor_name": self.name,
            "valence": valence,
            "arousal": arousal,
            "emotion_label": self.va_to_label(valence, arousal),
            "confidence": 1.0,
        }

    def predict_sequence(self, seq: list[dict[str, float]]) -> dict:
        """Prediz a partir de uma sequência de frames (agrega V-A no tempo).

        Calcula V-A por frame e agrega: valência média (alinha com o rótulo de
        janela do OMG), arousal médio, e também o pico de valência (|v| máximo).
        """
        if not seq:
            raise ValueError("Sequência de blendshapes vazia.")
        vas = [self.blendshape_to_va(c) for c in seq]
        valences = [v for v, _ in vas]
        arousals = [a for _, a in vas]
        v_mean = sum(valences) / len(valences)
        a_mean = sum(arousals) / len(arousals)
        v_peak = max(valences, key=abs)
        return {
            "predictor_name": self.name,
            "valence": v_mean,
            "arousal": a_mean,
            "valence_peak": v_peak,
            "emotion_label": self.va_to_label(v_mean, a_mean),
            "confidence": 1.0,
            "n_frames": len(seq),
        }

    def blendshape_to_va(self, coefs: dict[str, float]) -> tuple[float, float]:
        """Extrai valência-arousal a partir dos coeficientes de blendshape."""
        valence = (sum(coefs.get(b, 0.0) for b in _VALENCE_POS)
                   - sum(coefs.get(b, 0.0) for b in _VALENCE_NEG))
        arousal = (sum(coefs.get(b, 0.0) for b in _AROUSAL_POS)
                   - sum(coefs.get(b, 0.0) for b in _AROUSAL_NEG))
        return _clip(valence), _clip(arousal)

    @staticmethod
    def va_to_label(valence: float, arousal: float) -> str:
        """Rótulo categórico = emoção mais próxima no espaço V-A (vizinho mais próximo)."""
        def dist(target: tuple[float, float]) -> float:
            return math.hypot(valence - target[0], arousal - target[1])

        return min(EMOTION_VA, key=lambda emo: dist(EMOTION_VA[emo]))
