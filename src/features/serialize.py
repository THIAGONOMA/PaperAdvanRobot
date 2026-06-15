"""Serialização de features para texto/imagem usados nos prompts do LLM."""
from __future__ import annotations

import base64
from pathlib import Path


def blendshapes_to_text(coefs: dict[str, float], top_n: int = 15) -> str:
    """Serializa os blendshapes mais salientes para o prompt (condição C2)."""
    top = sorted(coefs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return "\n".join(f"- {name}: {score:.2f}" for name, score in top)


def aggregate_sequence(seq: list[dict[str, float]]) -> dict[str, tuple[float, float, float]]:
    """Agrega uma sequência de blendshapes em (média, máximo, desvio) por coeficiente."""
    import numpy as np

    names = set()
    for f in seq:
        names.update(f.keys())
    stats: dict[str, tuple[float, float, float]] = {}
    for name in names:
        vals = np.array([f.get(name, 0.0) for f in seq], dtype=float)
        stats[name] = (float(vals.mean()), float(vals.max()), float(vals.std()))
    return stats


def blendshapes_sequence_to_text(seq: list[dict[str, float]], top_n: int = 15) -> str:
    """Descreve a janela temporal: média/máx/desvio dos blendshapes de maior pico (condição C2)."""
    stats = aggregate_sequence(seq)
    top = sorted(stats.items(), key=lambda kv: kv[1][1], reverse=True)[:top_n]  # ordena por pico
    lines = [f"- {name}: mean {m:.2f}, max {mx:.2f}, std {sd:.2f}"
             for name, (m, mx, sd) in top]
    return f"(aggregated over {len(seq)} frames)\n" + "\n".join(lines)


def facet_to_text(facet: list[float], top_n: int = 15) -> str:
    """Serializa as features FACET do MOSEI (proxy facial p/ condição C2)."""
    pairs = [(f"facet_{i}", v) for i, v in enumerate(facet)]
    top = sorted(pairs, key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return "\n".join(f"- {name}: {score:.3f}" for name, score in top)


def image_to_base64(image_path: str | Path) -> str:
    """Codifica uma imagem JPEG em base64 para envio multimodal (condição C3)."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("ascii")
