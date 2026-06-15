"""Métricas de avaliação: CCC (OMG) e F1/acurácia (MOSEI)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def ccc(y_true, y_pred) -> float:
    """Concordance Correlation Coefficient (métrica oficial do OMG-Empathy)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    cov = np.cov(yt, yp, bias=True)[0, 1]
    denom = yt.var() + yp.var() + (yt.mean() - yp.mean()) ** 2
    return float(2 * cov / denom) if denom else 0.0


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def weighted_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def confusion(y_true, y_pred, labels: list[str]):
    return confusion_matrix(y_true, y_pred, labels=labels)


def ccc_per_video(
    records: list[dict], min_points: int = 3, min_std: float = 1e-3
) -> dict:
    """CCC calculado na série temporal contínua de cada vídeo (protocolo OMG).

    Args:
        records: lista de dicts com chaves 'video', 'order', 'gt', 'pred'.
        min_points: mínimo de pontos por vídeo para calcular CCC.
        min_std: variância mínima (em ambas as séries) para o CCC ser válido.

    Returns:
        {'per_video': {video: ccc}, 'mean': float, 'std': float, 'n_videos': int}
    """
    by_video: dict[str, list[dict]] = {}
    for r in records:
        if r.get("pred") is None or r.get("gt") is None:
            continue
        by_video.setdefault(r["video"], []).append(r)

    per_video: dict[str, float] = {}
    for video, rows in by_video.items():
        rows = sorted(rows, key=lambda x: x["order"])
        yt = np.asarray([x["gt"] for x in rows], dtype=float)
        yp = np.asarray([x["pred"] for x in rows], dtype=float)
        if len(yt) < min_points or yt.std() < min_std or yp.std() < min_std:
            continue
        per_video[video] = ccc(yt, yp)

    if not per_video:
        return {"per_video": {}, "mean": float("nan"), "std": float("nan"), "n_videos": 0}
    vals = np.array(list(per_video.values()))
    return {"per_video": per_video, "mean": float(vals.mean()),
            "std": float(vals.std()), "n_videos": len(per_video)}
