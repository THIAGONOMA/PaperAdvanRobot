"""Testes de significância: McNemar (pareado) e bootstrap de intervalo de confiança."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """McNemar pareado entre dois classificadores nas mesmas amostras.

    Args:
        correct_a / correct_b: acertou (True) ou errou (False) por amostra.

    Returns:
        {'statistic': ..., 'pvalue': ...}
    """
    from statsmodels.stats.contingency_tables import mcnemar

    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum(a & ~b))   # A acerta, B erra
    n10 = int(np.sum(~a & b))   # A erra, B acerta
    table = [[0, n01], [n10, 0]]
    res = mcnemar(table, exact=(n01 + n10) < 25)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue),
            "n01": n01, "n10": n10}


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Intervalo de confiança por bootstrap para uma métrica arbitrária."""
    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    n = len(yt)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric_fn(yt[idx], yp[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"point": float(metric_fn(yt, yp)), "lo": float(lo), "hi": float(hi)}
