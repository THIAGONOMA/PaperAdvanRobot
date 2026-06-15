"""Splits oficiais e seleção de exemplos few-shot.

Os splits devem ser idênticos aos usados pelo baseline (a confirmar com a Léa).
Exemplos de few-shot são extraídos SOMENTE do split de treino para evitar
vazamento no teste.
"""
from __future__ import annotations

import random
from typing import Iterable

from .types import Sample


def few_shot_examples(train: Iterable[Sample], k: int, seed: int = 42) -> list[Sample]:
    """Seleciona k exemplos do split de treino de forma determinística."""
    pool = list(train)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:k]


def split_hash(samples: Iterable[Sample]) -> str:
    """Hash dos sample_ids do split para registrar reprodutibilidade."""
    import hashlib

    ids = ",".join(sorted(s.sample_id for s in samples))
    return hashlib.sha256(ids.encode()).hexdigest()[:12]
