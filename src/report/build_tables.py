"""Agrega os resultados (3 seeds) e monta as tabelas finais do paper."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import CFG


def load_results(dataset: str, condition: str) -> pd.DataFrame:
    """Carrega e concatena os parquet das múltiplas seeds."""
    out_dir = Path(CFG.run.results_dir)
    files = sorted(out_dir.glob(f"{dataset}_{condition}_seed*.parquet"))
    if not files:
        raise FileNotFoundError(f"Sem resultados para {dataset}/{condition} em {out_dir}.")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def aggregate_metrics(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas por seed em média ± desvio."""
    raise NotImplementedError(
        "TODO: calcular CCC/F1 por seed (via src.eval) e agregar média ± desvio."
    )
