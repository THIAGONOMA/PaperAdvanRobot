"""Runner: itera o split de teste, invoca o grafo e persiste os resultados."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import CFG
from ..data.types import Sample, Task
from ..features.extract import blendshapes_sequence_for, keyframes_for
from ..features.serialize import (
    blendshapes_sequence_to_text,
    facet_to_text,
    image_to_base64,
)
from .graph import build_graph
from .state import GraphState

TASK_BY_DATASET: dict[str, Task] = {"omg": "valence", "mosei": "emotion"}


def _features_for(sample: Sample, condition: str) -> dict:
    """Monta o dicionário de features conforme a condição de entrada."""
    feats: dict = {}
    if condition == "C1":
        feats["transcript"] = sample.transcript or ""
    elif condition == "C2":
        seq = blendshapes_sequence_for(sample)  # multi-frame (OMG) se houver vídeo
        if seq:
            feats["blendshapes_text"] = blendshapes_sequence_to_text(
                seq, CFG.data.top_n_blendshapes
            )
        elif sample.facet:
            feats["blendshapes_text"] = facet_to_text(
                sample.facet, CFG.data.top_n_blendshapes
            )
        else:
            raise ValueError(f"C2 sem blendshapes/FACET para {sample.sample_id}.")
    elif condition == "C3":
        paths = keyframes_for(sample)
        feats["images_b64"] = [image_to_base64(p) for p in paths]
    return feats


def run(
    samples: Iterable[Sample],
    dataset: str,
    condition: str,
    seed: int,
    k_shots: int = 0,
) -> pd.DataFrame:
    """Executa o grafo sobre todas as amostras e retorna um DataFrame de predições."""
    graph = build_graph()
    task = TASK_BY_DATASET[dataset]
    rows: list[dict] = []

    fewshot_examples: list[dict] = []
    if k_shots > 0:
        from .fewshot import build_fewshot
        fewshot_examples = build_fewshot(dataset, condition, k_shots, seed)
        if fewshot_examples:
            print(f">> {len(fewshot_examples)} exemplos few-shot do split de treino", flush=True)

    for i, sample in enumerate(samples, 1):
        print(f"   [{i}] {sample.sample_id} ...", flush=True)
        try:
            features = _features_for(sample, condition)
            if fewshot_examples:
                features["few_shot"] = fewshot_examples
        except Exception as exc:  # ex.: nenhum rosto detectado na janela
            rows.append({
                "sample_id": sample.sample_id, "dataset": dataset,
                "condition": condition, "seed": seed, "latency_ms": None,
                "fallback": True, "feature_error": str(exc)[:120],
                "gt_valence": sample.ground_truth.valence,
                "gt_emotions": ",".join(sample.ground_truth.emotions),
                "gt_sentiment": sample.ground_truth.sentiment,
            })
            continue
        state: GraphState = {
            "sample_id": sample.sample_id,
            "dataset": dataset,  # type: ignore[typeddict-item]
            "task": task,
            "condition": condition,  # type: ignore[typeddict-item]
            "k_shots": k_shots,
            "features": features,
            "attempts": 0,
            "max_attempts": CFG.run.max_attempts,
        }
        result = graph.invoke(state)
        pred = dict(result.get("prediction") or {})
        ekman = pred.pop("ekman", None)
        row = {
            "sample_id": sample.sample_id,
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "latency_ms": result.get("latency_ms"),
            "fallback": pred.pop("fallback", False),
            **{f"pred_{k}": v for k, v in pred.items()},
            "gt_valence": sample.ground_truth.valence,
            "gt_emotions": ",".join(sample.ground_truth.emotions),
            "gt_sentiment": sample.ground_truth.sentiment,
        }
        if isinstance(ekman, dict):
            row.update({f"ekman_{k}": v for k, v in ekman.items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(CFG.run.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{dataset}_{condition}_seed{seed}.parquet", index=False)
    return df
