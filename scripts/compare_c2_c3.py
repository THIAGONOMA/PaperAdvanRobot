"""Compara C2 (blendshapes), C3 (visão multimodal) e baseline no OMG.

Amostragem variada: janelas espalhadas por vários sujeitos/histórias.
Reporta CCC e intervalo de confiança (bootstrap) por condição.

Uso:
    .venv/bin/python scripts/compare_c2_c3.py --n 40 --per-video 3
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import defaultdict

import numpy as np

from src.baseline.rule_engine import BaselineRuleEngine
from src.data import omg_loader
from src.eval.metrics import ccc
from src.eval.stats import bootstrap_ci
from src.features.extract import blendshapes_sequence_for
from src.llm.runner import run as run_inference


def varied_samples(split: str, n_total: int, per_video: int) -> list:
    """Seleciona janelas espalhadas por vários vídeos (sujeitos/histórias)."""
    groups: dict[str, list] = defaultdict(list)
    for s in omg_loader.iter_samples(split, window_s=4):
        groups[s.video_path].append(s)

    # de cada vídeo, pega 'per_video' janelas igualmente espaçadas
    picked: list[list] = []
    for _, windows in sorted(groups.items()):
        idxs = sorted(set(np.linspace(0, len(windows) - 1, per_video).astype(int)))
        picked.append([windows[i] for i in idxs])

    # round-robin entre vídeos até atingir n_total (maximiza variedade de sujeitos)
    result: list = []
    depth = 0
    max_depth = max(len(p) for p in picked) if picked else 0
    while len(result) < n_total and depth < max_depth:
        for lst in picked:
            if depth < len(lst):
                result.append(lst[depth])
                if len(result) >= n_total:
                    break
        depth += 1
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--per-video", type=int, default=3)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    samples = varied_samples(args.split, args.n, args.per_video)
    n_videos = len({s.video_path for s in samples})
    print(f">> {len(samples)} amostras de {n_videos} vídeos (split={args.split})\n", flush=True)

    print(">> Rodando C2 (blendshapes multi-frame) ...", flush=True)
    df_c2 = run_inference(samples, dataset="omg", condition="C2", seed=13)
    print(">> Rodando C3 (visão multimodal, multi-frame) ...", flush=True)
    df_c3 = run_inference(samples, dataset="omg", condition="C3", seed=13)

    print(">> Baseline (regras, multi-frame) ...", flush=True)
    engine = BaselineRuleEngine()
    base = {}
    for s in samples:
        seq = blendshapes_sequence_for(s)
        base[s.sample_id] = engine.predict_sequence(seq)["valence"] if seq else None

    c2 = dict(zip(df_c2["sample_id"], df_c2["pred_valence"]))
    c3 = dict(zip(df_c3["sample_id"], df_c3["pred_valence"]))
    fb_c2 = dict(zip(df_c2["sample_id"], df_c2["fallback"]))
    fb_c3 = dict(zip(df_c3["sample_id"], df_c3["fallback"]))
    gt = {s.sample_id: s.ground_truth.valence for s in samples}
    ids = [s.sample_id for s in samples]

    def arrays(pred: dict):
        pairs = [(gt[i], pred[i]) for i in ids
                 if pred.get(i) is not None and not (isinstance(pred[i], float) and np.isnan(pred[i]))]
        return ([a for a, _ in pairs], [b for _, b in pairs])

    print("\n=== RESUMO ===")
    print(f"  fallbacks C2: {sum(bool(v) for v in fb_c2.values())}/{len(ids)} | "
          f"C3: {sum(bool(v) for v in fb_c3.values())}/{len(ids)}")

    print("\n=== CCC + IC95% (bootstrap) ===")
    for name, pred in [("C2 (blendshapes)", c2), ("C3 (visao)", c3), ("Baseline (regras)", base)]:
        yt, yp = arrays(pred)
        if len(yt) >= 2:
            ci = bootstrap_ci(yt, yp, ccc, n_boot=2000)
            print(f"  {name:20}: CCC={ci['point']:+.3f}  IC95%=[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  (n={len(yt)})")
        else:
            print(f"  {name:20}: amostras insuficientes")

    # salva CSV consolidado
    import pandas as pd
    rows = [{"sample_id": i, "gt": gt[i], "c2": c2.get(i), "c3": c3.get(i), "baseline": base.get(i)}
            for i in ids]
    out = "results/compare_c2_c3.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n>> resultados salvos em {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
