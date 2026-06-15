"""Smoke test: roda algumas amostras do OMG contra o Gemma (vLLM) e o baseline.

Uso:
    .venv/bin/python scripts/smoke_omg.py --n 2 --condition C3
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback

from src.baseline.rule_engine import BaselineRuleEngine
from src.data import omg_loader
from src.features.extract import blendshapes_sequence_for, keyframe_for


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="número de amostras")
    ap.add_argument("--condition", default="C3", choices=["C2", "C3"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=1, help="pega 1 a cada N janelas")
    args = ap.parse_args()

    print(f">> Carregando {args.n} amostra(s) do OMG split={args.split} (stride={args.stride}) ...")
    it = itertools.islice(omg_loader.iter_samples(args.split, window_s=4), 0, None, args.stride)
    samples = list(itertools.islice(it, args.n))
    print(f">> {len(samples)} amostra(s) carregada(s).")
    for s in samples:
        print(f"   - {s.sample_id} | GT valence={s.ground_truth.valence:+.3f} "
              f"| t={s.frame_time:.1f}s | {s.video_path.split('/')[-1]}")

    print("\n>> Extraindo features e chamando o LLM ...")
    from src.llm.runner import run as run_inference

    df = run_inference(samples, dataset="omg", condition=args.condition,
                       seed=13, k_shots=0)

    print("\n=== RESULTADO LLM ===")
    cols = [c for c in ("sample_id", "pred_valence", "pred_dominant_emotion",
                        "pred_confidence", "gt_valence", "latency_ms", "fallback")
            if c in df.columns]
    print(df[cols].to_string(index=False))

    ek_cols = [c for c in df.columns if c.startswith("ekman_")]
    if ek_cols:
        print("\n=== EMOÇÕES EKMAN (LLM) ===")
        show = ["sample_id"] + ek_cols
        print(df[show].round(2).to_string(index=False))

    if "pred_valence" in df.columns and df["pred_valence"].notna().any():
        from src.eval.metrics import ccc
        d = df.dropna(subset=["pred_valence", "gt_valence"])
        if len(d) >= 2:
            print(f"\n>> CCC (preliminar, {len(d)} amostras): "
                  f"{ccc(d['gt_valence'], d['pred_valence']):.3f}")

    print("\n=== BASELINE (regras, multi-frame, mesmos blendshapes do LLM) ===")
    engine = BaselineRuleEngine()
    base_rows = []
    for s in samples:
        seq = blendshapes_sequence_for(s)
        if seq:
            out = engine.predict_sequence(seq)
            base_rows.append((s.ground_truth.valence, out["valence"]))
            print(f"   {s.sample_id}: valence={out['valence']:+.3f} "
                  f"(peak={out['valence_peak']:+.3f}) arousal={out['arousal']:+.3f} "
                  f"label={out['emotion_label']} | {out['n_frames']} frames | GT={s.ground_truth.valence:+.3f}")
        else:
            print(f"   {s.sample_id}: (sem rosto detectado)")

    if len(base_rows) >= 2:
        from src.eval.metrics import ccc
        gt = [g for g, _ in base_rows]
        pr = [p for _, p in base_rows]
        print(f"\n>> CCC baseline (preliminar, {len(base_rows)} amostras): {ccc(gt, pr):.3f}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
