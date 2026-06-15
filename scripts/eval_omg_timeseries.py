"""Avaliação OMG no protocolo oficial: CCC na série temporal contínua por vídeo.

Para cada vídeo selecionado, prediz a valência em pontos ao longo do tempo,
calcula o CCC entre a sequência predita e a anotada, e reporta a média entre
vídeos (estilo track Generalized/Personalized).

Uso:
    .venv/bin/python scripts/eval_omg_timeseries.py --videos 6 --points 20 --conditions c2,c3,baseline
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import defaultdict

import numpy as np

from src.baseline.rule_engine import BaselineRuleEngine
from src.data import omg_loader
from src.eval.metrics import ccc_per_video
from src.features.extract import blendshapes_sequence_for
from src.llm.runner import run as run_inference


def _video_of(sample_id: str) -> str:
    return sample_id.rsplit("_w", 1)[0]


def select(split: str, n_videos: int, points: int) -> list:
    """Seleciona n_videos (variando sujeitos/histórias) e 'points' janelas por vídeo, em ordem temporal."""
    groups: dict[str, list] = defaultdict(list)
    for s in omg_loader.iter_samples(split, window_s=4):
        groups[s.video_path].append(s)

    videos = sorted(groups)
    idx_v = sorted(set(np.linspace(0, len(videos) - 1, n_videos).astype(int)))
    chosen = [videos[i] for i in idx_v]

    out: list = []
    for v in chosen:
        ws = groups[v]
        idx = sorted(set(np.linspace(0, len(ws) - 1, points).astype(int)))
        out.extend(ws[i] for i in idx)
    return out


def records_from(samples, pred_by_id: dict) -> list[dict]:
    recs = []
    for s in samples:
        recs.append({
            "video": _video_of(s.sample_id),
            "order": s.start_time,
            "gt": s.ground_truth.valence,
            "pred": pred_by_id.get(s.sample_id),
        })
    return recs


def report(name: str, res: dict) -> None:
    print(f"\n=== {name} — CCC por vídeo (protocolo OMG) ===")
    for v, c in sorted(res["per_video"].items()):
        print(f"   {v.split('/')[-1]:28} CCC={c:+.3f}")
    print(f"   >> média entre vídeos: {res['mean']:+.3f} ± {res['std']:.3f} "
          f"(n_vídeos={res['n_videos']})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", default="c2,baseline")
    ap.add_argument("--k-shots", type=int, default=0, help="exemplos few-shot (C2)")
    args = ap.parse_args()
    conds = [c.strip().lower() for c in args.conditions.split(",")]
    k = args.k_shots

    samples = select(args.split, args.videos, args.points)
    n_vid = len({_video_of(s.sample_id) for s in samples})
    shot_tag = f"few-shot k={k}" if k > 0 else "zero-shot"
    print(f">> {len(samples)} janelas de {n_vid} vídeos (split={args.split}) | {shot_tag}", flush=True)

    if "c2" in conds:
        print("\n>> C2 (blendshapes multi-frame) ...", flush=True)
        df = run_inference(samples, dataset="omg", condition="C2", seed=13, k_shots=k)
        pred = dict(zip(df["sample_id"], df["pred_valence"])) if "pred_valence" in df else {}
        report(f"C2 (blendshapes, {shot_tag})", ccc_per_video(records_from(samples, pred)))

    if "c3" in conds:
        print("\n>> C3 (visão multimodal) ...", flush=True)
        df = run_inference(samples, dataset="omg", condition="C3", seed=13)
        pred = dict(zip(df["sample_id"], df["pred_valence"])) if "pred_valence" in df else {}
        report("C3 (visão)", ccc_per_video(records_from(samples, pred)))

    if "baseline" in conds:
        print("\n>> Baseline (regras multi-frame) ...", flush=True)
        engine = BaselineRuleEngine()
        pred = {}
        for s in samples:
            seq = blendshapes_sequence_for(s)
            pred[s.sample_id] = engine.predict_sequence(seq)["valence"] if seq else None
        report("Baseline (regras)", ccc_per_video(records_from(samples, pred)))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
