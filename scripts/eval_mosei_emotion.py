"""Avaliação de classificação de emoção no CMU-MOSEI (emoção dominante).

Compara:
  - Baseline tradicional: Regressão Logística treinada nas features FACET (35-dim).
  - LLM C1 (texto/transcrição) e C2 (FACET serializado em texto), zero-shot.

Métricas: acurácia, F1 ponderado, matriz de confusão e McNemar (LLM vs baseline).

Uso:
    .venv/bin/python scripts/eval_mosei_emotion.py --n-train 3000 --n-test 80 --conditions c1,c2
"""
from __future__ import annotations

import argparse
import itertools
import sys
import traceback

import numpy as np

from src.data import mosei_loader
from src.data.types import dominant_emotion
from src.eval.metrics import accuracy, confusion, weighted_f1
from src.eval.stats import mcnemar_test
from src.llm.runner import run as run_inference

LABELS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]


def collect(split: str, n: int, need_facet: bool = True, need_text: bool = False) -> list:
    out = []
    for s in mosei_loader.iter_samples(split):
        if need_facet and not s.facet:
            continue
        if need_text and not s.transcript:
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--conditions", default="c1,c2")
    args = ap.parse_args()
    conds = [c.strip().lower() for c in args.conditions.split(",")]

    print(f">> Coletando treino (FACET) ... alvo={args.n_train}", flush=True)
    train = collect("train", args.n_train, need_facet=True)
    Xtr = np.array([s.facet for s in train])
    ytr = [dominant_emotion(s.ground_truth) for s in train]
    print(f"   treino: {len(train)} segmentos | classes: "
          f"{ {c: ytr.count(c) for c in set(ytr)} }", flush=True)

    print(f">> Coletando teste ... alvo={args.n_test}", flush=True)
    test = collect("test", args.n_test, need_facet=True, need_text=True)
    yte = [dominant_emotion(s.ground_truth) for s in test]
    print(f"   teste: {len(test)} segmentos | distribuição GT: "
          f"{ {c: yte.count(c) for c in set(yte)} }", flush=True)

    # --- Baseline tradicional: LogReg sobre FACET ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    print(">> Treinando baseline (LogReg sobre FACET) ...", flush=True)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    Xte = np.array([s.facet for s in test])
    pred_base = list(clf.predict(Xte))

    results = {"baseline(FACET+LogReg)": pred_base}

    # --- LLM ---
    if "c1" in conds:
        print(">> LLM C1 (texto) ...", flush=True)
        df = run_inference(test, dataset="mosei", condition="C1", seed=13)
        pm = dict(zip(df["sample_id"], df.get("pred_emotion_label", [])))
        results["LLM C1 (texto)"] = [pm.get(s.sample_id, "neutral") for s in test]
    if "c2" in conds:
        print(">> LLM C2 (FACET) ...", flush=True)
        df = run_inference(test, dataset="mosei", condition="C2", seed=13)
        pm = dict(zip(df["sample_id"], df.get("pred_emotion_label", [])))
        results["LLM C2 (FACET)"] = [pm.get(s.sample_id, "neutral") for s in test]

    print("\n=== RESULTADOS (emoção dominante, MOSEI) ===")
    print(f"{'sistema':28} {'acurácia':>9} {'F1_pond':>9}")
    for name, pred in results.items():
        print(f"{name:28} {accuracy(yte, pred):>9.3f} {weighted_f1(yte, pred):>9.3f}")

    # McNemar LLM vs baseline
    print("\n=== McNemar (LLM vs baseline) ===")
    base_correct = [p == t for p, t in zip(pred_base, yte)]
    for name, pred in results.items():
        if name.startswith("baseline"):
            continue
        llm_correct = [p == t for p, t in zip(pred, yte)]
        res = mcnemar_test(base_correct, llm_correct)
        print(f"  {name:20}: p={res['pvalue']:.4f} | baseline-acerta/LLM-erra={res['n01']} "
              f"| LLM-acerta/baseline-erra={res['n10']}")

    # matriz de confusão da melhor condição LLM (se houver)
    present = [c for c in LABELS if c in set(yte) | set(pred_base)]
    print("\n=== Matriz de confusão — baseline (linhas=GT, colunas=pred) ===")
    print("labels:", present)
    print(confusion(yte, pred_base, present))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
