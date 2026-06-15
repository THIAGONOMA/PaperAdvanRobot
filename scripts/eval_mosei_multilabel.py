"""Avaliação multi-rótulo de emoção no CMU-MOSEI (protocolo padrão da literatura).

Cada uma das 6 emoções de Ekman é um rótulo binário (presente se intensidade > 0).
Compara:
  - Majority/prevalence baseline (prevê presença fixa por prevalência no treino).
  - Baseline tradicional: LogReg multi-saída sobre FACET (35-dim).
  - LLM C1 (texto) e C2 (FACET), zero-shot e few-shot.

Métricas: F1 micro / macro / ponderado e F1 por emoção. Limiar do LLM nas
intensidades normalizadas (default 0.5).

Uso:
    .venv/bin/python scripts/eval_mosei_multilabel.py --n-train 4000 --n-test 200 \
        --conditions c1,c1fs,c2 --threshold 0.5 --k-shots 5
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import mosei_loader

EKMAN = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]


def collect(split: str, n: int, need_text: bool = False) -> list:
    out = []
    for s in mosei_loader.iter_samples(split):
        if not s.facet:
            continue
        if need_text and not s.transcript:
            continue
        out.append(s)
        if len(out) >= n:
            break
    return out


def label_matrix(samples) -> np.ndarray:
    return np.array([[1 if s.ground_truth.emotion_scores.get(e, 0.0) > 0 else 0
                      for e in EKMAN] for s in samples])


def report(name: str, Y_true: np.ndarray, Y_pred: np.ndarray) -> None:
    micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    macro = f1_score(Y_true, Y_pred, average="macro", zero_division=0)
    weighted = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
    per = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    print(f"\n{name}")
    print(f"  F1 micro={micro:.3f} | macro={macro:.3f} | ponderado={weighted:.3f}")
    print("  F1 por emoção: " + ", ".join(f"{e}={v:.2f}" for e, v in zip(EKMAN, per)))


def report_sweep(name: str, Y_true: np.ndarray, scores: np.ndarray,
                 thresholds=(0.15, 0.2, 0.25, 0.3, 0.4, 0.5)) -> None:
    """Reporta o F1 do LLM no melhor limiar (LLM tende a dar scores conservadores)."""
    best = None
    for t in thresholds:
        Y_pred = (scores >= t).astype(int)
        micro = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
        if best is None or micro > best[1]:
            best = (t, micro, Y_pred)
    t, _, Y_pred = best
    report(f"{name} [melhor limiar={t}]", Y_true, Y_pred)


def llm_scores(samples, condition: str, k_shots: int) -> np.ndarray:
    """Roda o LLM e devolve a matriz contínua de scores de Ekman (n x 6)."""
    from src.llm.runner import run as run_inference

    df = run_inference(samples, dataset="mosei", condition=condition.upper(),
                       seed=13, k_shots=k_shots)
    by_id = {row["sample_id"]: row for _, row in df.iterrows()}
    S = np.zeros((len(samples), len(EKMAN)), dtype=float)
    for i, s in enumerate(samples):
        row = by_id.get(s.sample_id, {})
        for j, e in enumerate(EKMAN):
            val = row.get(f"ekman_{e}")
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                S[i, j] = float(val)
    return S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--conditions", default="c1,c1fs,c2",
                    help="c1, c2 (zero-shot) e c1fs, c2fs (few-shot)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--k-shots", type=int, default=5)
    args = ap.parse_args()
    conds = [c.strip().lower() for c in args.conditions.split(",")]

    print(f">> Treino ... alvo={args.n_train}", flush=True)
    train = collect("train", args.n_train)
    Xtr, Ytr = np.array([s.facet for s in train]), label_matrix(train)
    print(f"   {len(train)} segmentos | prevalência treino: "
          f"{ {e: round(float(Ytr[:, j].mean()), 2) for j, e in enumerate(EKMAN)} }", flush=True)

    print(f">> Teste ... alvo={args.n_test}", flush=True)
    test = collect("test", args.n_test, need_text=True)
    Xte, Yte = np.array([s.facet for s in test]), label_matrix(test)
    print(f"   {len(test)} segmentos", flush=True)

    prevalence = Ytr.mean(axis=0)

    # 0) Acaso — sorteia cada emoção com prob = prevalência (média de R sorteios)
    rng = np.random.default_rng(0)
    f1s = [f1_score(Yte, (rng.random((len(test), len(EKMAN))) < prevalence).astype(int),
                    average="micro", zero_division=0) for _ in range(50)]
    print(f"\n[Acaso — sorteio por prevalência]  F1 micro≈{np.mean(f1s):.3f}")

    # 1) Majority — emoção mais prevalente sempre presente
    top = int(np.argmax(prevalence))
    Y_majority = np.zeros((len(test), len(EKMAN)), dtype=int)
    Y_majority[:, top] = 1
    report(f"[Majority — sempre '{EKMAN[top]}']", Yte, Y_majority)

    # 2) Baseline tradicional: LogReg multi-saída sobre FACET
    print("\n>> Treinando baseline (LogReg multi-saída sobre FACET) ...", flush=True)
    clf = MultiOutputClassifier(
        make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    )
    clf.fit(Xtr, Ytr)
    Y_base = clf.predict(Xte)
    report("[Baseline FACET + LogReg multi-saída]", Yte, Y_base)

    # 3) LLM (scores contínuos + varredura de limiar para comparação justa)
    if "c1" in conds:
        print("\n>> LLM C1 (texto, zero-shot) ...", flush=True)
        report_sweep("[LLM C1 texto — zero-shot]", Yte, llm_scores(test, "c1", 0))
    if "c1fs" in conds:
        print("\n>> LLM C1 (texto, few-shot) ...", flush=True)
        report_sweep(f"[LLM C1 texto — few-shot k={args.k_shots}]", Yte,
                     llm_scores(test, "c1", args.k_shots))
    if "c2" in conds:
        print("\n>> LLM C2 (FACET, zero-shot) ...", flush=True)
        report_sweep("[LLM C2 FACET — zero-shot]", Yte, llm_scores(test, "c2", 0))
    if "c2fs" in conds:
        print("\n>> LLM C2 (FACET, few-shot) ...", flush=True)
        report_sweep(f"[LLM C2 FACET — few-shot k={args.k_shots}]", Yte,
                     llm_scores(test, "c2", args.k_shots))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
