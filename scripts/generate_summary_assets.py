"""Gera testes estatísticos e gráficos para o summary.md do paper.

Uso:
    PYTHONPATH=. python scripts/generate_summary_assets.py

Saída:
    results/figures/omg_ccc_conditions.png
    results/figures/omg_ccc_per_video.png
    results/figures/mosei_f1_conditions.png
    results/figures/mosei_f1_per_emotion.png
    results/stats_report.txt
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "llm": "#4A90D9",
    "llm_light": "#7AB8F5",
    "baseline": "#E07B54",
    "baseline_light": "#F0A882",
    "weak": "#AAAAAA",
}

# ── OMG data (CCC per video, 15 videos) ────────────────────────────────────

VIDEOS = [
    "S10_St3", "S10_St7", "S1_St6", "S2_St3", "S2_St7",
    "S3_St6", "S4_St3", "S4_St7", "S5_St6", "S6_St3",
    "S6_St7", "S7_St6", "S8_St3", "S8_St7", "S9_St7",
]

CCC_C2_ZERO = np.array([
    -0.054, 0.002, -0.199, 0.338, 0.224,
    0.135, 0.343, -0.032, 0.004, 0.320,
    0.093, 0.009, 0.011, 0.257, -0.105,
])

CCC_C3_ZERO = np.array([
    0.201, 0.108, -0.027, 0.387, 0.312,
    0.019, 0.254, 0.248, 0.094, 0.345,
    -0.048, 0.114, 0.081, 0.334, -0.059,
])

CCC_C2_FEW = np.array([
    0.202, 0.041, -0.116, 0.299, 0.213,
    0.168, 0.345, -0.035, 0.022, 0.450,
    0.071, 0.006, -0.002, 0.317, -0.069,
])

CCC_BASELINE = np.array([
    0.116, 0.087, -0.174, 0.328, 0.258,
    0.240, 0.490, -0.025, 0.027, 0.565,
    -0.031, -0.010, 0.026, 0.386, -0.068,
])

# ── MOSEI data ──────────────────────────────────────────────────────────────

MOSEI_SYSTEMS = ["Acaso", "Majority", "LogReg\n(FACET)", "LLM C1\nzero", "LLM C1\nfew-shot", "LLM C2\nzero", "LLM C2\nfew-shot"]
MOSEI_MICRO = [0.313, 0.391, 0.376, 0.482, 0.499, 0.111, 0.289]
MOSEI_MACRO = [np.nan, 0.108, 0.333, 0.416, 0.416, 0.057, 0.130]
MOSEI_WEIGHTED = [np.nan, 0.214, 0.423, 0.493, 0.498, 0.096, 0.213]

EMOTIONS = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
F1_LLM_C1_FS = [0.57, 0.38, 0.52, 0.14, 0.63, 0.25]
F1_BASELINE   = [0.60, 0.37, 0.25, 0.07, 0.49, 0.21]
F1_LLM_C1_ZS  = [0.56, 0.40, 0.48, 0.16, 0.63, 0.26]

# ── Statistical tests ───────────────────────────────────────────────────────

def bootstrap_ci(data: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    means = np.array([rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0


def run_stats(report_file: Path):
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("TESTES ESTATÍSTICOS — OMG-Empathy (CCC por vídeo, n=15)")
    lines.append("=" * 70)

    pairs = [
        ("C3 (visão) vs Baseline", CCC_C3_ZERO, CCC_BASELINE),
        ("C3 (visão) vs C2 zero-shot", CCC_C3_ZERO, CCC_C2_ZERO),
        ("C2 few-shot vs C2 zero-shot", CCC_C2_FEW, CCC_C2_ZERO),
        ("C2 few-shot vs Baseline", CCC_C2_FEW, CCC_BASELINE),
        ("Baseline vs C2 zero-shot", CCC_BASELINE, CCC_C2_ZERO),
    ]

    for label, a, b in pairs:
        diff = a - b
        stat_w, p_w = sp_stats.wilcoxon(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        lines.append(f"\n--- {label} ---")
        lines.append(f"  Δ média  = {diff.mean():+.4f} ± {diff.std(ddof=1):.4f}")
        lines.append(f"  Wilcoxon signed-rank: W={stat_w:.1f}, p={p_w:.4f}")
        lines.append(f"  Cohen's d (pareado)  = {d:+.3f}")
        if p_w < 0.05:
            lines.append(f"  → SIGNIFICATIVO (p < 0.05)")
        else:
            lines.append(f"  → Não significativo (p ≥ 0.05)")

    lines.append("\n" + "=" * 70)
    lines.append("BOOTSTRAP CI 95% — CCC médio por condição (10.000 reamostragens)")
    lines.append("=" * 70)

    for name, data in [("C2 zero-shot", CCC_C2_ZERO), ("C2 few-shot k=3", CCC_C2_FEW),
                       ("C3 visão zero-shot", CCC_C3_ZERO), ("Baseline regras", CCC_BASELINE)]:
        lo, hi = bootstrap_ci(data)
        lines.append(f"  {name:25s}: média={data.mean():+.3f}  IC95%=[{lo:+.3f}, {hi:+.3f}]")

    lines.append("\n" + "=" * 70)
    lines.append("EFFECT SIZE — Cohen's d pareado (> 0 = 1º melhor que 2º)")
    lines.append("=" * 70)
    for label, a, b in pairs:
        d = cohens_d(a, b)
        mag = "grande" if abs(d) >= 0.8 else "médio" if abs(d) >= 0.5 else "pequeno"
        lines.append(f"  {label:40s}: d={d:+.3f} ({mag})")

    report = "\n".join(lines) + "\n"
    report_file.write_text(report)
    print(report)
    return report


# ── Charts ──────────────────────────────────────────────────────────────────

def plot_omg_ccc_conditions():
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["C2\nzero-shot", "C2\nfew-shot k=3", "C3\nvisão zero-shot", "Baseline\nregras"]
    means = [CCC_C2_ZERO.mean(), CCC_C2_FEW.mean(), CCC_C3_ZERO.mean(), CCC_BASELINE.mean()]
    stds = [CCC_C2_ZERO.std(), CCC_C2_FEW.std(), CCC_C3_ZERO.std(), CCC_BASELINE.std()]
    colors = [PALETTE["llm_light"], PALETTE["llm"], PALETTE["llm"], PALETTE["baseline"]]

    cis = [bootstrap_ci(d) for d in [CCC_C2_ZERO, CCC_C2_FEW, CCC_C3_ZERO, CCC_BASELINE]]
    err_lo = [m - ci[0] for m, ci in zip(means, cis)]
    err_hi = [ci[1] - m for m, ci in zip(means, cis)]

    bars = ax.bar(conditions, means, yerr=[err_lo, err_hi], capsize=6,
                  color=colors, edgecolor="white", linewidth=1.2, width=0.6)

    ax.set_ylabel("CCC médio (± IC 95% bootstrap)", fontsize=12)
    ax.set_title("OMG-Empathy — CCC por Condição (15 vídeos de teste)", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.05, 0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{m:+.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "omg_ccc_conditions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'omg_ccc_conditions.png'}")


def plot_omg_ccc_per_video():
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(VIDEOS))
    w = 0.2

    ax.bar(x - 1.5 * w, CCC_C2_ZERO, w, label="C2 zero-shot", color=PALETTE["weak"], edgecolor="white")
    ax.bar(x - 0.5 * w, CCC_C2_FEW, w, label="C2 few-shot k=3", color=PALETTE["llm_light"], edgecolor="white")
    ax.bar(x + 0.5 * w, CCC_C3_ZERO, w, label="C3 visão", color=PALETTE["llm"], edgecolor="white")
    ax.bar(x + 1.5 * w, CCC_BASELINE, w, label="Baseline regras", color=PALETTE["baseline"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(VIDEOS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("CCC", fontsize=12)
    ax.set_title("OMG-Empathy — CCC por Vídeo e Condição", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "omg_ccc_per_video.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'omg_ccc_per_video.png'}")


def plot_mosei_f1_conditions():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    systems = MOSEI_SYSTEMS
    x = np.arange(len(systems))
    w = 0.25

    ax.bar(x - w, MOSEI_MICRO, w, label="F1 micro", color=PALETTE["llm_light"], edgecolor="white")
    macro_vals = [v if not np.isnan(v) else 0 for v in MOSEI_MACRO]
    ax.bar(x, macro_vals, w, label="F1 macro", color=PALETTE["llm"], edgecolor="white")
    weighted_vals = [v if not np.isnan(v) else 0 for v in MOSEI_WEIGHTED]
    ax.bar(x + w, weighted_vals, w, label="F1 ponderado", color=PALETTE["baseline"], edgecolor="white")

    for i, (mi, ma, we) in enumerate(zip(MOSEI_MICRO, MOSEI_MACRO, MOSEI_WEIGHTED)):
        if not np.isnan(ma):
            ax.text(i, ma + 0.015, f"{ma:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("CMU-MOSEI — F1 Multi-rótulo por Sistema", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mosei_f1_conditions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'mosei_f1_conditions.png'}")


def plot_mosei_f1_per_emotion():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(EMOTIONS))
    w = 0.25

    ax.bar(x - w, F1_BASELINE, w, label="Baseline (LogReg FACET)", color=PALETTE["baseline"], edgecolor="white")
    ax.bar(x, F1_LLM_C1_ZS, w, label="LLM C1 zero-shot", color=PALETTE["llm_light"], edgecolor="white")
    ax.bar(x + w, F1_LLM_C1_FS, w, label="LLM C1 few-shot k=5", color=PALETTE["llm"], edgecolor="white")

    for i in range(len(EMOTIONS)):
        best = max(F1_BASELINE[i], F1_LLM_C1_ZS[i], F1_LLM_C1_FS[i])
        vals = [F1_BASELINE[i], F1_LLM_C1_ZS[i], F1_LLM_C1_FS[i]]
        offsets = [x[i] - w, x[i], x[i] + w]
        for v, o in zip(vals, offsets):
            if v == best:
                ax.text(o, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], fontsize=10)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_title("CMU-MOSEI — F1 por Emoção: LLM-texto vs. Baseline", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mosei_f1_per_emotion.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'mosei_f1_per_emotion.png'}")


if __name__ == "__main__":
    print("=== Testes estatísticos ===")
    run_stats(Path("results/stats_report.txt"))

    print("\n=== Gerando gráficos ===")
    plot_omg_ccc_conditions()
    plot_omg_ccc_per_video()
    plot_mosei_f1_conditions()
    plot_mosei_f1_per_emotion()

    print("\n✓ Tudo gerado em results/figures/ e results/stats_report.txt")
