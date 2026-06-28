"""Computational cost and latency analysis for the paper.

Usage:
    PYTHONPATH=. python scripts/compute_cost_report.py

Output:
    results/cost_report.txt
    results/figures/latency_boxplot.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.dpi": DPI,
})

# ── Load parquets ───────────────────────────────────────────────────────────

FILES = {
    "OMG C2 (blendshapes)": "results/omg_C2_seed13.parquet",
    "OMG C3 (vision)":      "results/omg_C3_seed13.parquet",
    "MOSEI C1 (text)":      "results/mosei_C1_seed13.parquet",
    "MOSEI C2 (FACET)":     "results/mosei_C2_seed13.parquet",
}

# ── Model architecture specs ────────────────────────────────────────────────

MODEL_SPECS = {
    "model_name": "Gemma 4 26B-A4B-it",
    "model_id": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
    "architecture": "Mixture-of-Experts (MoE)",
    "total_params_B": 25.2,
    "active_params_B": 3.8,
    "layers": 30,
    "experts_total": 128,
    "experts_active": 8,
    "experts_shared": 1,
    "context_length": 262144,
    "vocab_size": 262144,
    "vision_encoder_params_M": 550,
    "quantization": "AWQ 4-bit (weight-only)",
    "memory_footprint_GB": 13.5,
    "serving_engine": "vLLM (PagedAttention)",
    "gpu": "NVIDIA L4 24 GB (GCP Spot VM)",
    "gpu_memory_GB": 24,
    "gpu_compute_tflops_fp16": 121,
    "temperature": 0,
    "max_tokens": 512,
}

# ── Token estimation per condition ──────────────────────────────────────────

SYSTEM_VALENCE_TOKENS = 180
SYSTEM_EMOTION_TOKENS = 230
BLENDSHAPE_LEGEND_TOKENS = 80

TOKEN_ESTIMATES = {
    "OMG C2 zero-shot": {
        "system_prompt": SYSTEM_VALENCE_TOKENS,
        "user_prompt": 280,
        "blendshape_legend": BLENDSHAPE_LEGEND_TOKENS,
        "images": 0,
        "few_shot_examples": 0,
        "output": 30,
        "description": "System + 15 blendshapes (mean/max/std) + legend",
    },
    "OMG C2 few-shot (k=3)": {
        "system_prompt": SYSTEM_VALENCE_TOKENS,
        "user_prompt": 280,
        "blendshape_legend": BLENDSHAPE_LEGEND_TOKENS,
        "images": 0,
        "few_shot_examples": 3 * 320,
        "output": 30,
        "description": "System + 3 few-shot pairs + blendshapes + legend",
    },
    "OMG C3 zero-shot": {
        "system_prompt": SYSTEM_VALENCE_TOKENS,
        "user_prompt": 40,
        "blendshape_legend": 0,
        "images": 3 * 1200,
        "few_shot_examples": 0,
        "output": 30,
        "description": "System + 3 JPEG keyframes (~1200 tokens each via vision encoder)",
    },
    "MOSEI C1 zero-shot": {
        "system_prompt": SYSTEM_EMOTION_TOKENS,
        "user_prompt": 80,
        "blendshape_legend": 0,
        "images": 0,
        "few_shot_examples": 0,
        "output": 80,
        "description": "System + transcript (~50-80 words)",
    },
    "MOSEI C1 few-shot (k=5)": {
        "system_prompt": SYSTEM_EMOTION_TOKENS,
        "user_prompt": 80,
        "blendshape_legend": 0,
        "images": 0,
        "few_shot_examples": 5 * 200,
        "output": 80,
        "description": "System + 5 few-shot pairs + transcript",
    },
    "MOSEI C2 zero-shot": {
        "system_prompt": SYSTEM_EMOTION_TOKENS,
        "user_prompt": 200,
        "blendshape_legend": BLENDSHAPE_LEGEND_TOKENS,
        "images": 0,
        "few_shot_examples": 0,
        "output": 80,
        "description": "System + 15 FACET features + legend",
    },
    "MOSEI C2 few-shot (k=5)": {
        "system_prompt": SYSTEM_EMOTION_TOKENS,
        "user_prompt": 200,
        "blendshape_legend": BLENDSHAPE_LEGEND_TOKENS,
        "images": 0,
        "few_shot_examples": 5 * 280,
        "output": 80,
        "description": "System + 5 few-shot pairs + FACET + legend",
    },
}


def total_tokens(est: dict) -> int:
    return sum(v for k, v in est.items() if isinstance(v, (int, float)) and k != "output") + est["output"]


def input_tokens(est: dict) -> int:
    return sum(v for k, v in est.items() if isinstance(v, (int, float)) and k != "output")


# ── Generate report ─────────────────────────────────────────────────────────

def generate_report():
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append("COMPUTATIONAL COST AND LATENCY REPORT")
    lines.append("=" * 80)

    lines.append("\n" + "-" * 80)
    lines.append("1. MODEL ARCHITECTURE")
    lines.append("-" * 80)
    for k, v in MODEL_SPECS.items():
        lines.append(f"  {k:35s}: {v}")

    lines.append("\n" + "-" * 80)
    lines.append("2. TOKEN ESTIMATES PER CONDITION")
    lines.append("-" * 80)
    lines.append(f"  {'Condition':<30s} {'Input':>8s} {'Output':>8s} {'Total':>8s}  Description")
    lines.append(f"  {'-'*30:s} {'-'*8:s} {'-'*8:s} {'-'*8:s}  {'-'*40}")
    for name, est in TOKEN_ESTIMATES.items():
        inp = input_tokens(est)
        out = est["output"]
        tot = total_tokens(est)
        lines.append(f"  {name:<30s} {inp:>8d} {out:>8d} {tot:>8d}  {est['description']}")

    lines.append("\n" + "-" * 80)
    lines.append("3. MEASURED INFERENCE LATENCY (from parquet files)")
    lines.append("-" * 80)
    lines.append(f"  {'Condition':<30s} {'N':>5s} {'Mean':>9s} {'Median':>9s} {'Std':>9s} {'P5':>9s} {'P95':>9s} {'Total':>10s}")
    lines.append(f"  {'-'*30:s} {'-'*5:s} {'-'*9:s} {'-'*9:s} {'-'*9:s} {'-'*9:s} {'-'*9:s} {'-'*10:s}")

    all_dfs = {}
    for label, path in FILES.items():
        df = pd.read_parquet(path)
        all_dfs[label] = df
        lat = df["latency_ms"].dropna()
        n = len(lat)
        total_s = lat.sum() / 1000
        lines.append(
            f"  {label:<30s} {n:>5d} {lat.mean():>8.0f}ms {lat.median():>8.0f}ms "
            f"{lat.std():>8.0f}ms {lat.quantile(0.05):>8.0f}ms {lat.quantile(0.95):>8.0f}ms "
            f"{total_s:>9.1f}s"
        )

    lines.append("\n" + "-" * 80)
    lines.append("4. THROUGHPUT AND TOTAL EXPERIMENT TIME")
    lines.append("-" * 80)

    total_calls = sum(len(df) for df in all_dfs.values())
    total_time_s = sum(df["latency_ms"].sum() for df in all_dfs.values()) / 1000
    total_time_min = total_time_s / 60

    lines.append(f"  Total LLM API calls:          {total_calls:,}")
    lines.append(f"  Total LLM inference time:      {total_time_s:,.0f}s ({total_time_min:.1f} min)")
    lines.append(f"  Average throughput:             {total_calls / total_time_s:.2f} calls/s")
    lines.append(f"  Average latency (all):          {total_time_s * 1000 / total_calls:.0f}ms/call")

    for label, df in all_dfs.items():
        lat = df["latency_ms"]
        est_key = None
        if "OMG" in label and "C2" in label:
            est_key = "OMG C2 zero-shot"
        elif "OMG" in label and "C3" in label:
            est_key = "OMG C3 zero-shot"
        elif "MOSEI" in label and "C1" in label:
            est_key = "MOSEI C1 zero-shot"
        elif "MOSEI" in label and "C2" in label:
            est_key = "MOSEI C2 zero-shot"

        if est_key and est_key in TOKEN_ESTIMATES:
            est = TOKEN_ESTIMATES[est_key]
            tot = total_tokens(est)
            tps = tot / (lat.mean() / 1000) if lat.mean() > 0 else 0
            lines.append(f"  {label:<30s}: ~{tot} tokens/call, ~{tps:.0f} tokens/s effective")

    lines.append("\n" + "-" * 80)
    lines.append("5. INFRASTRUCTURE COST ESTIMATE")
    lines.append("-" * 80)
    gcp_l4_spot_hr = 0.24
    lines.append(f"  GPU:                            NVIDIA L4 24 GB")
    lines.append(f"  GCP Spot VM price (approx):     ${gcp_l4_spot_hr:.2f}/hr")
    lines.append(f"  Total GPU time (LLM only):      {total_time_min:.1f} min")
    lines.append(f"  Estimated GPU cost (LLM only):  ${gcp_l4_spot_hr * total_time_min / 60:.3f}")
    lines.append(f"  Note: excludes feature extraction (MediaPipe), data loading, and idle time")

    lines.append("\n" + "-" * 80)
    lines.append("6. COMPARISON: LLM vs BASELINE COMPUTATIONAL REQUIREMENTS")
    lines.append("-" * 80)
    lines.append(f"  {'Metric':<35s} {'LLM (Gemma 4)':>20s} {'Baseline (Rules/LogReg)':>25s}")
    lines.append(f"  {'-'*35:s} {'-'*20:s} {'-'*25:s}")
    lines.append(f"  {'GPU required':35s} {'Yes (L4 24 GB)':>20s} {'No (CPU only)':>25s}")
    lines.append(f"  {'Model memory':35s} {'~13.5 GB (AWQ 4-bit)':>20s} {'< 1 MB':>25s}")
    lines.append(f"  {'Active parameters':35s} {'3.8 B per token':>20s} {'< 1 K (rules/weights)':>25s}")
    lines.append(f"  {'Latency per sample (text C1)':35s} {'~1,828 ms':>20s} {'< 1 ms':>25s}")
    lines.append(f"  {'Latency per sample (vision C3)':35s} {'~4,553 ms':>20s} {'~15 ms (MediaPipe)':>25s}")
    lines.append(f"  {'Latency per sample (features C2)':35s} {'~1,035 ms':>20s} {'< 1 ms':>25s}")
    lines.append(f"  {'Real-time capable (30 fps)':35s} {'No':>20s} {'Yes':>25s}")
    lines.append(f"  {'Requires internet/API':35s} {'Yes (or local GPU)':>20s} {'No':>25s}")
    lines.append(f"  {'Energy per inference (est.)':35s} {'~0.3–1.5 Wh':>20s} {'~0.001 Wh':>25s}")

    lines.append("\n" + "-" * 80)
    lines.append("7. VISION CONDITION (C3) — IMAGE PROCESSING DETAIL")
    lines.append("-" * 80)
    lines.append(f"  Images per window:              3 JPEG keyframes (4s window)")
    lines.append(f"  Typical image size:             ~30–60 KB per JPEG (640px crop)")
    lines.append(f"  Vision encoder:                 SigLIP (~550M params, frozen)")
    lines.append(f"  Est. vision tokens per image:   ~1,200 tokens")
    lines.append(f"  Est. total visual tokens:       ~3,600 tokens (3 images)")
    lines.append(f"  This explains the 4.4x latency increase vs text-only conditions")

    report = "\n".join(lines) + "\n"
    Path("results/cost_report.txt").write_text(report)
    print(report)
    return all_dfs


def plot_latency_boxplot(all_dfs: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    labels = list(all_dfs.keys())
    data = [df["latency_ms"].dropna().values for df in all_dfs.values()]

    colors = ["#7AB8F5", "#4A90D9", "#7AB8F5", "#4A90D9"]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.5},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.5})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    for i, (d, label) in enumerate(zip(data, labels)):
        median = np.median(d)
        ax.text(i + 1, median + 100, f"{median:.0f}ms",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Latency per Sample (ms)")
    ax.set_xlabel("Condition")
    ax.set_title("LLM Inference Latency Distribution per Condition")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGURES_DIR / "latency_boxplot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    dfs = generate_report()
    plot_latency_boxplot(dfs)
    print("\nDone — results/cost_report.txt and results/figures/latency_boxplot.png")
