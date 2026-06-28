"""Generate statistical tests and publication-quality figures.

Usage:
    PYTHONPATH=. python scripts/generate_summary_assets.py

Output:
    results/figures/omg_ccc_conditions.png
    results/figures/omg_ccc_per_video.png
    results/figures/mosei_f1_conditions.png
    results/figures/mosei_f1_per_emotion.png
    results/stats_report.txt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": DPI,
})

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

MOSEI_SYSTEMS = [
    "Random\nChance", "Majority", "LogReg\n(FACET)",
    "LLM C1\nzero-shot", "LLM C1\nfew-shot",
    "LLM C2\nzero-shot", "LLM C2\nfew-shot",
]
MOSEI_MICRO    = [0.313, 0.391, 0.376, 0.482, 0.499, 0.111, 0.289]
MOSEI_MACRO    = [np.nan, 0.108, 0.333, 0.416, 0.416, 0.057, 0.130]
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


def _load_mosei_per_sample():
    """Load MOSEI parquets and compute per-sample correctness for multi-label."""
    import pandas as pd

    EKMAN = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    THRESHOLD_LLM = 0.15

    c1 = pd.read_parquet("results/mosei_C1_seed13.parquet")
    c2 = pd.read_parquet("results/mosei_C2_seed13.parquet")

    def _gt_set(gt_str: str) -> set[str]:
        if not gt_str or gt_str == "neutral":
            return set()
        return set(gt_str.split(","))

    def _pred_set_llm(row, threshold: float) -> set[str]:
        return {e for e in EKMAN if row.get(f"ekman_{e}", 0) >= threshold}

    def _sample_f1(pred: set, gt: set) -> float:
        if not pred and not gt:
            return 1.0
        if not pred or not gt:
            return 0.0
        tp = len(pred & gt)
        prec = tp / len(pred)
        rec = tp / len(gt)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    gt_sets = [_gt_set(s) for s in c1["gt_emotions"]]
    c1_pred = [_pred_set_llm(c1.iloc[i], THRESHOLD_LLM) for i in range(len(c1))]
    c2_pred = [_pred_set_llm(c2.iloc[i], THRESHOLD_LLM) for i in range(len(c2))]

    c1_f1 = np.array([_sample_f1(p, g) for p, g in zip(c1_pred, gt_sets)])
    c2_f1 = np.array([_sample_f1(p, g) for p, g in zip(c2_pred, gt_sets)])

    c1_correct = np.array([int(p == g) for p, g in zip(c1_pred, gt_sets)])
    c2_correct = np.array([int(p == g) for p, g in zip(c2_pred, gt_sets)])

    return c1_f1, c2_f1, c1_correct, c2_correct, gt_sets


def run_stats(report_file: Path):
    lines: list[str] = []

    # ══════════════════════════════════════════════════════════════════════
    # SECTION A: OMG-Empathy — Pairwise Wilcoxon
    # ══════════════════════════════════════════════════════════════════════
    lines.append("=" * 70)
    lines.append("A. PAIRWISE WILCOXON — OMG-Empathy (CCC per video, n=15)")
    lines.append("=" * 70)

    pairs = [
        ("C3 (vision) vs Baseline", CCC_C3_ZERO, CCC_BASELINE),
        ("C3 (vision) vs C2 zero-shot", CCC_C3_ZERO, CCC_C2_ZERO),
        ("C2 few-shot vs C2 zero-shot", CCC_C2_FEW, CCC_C2_ZERO),
        ("C2 few-shot vs Baseline", CCC_C2_FEW, CCC_BASELINE),
        ("Baseline vs C2 zero-shot", CCC_BASELINE, CCC_C2_ZERO),
    ]

    for label, a, b in pairs:
        diff = a - b
        stat_w, p_w = sp_stats.wilcoxon(a, b, alternative="two-sided")
        d = cohens_d(a, b)
        lines.append(f"\n--- {label} ---")
        lines.append(f"  Mean diff = {diff.mean():+.4f} +/- {diff.std(ddof=1):.4f}")
        lines.append(f"  Wilcoxon signed-rank: W={stat_w:.1f}, p={p_w:.4f}")
        lines.append(f"  Cohen's d (paired)   = {d:+.3f}")
        if p_w < 0.05:
            lines.append(f"  -> SIGNIFICANT (p < 0.05)")
        else:
            lines.append(f"  -> Not significant (p >= 0.05)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION B: OMG-Empathy — Friedman test (omnibus)
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("B. FRIEDMAN TEST — OMG-Empathy (omnibus, 4 conditions × 15 videos)")
    lines.append("=" * 70)

    omg_conditions = {
        "C2 zero-shot": CCC_C2_ZERO,
        "C2 few-shot (k=3)": CCC_C2_FEW,
        "C3 vision": CCC_C3_ZERO,
        "Baseline (rules)": CCC_BASELINE,
    }
    stat_fr, p_fr = sp_stats.friedmanchisquare(*omg_conditions.values())
    k = len(omg_conditions)
    n = len(CCC_C2_ZERO)
    W_kendall = stat_fr / (n * (k - 1))
    lines.append(f"  Conditions (k):       {k}")
    lines.append(f"  Blocks (n videos):    {n}")
    lines.append(f"  Friedman chi²:        {stat_fr:.3f}")
    lines.append(f"  p-value:              {p_fr:.4f}")
    lines.append(f"  Kendall's W:          {W_kendall:.3f}")
    if p_fr < 0.05:
        lines.append(f"  -> SIGNIFICANT (p < 0.05): at least one condition differs")
    else:
        lines.append(f"  -> NOT significant (p >= 0.05): no omnibus difference detected")

    # Post-hoc: Nemenyi-like pairwise Wilcoxon with Holm correction
    lines.append(f"\n  Post-hoc: pairwise Wilcoxon with Holm-Bonferroni correction")
    names = list(omg_conditions.keys())
    arrays = list(omg_conditions.values())
    posthoc_results = []
    for i in range(k):
        for j in range(i + 1, k):
            _, pw = sp_stats.wilcoxon(arrays[i], arrays[j], alternative="two-sided")
            posthoc_results.append((f"{names[i]} vs {names[j]}", pw))

    posthoc_results.sort(key=lambda x: x[1])
    m = len(posthoc_results)
    for rank, (label, p_raw) in enumerate(posthoc_results, 1):
        p_adj = min(p_raw * (m - rank + 1), 1.0)  # Holm correction
        sig = "**" if p_adj < 0.05 else "ns"
        lines.append(f"    {label:40s}  p_raw={p_raw:.4f}  p_adj(Holm)={p_adj:.4f}  {sig}")

    # Mean ranks
    lines.append(f"\n  Mean ranks (higher = better CCC):")
    from scipy.stats import rankdata
    rank_matrix = np.column_stack([
        rankdata(-arr) for arr in arrays  # negative so higher CCC = lower rank number, then invert
    ])
    # Actually: rank within each video (row), Friedman convention
    data_matrix = np.column_stack(arrays)  # shape (15, 4)
    ranks = np.apply_along_axis(lambda row: rankdata(-row), axis=1, arr=data_matrix)
    mean_ranks = ranks.mean(axis=0)
    for name, mr in zip(names, mean_ranks):
        lines.append(f"    {name:25s}: {mr:.2f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION C: Bootstrap 95% CI — OMG
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("C. BOOTSTRAP 95% CI — OMG Mean CCC (10,000 resamples)")
    lines.append("=" * 70)

    for name, data in omg_conditions.items():
        lo, hi = bootstrap_ci(data)
        lines.append(f"  {name:25s}: mean={data.mean():+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION D: Effect Size — OMG
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("D. EFFECT SIZE — OMG Paired Cohen's d")
    lines.append("=" * 70)
    for label, a, b in pairs:
        d = cohens_d(a, b)
        mag = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        lines.append(f"  {label:40s}: d={d:+.3f} ({mag})")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION E: MOSEI — Per-sample paired tests (C1 text vs C2 FACET)
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("E. MOSEI — Per-sample Statistics (C1 text vs C2 FACET, n=400)")
    lines.append("=" * 70)

    c1_f1, c2_f1, c1_correct, c2_correct, gt_sets = _load_mosei_per_sample()

    lines.append(f"\n  --- Per-sample F1 (paired) ---")
    lines.append(f"  C1 (text)   mean F1: {c1_f1.mean():.4f} ± {c1_f1.std():.4f}")
    lines.append(f"  C2 (FACET)  mean F1: {c2_f1.mean():.4f} ± {c2_f1.std():.4f}")
    diff_f1 = c1_f1 - c2_f1
    lines.append(f"  Mean diff (C1 - C2): {diff_f1.mean():+.4f} ± {diff_f1.std():.4f}")

    # Wilcoxon on per-sample F1
    stat_w, p_w = sp_stats.wilcoxon(c1_f1, c2_f1, alternative="two-sided")
    lines.append(f"  Wilcoxon signed-rank: W={stat_w:.1f}, p={p_w:.6f}")
    d_mosei = cohens_d(c1_f1, c2_f1)
    mag = "large" if abs(d_mosei) >= 0.8 else "medium" if abs(d_mosei) >= 0.5 else "small"
    lines.append(f"  Cohen's d (paired):  {d_mosei:+.3f} ({mag})")
    if p_w < 0.05:
        lines.append(f"  -> SIGNIFICANT (p < 0.05)")
    else:
        lines.append(f"  -> Not significant (p >= 0.05)")

    # McNemar's test (exact match correctness)
    lines.append(f"\n  --- McNemar's test (exact set match) ---")
    b_val = int(((c1_correct == 1) & (c2_correct == 0)).sum())  # C1 right, C2 wrong
    c_val = int(((c1_correct == 0) & (c2_correct == 1)).sum())  # C1 wrong, C2 right
    both_right = int(((c1_correct == 1) & (c2_correct == 1)).sum())
    both_wrong = int(((c1_correct == 0) & (c2_correct == 0)).sum())
    lines.append(f"  Contingency: both_right={both_right}, both_wrong={both_wrong}, "
                 f"C1_only={b_val}, C2_only={c_val}")

    if b_val + c_val > 0:
        if b_val + c_val < 25:
            from scipy.stats import binom_test
            p_mcn = binom_test(b_val, b_val + c_val, 0.5)
            lines.append(f"  McNemar (exact binomial): p={p_mcn:.6f}")
        else:
            chi2_mcn = (abs(b_val - c_val) - 1) ** 2 / (b_val + c_val)
            p_mcn = 1 - sp_stats.chi2.cdf(chi2_mcn, df=1)
            lines.append(f"  McNemar (chi² with continuity correction): χ²={chi2_mcn:.3f}, p={p_mcn:.6f}")
        if p_mcn < 0.05:
            lines.append(f"  -> SIGNIFICANT (p < 0.05)")
        else:
            lines.append(f"  -> Not significant (p >= 0.05)")
    else:
        lines.append(f"  McNemar: not applicable (no discordant pairs)")

    # Bootstrap CI for MOSEI per-sample F1 difference
    lines.append(f"\n  --- Bootstrap 95% CI for mean F1 difference (C1 - C2) ---")
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(10_000):
        idx = rng.choice(len(diff_f1), size=len(diff_f1), replace=True)
        boot_diffs.append(diff_f1[idx].mean())
    boot_diffs = np.array(boot_diffs)
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])
    lines.append(f"  Mean diff: {diff_f1.mean():+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        lines.append(f"  CI excludes zero -> C1 significantly better than C2")
    elif hi < 0:
        lines.append(f"  CI excludes zero -> C2 significantly better than C1")
    else:
        lines.append(f"  CI includes zero -> no significant difference")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION F: MOSEI — Bootstrap CI per system (macro-F1)
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("F. MOSEI — Bootstrap 95% CI per condition (per-sample F1)")
    lines.append("=" * 70)
    for name, f1_arr in [("C1 (text)", c1_f1), ("C2 (FACET)", c2_f1)]:
        lo, hi = bootstrap_ci(f1_arr)
        lines.append(f"  {name:20s}: mean={f1_arr.mean():.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION G: Latency — Kruskal-Wallis + Dunn post-hoc
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("G. LATENCY — Kruskal-Wallis (4 conditions) + Dunn post-hoc")
    lines.append("=" * 70)

    import pandas as pd
    import scikit_posthocs as sp

    latency_files = {
        "OMG C2 (blendshapes)": "results/omg_C2_seed13.parquet",
        "OMG C3 (vision)":      "results/omg_C3_seed13.parquet",
        "MOSEI C1 (text)":      "results/mosei_C1_seed13.parquet",
        "MOSEI C2 (FACET)":     "results/mosei_C2_seed13.parquet",
    }
    lat_groups = {}
    for lbl, path in latency_files.items():
        lat_groups[lbl] = pd.read_parquet(path)["latency_ms"].dropna().values

    for lbl, arr in lat_groups.items():
        lines.append(f"  {lbl:<25s}: n={len(arr):>4d}  mean={np.mean(arr):>8.0f}ms  "
                      f"median={np.median(arr):>8.0f}ms  std={np.std(arr):>8.0f}ms")

    stat_kw, p_kw = sp_stats.kruskal(*lat_groups.values())
    lines.append(f"\n  Kruskal-Wallis H = {stat_kw:.3f}, p = {p_kw:.2e}")
    eta2_kw = (stat_kw - len(lat_groups) + 1) / (sum(len(v) for v in lat_groups.values()) - len(lat_groups))
    lines.append(f"  Effect size (eta² approx): {eta2_kw:.4f}")

    if p_kw < 0.05:
        lines.append(f"  -> SIGNIFICANT (p < 0.05): at least one condition's latency differs\n")
        # Dunn post-hoc
        lat_df = pd.DataFrame([
            {"condition": lbl, "latency_ms": v}
            for lbl, arr in lat_groups.items()
            for v in arr
        ])
        dunn_result = sp.posthoc_dunn(
            lat_df, val_col="latency_ms", group_col="condition", p_adjust="bonferroni"
        )
        lines.append("  Dunn post-hoc (Bonferroni correction):")
        cond_names = list(lat_groups.keys())
        for i in range(len(cond_names)):
            for j in range(i + 1, len(cond_names)):
                p_val = dunn_result.loc[cond_names[i], cond_names[j]]
                sig = "**" if p_val < 0.05 else "ns"
                lines.append(f"    {cond_names[i]:25s} vs {cond_names[j]:25s}  p={p_val:.4e}  {sig}")
    else:
        lines.append(f"  -> NOT significant (p >= 0.05)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION H: MOSEI — Kruskal-Wallis across systems (per-sample F1)
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("H. MOSEI — Kruskal-Wallis (4 systems, per-sample F1)")
    lines.append("=" * 70)

    EKMAN = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    THRESHOLD_LLM = 0.15

    c1_df = pd.read_parquet("results/mosei_C1_seed13.parquet")
    c2_df = pd.read_parquet("results/mosei_C2_seed13.parquet")

    def _gt_set_h(gt_str):
        if not gt_str or gt_str == "neutral":
            return set()
        return set(gt_str.split(","))

    def _pred_set_llm_h(row, threshold):
        return {e for e in EKMAN if row.get(f"ekman_{e}", 0) >= threshold}

    def _sample_f1_h(pred, gt):
        if not pred and not gt:
            return 1.0
        if not pred or not gt:
            return 0.0
        tp = len(pred & gt)
        prec = tp / len(pred)
        rec = tp / len(gt)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    gt_sets_h = [_gt_set_h(s) for s in c1_df["gt_emotions"]]
    c1_pred_h = [_pred_set_llm_h(c1_df.iloc[i], THRESHOLD_LLM) for i in range(len(c1_df))]
    c2_pred_h = [_pred_set_llm_h(c2_df.iloc[i], THRESHOLD_LLM) for i in range(len(c2_df))]

    c1_f1_h = np.array([_sample_f1_h(p, g) for p, g in zip(c1_pred_h, gt_sets_h)])
    c2_f1_h = np.array([_sample_f1_h(p, g) for p, g in zip(c2_pred_h, gt_sets_h)])

    # Compute deterministic baselines per sample
    # Majority baseline: always predict the single most common emotion
    from collections import Counter
    emotion_counts = Counter()
    for gs in gt_sets_h:
        emotion_counts.update(gs)
    majority_label = emotion_counts.most_common(1)[0][0] if emotion_counts else "happiness"
    lines.append(f"\n  Majority baseline predicts: {{{majority_label}}}")
    majority_pred = [{majority_label}] * len(gt_sets_h)
    majority_f1 = np.array([_sample_f1_h(p, g) for p, g in zip(majority_pred, gt_sets_h)])

    # Random baseline (Monte Carlo, averaged over 100 runs for stability)
    rng_mc = np.random.default_rng(42)
    base_rates = {e: sum(1 for gs in gt_sets_h if e in gs) / len(gt_sets_h) for e in EKMAN}
    lines.append(f"  Emotion base rates: { {e: f'{r:.3f}' for e, r in base_rates.items()} }")

    random_f1_runs = []
    for _ in range(100):
        run_f1 = []
        for gs in gt_sets_h:
            pred = {e for e in EKMAN if rng_mc.random() < base_rates[e]}
            run_f1.append(_sample_f1_h(pred, gs))
        random_f1_runs.append(run_f1)
    random_f1 = np.array(random_f1_runs).mean(axis=0)

    systems_f1 = {
        "Random Chance": random_f1,
        "Majority": majority_f1,
        "LLM C1 (text)": c1_f1_h,
        "LLM C2 (FACET)": c2_f1_h,
    }

    for sname, sf1 in systems_f1.items():
        lines.append(f"  {sname:25s}: mean F1 = {sf1.mean():.4f} ± {sf1.std():.4f}")

    stat_kw_m, p_kw_m = sp_stats.kruskal(*systems_f1.values())
    N_total = sum(len(v) for v in systems_f1.values())
    eta2_m = (stat_kw_m - len(systems_f1) + 1) / (N_total - len(systems_f1))
    lines.append(f"\n  Kruskal-Wallis H = {stat_kw_m:.3f}, p = {p_kw_m:.2e}")
    lines.append(f"  Effect size (eta² approx): {eta2_m:.4f}")

    if p_kw_m < 0.05:
        lines.append(f"  -> SIGNIFICANT (p < 0.05): at least one system differs\n")
        # Dunn post-hoc
        mosei_df_kw = pd.DataFrame([
            {"system": lbl, "f1": v}
            for lbl, arr in systems_f1.items()
            for v in arr
        ])
        dunn_mosei = sp.posthoc_dunn(
            mosei_df_kw, val_col="f1", group_col="system", p_adjust="bonferroni"
        )
        lines.append("  Dunn post-hoc (Bonferroni correction):")
        sys_names = list(systems_f1.keys())
        for i in range(len(sys_names)):
            for j in range(i + 1, len(sys_names)):
                p_val = dunn_mosei.loc[sys_names[i], sys_names[j]]
                sig = "**" if p_val < 0.05 else "ns"
                lines.append(f"    {sys_names[i]:25s} vs {sys_names[j]:25s}  p={p_val:.4e}  {sig}")
    else:
        lines.append(f"  -> NOT significant (p >= 0.05)")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION I: MOSEI — Bootstrap 95% CI per emotion (F1)
    # ══════════════════════════════════════════════════════════════════════
    lines.append("\n\n" + "=" * 70)
    lines.append("I. MOSEI — Bootstrap 95% CI per Emotion F1 (LLM C1 vs Baseline LogReg)")
    lines.append("=" * 70)

    rng_emo = np.random.default_rng(42)

    # Per-emotion binary predictions from LLM C1 and ground truth
    for emotion in EKMAN:
        c1_scores = c1_df[f"ekman_{emotion}"].fillna(0).values
        c1_binary = (c1_scores >= THRESHOLD_LLM).astype(int)
        gt_binary = np.array([1 if emotion in gs else 0 for gs in gt_sets_h])

        def _boot_f1(pred_bin, gt_bin, rng, n_boot=10_000):
            f1s = []
            for _ in range(n_boot):
                idx = rng.choice(len(pred_bin), size=len(pred_bin), replace=True)
                p_b, g_b = pred_bin[idx], gt_bin[idx]
                tp = ((p_b == 1) & (g_b == 1)).sum()
                fp = ((p_b == 1) & (g_b == 0)).sum()
                fn = ((p_b == 0) & (g_b == 1)).sum()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1s.append(f1)
            return np.array(f1s)

        boot_c1 = _boot_f1(c1_binary, gt_binary, rng_emo)
        lo_c1, hi_c1 = np.percentile(boot_c1, [2.5, 97.5])

        # Point estimate
        tp_c1 = ((c1_binary == 1) & (gt_binary == 1)).sum()
        fp_c1 = ((c1_binary == 1) & (gt_binary == 0)).sum()
        fn_c1 = ((c1_binary == 0) & (gt_binary == 1)).sum()
        prec_c1 = tp_c1 / (tp_c1 + fp_c1) if (tp_c1 + fp_c1) > 0 else 0
        rec_c1 = tp_c1 / (tp_c1 + fn_c1) if (tp_c1 + fn_c1) > 0 else 0
        f1_c1 = 2 * prec_c1 * rec_c1 / (prec_c1 + rec_c1) if (prec_c1 + rec_c1) > 0 else 0

        lines.append(f"\n  {emotion.capitalize():12s}:")
        lines.append(f"    LLM C1   F1={f1_c1:.3f}  95%CI=[{lo_c1:.3f}, {hi_c1:.3f}]  "
                      f"support={int(gt_binary.sum())} positive samples")

    # Bootstrap CI for the F1 DIFFERENCE (C1 - LogReg) per emotion
    lines.append(f"\n  --- Bootstrap 95% CI for F1 difference (LLM C1 minus LogReg baseline) ---")
    lines.append(f"  Note: LogReg F1 from hardcoded per-emotion values (no per-sample predictions available)")
    F1_BASELINE_VALS = {"happiness": 0.60, "sadness": 0.37, "anger": 0.25,
                         "fear": 0.07, "disgust": 0.49, "surprise": 0.21}
    F1_C1_FS_VALS = {"happiness": 0.57, "sadness": 0.38, "anger": 0.52,
                      "fear": 0.14, "disgust": 0.63, "surprise": 0.25}
    F1_C1_ZS_VALS = {"happiness": 0.56, "sadness": 0.40, "anger": 0.48,
                      "fear": 0.16, "disgust": 0.63, "surprise": 0.26}

    for emotion in EKMAN:
        c1_scores = c1_df[f"ekman_{emotion}"].fillna(0).values
        c1_binary = (c1_scores >= THRESHOLD_LLM).astype(int)
        gt_binary = np.array([1 if emotion in gs else 0 for gs in gt_sets_h])

        boot_f1s = _boot_f1(c1_binary, gt_binary, rng_emo)
        bl_f1 = F1_BASELINE_VALS[emotion]
        boot_diffs = boot_f1s - bl_f1
        lo_d, hi_d = np.percentile(boot_diffs, [2.5, 97.5])
        mean_d = boot_diffs.mean()
        sig_str = "CI excludes 0" if (lo_d > 0 or hi_d < 0) else "CI includes 0"
        lines.append(f"    {emotion.capitalize():12s}: diff={mean_d:+.3f}  95%CI=[{lo_d:+.3f}, {hi_d:+.3f}]  {sig_str}")

    report = "\n".join(lines) + "\n"
    report_file.write_text(report)
    print(report)
    return report


# ── Charts ──────────────────────────────────────────────────────────────────

def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_omg_ccc_conditions():
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = [
        "C2\nZero-shot",
        "C2\nFew-shot (k=3)",
        "C3\nVision (zero-shot)",
        "Baseline\n(Rule-based)",
    ]
    means = [CCC_C2_ZERO.mean(), CCC_C2_FEW.mean(), CCC_C3_ZERO.mean(), CCC_BASELINE.mean()]
    colors = [PALETTE["llm_light"], PALETTE["llm"], PALETTE["llm"], PALETTE["baseline"]]

    cis = [bootstrap_ci(d) for d in [CCC_C2_ZERO, CCC_C2_FEW, CCC_C3_ZERO, CCC_BASELINE]]
    err_lo = [m - ci[0] for m, ci in zip(means, cis)]
    err_hi = [ci[1] - m for m, ci in zip(means, cis)]

    bars = ax.bar(conditions, means, yerr=[err_lo, err_hi], capsize=6,
                  color=colors, edgecolor="white", linewidth=1.2, width=0.6,
                  error_kw={"linewidth": 1.5})

    ax.set_ylabel("Mean CCC ($\\pm$ 95% Bootstrap CI)")
    ax.set_title("OMG-Empathy — CCC per Condition (15 Test Videos)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(-0.05, 0.30)
    _clean_axes(ax)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{m:+.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "omg_ccc_conditions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'omg_ccc_conditions.png'}")


def plot_omg_ccc_per_video():
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(VIDEOS))
    w = 0.2

    ax.bar(x - 1.5 * w, CCC_C2_ZERO, w, label="C2 Zero-shot", color=PALETTE["weak"], edgecolor="white")
    ax.bar(x - 0.5 * w, CCC_C2_FEW, w, label="C2 Few-shot (k=3)", color=PALETTE["llm_light"], edgecolor="white")
    ax.bar(x + 0.5 * w, CCC_C3_ZERO, w, label="C3 Vision", color=PALETTE["llm"], edgecolor="white")
    ax.bar(x + 1.5 * w, CCC_BASELINE, w, label="Baseline (Rule-based)", color=PALETTE["baseline"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(VIDEOS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("CCC")
    ax.set_xlabel("Video (Subject_Story)")
    ax.set_title("OMG-Empathy — Per-Video CCC Across Conditions")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper left", framealpha=0.9)
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "omg_ccc_per_video.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'omg_ccc_per_video.png'}")


def plot_mosei_f1_conditions():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(MOSEI_SYSTEMS))
    w = 0.25

    ax.bar(x - w, MOSEI_MICRO, w, label="F1 Micro", color=PALETTE["llm_light"], edgecolor="white")
    macro_vals = [v if not np.isnan(v) else 0 for v in MOSEI_MACRO]
    ax.bar(x, macro_vals, w, label="F1 Macro", color=PALETTE["llm"], edgecolor="white")
    weighted_vals = [v if not np.isnan(v) else 0 for v in MOSEI_WEIGHTED]
    ax.bar(x + w, weighted_vals, w, label="F1 Weighted", color=PALETTE["baseline"], edgecolor="white")

    for i, (mi, ma, we) in enumerate(zip(MOSEI_MICRO, MOSEI_MACRO, MOSEI_WEIGHTED)):
        if not np.isnan(ma):
            ax.text(i, ma + 0.015, f"{ma:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MOSEI_SYSTEMS, fontsize=9)
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("System")
    ax.set_title("CMU-MOSEI — Multi-Label F1-Score per System")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper right")
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mosei_f1_conditions.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'mosei_f1_conditions.png'}")


def plot_mosei_f1_per_emotion():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(EMOTIONS))
    w = 0.25

    ax.bar(x - w, F1_BASELINE, w, label="Baseline (LogReg on FACET)", color=PALETTE["baseline"], edgecolor="white")
    ax.bar(x, F1_LLM_C1_ZS, w, label="LLM C1 Zero-shot", color=PALETTE["llm_light"], edgecolor="white")
    ax.bar(x + w, F1_LLM_C1_FS, w, label="LLM C1 Few-shot (k=5)", color=PALETTE["llm"], edgecolor="white")

    for i in range(len(EMOTIONS)):
        best = max(F1_BASELINE[i], F1_LLM_C1_ZS[i], F1_LLM_C1_FS[i])
        vals = [F1_BASELINE[i], F1_LLM_C1_ZS[i], F1_LLM_C1_FS[i]]
        offsets = [x[i] - w, x[i], x[i] + w]
        for v, o in zip(vals, offsets):
            if v == best:
                ax.text(o, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS])
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Ekman Emotion Category")
    ax.set_title("CMU-MOSEI — Per-Emotion F1: LLM (Text) vs. Baseline")
    ax.set_ylim(0, 0.78)
    ax.legend(loc="upper right")
    _clean_axes(ax)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mosei_f1_per_emotion.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'mosei_f1_per_emotion.png'}")


if __name__ == "__main__":
    print("=== Statistical tests ===")
    run_stats(Path("results/stats_report.txt"))

    print("\n=== Generating figures ===")
    plot_omg_ccc_conditions()
    plot_omg_ccc_per_video()
    plot_mosei_f1_conditions()
    plot_mosei_f1_per_emotion()

    print("\n Done — results/figures/ and results/stats_report.txt")
