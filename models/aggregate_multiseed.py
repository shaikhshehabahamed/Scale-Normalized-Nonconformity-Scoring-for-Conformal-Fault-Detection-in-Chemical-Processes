"""
Aggregate per-seed evaluation results into mean +/- std tables, and run
seed-level significance tests.

Inputs:  results/seeds/seed_<S>/{tcnae_results,dense_ae_results,
         tcnae_mahal_results,pvbc_results,sample_ae_results,conv_ae_results,
         overlap_robustness,noise_robustness,detection_delay_summary}.csv
         for S in MAIN_SEEDS
         results/ablation_seeds/seed_<S>/ablation_{window,latent,data_size,
         alpha,calsize}.csv for S in ABLATION_SEEDS
         results/pca_baseline_results.csv (deterministic, single run)

Outputs: results/agg/{tcnae_results,dense_ae_results,tcnae_mahal_results,
         pvbc_results,sample_ae_results,conv_ae_results}_agg.csv -- one row
         per (fault_id, method) with FAR_mean/FAR_std/FDR_mean/FDR_std/
         ADD_mean/ADD_std/n_seeds
         results/agg/factorial_agg.csv -- 4 architectures x 3 scores x 20 faults
         results/agg/ablation_{window,latent,data_size,alpha,calsize}_agg.csv
         results/agg/{overlap_robustness,noise_robustness,detection_delay}_agg.csv
         results/agg/significance_tests.csv

Usage: uv run python models/aggregate_multiseed.py  (after run_multiseed.py)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    ABLATION_SEEDS,
    AGG_DIR,
    MAIN_SEEDS,
    OUTDIR,
    ablation_seed_results_dir,
    seed_results_dir,
)

AGG_DIR.mkdir(parents=True, exist_ok=True)
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}


def load_across_seeds(filename: str, seeds, dir_fn) -> pd.DataFrame:
    frames = []
    missing = []
    for s in seeds:
        path = dir_fn(s) / filename
        if not path.exists():
            missing.append(s)
            continue
        df = pd.read_csv(path)
        df["seed"] = s
        frames.append(df)
    if missing:
        print(f"  WARNING: {filename} missing for seeds {missing} "
              f"(expected {len(seeds)}, found {len(frames)})")
    if not frames:
        raise FileNotFoundError(f"No seed produced {filename}")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    agg_spec = {}
    for c in metric_cols:
        agg_spec[c] = ["mean", "std", "count"]
    out = df.groupby(group_cols).agg(agg_spec)
    out.columns = [f"{c}_{stat}" if stat != "count" else f"{c}_n" for c, stat in out.columns]
    out = out.reset_index()
    # std of a single-seed group is NaN -> 0 (no variability observed, not "unknown")
    for c in metric_cols:
        out[f"{c}_std"] = out[f"{c}_std"].fillna(0.0)
    return out


def normalize_mahal_methods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = df["method"].str.replace("^TCN-", "TCNAE-", regex=True)
    return df


def seed_level_means(raw_df: pd.DataFrame, method: str, fault_subset=None) -> np.ndarray:
    """Per-seed mean FDR across faults (one number per seed) for `method`."""
    sub = raw_df[raw_df["method"] == method]
    if fault_subset is not None:
        sub = sub[sub["fault_id"].isin(fault_subset)]
    return sub.groupby("seed")["FDR"].mean().sort_index().to_numpy()


def one_sample_test(values: np.ndarray, constant: float, label: str, alternative: str = "greater") -> dict:
    """Is `values` (n_seeds) greater (or, if alternative='less', smaller) than
    the fixed deterministic `constant`?"""
    n = len(values)
    diffs = values - constant
    t_stat, p_t = stats.ttest_1samp(values, constant, alternative=alternative)
    try:
        w_stat, p_w = stats.wilcoxon(diffs, alternative=alternative)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    return {
        "comparison": label, "test_type": "one_sample_vs_fixed_baseline",
        "n_seeds": n, "mean": float(values.mean()), "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "baseline": float(constant),
        "t_stat": float(t_stat), "p_ttest_1samp": float(p_t),
        "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "alternative": alternative,
    }


def paired_test(values_a: np.ndarray, values_b: np.ndarray, label: str) -> dict:
    """Paired-by-seed comparison of two methods' seed-level mean FDR."""
    n = len(values_a)
    t_stat, p_t = stats.ttest_rel(values_a, values_b, alternative="greater")
    try:
        w_stat, p_w = stats.wilcoxon(values_a - values_b, alternative="greater")
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    return {
        "comparison": label, "test_type": "paired_by_seed",
        "n_seeds": n,
        "mean_a": float(values_a.mean()), "mean_b": float(values_b.mean()),
        "mean_diff": float((values_a - values_b).mean()),
        "t_stat": float(t_stat), "p_ttest_rel": float(p_t),
        "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "p_wilcoxon": float(p_w) if not np.isnan(p_w) else None,
        "alternative": "greater",
    }


print("=" * 70)
print("AGGREGATING MULTI-SEED RESULTS")
print(f"MAIN_SEEDS={MAIN_SEEDS}  ABLATION_SEEDS={ABLATION_SEEDS}")
print("=" * 70)

# ── 1. Main comparison result types ──────────────────────────────────────────
print("\n1. Main comparison CSVs")
tcnae_raw = load_across_seeds("tcnae_results.csv", MAIN_SEEDS, seed_results_dir)
dae_raw = load_across_seeds("dense_ae_results.csv", MAIN_SEEDS, seed_results_dir)
mahal_raw = normalize_mahal_methods(load_across_seeds("tcnae_mahal_results.csv", MAIN_SEEDS, seed_results_dir))
pvbc_raw = load_across_seeds("pvbc_results.csv", MAIN_SEEDS, seed_results_dir)
sample_raw = load_across_seeds("sample_ae_results.csv", MAIN_SEEDS, seed_results_dir)
conv_raw = load_across_seeds("conv_ae_results.csv", MAIN_SEEDS, seed_results_dir)

tcnae_agg = aggregate(tcnae_raw, ["fault_id", "fault_name", "method"], ["FAR", "FDR", "ADD"])
dae_agg = aggregate(dae_raw, ["fault_id", "fault_name", "method"], ["FAR", "FDR", "ADD"])
mahal_agg = aggregate(mahal_raw, ["fault_id", "method"], ["FAR", "FDR", "ADD"])
pvbc_agg = aggregate(pvbc_raw, ["fault_id", "method"], ["FAR", "FDR", "ADD"])
sample_agg = aggregate(sample_raw, ["fault_id", "fault_name", "method"], ["FAR", "FDR", "ADD"])
conv_agg = aggregate(conv_raw, ["fault_id", "fault_name", "method"], ["FAR", "FDR", "ADD"])

tcnae_agg.to_csv(AGG_DIR / "tcnae_results_agg.csv", index=False)
dae_agg.to_csv(AGG_DIR / "dense_ae_results_agg.csv", index=False)
mahal_agg.to_csv(AGG_DIR / "tcnae_mahal_results_agg.csv", index=False)
pvbc_agg.to_csv(AGG_DIR / "pvbc_results_agg.csv", index=False)
sample_agg.to_csv(AGG_DIR / "sample_ae_results_agg.csv", index=False)
conv_agg.to_csv(AGG_DIR / "conv_ae_results_agg.csv", index=False)
print(f"  Saved {AGG_DIR / 'tcnae_results_agg.csv'}")
print(f"  Saved {AGG_DIR / 'dense_ae_results_agg.csv'}")
print(f"  Saved {AGG_DIR / 'tcnae_mahal_results_agg.csv'}")
print(f"  Saved {AGG_DIR / 'pvbc_results_agg.csv'}")
print(f"  Saved {AGG_DIR / 'sample_ae_results_agg.csv'}")
print(f"  Saved {AGG_DIR / 'conv_ae_results_agg.csv'}")

# ── 1b. Architecture x score factorial table (round-2 review, point 4) ───────
print("\n1b. Factorial table: 4 architectures x 3 scores")
FACTORIAL_SOURCES = {
    ("TCN-AE", "MSE"):       (tcnae_agg, "TCNAE-MSE-CP"),
    ("TCN-AE", "MaxPV"):     (tcnae_agg, "TCNAE-MaxPV-CP"),
    ("TCN-AE", "NormMaxPV"): (pvbc_agg, "NormMaxPV-solo"),
    ("DenseAE", "MSE"):       (dae_agg, "DenseAE-CP"),
    ("DenseAE", "MaxPV"):     (dae_agg, "DenseAE-MaxPV"),
    ("DenseAE", "NormMaxPV"): (dae_agg, "DenseAE-NormMaxPV"),
    ("SampleAE", "MSE"):       (sample_agg, "SampleAE-CP"),
    ("SampleAE", "MaxPV"):     (sample_agg, "SampleAE-MaxPV"),
    ("SampleAE", "NormMaxPV"): (sample_agg, "SampleAE-NormMaxPV"),
    ("ConvAE", "MSE"):       (conv_agg, "ConvAE-CP"),
    ("ConvAE", "MaxPV"):     (conv_agg, "ConvAE-MaxPV"),
    ("ConvAE", "NormMaxPV"): (conv_agg, "ConvAE-NormMaxPV"),
}
factorial_rows = []
for (arch, score), (src_df, method) in FACTORIAL_SOURCES.items():
    sub = src_df[src_df["method"] == method].copy()
    if sub.empty:
        raise ValueError(f"Factorial table: no rows for architecture={arch} score={score} method={method}")
    sub["architecture"] = arch
    sub["score"] = score
    factorial_rows.append(sub[["fault_id", "architecture", "score",
                                "FAR_mean", "FAR_std", "FDR_mean", "FDR_std", "ADD_mean", "ADD_std"]])
factorial_df = pd.concat(factorial_rows, ignore_index=True)
factorial_df.to_csv(AGG_DIR / "factorial_agg.csv", index=False)
print(f"  Saved {AGG_DIR / 'factorial_agg.csv'} ({len(factorial_df)} rows = 12 cells x 20 faults)")
print("\n  Factorial summary (avg FDR %% across 20 faults):")
piv = factorial_df.groupby(["architecture", "score"])["FDR_mean"].mean().unstack() * 100
print(piv.round(1).to_string())

for name, agg in [("TCNAE-MSE-CP", tcnae_agg), ("DenseAE-CP", dae_agg)]:
    sub = agg[agg["method"] == name] if "method" in agg.columns else agg
    if not sub.empty:
        print(f"  {name}: mean FDR={sub['FDR_mean'].mean()*100:.2f}%  "
              f"(seed std avg={sub['FDR_std'].mean()*100:.2f} pp, n_seeds={int(sub['FDR_n'].iloc[0])})")

# ── 2. Ablation result types ─────────────────────────────────────────────────
print("\n2. Ablation CSVs")
abl_window_raw = load_across_seeds("ablation_window.csv", ABLATION_SEEDS, ablation_seed_results_dir)
abl_latent_raw = load_across_seeds("ablation_latent.csv", ABLATION_SEEDS, ablation_seed_results_dir)
abl_data_raw = load_across_seeds("ablation_data_size.csv", ABLATION_SEEDS, ablation_seed_results_dir)
abl_alpha_raw = load_across_seeds("ablation_alpha.csv", ABLATION_SEEDS, ablation_seed_results_dir)
abl_calsize_raw = load_across_seeds("ablation_calsize.csv", ABLATION_SEEDS, ablation_seed_results_dir)

ABL_METRICS = ["FAR_mse", "FDR_mse", "ADD_mse", "FAR_mpv", "FDR_mpv", "ADD_mpv",
               "FAR_normmaxpv", "FDR_normmaxpv", "ADD_normmaxpv"]
abl_window_agg = aggregate(abl_window_raw, ["fault_id", "config"], ABL_METRICS)
abl_latent_agg = aggregate(abl_latent_raw, ["fault_id", "config"], ABL_METRICS)
abl_data_agg = aggregate(abl_data_raw, ["fault_id", "config"], ABL_METRICS)
# Alpha and calibration-size sweeps use NormMaxPV throughout (see
# ablation_study.py); their FAR/FDR/ADD columns already refer to NormMaxPV,
# not MSE (unlike the other three sub-studies).
abl_alpha_agg = aggregate(abl_alpha_raw, ["fault_id", "alpha", "config"], ["FAR", "FDR", "ADD"])
abl_calsize_agg = aggregate(abl_calsize_raw, ["fault_id", "n_cal_runs", "config"], ["FAR", "FDR", "ADD"])

abl_window_agg.to_csv(AGG_DIR / "ablation_window_agg.csv", index=False)
abl_latent_agg.to_csv(AGG_DIR / "ablation_latent_agg.csv", index=False)
abl_data_agg.to_csv(AGG_DIR / "ablation_data_size_agg.csv", index=False)
abl_alpha_agg.to_csv(AGG_DIR / "ablation_alpha_agg.csv", index=False)
abl_calsize_agg.to_csv(AGG_DIR / "ablation_calsize_agg.csv", index=False)
print(f"  Saved 5 ablation_*_agg.csv files (n_seeds={len(ABLATION_SEEDS)})")

# ── 2b. Robustness sweeps + censored detection-delay (round-2 review, points 2/6a/7) ─
print("\n2b. Robustness + detection-delay CSVs (main seeds)")
overlap_raw = load_across_seeds("overlap_robustness.csv", MAIN_SEEDS, seed_results_dir)
noise_raw = load_across_seeds("noise_robustness.csv", MAIN_SEEDS, seed_results_dir)
delay_raw = load_across_seeds("detection_delay_summary.csv", MAIN_SEEDS, seed_results_dir)

overlap_agg = aggregate(overlap_raw, ["scheme", "exchangeable_unit"], ["far_test"])
noise_agg = aggregate(noise_raw, ["noise_level", "method"], ["FAR_all", "FDR_all", "FDR_hard", "FDR_easy"])
# detection_probability/conditional_*_delay are means-of-per-seed-summaries here
# (not re-derived from pooled raw runs), consistent with this file's seed-level
# aggregation convention everywhere else.
delay_agg = aggregate(delay_raw, ["tier"], ["n_total", "n_detected", "detection_probability",
                                             "conditional_median_delay", "conditional_mean_delay"])

overlap_agg.to_csv(AGG_DIR / "overlap_robustness_agg.csv", index=False)
noise_agg.to_csv(AGG_DIR / "noise_robustness_agg.csv", index=False)
delay_agg.to_csv(AGG_DIR / "detection_delay_agg.csv", index=False)
print(f"  Saved {AGG_DIR / 'overlap_robustness_agg.csv'}")
print(overlap_agg[["scheme", "exchangeable_unit", "far_test_mean", "far_test_std"]].to_string(index=False))
print(f"  Saved {AGG_DIR / 'noise_robustness_agg.csv'}")
print(noise_agg[["noise_level", "FAR_all_mean", "FDR_all_mean"]].to_string(index=False))
print(f"  Saved {AGG_DIR / 'detection_delay_agg.csv'}")
print(delay_agg[["tier", "detection_probability_mean", "conditional_median_delay_mean"]].to_string(index=False))

# ── 3. Combined long-format all_results_agg.csv (+ deterministic PCA) ────────
print("\n3. Combined all_results_agg.csv")
pca_df = pd.read_csv(OUTDIR / "pca_baseline_results.csv")
pca_df["FAR_mean"], pca_df["FAR_std"], pca_df["FAR_n"] = pca_df["FAR"], 0.0, 1
pca_df["FDR_mean"], pca_df["FDR_std"], pca_df["FDR_n"] = pca_df["FDR"], 0.0, 1
pca_df["ADD_mean"], pca_df["ADD_std"], pca_df["ADD_n"] = pca_df["ADD"], 0.0, 1
pca_df = pca_df.drop(columns=["FAR", "FDR", "ADD"])

or_row = mahal_agg[mahal_agg["method"] == "TCNAE-OR(MSE+MaxPV)-CP"].copy()
mahal_solo_row = mahal_agg[mahal_agg["method"] == "TCNAE-Mahal-CP"].copy()
pvbc_keep = pvbc_agg[pvbc_agg["method"].isin(["NormMaxPV-solo", "OR(MSE+NormMaxPV)", "OR(MSE+PVBC)"])].copy()

common_cols = ["fault_id", "method", "FAR_mean", "FAR_std", "FAR_n",
               "FDR_mean", "FDR_std", "FDR_n", "ADD_mean", "ADD_std", "ADD_n"]
all_agg = pd.concat([
    pca_df[["fault_id", "fault_name", "method"] + common_cols[2:]],
    dae_agg[["fault_id", "fault_name", "method"] + common_cols[2:]],
    tcnae_agg[["fault_id", "fault_name", "method"] + common_cols[2:]],
    or_row[common_cols].assign(fault_name=None),
    mahal_solo_row[common_cols].assign(fault_name=None),
    pvbc_keep[common_cols].assign(fault_name=None),
], ignore_index=True)
all_agg.to_csv(AGG_DIR / "all_results_agg.csv", index=False)
print(f"  Saved {AGG_DIR / 'all_results_agg.csv'}  ({len(all_agg)} rows)")

# ── 4. Significance tests ────────────────────────────────────────────────────
print("\n4. Significance tests (seed-level, n_seeds={})".format(len(MAIN_SEEDS)))
pca_q_mean = float(pca_df[pca_df["method"] == "PCA-Q(n=7)"]["FDR_mean"].mean())
pca_comb_mean = float(pca_df[pca_df["method"] == "PCA-Comb(n=29)"]["FDR_mean"].mean())

tcn_mse_seedmeans = seed_level_means(tcnae_raw, "TCNAE-MSE-CP")
tcn_maxpv_seedmeans = seed_level_means(tcnae_raw, "TCNAE-MaxPV-CP")
tcn_or_seedmeans = seed_level_means(mahal_raw, "TCNAE-OR(MSE+MaxPV)-CP")
tcn_mahal_seedmeans = seed_level_means(mahal_raw, "TCNAE-Mahal-CP")
dae_seedmeans = seed_level_means(dae_raw, "DenseAE-CP")
dae_normmaxpv_seedmeans = seed_level_means(dae_raw, "DenseAE-NormMaxPV")
normmaxpv_seedmeans = seed_level_means(pvbc_raw, "NormMaxPV-solo")
ornorm_seedmeans = seed_level_means(pvbc_raw, "OR(MSE+NormMaxPV)")
orpvbc_seedmeans = seed_level_means(pvbc_raw, "OR(MSE+PVBC)")

# Every headline TCN-AE variant (all methods appearing in Table 4 except the
# already-flagged Mahal underperformer) vs both PCA baselines -- this backs
# the "every TCN-AE variant significantly outperforms both PCA baselines"
# claim in Section 5.2 with an actual test per variant, rather than asserting
# it from only a couple of spot checks.
VARIANTS = {
    "TCN-MSE-CP": tcn_mse_seedmeans,
    "TCN-MaxPV-CP": tcn_maxpv_seedmeans,
    "TCN-OR(MSE+MaxPV)": tcn_or_seedmeans,
    "NormMaxPV-solo": normmaxpv_seedmeans,
    "OR(MSE+NormMaxPV)": ornorm_seedmeans,
    "OR(MSE+PVBC)": orpvbc_seedmeans,
}

tests = []
for name, seedmeans in VARIANTS.items():
    tests.append(one_sample_test(seedmeans, pca_q_mean, f"{name} vs PCA-Q(n=7)"))
    tests.append(one_sample_test(seedmeans, pca_comb_mean, f"{name} vs PCA-Comb(n=29)"))
tests.append(one_sample_test(dae_seedmeans, pca_q_mean, "DenseAE-CP vs PCA-Q(n=7)"))
tests.append(one_sample_test(dae_seedmeans, pca_comb_mean, "DenseAE-CP vs PCA-Comb(n=29)"))
# Mahal is the one method flagged in Section 5.7 as UNDERperforming PCA; test
# with alternative='less' so the reported p-value means what it says.
tests.append(one_sample_test(tcn_mahal_seedmeans, pca_q_mean, "TCN-Mahal-CP vs PCA-Q(n=7)", alternative="less"))
tests.append(one_sample_test(tcn_mahal_seedmeans, pca_comb_mean, "TCN-Mahal-CP vs PCA-Comb(n=29)", alternative="less"))

tests.append(paired_test(tcn_or_seedmeans, dae_seedmeans, "TCN-OR(MSE+MaxPV) vs DenseAE-CP (paired by seed)"))
tests.append(paired_test(normmaxpv_seedmeans, tcn_or_seedmeans, "NormMaxPV-solo vs TCN-OR(MSE+MaxPV) (paired by seed)"))
tests.append(paired_test(orpvbc_seedmeans, tcn_or_seedmeans, "OR(MSE+PVBC) vs TCN-OR(MSE+MaxPV) (paired by seed)"))

# M2 (reviewer): is NormMaxPV's benefit architecture-agnostic (works for the
# Dense AE too) or specific to the TCN encoder? Both comparisons needed.
tests.append(paired_test(dae_normmaxpv_seedmeans, dae_seedmeans, "DenseAE-NormMaxPV vs DenseAE-CP (paired by seed)"))
tests.append(one_sample_test(dae_normmaxpv_seedmeans, pca_q_mean, "DenseAE-NormMaxPV vs PCA-Q(n=7)"))
tests.append(one_sample_test(dae_normmaxpv_seedmeans, pca_comb_mean, "DenseAE-NormMaxPV vs PCA-Comb(n=29)"))
tests.append(paired_test(normmaxpv_seedmeans, dae_normmaxpv_seedmeans, "NormMaxPV-solo (TCN) vs DenseAE-NormMaxPV (paired by seed)"))

# Round-2 review, point 4: full 4-architecture x 3-score factorial. Does
# NormMaxPV's gain over MSE hold for every architecture (not just TCN-AE and
# Dense AE), including the true temporal-blind SampleAE control and the
# conventional (non-dilated, non-residual) ConvAE?
sample_mse_seedmeans = seed_level_means(sample_raw, "SampleAE-CP")
sample_normmaxpv_seedmeans = seed_level_means(sample_raw, "SampleAE-NormMaxPV")
conv_mse_seedmeans = seed_level_means(conv_raw, "ConvAE-CP")
conv_normmaxpv_seedmeans = seed_level_means(conv_raw, "ConvAE-NormMaxPV")

for arch_label, mse_sm, norm_sm in [
    ("TCN-AE", tcn_mse_seedmeans, normmaxpv_seedmeans),
    ("DenseAE", dae_seedmeans, dae_normmaxpv_seedmeans),
    ("SampleAE", sample_mse_seedmeans, sample_normmaxpv_seedmeans),
    ("ConvAE", conv_mse_seedmeans, conv_normmaxpv_seedmeans),
]:
    tests.append(paired_test(norm_sm, mse_sm, f"{arch_label}-NormMaxPV vs {arch_label}-MSE (paired by seed)"))

for arch_label, norm_sm in [("SampleAE", sample_normmaxpv_seedmeans), ("ConvAE", conv_normmaxpv_seedmeans)]:
    tests.append(one_sample_test(norm_sm, pca_q_mean, f"{arch_label}-NormMaxPV vs PCA-Q(n=7)"))
    tests.append(one_sample_test(norm_sm, pca_comb_mean, f"{arch_label}-NormMaxPV vs PCA-Comb(n=29)"))
    tests.append(paired_test(normmaxpv_seedmeans, norm_sm, f"NormMaxPV-solo (TCN) vs {arch_label}-NormMaxPV (paired by seed)"))

# Hard-fault-only seed-level test (the paper's headline claim)
tcn_or_hard = seed_level_means(mahal_raw, "TCNAE-OR(MSE+MaxPV)-CP", fault_subset=HARD_FAULTS)
normmaxpv_hard = seed_level_means(pvbc_raw, "NormMaxPV-solo", fault_subset=HARD_FAULTS)
pca_q_hard = float(pca_df[(pca_df["method"] == "PCA-Q(n=7)") & (pca_df["fault_id"].isin(HARD_FAULTS))]["FDR_mean"].mean())
pca_comb_hard = float(pca_df[(pca_df["method"] == "PCA-Comb(n=29)") & (pca_df["fault_id"].isin(HARD_FAULTS))]["FDR_mean"].mean())
tests.append(one_sample_test(tcn_or_hard, pca_q_hard, "TCN-OR(MSE+MaxPV) vs PCA-Q(n=7), hard faults only"))
tests.append(one_sample_test(normmaxpv_hard, pca_q_hard, "NormMaxPV-solo vs PCA-Q(n=7), hard faults only"))
tests.append(one_sample_test(normmaxpv_hard, pca_comb_hard, "NormMaxPV-solo vs PCA-Comb(n=29), hard faults only"))

sig_df = pd.DataFrame(tests)
sig_df.to_csv(AGG_DIR / "significance_tests.csv", index=False)
print(f"\nSaved {AGG_DIR / 'significance_tests.csv'}")
print(sig_df[["comparison", "test_type", "n_seeds", "p_ttest_1samp" if "p_ttest_1samp" in sig_df.columns else "p_ttest_rel"]].to_string() if False else "")
for row in tests:
    p = row.get("p_ttest_1samp", row.get("p_ttest_rel"))
    print(f"  {row['comparison']:55s} n={row.get('n_seeds','?')}  p={p:.4f}  "
          f"({'SIGNIFICANT' if p < 0.05 else 'not significant'} at 0.05)")

print("\n" + "=" * 70)
print("AGGREGATION COMPLETE")
print("Next: uv run python models/final_comparison.py")
print("=" * 70)
