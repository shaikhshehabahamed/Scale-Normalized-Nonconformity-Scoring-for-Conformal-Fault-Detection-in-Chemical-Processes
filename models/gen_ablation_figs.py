"""
Generate paper-quality ablation figures from the aggregated multi-seed CSVs.
Run AFTER aggregate_multiseed.py has produced:
  results/agg/ablation_window_agg.csv
  results/agg/ablation_latent_agg.csv
  results/agg/ablation_data_size_agg.csv
  results/agg/ablation_alpha_agg.csv
(mean+std across ABLATION_SEEDS; see tcnae_utils.py)

Outputs (300 DPI, Elsevier format):
  fig12_ablation_window.png
  fig13_ablation_latent.png
  fig14_ablation_data_size.png
  fig15_ablation_alpha.png
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import paper_style
from tcnae_utils import ABLATION_SEEDS, AGG_DIR, OUTDIR

paper_style.apply()

HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
N_ABL_SEEDS = len(ABLATION_SEEDS)

def category_mean_std(df, config_val, mean_col, std_col, fault_subset=None):
    """Mean across faults + propagated seed-std (assumes independent faults)."""
    sub = df[df['config'] == config_val]
    if fault_subset is not None:
        sub = sub[sub['fault_id'].isin(fault_subset)]
    if sub.empty:
        return 0.0, 0.0
    mean_v = sub[mean_col].mean()
    std_v = (sub[std_col] ** 2).mean() ** 0.5 / max(len(sub), 1) ** 0.5
    return mean_v, std_v


# ── fig12: Window size ────────────────────────────────────────────────────────────
# Aggregate summary (avg FDR / avg ADD vs W), matching what the caption and
# body text describe -- a per-fault breakdown previously shown here did not
# match either.
abl_w = pd.read_csv(AGG_DIR / "ablation_window_agg.csv")
window_configs = sorted(
    abl_w['config'].str.extract(r'W=(\d+)')[0].dropna().astype(int).unique()
)

print("fig12: Window size ablation (aggregate, with seed-std error bars) ...")
_, axes = plt.subplots(1, 2, figsize=(paper_style.W2 + 1.0, 3.5))
ax = axes[0]
avg_fdr_w, avg_fdr_w_err, hard_fdr_w, hard_fdr_w_err = [], [], [], []
for W in window_configs:
    m, s = category_mean_std(abl_w, f'W={W}', 'FDR_normmaxpv_mean', 'FDR_normmaxpv_std')
    avg_fdr_w.append(m * 100); avg_fdr_w_err.append(s * 100)
    m, s = category_mean_std(abl_w, f'W={W}', 'FDR_normmaxpv_mean', 'FDR_normmaxpv_std', HARD_FAULTS)
    hard_fdr_w.append(m * 100); hard_fdr_w_err.append(s * 100)
ax.errorbar(window_configs, avg_fdr_w, yerr=avg_fdr_w_err, fmt='bo-', capsize=3,
            label='All faults', linewidth=1.5, markersize=5)
ax.errorbar(window_configs, hard_fdr_w, yerr=hard_fdr_w_err, fmt='rs--', capsize=3,
            label='Hard faults', linewidth=1.5, markersize=5)
ax.axvline(20, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Chosen (W=20)')
ax.set_xlabel('Window size W', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title('Effect of W on FDR', fontsize=9)
ax.set_xticks(window_configs)
ax.legend(fontsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

ax = axes[1]
avg_add_w, avg_add_w_err = [], []
for W in window_configs:
    m, s = category_mean_std(abl_w, f'W={W}', 'ADD_normmaxpv_mean', 'ADD_normmaxpv_std')
    avg_add_w.append(m); avg_add_w_err.append(s)
ax.errorbar(window_configs, avg_add_w, yerr=avg_add_w_err, fmt='go-', capsize=3,
            linewidth=1.5, markersize=5)
ax.axvline(20, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Chosen (W=20)')
ax.set_xlabel('Window size W', fontsize=9)
ax.set_ylabel('Avg ADD (samples)', fontsize=9)
ax.set_title('Effect of W on ADD', fontsize=9)
ax.set_xticks(window_configs)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

plt.suptitle(f'Ablation Study 1: Window Size W (NormMaxPV; mean +/- std, n={N_ABL_SEEDS} seeds)', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig12_ablation_window.png")
plt.close()
print("  Saved fig12_ablation_window.png")

# ── fig13: Latent dimension ───────────────────────────────────────────────────────
abl_l = pd.read_csv(AGG_DIR / "ablation_latent_agg.csv")
latent_configs = sorted(
    abl_l['config'].str.extract(r'lat=(\d+)')[0].dropna().astype(int).unique()
)

print("fig13: Latent dimension (with seed-std error bars) ...")
_, ax = plt.subplots(figsize=(paper_style.W1 + 0.6, 3.2))
avg_fdr, avg_fdr_err, hard_fdr, hard_fdr_err = [], [], [], []
for ld in latent_configs:
    m, s = category_mean_std(abl_l, f'lat={ld}', 'FDR_normmaxpv_mean', 'FDR_normmaxpv_std')
    avg_fdr.append(m * 100); avg_fdr_err.append(s * 100)
    m, s = category_mean_std(abl_l, f'lat={ld}', 'FDR_normmaxpv_mean', 'FDR_normmaxpv_std', HARD_FAULTS)
    hard_fdr.append(m * 100); hard_fdr_err.append(s * 100)
ax.errorbar(latent_configs, avg_fdr, yerr=avg_fdr_err, fmt='bo-', capsize=3,
            label='All faults', linewidth=1.5, markersize=5)
ax.errorbar(latent_configs, hard_fdr, yerr=hard_fdr_err, fmt='rs--', capsize=3,
            label='Hard faults', linewidth=1.5, markersize=5)
ax.axvline(24, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Chosen (d=24)')
ax.set_xlabel('Latent dimension', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title(f'Ablation Study 2: Latent Dimension (NormMaxPV; n={N_ABL_SEEDS} seeds)', fontsize=9)
ax.set_xticks(latent_configs)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig13_ablation_latent.png")
plt.close()
print("  Saved fig13_ablation_latent.png")

# ── fig14: Training data size ─────────────────────────────────────────────────────
abl_d = pd.read_csv(AGG_DIR / "ablation_data_size_agg.csv")
data_configs = sorted(
    abl_d['config'].str.extract(r'n_tr=(\d+)')[0].dropna().astype(int).unique()
)

print("fig14: Training data size (with seed-std error bars) ...")
_, ax = plt.subplots(figsize=(paper_style.W1 + 0.6, 3.2))
avg_fdr_d, avg_fdr_d_err = [], []
for n in data_configs:
    m, s = category_mean_std(abl_d, f'n_tr={n}', 'FDR_normmaxpv_mean', 'FDR_normmaxpv_std')
    avg_fdr_d.append(m * 100); avg_fdr_d_err.append(s * 100)
ax.errorbar(data_configs, avg_fdr_d, yerr=avg_fdr_d_err, fmt='go-', capsize=3,
            linewidth=1.8, markersize=6)
ax.axvline(400, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Chosen (400 runs)')
for x, y in zip(data_configs, avg_fdr_d):
    # x/500 = fraction of the 500-run fault-free pool used for training;
    # labeled explicitly so it isn't misread as a second FDR-like quantity
    ax.annotate(f'{x // 5}% of pool', (x, y), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=7)
ax.set_xlabel('Training runs', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title(f'Ablation Study 3: Training Data Size (NormMaxPV; n={N_ABL_SEEDS} seeds)\n'
             '(point labels: % of the 500-run fault-free pool used for training)', fontsize=9)
ax.set_xticks(data_configs)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig14_ablation_data_size.png")
plt.close()
print("  Saved fig14_ablation_data_size.png")

# ── fig15: Alpha sensitivity (FAR–FDR tradeoff) ───────────────────────────────────
abl_a = pd.read_csv(AGG_DIR / "ablation_alpha_agg.csv")
alphas  = sorted(abl_a['alpha'].unique())
avg_far = [abl_a[abl_a['alpha'] == a]['FAR_mean'].mean() * 100 for a in alphas]
avg_fdr = [abl_a[abl_a['alpha'] == a]['FDR_mean'].mean() * 100 for a in alphas]
avg_fdr_err = [(abl_a[abl_a['alpha'] == a]['FDR_std'] ** 2).mean() ** 0.5 * 100 for a in alphas]

print("fig15: Alpha sensitivity (with seed-std error bars) ...")
_, ax = plt.subplots(figsize=(paper_style.W1 + 0.6, 3.5))
ax.errorbar(avg_far, avg_fdr, yerr=avg_fdr_err, fmt='bo-', capsize=3,
            linewidth=1.8, markersize=7)
for far, fdr, a in zip(avg_far, avg_fdr, alphas):
    ax.annotate(f'α={a}', (far, fdr), textcoords='offset points',
                xytext=(6, 3), fontsize=7.5)
ax.set_xlabel('Avg FAR (%)', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title(f'Ablation Study 4: FAR-FDR Trade-off (NormMaxPV; n={N_ABL_SEEDS} seeds)\n(alpha=0.01 used in paper)', fontsize=9)
ax.set_xlim(left=-0.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig15_ablation_alpha.png")
plt.close()
print("  Saved fig15_ablation_alpha.png")

# ── fig15b: Limited calibration-data sensitivity (round-2 review, point 7) ───
abl_c = pd.read_csv(AGG_DIR / "ablation_calsize_agg.csv")
cal_sizes = sorted(abl_c['n_cal_runs'].unique())
avg_fdr_c = [abl_c[abl_c['n_cal_runs'] == c]['FDR_mean'].mean() * 100 for c in cal_sizes]
avg_fdr_c_err = [(abl_c[abl_c['n_cal_runs'] == c]['FDR_std'] ** 2).mean() ** 0.5 * 100 for c in cal_sizes]
avg_far_c = [abl_c[abl_c['n_cal_runs'] == c]['FAR_mean'].mean() * 100 for c in cal_sizes]
avg_far_c_err = [(abl_c[abl_c['n_cal_runs'] == c]['FAR_std'] ** 2).mean() ** 0.5 * 100 for c in cal_sizes]

print("fig15b: Limited calibration-data sensitivity (with seed-std error bars) ...")
_, axes = plt.subplots(1, 2, figsize=(paper_style.W2 + 1.0, 3.5))
ax = axes[0]
ax.errorbar(cal_sizes, avg_fdr_c, yerr=avg_fdr_c_err, fmt='bo-', capsize=3,
            linewidth=1.5, markersize=5)
ax.axvline(50, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Used in paper (n=50)')
ax.set_xlabel('Calibration runs used', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title('Effect on FDR (NormMaxPV)', fontsize=9)
ax.set_xticks(cal_sizes)
ax.legend(fontsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

ax = axes[1]
ax.errorbar(cal_sizes, avg_far_c, yerr=avg_far_c_err, fmt='rs-', capsize=3,
            linewidth=1.5, markersize=5)
ax.axhline(1.0, color='k', lw=1.0, linestyle='--', alpha=0.5, label='Target alpha=1%')
ax.axvline(50, color='gray', lw=1.0, linestyle=':', alpha=0.7, label='Used in paper (n=50)')
ax.set_xlabel('Calibration runs used', fontsize=9)
ax.set_ylabel('Avg FAR (%)', fontsize=9)
ax.set_title('Effect on FAR (NormMaxPV)', fontsize=9)
ax.set_xticks(cal_sizes)
ax.legend(fontsize=7.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

plt.suptitle(f'Ablation Study 5: Limited Calibration-Data Sensitivity (n={N_ABL_SEEDS} seeds)', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig15b_ablation_calsize.png")
plt.close()
print("  Saved fig15b_ablation_calsize.png")

print("\nDone. Ablation figures saved at 300 DPI.")
