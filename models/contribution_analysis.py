"""
Per-variable reconstruction contribution analysis for TCN-AE (Figure 2), an
illustrative detection-delay CDF (Figure 7), and a category comparison chart
(Figure 3-style summary). All three use the reference seed-42 model
(models/seeds/seed_42/), matching every other "illustrative, single-seed"
figure in this paper (Figures 4, 6, 25); rigorous multi-seed numbers for
detection delay/probability come from detection_delay_eval.py + the
aggregated results/agg/detection_delay_agg.csv, not from this script.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae import TCNAE
from tcnae_utils import (
    FEAT,
    MODDIR,
    OUTDIR,
    configured_batch_size,
    cp_threshold,
    load_faulty_test_cache,
    make_run_windows,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    select_device,
    split_fault_test_runs,
    tensor_to_numpy,
)
seed_everything(42)
REF_SEED = 42
SDIR = seed_model_dir(REF_SEED)

os.makedirs(OUTDIR, exist_ok=True)

FAULT_START = 161
WINDOW      = 20
ALPHA       = 0.01
DEVICE      = select_device()
INFER_BS    = configured_batch_size("TCNAE_INFER_BS", 2048)
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
EASY_FAULTS = {1, 2, 4, 5, 6, 7, 8, 12, 13, 14}

FAULT_NAMES = {
    1:"A/C feed ratio step", 2:"B composition step", 3:"D feed temp step",
    4:"Reactor CW temp step", 5:"Condenser CW temp step", 6:"A feed loss",
    7:"C pressure loss step", 8:"A,B,C feed comp step", 9:"D feed temp random",
    10:"C feed temp random", 11:"Reactor CW random", 12:"Condenser CW random",
    13:"Reaction kinetics slow drift", 14:"Reactor CW valve sticking",
    15:"Condenser CW valve sticking", 16:"Unknown-16", 17:"Unknown-17",
    18:"Unknown-18", 19:"Unknown-19", 20:"Unknown-20"
}

print("="*65)
print("CONTRIBUTION ANALYSIS & VISUALIZATION")
print("="*65)
print_device(DEVICE)

# ── Load model (reference seed) ───────────────────────────────────────────────
scaler = joblib.load(MODDIR / "scaler.pkl")   # shared across seeds (StandardScaler has no RNG)

model = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print(f"Model loaded from {SDIR}.")

# std_pv from the norm-estimation set (round-2 review, point 1): the same
# fixed statistics NormMaxPV uses at deployment, never recomputed from cal_pv.
norm_stats = joblib.load(SDIR / "norm_stats.pkl")
std_pv_ref = norm_stats['std_pv']
cal_pv_ref = np.load(SDIR / "cal_pv_scores.npy")
tau_normmaxpv_ref = cp_threshold((cal_pv_ref / std_pv_ref).max(axis=1), ALPHA)
print(f"NormMaxPV threshold (reference model): {tau_normmaxpv_ref:.5f}")

# ── Per-variable contribution over time for selected faults ──────────────────
print("\n1. PER-VARIABLE CONTRIBUTION ANALYSIS")

def get_per_var_scores(X_run_raw):
    """Returns per-variable MSE contribution per sample. Shape: (960, 52)."""
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)
    pv_windows = []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i : i + INFER_BS], DEVICE)
            recon, _ = model(batch)
            pv  = ((batch - recon)**2).mean(dim=2)   # (B, V)
            pv_windows.append(tensor_to_numpy(pv))
    pv_w = np.concatenate(pv_windows)  # (n-W+1, 52)
    # Assign to last sample index
    pv_sample = np.zeros((n, 52))
    pv_sample[:WINDOW-1] = pv_w[0]
    for i, w_idx in enumerate(range(WINDOW-1, n)):
        pv_sample[w_idx] = pv_w[i]
    return pv_sample

faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faulty cache: {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

# Per Eq. 25 (Section 3.7): c_j(X) = [e_j(X)/sigma_j^2] / sum_k [e_k(X)/sigma_k^2]
# for a single alarmed window, averaged over all alarmed windows during a
# fault episode. Normalizing by sigma_j^2 (round-2 review, point 6b) makes
# c_j(X) reflect what actually drove the NormMaxPV alarm that selected the
# window in the first place; the raw (unnormalized) e_j(X) version used
# previously could rank a variable as the top "contributor" purely because it
# has a large raw calibration variance, even when it was not the variable
# whose *normalized* error exceeded the threshold. c_j(X) is non-negative and
# sums to 1 across variables by construction, same as before.
print("Computing per-variable fault signatures (Eq. 25 contribution maps, "
      "normalized by std_pv)...")
contrib_records = []
ANALYZE_FAULTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
N_RUNS_ANALYZE = 50   # first 50 FAULT_TEST_RUNS per fault (was 10; cheap, reuses trained model)
CONTRIB_RUN_IDS = FAULT_TEST_RUNS[:N_RUNS_ANALYZE]

for fid in ANALYZE_FAULTS:
    alarmed_c = []       # c_j(X) for each alarmed window, this fault
    all_fault_c = []     # fallback: c_j(X) for every fault-period window
    n_runs = 0
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=CONTRIB_RUN_IDS):
        if len(X_run) < 960:
            continue
        pv = get_per_var_scores(X_run)                         # (960, 52), e_j(X_t)
        pv_norm = pv / std_pv_ref                                # (960, 52), e_j(X_t)/sigma_j^2
        normmaxpv_t = pv_norm.max(axis=1)                        # (960,) -- NormMaxPV itself
        alarm_t = normmaxpv_t > tau_normmaxpv_ref                # (960,)
        row_sums = pv_norm.sum(axis=1, keepdims=True) + 1e-12
        c_t = pv_norm / row_sums                                 # (960, 52), sums to 1 per row
        fault_period_idx = np.arange(FAULT_START - 1, 960)
        alarmed_idx = fault_period_idx[alarm_t[fault_period_idx]]
        alarmed_c.extend(c_t[alarmed_idx])
        all_fault_c.extend(c_t[fault_period_idx])
        n_runs += 1
    if n_runs == 0:
        continue
    if len(alarmed_c) > 0:
        contribution = np.mean(alarmed_c, axis=0)
        n_windows = len(alarmed_c)
    else:
        # No alarmed windows found (e.g. IDV3/9/15, consistent with their
        # near-zero FDR): fall back to all fault-period windows so the figure
        # still renders something, rather than an empty/undefined row.
        contribution = np.mean(all_fault_c, axis=0)
        n_windows = len(all_fault_c)

    for j, var in enumerate(FEAT):
        contrib_records.append({
            'fault_id': fid, 'variable': var, 'var_idx': j,
            'contribution': float(contribution[j]),
            'n_alarmed_windows': n_windows,
        })

contrib_df = pd.DataFrame(contrib_records)
contrib_df.to_csv(OUTDIR / "variable_contributions.csv", index=False)
print("Saved variable_contributions.csv")

# ── Figure: Per-variable contribution heatmap (all faults × top 20 vars) ─────
# Rows = fault types, columns = process variables (see Section 3.7 text).
pivot = contrib_df.pivot(index='fault_id', columns='variable', values='contribution')
# Select top 20 variables by max contribution across any fault
top_vars = pivot.abs().max(axis=0).nlargest(20).index.tolist()
pivot_top = pivot[top_vars]
# Row-normalized for visibility; contribution is already non-negative
# (Eq. 25), so this rescales each fault's own max to 1.0 without introducing
# the negative values a diverging colormap would suggest.
pivot_norm = pivot_top.div(pivot_top.max(axis=1) + 1e-9, axis=0)

_, ax = plt.subplots(figsize=(18, 8))
sns.heatmap(pivot_norm, cmap='Reds', vmin=0, ax=ax,
            cbar_kws={'label': 'Row-normalized contribution'}, linewidths=0.3,
            annot=False)
ax.set_title('Per-Variable Reconstruction Contribution by Fault\n'
             '(Top 20 most discriminative variables; normalized by calibration std, Eq. 25)', fontsize=13)
ax.set_xlabel('Process Variable')
ax.set_ylabel('Fault ID')
ax.tick_params(axis='x', rotation=45, labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig16_variable_contribution_heatmap.png", dpi=150)
plt.close()
print("Saved fig16_variable_contribution_heatmap.png")

# ── Figure: Time-series contribution for key faults ──────────────────────────
KEY_FAULTS_PLOT = [1, 3, 9, 15, 16]
_, axes = plt.subplots(len(KEY_FAULTS_PLOT), 1, figsize=(14, 4*len(KEY_FAULTS_PLOT)))

for ax_idx, fid in enumerate(KEY_FAULTS_PLOT):
    ax = axes[ax_idx]
    # Get top 5 contributing variables for this fault
    top5_this = (contrib_df[contrib_df['fault_id'] == fid]
                 .sort_values('contribution', ascending=False)
                 .head(5)['variable'].tolist())

    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS[:1]):   # one run for clarity
        if len(X_run) < 960:
            continue
        pv = get_per_var_scores(X_run)    # (960, 52)
        for var in top5_this:
            j = FEAT.index(var)
            ax.plot(range(1, 961), pv[:, j], linewidth=1.0, alpha=0.8, label=var)

    ax.axvline(FAULT_START, color='red', linestyle='--', linewidth=1.5, label='Fault injection')
    ax.set_title(f'Fault {fid}: {FAULT_NAMES[fid]} — Per-Variable Contribution')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Reconstruction Error (per variable)')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

plt.suptitle('Per-Variable Reconstruction Error Over Time', fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig17_per_var_time_series.png", dpi=150)
plt.close()
print("Saved fig17_per_var_time_series.png")

# ── Detection Delay CDF across FAULT_TEST_RUNS (censoring-aware) ─────────────
# Scored with NormMaxPV (matching the manuscript's Figure 7 text/caption --
# the previous version scored with MSE, a content mismatch: round-2 review,
# point 6a). Undetected runs are tracked explicitly (NaN, not the in-range
# sentinel 800 used previously, which silently let them count as "detected at
# delay 800" once the >800 filter became a no-op) and the plotted curve is a
# proper cause-specific cumulative-incidence curve: at delay x, the height is
# (# runs detected by x) / (total runs in the tier), so its asymptote is the
# tier's detection probability, not forced to 100%.
print("\n2. DETECTION DELAY CDF (NormMaxPV, censoring-aware)")
tier_delays = {'All': [], 'Easy': [], 'Hard': []}   # detected delays only
tier_totals = {'All': 0, 'Easy': 0, 'Hard': 0}

for fid in fault_ids:
    tiers_for_fault = ['All'] + (['Easy'] if fid in EASY_FAULTS else []) + \
                       (['Hard'] if fid in HARD_FAULTS else [])
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        Xs = scaler.transform(X_run).astype(np.float32)
        n  = len(Xs)
        W_arr = make_run_windows(Xs, WINDOW)
        pv_w = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                b = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                recon, _ = model(b)
                pv_w.append(tensor_to_numpy(((b-recon)**2).mean(dim=2)))
        pv_w = np.concatenate(pv_w)                       # (n-W+1, 52)
        normmaxpv_w = (pv_w / std_pv_ref).max(axis=1)      # (n-W+1,)
        alarm_w = normmaxpv_w > tau_normmaxpv_ref
        fault_w_start = FAULT_START - WINDOW  # first window whose last sample is the fault start
        first = np.where(alarm_w[fault_w_start:])[0]
        delay = int(first[0]) + 1 if len(first) > 0 else None   # None = never detected (censored)

        for tier in tiers_for_fault:
            tier_totals[tier] += 1
            if delay is not None:
                tier_delays[tier].append(delay)

_, ax = plt.subplots(figsize=(10, 6))
delay_summary = {}
for label, color in [('All', '#333333'), ('Easy', '#2196F3'), ('Hard', '#E85D4A')]:
    delays = np.array(sorted(tier_delays[label]))
    total = tier_totals[label]
    n_detected = len(delays)
    det_prob = n_detected / total if total else float('nan')
    cond_median = float(np.median(delays)) if n_detected else float('nan')
    cond_mean = float(np.mean(delays)) if n_detected else float('nan')
    delay_summary[label] = {
        'n_total': total, 'n_detected': n_detected,
        'detection_probability': det_prob,
        'conditional_median_delay': cond_median,
        'conditional_mean_delay': cond_mean,
    }
    print(f"  {label:5s}: n_total={total}  n_detected={n_detected}  "
          f"detection_prob={det_prob*100:.1f}%  "
          f"conditional median delay={cond_median:.1f}  mean={cond_mean:.1f}")
    if n_detected == 0:
        continue
    cum_detected = np.arange(1, n_detected + 1)
    cdf = cum_detected / total * 100   # asymptotes at detection_probability*100, not 100
    ax.plot(delays, cdf, linewidth=2,
            label=f'{label} faults (n={total}, detected={n_detected}, '
                  f'{det_prob*100:.0f}% ever-detected)', color=color)

pd.DataFrame(delay_summary).T.to_csv(OUTDIR / "detection_delay_reference_seed_summary.csv")
print(f"Saved detection_delay_reference_seed_summary.csv "
      f"(illustrative reference-seed={REF_SEED}; see results/agg/detection_delay_agg.csv "
      f"for the pooled 5-seed censoring-aware summary)")

ax.set_xlabel('Detection Delay (samples after fault injection)', fontsize=12)
ax.set_ylabel('Cumulative Detection Rate (%)\n(asymptotes at the tier detection probability, not 100%)', fontsize=11)
ax.set_title('Detection Delay CDF — NormMaxPV + Conformal Prediction (censoring-aware)', fontsize=13)
ax.set_xlim(0, 200)
ax.set_ylim(0, 102)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig18_detection_delay_cdf.png", dpi=150)
plt.close()
print("Saved fig18_detection_delay_cdf.png")

del faulty_cache
gc.collect()

# ── Summary figure: Mean FDR bar chart by fault category (reference model) ───
# NOTE: this uses the single reference-seed model (models/seeds/seed_42/,
# seed=42) purely to illustrate the category breakdown. The paper's actual
# statistical-significance claims are computed across the full multi-seed
# ensemble by aggregate_multiseed.py (results/agg/significance_tests.csv) --
# see that script rather than this one for the reported p-values.
print("\n3. CATEGORY COMPARISON FIGURE (reference model, seed=42)")
tcn_res  = pd.read_csv(OUTDIR / "seeds" / "seed_42" / "tcnae_results.csv")
pca_res  = pd.read_csv(OUTDIR / "pca_baseline_results.csv")
pca_best = pca_res[pca_res['method'] == 'PCA-Q(n=7)'].set_index('fault_id')
tcn_mse  = tcn_res[tcn_res['method'] == 'TCNAE-MSE-CP'].set_index('fault_id')
fids = sorted(set(pca_best.index) & set(tcn_mse.index))

categories = {
    'Easy\n(1,2,4-8,12-14)': [f for f in fids if f in EASY_FAULTS],
    'Medium\n(10,11)':       [f for f in fids if f in {10, 11}],
    'Hard\n(3,9,15-20)':     [f for f in fids if f in HARD_FAULTS],
    'All': fids
}
pca_cat  = [np.mean([pca_best.loc[f,'FDR']*100 for f in fs]) for fs in categories.values()]
tcn_cat  = [np.mean([tcn_mse.loc[f,'FDR']*100  for f in fs]) for fs in categories.values()]

_, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, pca_cat, w, label='PCA-Q (n=7)', color='#5C85D6', alpha=0.9)
b2 = ax.bar(x + w/2, tcn_cat, w, label='TCN-AE + CP', color='#E85D4A', alpha=0.9)
ax.bar_label(b1, [f'{v:.1f}%' for v in pca_cat], fontsize=9)
ax.bar_label(b2, [f'{v:.1f}%' for v in tcn_cat], fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(list(categories.keys()), fontsize=10)
ax.set_ylabel('Mean FDR (%)', fontsize=12)
ax.set_title('Mean FDR by Fault Category: TCN-AE+CP vs PCA-Q Baseline\n'
             '(reference seed=42; see results/agg/significance_tests.csv for the '
             'multi-seed significance test)', fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig19_category_comparison.png", dpi=150)
plt.close()
print("\nSaved fig19_category_comparison.png")

print("\n" + "="*65)
print("CONTRIBUTION ANALYSIS COMPLETE")
print("="*65)
