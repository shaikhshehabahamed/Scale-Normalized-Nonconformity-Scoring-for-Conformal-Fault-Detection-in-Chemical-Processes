"""
Regenerate the main single-model-comparison figures (fig08-11, fig20) from
the aggregated multi-seed results, with error bars showing std across seeds.
These used to be generated inline (single-run, no error bars) by
evaluate_tcnae.py and latent_mahal_eval.py; they now live here so they run
exactly once, after aggregate_multiseed.py, on the full multi-seed ensemble.

Inputs: results/agg/all_results_agg.csv, results/agg/tcnae_mahal_results_agg.csv
Outputs: fig08_fdr_comparison.png, fig09_fdr_scatter.png,
         fig10_add_comparison.png, fig11_full_heatmap.png,
         fig20_four_way_comparison.png, all_results.csv (mean-only, for any
         downstream script still expecting the legacy single-value schema)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import AGG_DIR, MAIN_SEEDS, OUTDIR

HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}

print("Loading aggregated multi-seed results...")
all_agg = pd.read_csv(AGG_DIR / "all_results_agg.csv")
mahal_agg = pd.read_csv(AGG_DIR / "tcnae_mahal_results_agg.csv")
fault_ids = sorted(all_agg['fault_id'].unique())

# Legacy-schema all_results.csv (mean values only) for any script still reading it
legacy = all_agg.rename(columns={"FAR_mean": "FAR", "FDR_mean": "FDR", "ADD_mean": "ADD"})
legacy = legacy[["fault_id", "fault_name", "method", "FAR", "FDR", "ADD"]]
legacy.to_csv(OUTDIR / "all_results.csv", index=False)
print(f"Saved {OUTDIR / 'all_results.csv'} (mean-only legacy schema)")

pca_best = all_agg[all_agg['method'] == 'PCA-Q(n=7)'].set_index('fault_id')
tcn_mse = all_agg[all_agg['method'] == 'TCNAE-MSE-CP'].set_index('fault_id')

x = np.arange(len(fault_ids))
w = 0.35
pca_fdrs = [pca_best.loc[f, 'FDR_mean']*100 if f in pca_best.index else 0.0 for f in fault_ids]
tcn_fdrs = [tcn_mse.loc[f, 'FDR_mean']*100 if f in tcn_mse.index else 0.0 for f in fault_ids]
tcn_fdr_stds = [tcn_mse.loc[f, 'FDR_std']*100 if f in tcn_mse.index else 0.0 for f in fault_ids]

# ─ Figure 1: FDR grouped bar (TCN vs PCA), TCN bars with seed-std error bars ──
print("\nfig08: FDR comparison bar chart (with seed-std error bars)...")
_, ax = plt.subplots(figsize=(16, 7))
bars1 = ax.bar(x - w/2, pca_fdrs, w, label='PCA-Q (n=7, baseline; deterministic)',
               color='#5C85D6', alpha=0.85)
bars2 = ax.bar(x + w/2, tcn_fdrs, w, yerr=tcn_fdr_stds, capsize=3,
               label=f'TCN-AE + CP (proposed, mean +/- std, n={len(MAIN_SEEDS)} seeds)',
               color='#E85D4A', alpha=0.85)
for f_idx, fid in enumerate(fault_ids):
    if fid in HARD_FAULTS:
        bars1[f_idx].set_hatch('///')
        bars2[f_idx].set_hatch('///')
ax.set_xticks(x)
ax.set_xticklabels([str(f) for f in fault_ids], fontsize=10)
ax.set_xlabel('Fault ID', fontsize=12)
ax.set_ylabel('Fault Detection Rate (%)', fontsize=12)
ax.set_title(f'FDR: TCN-AE + Conformal Prediction vs. PCA-Q Baseline\n'
             f'(Hatched bars = hard faults; target FAR = 1%; TCN-AE error bars = std across {len(MAIN_SEEDS)} seeds)',
             fontsize=13)
ax.legend(fontsize=11)
ax.set_ylim(0, 112)
ax.axhline(100, color='k', linestyle='--', linewidth=0.8, alpha=0.4)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig08_fdr_comparison.png", dpi=150)
plt.close()
print("Saved fig08_fdr_comparison.png")

# ─ Figure 2: FDR scatter (PCA vs TCN mean, with vertical error bars) ─────────
print("fig09: FDR scatter with seed-std error bars...")
_, ax = plt.subplots(figsize=(8, 8))
colors = ['#E85D4A' if f in HARD_FAULTS else '#5C85D6' for f in fault_ids]
ax.errorbar(pca_fdrs, tcn_fdrs, yerr=tcn_fdr_stds, fmt='none', ecolor='gray',
            alpha=0.5, capsize=2, zorder=4)
ax.scatter(pca_fdrs, tcn_fdrs, c=colors, s=120, zorder=5, edgecolors='k', linewidths=0.5)
for f_idx, fid in enumerate(fault_ids):
    ax.annotate(str(fid), (pca_fdrs[f_idx], tcn_fdrs[f_idx]),
                textcoords='offset points', xytext=(5, 3), fontsize=8)
ax.plot([0, 100], [0, 100], 'k--', linewidth=1, alpha=0.5, label='Equal performance')
ax.fill_between([0, 100], [0, 100], [100, 100], alpha=0.05, color='green')
ax.set_xlabel('PCA-Q FDR (%)', fontsize=12)
ax.set_ylabel(f'TCN-AE + CP FDR (%)  (mean +/- std, n={len(MAIN_SEEDS)} seeds)', fontsize=12)
ax.set_title('Per-Fault FDR: TCN-AE vs PCA-Q\n(Red = hard faults)', fontsize=12)
ax.set_xlim(-2, 105)
ax.set_ylim(-2, 105)
ax.grid(alpha=0.3)
legend_elements = [Patch(facecolor='#5C85D6', label='Easy faults'),
                   Patch(facecolor='#E85D4A', label='Hard faults')]
ax.legend(handles=legend_elements, fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig09_fdr_scatter.png", dpi=150)
plt.close()
print("Saved fig09_fdr_scatter.png")

# ─ Figure 3: ADD comparison (lower is better), with error bars ───────────────
print("fig10: ADD comparison with seed-std error bars...")
_, ax = plt.subplots(figsize=(16, 6))
pca_adds = [pca_best.loc[f, 'ADD_mean'] if f in pca_best.index else 0.0 for f in fault_ids]
tcn_adds = [tcn_mse.loc[f, 'ADD_mean'] if f in tcn_mse.index else 0.0 for f in fault_ids]
tcn_add_stds = [tcn_mse.loc[f, 'ADD_std'] if f in tcn_mse.index else 0.0 for f in fault_ids]
ax.bar(x - w/2, pca_adds, w, label='PCA-Q (n=7)', color='#5C85D6', alpha=0.85)
ax.bar(x + w/2, tcn_adds, w, yerr=tcn_add_stds, capsize=3,
       label=f'TCN-AE + CP (mean +/- std, n={len(MAIN_SEEDS)} seeds)', color='#E85D4A', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([str(f) for f in fault_ids])
ax.set_xlabel('Fault ID')
ax.set_ylabel('Avg Detection Delay (samples)')
ax.set_title('Average Detection Delay -- Lower is Better')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig10_add_comparison.png", dpi=150)
plt.close()
print("Saved fig10_add_comparison.png")

# ─ Figure 4: Full results heatmap (mean FDR across seeds) ────────────────────
print("fig11: Full heatmap (seed-mean FDR)...")
methods_heat = ['PCA-Q(n=7)', 'PCA-Comb(n=29)', 'TCNAE-MSE-CP', 'TCNAE-MaxPV-CP']
heat_data = {}
for m in methods_heat:
    sub = all_agg[all_agg['method'] == m]
    row = {}
    for fid in fault_ids:
        v = sub[sub['fault_id'] == fid]['FDR_mean'].values
        row[fid] = round(float(v[0]) * 100, 1) if len(v) > 0 else 0.0
    heat_data[m] = row

heat_df = pd.DataFrame(heat_data).T
_, ax = plt.subplots(figsize=(20, 4))
sns.heatmap(heat_df, annot=True, fmt='.0f', cmap='RdYlGn',
            vmin=0, vmax=100, ax=ax,
            cbar_kws={'label': 'Mean FDR (%)'}, linewidths=0.4, annot_kws={'size': 8})
ax.set_title(f'Fault Detection Rate (%) -- All Methods and All Faults (mean across {len(MAIN_SEEDS)} seeds)',
             fontsize=13)
ax.set_xlabel('Fault ID')
ax.set_ylabel('Method')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig11_full_heatmap.png", dpi=150)
plt.close()
print("Saved fig11_full_heatmap.png")

# ─ Figure 5: Four-way comparison (PCA-Q, TCN-MSE, TCN-Mahal, TCN-OR) ─────────
print("fig20: Four-way comparison with seed-std error bars...")
tcn_mse_df = mahal_agg[mahal_agg['method'] == 'TCNAE-MSE-CP'].set_index('fault_id')
tcn_mh_df = mahal_agg[mahal_agg['method'] == 'TCNAE-Mahal-CP'].set_index('fault_id')
tcn_or_df = mahal_agg[mahal_agg['method'] == 'TCNAE-OR(MSE+Mahal)-CP'].set_index('fault_id')

_, ax = plt.subplots(figsize=(18, 7))
w4 = 0.22
for k, (df_m, color, label) in enumerate([
    (pca_best, '#5C85D6', 'PCA-Q (n=7)'),
    (tcn_mse_df, '#E85D4A', 'TCN-AE+CP (MSE)'),
    (tcn_mh_df, '#4CAF50', 'TCN-AE+CP (Mahal)'),
    (tcn_or_df, '#FF9800', 'TCN-AE+CP (MSE OR Mahal)'),
]):
    vals = [df_m.loc[f, 'FDR_mean']*100 if f in df_m.index else 0.0 for f in fault_ids]
    errs = [df_m.loc[f, 'FDR_std']*100 if f in df_m.index else 0.0 for f in fault_ids]
    ax.bar(x + (k - 1.5) * w4, vals, w4, yerr=errs, capsize=2, label=label, color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([str(f) for f in fault_ids], fontsize=9)
ax.set_xlabel('Fault ID', fontsize=12)
ax.set_ylabel('FDR (%)', fontsize=12)
ax.set_title(f'FDR: Four-Way Comparison (PCA-Q vs TCN-AE Detection Statistics)\n'
             f'Target FAR = 1%; error bars = std across {len(MAIN_SEEDS)} seeds', fontsize=12)
ax.legend(fontsize=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig20_four_way_comparison.png", dpi=150)
plt.close()
print("Saved fig20_four_way_comparison.png")

print("\nDone. All main comparison figures regenerated from the multi-seed aggregate.")
