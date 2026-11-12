"""
Figures for the round-2 peer review robustness checks (points 2 and 7).
Run after aggregate_multiseed.py.
Outputs: fig27_noise_robustness.png, fig28_overlap_robustness.png
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import paper_style
from tcnae_utils import AGG_DIR, MAIN_SEEDS, OUTDIR

paper_style.apply()

# ── fig27: Sensor-noise robustness (point 7) ──────────────────────────────────
print("fig27: Sensor-noise robustness ...")
noise = pd.read_csv(AGG_DIR / "noise_robustness_agg.csv").sort_values('noise_level')

_, axes = plt.subplots(1, 2, figsize=(paper_style.W2 + 1.0, 3.5))
ax = axes[0]
ax.errorbar(noise['noise_level'] * 100, noise['FDR_all_mean'] * 100,
            yerr=(noise['FDR_all_std'] ** 2) ** 0.5 * 100, fmt='bo-', capsize=3,
            label='All faults', linewidth=1.5, markersize=5)
ax.errorbar(noise['noise_level'] * 100, noise['FDR_hard_mean'] * 100,
            yerr=(noise['FDR_hard_std'] ** 2) ** 0.5 * 100, fmt='rs--', capsize=3,
            label='Hard faults', linewidth=1.5, markersize=5)
ax.set_xlabel('Sensor noise (% of standardized unit variance)', fontsize=9)
ax.set_ylabel('Avg FDR (%)', fontsize=9)
ax.set_title('Detection rate under sensor noise\n(tau fixed from clean calibration)', fontsize=9)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

ax = axes[1]
ax.errorbar(noise['noise_level'] * 100, noise['FAR_all_mean'] * 100,
            yerr=(noise['FAR_all_std'] ** 2) ** 0.5 * 100, fmt='go-', capsize=3,
            linewidth=1.5, markersize=5)
ax.axhline(1.0, color='k', lw=1.0, linestyle='--', alpha=0.5, label='Target alpha=1%')
ax.set_xlabel('Sensor noise (% of standardized unit variance)', fontsize=9)
ax.set_ylabel('Avg FAR (%)', fontsize=9)
ax.set_title('False alarm rate under sensor noise', fontsize=9)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.35)

plt.suptitle(f'NormMaxPV Robustness to Sensor Noise (n={len(MAIN_SEEDS)} seeds)', fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig27_noise_robustness.png")
plt.close()
print("  Saved fig27_noise_robustness.png")

# ── fig28: Overlapping-window dependence robustness (point 2) ────────────────
print("fig28: Overlapping-window dependence robustness ...")
overlap = pd.read_csv(AGG_DIR / "overlap_robustness_agg.csv")
scheme_labels = {
    'overlapping_stride1': 'Overlapping\n(stride=1, main scheme)',
    'non_overlapping_strideW': 'Non-overlapping\n(stride=W=20)',
    'run_level_block': 'Run-level block\n(1 score/run)',
}
overlap['label'] = overlap['scheme'].map(scheme_labels)
overlap = overlap.set_index('scheme').loc[list(scheme_labels.keys())].reset_index()

_, ax = plt.subplots(figsize=(paper_style.W1 + 0.8, 3.5))
x = range(len(overlap))
ax.bar(x, overlap['far_test_mean'] * 100, yerr=(overlap['far_test_std'] ** 2) ** 0.5 * 100,
       capsize=4, color=['#5C85D6', '#E85D4A', '#4CAF50'], alpha=0.85)
ax.axhline(1.0, color='k', lw=1.2, linestyle='--', alpha=0.6, label='Target alpha=1%')
ax.set_xticks(x)
ax.set_xticklabels(overlap['label'], fontsize=8)
ax.set_ylabel('Empirical test FAR (%)', fontsize=9)
ax.set_title(f'FAR Under Three Exchangeable-Unit Definitions\n(NormMaxPV; n={len(MAIN_SEEDS)} seeds)', fontsize=9.5)
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.35)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig28_overlap_robustness.png")
plt.close()
print("  Saved fig28_overlap_robustness.png")

print("\nDone.")
