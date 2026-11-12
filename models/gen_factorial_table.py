"""
Round-2 peer review, point 4: render the 4-architecture x 3-score factorial
(results/agg/factorial_agg.csv, built by aggregate_multiseed.py) as a paper
table and a heatmap figure, replacing the two-architecture comparison
previously in Section 5.3.
Run after aggregate_multiseed.py.
Outputs: results/factorial_table.csv, results/fig26_factorial_heatmap.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import paper_style
from tcnae_utils import AGG_DIR, MAIN_SEEDS, OUTDIR

paper_style.apply()

HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
ARCH_ORDER = ["SampleAE", "DenseAE", "ConvAE", "TCN-AE"]
SCORE_ORDER = ["MSE", "MaxPV", "NormMaxPV"]

print("Loading factorial_agg.csv ...")
df = pd.read_csv(AGG_DIR / "factorial_agg.csv")

# ── Table: mean FDR% (+/- seed std) per architecture x score, All and Hard ───
rows = []
for arch in ARCH_ORDER:
    row = {"architecture": arch}
    for score in SCORE_ORDER:
        sub = df[(df["architecture"] == arch) & (df["score"] == score)]
        all_mean = sub["FDR_mean"].mean() * 100
        all_std = sub["FDR_std"].mean() * 100
        hard = sub[sub["fault_id"].isin(HARD_FAULTS)]
        hard_mean = hard["FDR_mean"].mean() * 100
        row[f"{score}_All_FDR"] = round(all_mean, 1)
        row[f"{score}_All_FDR_std"] = round(all_std, 2)
        row[f"{score}_Hard_FDR"] = round(hard_mean, 1)
    rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUTDIR / "factorial_table.csv", index=False)
print(f"Saved {OUTDIR / 'factorial_table.csv'}")
print(table.to_string(index=False))

# ── Figure: heatmap, architectures x scores, All-fault avg FDR% ──────────────
heat_all = np.zeros((len(ARCH_ORDER), len(SCORE_ORDER)))
heat_hard = np.zeros((len(ARCH_ORDER), len(SCORE_ORDER)))
for i, arch in enumerate(ARCH_ORDER):
    for j, score in enumerate(SCORE_ORDER):
        sub = df[(df["architecture"] == arch) & (df["score"] == score)]
        heat_all[i, j] = sub["FDR_mean"].mean() * 100
        heat_hard[i, j] = sub[sub["fault_id"].isin(HARD_FAULTS)]["FDR_mean"].mean() * 100

_, axes = plt.subplots(1, 2, figsize=(paper_style.W2, 3.6))
for ax, mat, title in [(axes[0], heat_all, "All 20 faults"), (axes[1], heat_hard, "Hard tier (8 faults)")]:
    sns.heatmap(mat, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
                xticklabels=SCORE_ORDER, yticklabels=ARCH_ORDER, ax=ax,
                cbar_kws={'label': 'Mean FDR (%)'}, linewidths=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Nonconformity score", fontsize=9)
plt.suptitle(f"Architecture x Score Factorial (mean FDR %, n={len(MAIN_SEEDS)} seeds)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig26_factorial_heatmap.png")
plt.close()
print(f"Saved {OUTDIR}/fig26_factorial_heatmap.png")
