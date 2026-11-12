"""
Final consolidated comparison: all methods vs all faults, aggregated across
the multi-seed protocol (MAIN_SEEDS in tcnae_utils.py).
Run this AFTER aggregate_multiseed.py.
Produces:
  table1_full.csv  -- per-fault, per-method table with FDR_<method>/ADD_<method>
                       (seed mean) and FDR_std_<method>/ADD_std_<method>
                       (seed std) columns. Input for gen_paper_figures.py.
Statistical significance is NOT recomputed here -- it is read from
results/agg/significance_tests.csv (seed-level tests), which
aggregate_multiseed.py already produced.
Figures (fig22-24) are generated from table1_full.csv by gen_paper_figures.py.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import AGG_DIR, MAIN_SEEDS, OUTDIR

os.makedirs(OUTDIR, exist_ok=True)

HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
EASY_FAULTS = {1, 2, 4, 5, 6, 7, 8, 12, 13, 14}

FAULT_NAMES = {
    1:"A/C feed ratio", 2:"B comp step", 3:"D feed temp step",
    4:"Reactor CW temp", 5:"Condenser CW temp", 6:"A feed loss",
    7:"C pressure loss", 8:"A,B,C feed comp", 9:"D feed temp random",
    10:"C feed temp random", 11:"Reactor CW random", 12:"Condenser CW random",
    13:"Reaction kinetics", 14:"Reactor CW valve", 15:"Cond. CW valve",
    16:"Unknown-16", 17:"Unknown-17", 18:"Unknown-18",
    19:"Unknown-19", 20:"Unknown-20"
}

METHOD_ORDER = [
    'PCA-Q(n=7)', 'PCA-Comb(n=29)', 'DenseAE-CP', 'TCNAE-MSE-CP', 'TCNAE-MaxPV-CP',
    'TCNAE-OR(MSE+MaxPV)-CP', 'NormMaxPV-solo', 'OR(MSE+NormMaxPV)', 'OR(MSE+PVBC)', 'TCNAE-Mahal-CP'
]
METHOD_LABELS = {
    'PCA-Q(n=7)':              'PCA-Q (n=7)',
    'PCA-Comb(n=29)':          'PCA-Comb (n=29)',
    'DenseAE-CP':              'Dense AE+CP',
    'TCNAE-MSE-CP':            'TCN-AE+CP (MSE)',
    'TCNAE-MaxPV-CP':          'TCN-AE+CP (MaxPV)',
    'TCNAE-OR(MSE+MaxPV)-CP':  'TCN-AE+CP (OR: MSE+MaxPV)',
    'NormMaxPV-solo':          'TCN-AE+CP (NormMaxPV)',
    'OR(MSE+NormMaxPV)':       'TCN-AE+CP (OR: MSE+NormMaxPV)',
    'OR(MSE+PVBC)':            'TCN-AE+CP (OR: MSE+PVBC)',
    'TCNAE-Mahal-CP':          'TCN-AE+CP (Mahal)',
}


def require_methods(df: pd.DataFrame, source: str, methods: list[str]) -> None:
    missing = sorted(set(methods) - set(df["method"].unique()))
    if missing:
        raise ValueError(f"{source} is missing required methods: {missing}")


print("="*65)
print(f"FINAL CONSOLIDATED COMPARISON (aggregated across {len(MAIN_SEEDS)} seeds: {MAIN_SEEDS})")
print("="*65)

# ── Load aggregated CSVs (mean/std across seeds; PCA is deterministic, std=0) ─
all_agg = pd.read_csv(AGG_DIR / "all_results_agg.csv")
require_methods(all_agg, "all_results_agg.csv",
                 ["PCA-Q(n=7)", "PCA-Comb(n=29)", "DenseAE-CP", "TCNAE-MSE-CP",
                  "TCNAE-MaxPV-CP", "TCNAE-OR(MSE+MaxPV)-CP", "TCNAE-Mahal-CP",
                  "NormMaxPV-solo", "OR(MSE+NormMaxPV)", "OR(MSE+PVBC)"])

fault_ids = sorted(all_agg['fault_id'].unique())

# ── Summary stats ──────────────────────────────────────────────────────────────
print("\n" + "-"*74)
print(f"{'Method':<35} {'Avg FDR':>12} {'Hard FDR':>12} {'Easy FDR':>12} {'Avg FAR':>10}")
print("-"*74)

for meth in METHOD_ORDER:
    sub = all_agg[all_agg['method'] == meth]
    if sub.empty:
        continue
    hard = sub[sub['fault_id'].isin(HARD_FAULTS)]['FDR_mean'].mean()
    easy = sub[sub['fault_id'].isin(EASY_FAULTS)]['FDR_mean'].mean()
    avg  = sub['FDR_mean'].mean()
    avg_std = sub['FDR_std'].mean()
    far  = sub['FAR_mean'].mean()
    n = int(sub['FDR_n'].iloc[0])
    print(f"  {METHOD_LABELS.get(meth, meth):<33} {avg*100:>6.2f}%+/-{avg_std*100:<4.2f} "
          f"{hard*100:>10.2f}% {easy*100:>10.2f}% {far*100:>8.3f}%  (n_seeds={n})")

# ── Significance tests (already computed at the seed level) ──────────────────
print("\n" + "-"*74)
print("Seed-level significance tests (see results/agg/significance_tests.csv "
      "for full detail):")
sig_path = AGG_DIR / "significance_tests.csv"
if sig_path.exists():
    sig_df = pd.read_csv(sig_path)
    for _, row in sig_df.iterrows():
        p = row.get("p_ttest_1samp")
        if pd.isna(p):
            p = row.get("p_ttest_rel")
        mark = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {row['comparison']:<55} n_seeds={int(row['n_seeds'])}  p={p:.4f}  ({mark} at p<0.05)")
else:
    print("  [!] results/agg/significance_tests.csv not found -- run aggregate_multiseed.py first")

# ── Build wide table for paper (mean + std columns) ───────────────────────────
rows = []
for fid in sorted(fault_ids):
    row = {'fault_id': fid, 'fault_name': FAULT_NAMES.get(fid, f'F{fid}')}
    for meth in METHOD_ORDER:
        sub = all_agg[(all_agg['fault_id'] == fid) & (all_agg['method'] == meth)]
        if sub.empty:
            row[f'FDR_{meth}'] = np.nan
            row[f'FDR_std_{meth}'] = np.nan
            row[f'ADD_{meth}'] = np.nan
            row[f'ADD_std_{meth}'] = np.nan
        else:
            row[f'FDR_{meth}'] = round(sub['FDR_mean'].values[0] * 100, 2)
            row[f'FDR_std_{meth}'] = round(sub['FDR_std'].values[0] * 100, 2)
            row[f'ADD_{meth}'] = round(sub['ADD_mean'].values[0], 1)
            row[f'ADD_std_{meth}'] = round(sub['ADD_std'].values[0], 1)
    rows.append(row)

table1 = pd.DataFrame(rows)
table1.to_csv(f"{OUTDIR}/table1_full.csv", index=False)
print(f"\nSaved table1_full.csv  ({len(table1)} faults x {len(METHOD_ORDER)} methods, mean+std)")
print("\nRun gen_paper_figures.py to generate fig22-24 from this table.")
print("Run gen_main_comparison_figs.py to generate fig08-11 and fig20 from all_results_agg.csv.")

print("\n" + "="*65)
print("FINAL COMPARISON COMPLETE")
print("="*65)
