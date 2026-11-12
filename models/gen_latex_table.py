"""
Generate LaTeX table from table1_full.csv for the paper.
Run after final_comparison.py has produced table1_full.csv (mean +/- std
across MAIN_SEEDS; see tcnae_utils.py).
"""
import decimal
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import MAIN_SEEDS, OUTDIR


def round1(x: float):
    """Round-half-up to 1 decimal via Decimal. float round()/pandas .round()
    use binary floating point and can mis-round exact .x5 boundaries (e.g.
    round(99.65, 1) == 99.6, not 99.7) -- str(x) first captures the value as
    the decimal literal the CSV actually stored."""
    if pd.isna(x):
        return np.nan
    return float(decimal.Decimal(str(x)).quantize(decimal.Decimal("0.1"), rounding=decimal.ROUND_HALF_UP))

HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
EASY_FAULTS = {1, 2, 4, 5, 6, 7, 8, 12, 13, 14}

df = pd.read_csv(OUTDIR / "table1_full.csv")

# Determine which methods are present. table1_full.csv now also has
# FDR_std_<method> columns (also start with "FDR_"), so exclude those.
fdr_cols = [c for c in df.columns if c.startswith('FDR_') and not c.startswith('FDR_std_')]
methods  = [c[4:] for c in fdr_cols]  # strip 'FDR_'

METHOD_SHORT = {
    'PCA-Q(n=7)':              'PCA-Q',
    'PCA-Comb(n=29)':          'PCA-Comb',
    'DenseAE-CP':              'DAE+CP',
    'TCNAE-MSE-CP':            'TCN (MSE)',
    'TCNAE-MaxPV-CP':          'TCN (MaxPV)',
    'TCNAE-OR(MSE+MaxPV)-CP':  'TCN (OR)',
    'NormMaxPV-solo':          'TCN (NormMaxPV)',
    'OR(MSE+NormMaxPV)':       'TCN (OR+Norm)',
    'OR(MSE+PVBC)':            'TCN (OR+PVBC)',
    'TCNAE-Mahal-CP':          'TCN (Mahal)',
}

FAULT_NAMES_SHORT = {
    1:"A/C feed ratio",  2:"B comp. step",    3:"D feed temp (step)",
    4:"Reactor CW temp", 5:"Condenser CW",     6:"A feed loss",
    7:"C press. loss",   8:"A,B,C feed comp.", 9:"D feed temp (rand.)",
   10:"C feed temp (r.)",11:"Reactor CW (r.)",12:"Condenser CW (r.)",
   13:"React. kinetics", 14:"React. CW valve",15:"Cond. CW valve",
   16:"Unknown F16",     17:"Unknown F17",    18:"Unknown F18",
   19:"Unknown F19",     20:"Unknown F20",
}

# Keep the raw (unrounded) FDR columns for the summary rows below -- their
# category average must be computed from raw per-fault values and rounded
# once, matching final_comparison.py/table1_full.csv, not from values that
# were already rounded per-fault (averaging pre-rounded numbers then
# rounding again is a different, inconsistent number).
df_raw = df.copy()

# Round all FDR mean/std columns to 1 decimal (decimal-based: see round1())
for m in methods:
    if f'FDR_{m}' in df.columns:
        df[f'FDR_{m}'] = df[f'FDR_{m}'].map(round1)
    if f'FDR_std_{m}' in df.columns:
        df[f'FDR_std_{m}'] = df[f'FDR_std_{m}'].map(round1)

# Bold best value per fault (among non-Mahal, non-Mahal methods)
highlight_methods = [m for m in methods if 'Mahal' not in m]

n_seeds = len(MAIN_SEEDS)
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(
    r"\caption{Fault Detection Rate (\%, mean $\pm$ std across " + str(n_seeds) +
    r" random seeds) and Average Detection Delay (ADD) for all methods on the "
    r"Rieth 2017 TEP benchmark ($\alpha=0.01$). Hard faults (\textbf{bold rows}) "
    r"are faults 3, 9, 15--20. Best mean FDR per fault (excluding Mahal) in "
    r"\textcolor{blue}{\textbf{blue}}.}"
)
lines.append(r"\label{tab:fdr_all}")
lines.append(r"\setlength{\tabcolsep}{3pt}")

# Column spec: Fault | FaultName | FDR for each method
n_methods = len(methods)
colspec = "r l " + " r" * n_methods
lines.append(f"\\begin{{tabular}}{{{colspec}}}")
lines.append(r"\toprule")

# Header row
header = "F & Fault Type & " + " & ".join(
    f"\\rotatebox{{90}}{{{METHOD_SHORT.get(m, m)}}}"
    for m in methods
) + r" \\"
lines.append(header)
lines.append(r"\midrule")

for _, row in df.iterrows():
    fid = int(row['fault_id'])
    fname = FAULT_NAMES_SHORT.get(fid, f"F{fid}")
    is_hard = fid in HARD_FAULTS

    fdrs = {}
    fdr_stds = {}
    for m in methods:
        fdrs[m] = row.get(f'FDR_{m}', np.nan)
        fdr_stds[m] = row.get(f'FDR_std_{m}', np.nan)

    # Find best among highlight_methods (ranked by mean)
    valid = {m: fdrs[m] for m in highlight_methods if not np.isnan(fdrs.get(m, np.nan))}
    best_v = max(valid.values()) if valid else None

    cells = []
    for m in methods:
        v = fdrs[m]
        sd = fdr_stds.get(m, np.nan)
        if np.isnan(v):
            cells.append("--")
        else:
            s = f"{v:.1f}" + (rf"$\pm${sd:.1f}" if not np.isnan(sd) else "")
            # Exact tie at displayed (rounded) precision, not a fuzzy raw tolerance
            if best_v is not None and v == best_v and m in highlight_methods:
                s = r"\textcolor{blue}{\textbf{" + s + "}}"
            cells.append(s)

    row_str = f"{fid} & {fname} & " + " & ".join(cells) + r" \\"
    if is_hard:
        row_str = r"\rowcolor{gray!15}" + row_str
    lines.append(row_str)

lines.append(r"\midrule")

# Summary rows (error bar = propagated seed-std across faults in the category,
# assuming faults are independent draws: sqrt(mean(std^2) / n_faults))
for label, fset in [("All", set(range(1,21))), ("Easy", EASY_FAULTS), ("Hard", HARD_FAULTS)]:
    cells = []
    for m in methods:
        sub = df_raw[df_raw['fault_id'].isin(fset)][f'FDR_{m}'].dropna()
        std_col = f'FDR_std_{m}'
        if len(sub):
            if std_col in df_raw.columns:
                sub_std = df_raw.loc[sub.index, std_col].dropna()
                cat_std = (sub_std ** 2).mean() ** 0.5 / max(len(sub_std), 1) ** 0.5 if len(sub_std) else np.nan
                cells.append(f"{sub.mean():.1f}" + (rf"$\pm${cat_std:.1f}" if not np.isnan(cat_std) else ""))
            else:
                cells.append(f"{sub.mean():.1f}")
        else:
            cells.append("--")
    lines.append(f"\\textbf{{{label} avg}} & & " + " & ".join(cells) + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

table_latex = "\n".join(lines)
out_path = OUTDIR / "table1_latex.tex"
with open(out_path, 'w') as f:
    f.write(table_latex)
print(f"Saved {out_path}")
print(table_latex[:2000])
