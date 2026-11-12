"""
PCA baseline for TEP fault detection.
Must run AFTER train_tcnae.py (needs scaler.pkl).
Must run BEFORE evaluate_tcnae.py (produces pca_baseline_results.csv).

Methods evaluated:
  PCA-Q(n=7):    SPE/Q statistic, 7 PCs, conformal threshold at alpha=0.01
  PCA-Comb(n=29): Bonferroni OR-rule T² | Q, 29 PCs, each threshold at alpha/2

Saves: models/pca_nc29.pkl, results/pca_baseline_results.csv
Usage: uv run python models/pca_baseline.py
"""
import gc
import os
import sys
import numpy as np
import pandas as pd
import pyreadr
import joblib
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    ARCHIVE,
    FEAT,
    MODDIR,
    OUTDIR,
    cp_threshold,
    load_faulty_test_cache,
    seed_everything,
    split_faultfree_runs,
    split_fault_test_runs,
)
seed_everything(42)

os.makedirs(OUTDIR, exist_ok=True)

FAULT_START = 161
ALPHA       = 0.01
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
FAULT_NAMES = {
    1:"A/C feed ratio", 2:"B composition", 3:"D feed temp step",
    4:"Reactor CW temp", 5:"Condenser CW temp", 6:"A feed loss",
    7:"C pressure loss", 8:"A,B,C feed comp", 9:"D feed temp random",
    10:"C feed temp random", 11:"Reactor CW random", 12:"Condenser CW random",
    13:"Reaction kinetics", 14:"Reactor CW valve", 15:"Condenser CW valve",
    16:"Unknown-16", 17:"Unknown-17", 18:"Unknown-18",
    19:"Unknown-19", 20:"Unknown-20"
}


def q_score(pca: PCA, X: np.ndarray) -> np.ndarray:
    """SPE / Q statistic: per-sample squared reconstruction residual."""
    recon = pca.inverse_transform(pca.transform(X))
    return ((X - recon) ** 2).sum(axis=1)


def t2_score(pca: PCA, X: np.ndarray) -> np.ndarray:
    """Hotelling T² statistic normalised by eigenvalues."""
    scores = pca.transform(X)
    return (scores ** 2 / pca.explained_variance_).sum(axis=1)


def eval_run_metrics(scores: np.ndarray, threshold: float):
    n_pre  = FAULT_START - 1
    n_post = len(scores) - n_pre
    alarms = scores > threshold
    FAR = float(alarms[:n_pre].mean())
    FDR = float(alarms[n_pre:].mean())
    first = np.where(alarms[n_pre:])[0]
    ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
    return FAR, FDR, ADD


def eval_alarm_metrics(alarm: np.ndarray):
    """Evaluate pre-computed boolean alarm array."""
    n_pre  = FAULT_START - 1
    n_post = len(alarm) - n_pre
    FAR = float(alarm[:n_pre].mean())
    FDR = float(alarm[n_pre:].mean())
    first = np.where(alarm[n_pre:])[0]
    ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
    return FAR, FDR, ADD


print("=" * 65)
print("PCA BASELINE (Q statistic + Combined T²/Q)")
print("=" * 65)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\n1. LOADING FAULT-FREE TRAINING DATA")
scaler = joblib.load(MODDIR / "scaler.pkl")

r = pyreadr.read_r(str(ARCHIVE / "TEP_FaultFree_Training.RData"))
df_ff = list(r.values())[0]
del r
df_ff[FEAT] = df_ff[FEAT].astype(np.float32)
all_runs = sorted(df_ff["simulationRun"].unique())
# PCA's T^2/Q statistics involve no per-variable normalization step (no
# analogue of NormMaxPV's std_pv or Mahalanobis's mu_z/cov_inv), so PCA has
# no round-2-review point-1 leakage exposure and could legitimately calibrate
# on all 100 held-out runs. It uses only the 50-run cal_runs split (ignoring
# norm_runs) instead, so every compared method calibrates its threshold from
# the same amount of held-out data -- otherwise PCA would get a systematically
# more precise threshold estimate than TCN-AE/DenseAE/SampleAE/ConvAE purely
# from a larger calibration sample, which would bias the comparison.
tr_runs, _norm_runs, cal_runs = split_faultfree_runs(all_runs)

X_tr_raw  = df_ff[df_ff["simulationRun"].isin(tr_runs)][FEAT].to_numpy(copy=True)
X_cal_raw = df_ff[df_ff["simulationRun"].isin(cal_runs)][FEAT].to_numpy(copy=True)
del df_ff
gc.collect()

X_tr  = scaler.transform(X_tr_raw).astype(np.float32)
X_cal = scaler.transform(X_cal_raw).astype(np.float32)
print(f"Train: {X_tr.shape}   Cal: {X_cal.shape}")

# ── 2. Fit PCA models + calibrate ─────────────────────────────────────────────
print("\n2. FITTING PCA MODELS AND CALIBRATING")

# PCA-Q with n=7 components
pca7 = PCA(n_components=7)
pca7.fit(X_tr)
print(f"PCA-7  cumul. var: {pca7.explained_variance_ratio_.sum() * 100:.1f}%")

cal_q7 = q_score(pca7, X_cal)
tau_q7 = cp_threshold(cal_q7, ALPHA)
print(f"PCA-Q(7): tau_Q={tau_q7:.5f}  cal FAR={(cal_q7 > tau_q7).mean():.5f}")

# PCA-Comb with n=29 components
pca29 = PCA(n_components=29)
pca29.fit(X_tr)
print(f"PCA-29 cumul. var: {pca29.explained_variance_ratio_.sum() * 100:.1f}%")

cal_t2_29 = t2_score(pca29, X_cal)
cal_q29   = q_score(pca29, X_cal)
tau_t2b   = cp_threshold(cal_t2_29, ALPHA / 2)  # Bonferroni: alpha/2 each
tau_qb    = cp_threshold(cal_q29,   ALPHA / 2)
cal_far_comb = ((cal_t2_29 > tau_t2b) | (cal_q29 > tau_qb)).mean()
print(f"PCA-Comb(29): tau_T2={tau_t2b:.4f}  tau_Q={tau_qb:.5f}  "
      f"cal FAR={cal_far_comb:.5f}  (target={ALPHA})")

# Save PCA-29 model for gen_detection_showcase.py
joblib.dump(
    {"pca": pca29, "nc": 29, "tau_t2b": float(tau_t2b), "tau_qb": float(tau_qb)},
    MODDIR / "pca_nc29.pkl",
)
print("Saved pca_nc29.pkl")

del X_tr, X_cal
gc.collect()

# ── 3. Evaluate on faulty test data ───────────────────────────────────────────
print("\n3. EVALUATING ON FAULTY TEST DATA")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault (same test split as "
      f"every other method's headline result, for a fair comparison)")

records = []
for fid in fault_ids:
    res_q7   = []
    res_comb = []

    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        Xs = scaler.transform(X_run).astype(np.float32)

        sq7  = q_score(pca7,  Xs)
        st2  = t2_score(pca29, Xs)
        sq29 = q_score(pca29,  Xs)

        res_q7.append(eval_run_metrics(sq7, tau_q7))
        res_comb.append(eval_alarm_metrics((st2 > tau_t2b) | (sq29 > tau_qb)))

    m_q7   = np.mean(res_q7,   axis=0)
    m_comb = np.mean(res_comb, axis=0)
    records.append({
        "fault_id": fid, "fault_name": FAULT_NAMES[fid],
        "method": "PCA-Q(n=7)",
        "FAR": round(float(m_q7[0]),   5),
        "FDR": round(float(m_q7[1]),   5),
        "ADD": round(float(m_q7[2]),   2),
    })
    records.append({
        "fault_id": fid, "fault_name": FAULT_NAMES[fid],
        "method": "PCA-Comb(n=29)",
        "FAR": round(float(m_comb[0]), 5),
        "FDR": round(float(m_comb[1]), 5),
        "ADD": round(float(m_comb[2]), 2),
    })
    print(f"  F{fid:2d}: Q7 FDR={m_q7[1]:.4f} FAR={m_q7[0]:.4f} | "
          f"Comb FDR={m_comb[1]:.4f} FAR={m_comb[0]:.4f}")

del faulty_cache
gc.collect()

res_df = pd.DataFrame(records)
res_df.to_csv(OUTDIR / "pca_baseline_results.csv", index=False)
print("\nSaved pca_baseline_results.csv")

# ── 4. Summary ─────────────────────────────────────────────────────────────────
pca_q7  = res_df[res_df["method"] == "PCA-Q(n=7)"]
pca_c29 = res_df[res_df["method"] == "PCA-Comb(n=29)"]

hard_q7  = pca_q7[pca_q7["fault_id"].isin(HARD_FAULTS)]["FDR"].mean() * 100
hard_c29 = pca_c29[pca_c29["fault_id"].isin(HARD_FAULTS)]["FDR"].mean() * 100

print(f"\nPCA-Q(n=7)     avg FDR: {pca_q7['FDR'].mean()*100:.2f}%  "
      f"hard FDR: {hard_q7:.2f}%  avg FAR: {pca_q7['FAR'].mean()*100:.4f}%")
print(f"PCA-Comb(n=29) avg FDR: {pca_c29['FDR'].mean()*100:.2f}%  "
      f"hard FDR: {hard_c29:.2f}%  avg FAR: {pca_c29['FAR'].mean()*100:.4f}%")

print("\n" + "=" * 65)
print("PCA BASELINE COMPLETE")
print("=" * 65)
