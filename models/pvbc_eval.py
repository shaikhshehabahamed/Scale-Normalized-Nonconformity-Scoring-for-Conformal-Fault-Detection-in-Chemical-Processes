"""
Per-Variable Bonferroni Conformal (PVBC) evaluation.

MaxPV uses: tau = quantile(max_j(pv_j) over cal, 1-alpha/2) — threshold on the MAXIMUM.
PVBC uses:  tau_j = quantile(pv_j over cal, 1-alpha/52) — per-variable threshold.
Standalone PVBC FAR guarantee: P(alarm|null) <= 52*(alpha/52) = alpha.

OR rules tested:
  - MSE OR PVBC      (Bonferroni outer: alpha/2 each)
  - MSE OR MaxPV OR PVBC  (Bonferroni: alpha/3 each)

Also tests:
  - NormMaxPV: max_j(pv_j / std_j) — scale-normalized max, single threshold.
    std_j comes from norm_stats.pkl (the norm-estimation set fit by
    train_tcnae.py), never from cal_pv itself -- otherwise each calibration
    NormMaxPV score would be computed with a std that includes that very
    observation, while a fresh test score would not (round-2 peer review,
    point 1). PVBC's per-variable thresholds use raw (unnormalized) cal_pv
    directly, same as MSE/MaxPV, since they involve no nuisance parameter.

Requires trained TCN-AE artifacts from train_tcnae.py (conformal_thresholds.pkl,
norm_stats.pkl, cal_scores.npy, cal_pv_scores.npy).
Uses the shared faulty-test memmap cache for full-benchmark evaluation, on the
FAULT_TEST_RUNS split (disjoint from the runs ablation_study.py tunes on).
Multi-seed: set TCNAE_SEED=<int>; writes results/seeds/seed_<TCNAE_SEED>/pvbc_results.csv.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import joblib
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae import TCNAE
from tcnae_utils import (
    MODDIR,
    OUTDIR,
    configured_batch_size,
    configured_seed,
    cp_threshold,
    load_faulty_test_cache,
    make_run_windows,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    seed_results_dir,
    select_device,
    split_fault_test_runs,
    tensor_to_numpy,
)
SEED = configured_seed()
seed_everything(SEED)
SDIR = seed_model_dir(SEED)
RDIR = seed_results_dir(SEED)

FAULT_START = 161
ALPHA  = 0.01
WINDOW = 20
DEVICE = select_device()
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
N_VARS = 52
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
os.makedirs(OUTDIR, exist_ok=True)

print("="*65)
print("PVBC + NORMMAXPV EVALUATION")
print("="*65)
print_device(DEVICE)

# ── Load model and calibration data ──────────────────────────────────────────
scaler = joblib.load(MODDIR / "scaler.pkl")
model  = TCNAE(n_vars=N_VARS, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("Model loaded.")

cal_mse = np.load(SDIR / "cal_scores.npy")
cal_pv  = np.load(SDIR / "cal_pv_scores.npy")
norm_stats = joblib.load(SDIR / "norm_stats.pkl")
print(f"Cal MSE shape: {cal_mse.shape}  Cal PV shape: {cal_pv.shape}  "
      f"(norm-estimation set: {norm_stats['n_norm_runs']} runs, disjoint from cal)")

# ── Compute all thresholds ────────────────────────────────────────────────────
# Existing: solo alpha, Bonferroni-2 (for OR with MSE)
tau_mse_solo  = cp_threshold(cal_mse, ALPHA)
tau_mse_b2    = cp_threshold(cal_mse, ALPHA / 2)   # Bonferroni for 2-way OR

# MaxPV
cal_maxpv     = cal_pv.max(axis=1)                  # (48100,)
tau_maxpv_b2  = cp_threshold(cal_maxpv, ALPHA / 2)  # existing OR(MSE+MaxPV)

# PVBC: per-variable thresholds at alpha/52 each
# Standalone PVBC: tau_j at 1 - alpha/52
tau_pvbc_solo = np.array([cp_threshold(cal_pv[:, j], ALPHA / N_VARS) for j in range(N_VARS)])
# Bonferroni-2 PVBC (for OR with MSE): tau_j at 1 - (alpha/2)/52
tau_pvbc_b2   = np.array([cp_threshold(cal_pv[:, j], (ALPHA / 2) / N_VARS) for j in range(N_VARS)])
# Bonferroni-3 PVBC (for OR with MSE+MaxPV): tau_j at 1 - (alpha/3)/52
tau_pvbc_b3   = np.array([cp_threshold(cal_pv[:, j], (ALPHA / 3) / N_VARS) for j in range(N_VARS)])
tau_mse_b3    = cp_threshold(cal_mse, ALPHA / 3)
tau_maxpv_b3  = cp_threshold(cal_maxpv, ALPHA / 3)

# NormMaxPV: normalize each variable by its std from the norm-estimation set
# (fixed, disjoint from cal_pv -- see module docstring).
std_pv        = norm_stats['std_pv']                 # (52,)
cal_normmaxpv = (cal_pv / std_pv).max(axis=1)        # (N_cal,)
tau_normmaxpv = cp_threshold(cal_normmaxpv, ALPHA)    # standalone, no Bonferroni
tau_normmaxpv_b2 = cp_threshold(cal_normmaxpv, ALPHA / 2)  # for OR with MSE

print("\nThresholds:")
print(f"  tau_mse (solo)     = {tau_mse_solo:.6f}")
print(f"  tau_mse (Bonf/2)   = {tau_mse_b2:.6f}")
print(f"  tau_maxpv (Bonf/2) = {tau_maxpv_b2:.6f}  [existing]")
print(f"  tau_pvbc (solo)    = mean {tau_pvbc_solo.mean():.6f}, range [{tau_pvbc_solo.min():.4f}, {tau_pvbc_solo.max():.4f}]")
print(f"  tau_normmaxpv      = {tau_normmaxpv:.6f}")

# ── Verify calibration FAR ────────────────────────────────────────────────────
far_mse   = (cal_mse > tau_mse_solo).mean()
pvbc_alarm = (cal_pv > tau_pvbc_solo).any(axis=1)
far_pvbc  = pvbc_alarm.mean()
far_normmax = (cal_normmaxpv > tau_normmaxpv).mean()
far_or_pvbc     = ((cal_mse > tau_mse_b2) | (cal_pv > tau_pvbc_b2).any(axis=1)).mean()
far_or_mse_norm = ((cal_mse > tau_mse_b2) | (cal_normmaxpv > tau_normmaxpv_b2)).mean()
far_or_3way     = ((cal_mse > tau_mse_b3) | (cal_maxpv > tau_maxpv_b3) |
                   (cal_pv > tau_pvbc_b3).any(axis=1)).mean()

print(f"\nCalibration FAR verification (target={ALPHA:.3f}):")
print(f"  MSE solo:           FAR={far_mse:.5f}")
print(f"  PVBC standalone:    FAR={far_pvbc:.5f}")
print(f"  NormMaxPV solo:     FAR={far_normmax:.5f}")
print(f"  OR(MSE+NormMaxPV):  FAR={far_or_mse_norm:.5f}")
print(f"  OR(MSE+PVBC):       FAR={far_or_pvbc:.5f}")
print(f"  OR(MSE+MaxPV+PVBC): FAR={far_or_3way:.5f}")

# ── Inference helper ──────────────────────────────────────────────────────────
def score_run(X_run_raw):
    """Returns (n,) MSE, (n,) MaxPV, (n,52) per-var scores."""
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)
    mse_w, pv_w = [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            b = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, _ = model(b)
            err = (b - recon) ** 2
            mse_w.append(tensor_to_numpy(err.mean(dim=(1, 2))))
            pv_w.append(tensor_to_numpy(err.mean(dim=2)))   # (B, 52)
    mse_w = np.concatenate(mse_w)     # (n-W+1,)
    pv_w  = np.vstack(pv_w)           # (n-W+1, 52)

    s_mse = np.full(n, mse_w[0])
    s_pv  = np.tile(pv_w[0], (n, 1))
    for i, w_idx in enumerate(range(WINDOW - 1, n)):
        s_mse[w_idx]   = mse_w[i]
        s_pv[w_idx, :] = pv_w[i]
    return s_mse, s_pv

def eval_metrics(alarm, n_pre=FAULT_START - 1):
    n_post = len(alarm) - n_pre
    FAR = alarm[:n_pre].mean()
    FDR = alarm[n_pre:].mean()
    first = np.where(alarm[n_pre:])[0]
    ADD = int(first[0]) + 1 if len(first) else n_post + 1
    return float(FAR), float(FDR), int(ADD)

# ── Load faulty test data ─────────────────────────────────────────────────────
print("\nOpening faulty test data cache ...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Loaded: {len(fault_ids)} faults  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

# ── Evaluate all methods ──────────────────────────────────────────────────────
METHODS = ['PVBC-solo', 'NormMaxPV-solo', 'OR(MSE+NormMaxPV)', 'OR(MSE+PVBC)', 'OR(MSE+MaxPV+PVBC)']
records = []

for fid in fault_ids:
    res = {m: [] for m in METHODS}
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        s_mse, s_pv = score_run(X_run)
        s_maxpv       = s_pv.max(axis=1)
        s_normmaxpv   = (s_pv / std_pv).max(axis=1)
        pvbc_alarm    = (s_pv > tau_pvbc_solo).any(axis=1)

        alarm_pvbc         = pvbc_alarm
        alarm_normmax      = s_normmaxpv > tau_normmaxpv
        alarm_or_mse_norm  = (s_mse > tau_mse_b2) | (s_normmaxpv > tau_normmaxpv_b2)
        alarm_or_b2        = (s_mse > tau_mse_b2) | (s_pv > tau_pvbc_b2).any(axis=1)
        alarm_or_3way      = ((s_mse > tau_mse_b3) | (s_maxpv > tau_maxpv_b3) |
                              (s_pv > tau_pvbc_b3).any(axis=1))

        res['PVBC-solo'].append(eval_metrics(alarm_pvbc))
        res['NormMaxPV-solo'].append(eval_metrics(alarm_normmax))
        res['OR(MSE+NormMaxPV)'].append(eval_metrics(alarm_or_mse_norm))
        res['OR(MSE+PVBC)'].append(eval_metrics(alarm_or_b2))
        res['OR(MSE+MaxPV+PVBC)'].append(eval_metrics(alarm_or_3way))

    for m in METHODS:
        m_arr = np.mean(res[m], axis=0)
        records.append({'fault_id': fid, 'method': m,
                        'FAR': round(m_arr[0], 5), 'FDR': round(m_arr[1], 5),
                        'ADD': round(m_arr[2], 2)})

del faulty_cache
gc.collect()
pvbc_df = pd.DataFrame(records)
pvbc_df.to_csv(RDIR / "pvbc_results.csv", index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY")
print("="*65)
print(f"{'Method':<30} {'Avg FDR':>8} {'Hard FDR':>9} {'FAR':>8}")
print("-"*65)
for m in METHODS:
    sub = pvbc_df[pvbc_df['method'] == m]
    avg  = sub['FDR'].mean() * 100
    hard = sub[sub['fault_id'].isin(HARD_FAULTS)]['FDR'].mean() * 100
    far  = sub['FAR'].mean() * 100
    print(f"  {m:<28} {avg:>7.2f}% {hard:>8.2f}% {far:>7.3f}%")

print(f"\nSaved {RDIR / 'pvbc_results.csv'}  (seed={SEED})")
print("="*65)
