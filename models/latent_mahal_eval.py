"""
Latent Mahalanobis evaluation — STEP 3 ONLY.
Assumes cal_z.npy and mahal_config.pkl already exist (run latent_mahal.py first).
Uses the shared faulty-test memmap cache to avoid full in-memory RData loading.
Multi-seed: set TCNAE_SEED=<int>; writes results/seeds/seed_<TCNAE_SEED>/tcnae_mahal_results.csv.
The four-way comparison figure (fig20) is regenerated once from aggregated
data by gen_main_comparison_figs.py, not per seed.
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

os.makedirs(OUTDIR, exist_ok=True)

FAULT_START = 161
ALPHA       = 0.01
WINDOW      = 20
DEVICE      = select_device()
INFER_BS    = configured_batch_size("TCNAE_INFER_BS", 2048)
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}

def eval_run_metrics(scores, threshold):
    n_pre  = FAULT_START - 1
    n_post = len(scores) - n_pre
    alarms = scores > threshold
    FAR = float(alarms[:n_pre].mean())
    FDR = float(alarms[n_pre:].mean())
    first = np.where(alarms[n_pre:])[0]
    ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
    return FAR, FDR, ADD

print("="*65)
print("LATENT MAHAL EVALUATION (step 3 only)")
print("="*65)
print_device(DEVICE)

# ── Load model and calibration ────────────────────────────────────────────────
scaler  = joblib.load(MODDIR / "scaler.pkl")
cfg     = joblib.load(SDIR / "mahal_config.pkl")
mu_z    = cfg['mu_z']
cov_inv = cfg['cov_inv']

cp_thresh = joblib.load(SDIR / "conformal_thresholds.pkl")
tau_mse_solo   = cp_thresh['tau_mse']
tau_maxpv_solo = cp_thresh['tau_maxpv']

# Re-compute all thresholds from calibration scores
cal_z           = np.load(SDIR / "cal_z.npy")           # (48100, 24)
cal_mse_scores  = np.load(SDIR / "cal_scores.npy")      # (48100,)
cal_pv          = np.load(SDIR / "cal_pv_scores.npy")   # (48100, 52)
cal_maxpv       = cal_pv.max(axis=1)                        # (48100,)

def mahal(Z, mu, cov_inv):
    diffs = Z - mu
    return np.sqrt((diffs @ cov_inv * diffs).sum(axis=1))

cal_mahal_scores = mahal(cal_z, mu_z, cov_inv)

# Solo thresholds (alpha=0.01)
tau_mahal_solo = cp_threshold(cal_mahal_scores, ALPHA)

# Bonferroni thresholds (alpha/2=0.005) for OR-rules
tau_mse_b    = cp_threshold(cal_mse_scores,   ALPHA / 2)
tau_maxpv_b  = cp_threshold(cal_maxpv,        ALPHA / 2)
tau_mahal_b  = cp_threshold(cal_mahal_scores, ALPHA / 2)

# Verify OR FAR on calibration
or_mse_mpv   = ((cal_mse_scores > tau_mse_b) | (cal_maxpv > tau_maxpv_b)).mean()
or_mse_mahal = ((cal_mse_scores > tau_mse_b) | (cal_mahal_scores > tau_mahal_b)).mean()
print("Calibration verification:")
print(f"  FAR MSE solo:    {(cal_mse_scores > tau_mse_solo).mean():.4f}")
print(f"  FAR MaxPV solo:  {(cal_maxpv > tau_maxpv_solo).mean():.4f}")
print(f"  FAR Mahal solo:  {(cal_mahal_scores > tau_mahal_solo).mean():.4f}")
print(f"  FAR OR(MSE+MaxPV):  {or_mse_mpv:.4f}  (target={ALPHA})")
print(f"  FAR OR(MSE+Mahal):  {or_mse_mahal:.4f}  (target={ALPHA})")

model = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

# ── Load faulty test data ─────────────────────────────────────────────────────
print("\nOpening faulty test data cache...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault "
      f"(runs {FAULT_TEST_RUNS[0]}-{FAULT_TEST_RUNS[-1]}; the first "
      f"{len(FAULT_VAL_RUNS)} are reserved for ablation_study.py tuning)")

# ── Inference ─────────────────────────────────────────────────────────────────
def score_run_all(X_run_raw):
    """Single pass: returns (mse, maxpv, mahal) per sample. All shape (960,)."""
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)
    mse_w, mpv_w, z_w = [], [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, z = model(batch)
            err = (batch - recon)**2
            mse_w.append(tensor_to_numpy(err.mean(dim=(1,2))))
            mpv_w.append(tensor_to_numpy(err.mean(dim=2).max(dim=1).values))
            z_w.append(tensor_to_numpy(z))
    mse_w = np.concatenate(mse_w)
    mpv_w = np.concatenate(mpv_w)
    mh_w  = mahal(np.concatenate(z_w), mu_z, cov_inv)

    def assign(w):
        s = np.full(n, w[0])
        for i, idx in enumerate(range(WINDOW-1, n)):
            s[idx] = w[i]
        return s

    return assign(mse_w), assign(mpv_w), assign(mh_w)

def or_metrics(s1, t1, s2, t2):
    alarm = (s1 > t1) | (s2 > t2)
    n_pre = FAULT_START - 1
    n_post = len(alarm) - n_pre
    FAR = float(alarm[:n_pre].mean())
    FDR = float(alarm[n_pre:].mean())
    fst = np.where(alarm[n_pre:])[0]
    return FAR, FDR, int(fst[0])+1 if len(fst) else n_post+1

print(f"\nEvaluating {len(fault_ids)} faults (5 methods in single pass)...")
records = []
for fid in fault_ids:
    r_mse, r_mpv, r_mahal, r_or_mm, r_or_mh = [], [], [], [], []
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        s_mse, s_mpv, s_mh = score_run_all(X_run)
        r_mse.append(eval_run_metrics(s_mse, tau_mse_solo))
        r_mpv.append(eval_run_metrics(s_mpv, tau_maxpv_solo))
        r_mahal.append(eval_run_metrics(s_mh, tau_mahal_solo))
        r_or_mm.append(or_metrics(s_mse, tau_mse_b, s_mpv, tau_maxpv_b))
        r_or_mh.append(or_metrics(s_mse, tau_mse_b, s_mh,  tau_mahal_b))

    for method, res in [('TCN-MSE-CP', r_mse), ('TCN-MaxPV-CP', r_mpv),
                        ('TCN-Mahal-CP', r_mahal),
                        ('TCN-OR(MSE+MaxPV)-CP', r_or_mm),
                        ('TCN-OR(MSE+Mahal)-CP', r_or_mh)]:
        m = np.mean(res, axis=0)
        records.append({'fault_id': fid, 'method': method,
                        'FAR': m[0], 'FDR': m[1], 'ADD': m[2]})

    m1 = np.mean(r_mse,axis=0)
    m2 = np.mean(r_mpv,axis=0)
    m3 = np.mean(r_mahal,axis=0)
    m4 = np.mean(r_or_mm,axis=0)
    print(f"  F{fid:2d}: MSE={m1[1]:.3f} MaxPV={m2[1]:.3f} Mahal={m3[1]:.3f} "
          f"OR(M+P)={m4[1]:.3f} FAR={m4[0]:.4f}")

mahal_df = pd.DataFrame(records)
mahal_df.to_csv(RDIR / "tcnae_mahal_results.csv", index=False)
del faulty_cache
gc.collect()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"FULL SUMMARY — ALL METHODS (seed={SEED})")
print("="*65)
pca_res  = pd.read_csv(OUTDIR / "pca_baseline_results.csv")
pca_best_row = pca_res[pca_res['method'] == 'PCA-Q(n=7)']
print(f"  PCA-Q(n=7):          All FDR={pca_best_row['FDR'].mean()*100:.2f}%  "
      f"Hard={pca_best_row[pca_best_row['fault_id'].isin(HARD_FAULTS)]['FDR'].mean()*100:.2f}%  "
      f"FAR={pca_best_row['FAR'].mean()*100:.4f}%")
for method in ['TCN-MSE-CP','TCN-MaxPV-CP','TCN-Mahal-CP',
               'TCN-OR(MSE+MaxPV)-CP','TCN-OR(MSE+Mahal)-CP']:
    sub = mahal_df[mahal_df['method'] == method]
    h   = sub[sub['fault_id'].isin(HARD_FAULTS)]
    e   = sub[~sub['fault_id'].isin(HARD_FAULTS)]
    print(f"  {method:28s}: All FDR={sub['FDR'].mean()*100:.2f}%  "
          f"Hard={h['FDR'].mean()*100:.2f}%  "
          f"Easy={e['FDR'].mean()*100:.2f}%  FAR={sub['FAR'].mean()*100:.4f}%")

print("\n(fig20_four_way_comparison.png is regenerated once from the aggregated")
print(" multi-seed results by gen_main_comparison_figs.py, not per seed.)")
print("Run gen_fig21.py to generate the latent-space PCA figure (reference seed).")

print("\n" + "="*65)
print(f"LATENT MAHAL EVALUATION COMPLETE (seed={SEED})")
print("="*65)
