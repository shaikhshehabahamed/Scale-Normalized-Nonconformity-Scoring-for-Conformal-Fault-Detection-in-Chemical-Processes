"""
Round-2 peer review, point 7: sensor-noise robustness check. Validation in
the main paper is restricted to one simulated, noise-free benchmark; this
script tests how much the already-calibrated NormMaxPV detector degrades
under realistic sensor noise added only at evaluation time -- the model is
trained and conformally calibrated exactly as in the main experiments (clean
data, unchanged tau_normmaxpv), then evaluated on FAULT_TEST_RUNS with
independent Gaussian jitter added to each standardized variable, at
sigma in {0, 0.01, 0.03, 0.05} (fraction of unit standardized variance --
data is already scaled to zero mean/unit variance by scaler.pkl, so these are
directly comparable across variables). This is a distribution-shift
robustness check, not a recalibration: the threshold is not re-fit to the
noisy distribution, matching how a detector calibrated once in a controlled
setting would actually be deployed.

Reuses the already-trained model, scaler, norm_stats (std_pv), and
cal_pv_scores.npy for the given seed -- no retraining, no recalibration.

Multi-seed: set TCNAE_SEED=<int>; writes
results/seeds/seed_<TCNAE_SEED>/noise_robustness.csv.
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
ALPHA = 0.01
WINDOW = 20
DEVICE = select_device()
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}
EASY_FAULTS = {1, 2, 4, 5, 6, 7, 8, 12, 13, 14}
NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05]

print("="*65)
print("SENSOR-NOISE ROBUSTNESS CHECK (point 7)")
print("="*65)
print_device(DEVICE)

scaler = joblib.load(MODDIR / "scaler.pkl")
model = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

std_pv = joblib.load(SDIR / "norm_stats.pkl")['std_pv']
cal_pv = np.load(SDIR / "cal_pv_scores.npy")
tau_normmaxpv = cp_threshold((cal_pv / std_pv).max(axis=1), ALPHA)
print(f"Loaded model + fixed (clean-calibration) tau_normmaxpv={tau_normmaxpv:.6f}")

faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

def eval_run_metrics(scores, threshold):
    n_pre  = FAULT_START - 1
    n_post = len(scores) - n_pre
    alarms = scores > threshold
    FAR = float(alarms[:n_pre].mean())
    FDR = float(alarms[n_pre:].mean())
    return FAR, FDR

def score_run_noisy(X_run_raw, noise_level, rng):
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    if noise_level > 0:
        Xs = Xs + rng.normal(0.0, noise_level, size=Xs.shape).astype(np.float32)
    n = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)
    normmaxpv_w = []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, _ = model(batch)
            pv = tensor_to_numpy(((batch - recon) ** 2).mean(dim=2))
            normmaxpv_w.append((pv / std_pv).max(axis=1))
    normmaxpv_w = np.concatenate(normmaxpv_w)
    s = np.full(n, normmaxpv_w[0])
    for i, w_idx in enumerate(range(WINDOW - 1, n)):
        s[w_idx] = normmaxpv_w[i]
    return s

records = []
for noise_level in NOISE_LEVELS:
    print(f"\nNoise level sigma={noise_level} (fraction of standardized unit variance)")
    rng = np.random.default_rng(SEED * 1000 + int(noise_level * 1000))
    fault_far, fault_fdr = {}, {}
    for fid in fault_ids:
        run_far, run_fdr = [], []
        for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
            if len(X_run) < 960:
                continue
            s = score_run_noisy(X_run, noise_level, rng)
            FAR, FDR = eval_run_metrics(s, tau_normmaxpv)
            run_far.append(FAR)
            run_fdr.append(FDR)
        fault_far[fid] = float(np.mean(run_far))
        fault_fdr[fid] = float(np.mean(run_fdr))

    all_far = float(np.mean(list(fault_far.values())))
    all_fdr = float(np.mean(list(fault_fdr.values())))
    hard_fdr = float(np.mean([fault_fdr[f] for f in fault_ids if f in HARD_FAULTS]))
    easy_fdr = float(np.mean([fault_fdr[f] for f in fault_ids if f in EASY_FAULTS]))
    print(f"  All: FAR={all_far:.5f} FDR={all_fdr:.4f}  Hard FDR={hard_fdr:.4f}  Easy FDR={easy_fdr:.4f}")
    records.append({'noise_level': noise_level, 'method': 'NormMaxPV-solo',
                    'FAR_all': all_far, 'FDR_all': all_fdr,
                    'FDR_hard': hard_fdr, 'FDR_easy': easy_fdr})

del faulty_cache
gc.collect()

df = pd.DataFrame(records)
df.to_csv(RDIR / "noise_robustness.csv", index=False)
print(f"\nSaved {RDIR / 'noise_robustness.csv'}")
print("\n" + "="*65)
print(f"SENSOR-NOISE ROBUSTNESS CHECK COMPLETE (seed={SEED})")
print("="*65)
