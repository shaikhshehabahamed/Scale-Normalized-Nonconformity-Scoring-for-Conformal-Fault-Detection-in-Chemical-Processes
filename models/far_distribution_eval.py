"""
Round-2 peer review, point 3: run-level FAR distribution, not just the pooled
mean. Table 5 reports one pooled FAR per method (mean over 400 test runs x 20
faults, itself already an independent-set number -- see Section 4.5). This
script additionally reports, for NormMaxPV, the distribution of per-run FAR
values across those same 8,000 independent test runs: what fraction of
individual runs ever exceed alpha, 2*alpha, etc., which the pooled mean can
obscure (a method could pool to ~1% while still spiking heavily on a handful
of runs). Reuses the already-trained model, scaler, norm_stats, and
cal_pv_scores.npy for the given seed -- no retraining.

Multi-seed: set TCNAE_SEED=<int>; writes
results/seeds/seed_<TCNAE_SEED>/far_distribution.csv (one row per test run).
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

print("="*65)
print("RUN-LEVEL FAR DISTRIBUTION (point 3)")
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
print(f"tau_normmaxpv={tau_normmaxpv:.6f}")

faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
n_pre = FAULT_START - 1

records = []
for fid in fault_ids:
    for run_id, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        X_pre = X_run[:n_pre]
        Xs = scaler.transform(X_pre).astype(np.float32)
        W_arr = make_run_windows(Xs, WINDOW)
        pv_list = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                recon, _ = model(batch)
                pv_list.append(tensor_to_numpy(((batch - recon) ** 2).mean(dim=2)))
        pv = np.concatenate(pv_list, axis=0)
        s = (pv / std_pv).max(axis=1)
        far_run = float((s > tau_normmaxpv).mean())
        records.append({'fault_id': fid, 'run_id': run_id, 'far_run': far_run})

del faulty_cache
gc.collect()

df = pd.DataFrame(records)
df.to_csv(RDIR / "far_distribution.csv", index=False)

far = df['far_run'].to_numpy()
print(f"\nn_runs={len(far)}  mean={far.mean():.5f}  "
      f"p95={np.percentile(far,95):.5f}  p99={np.percentile(far,99):.5f}  max={far.max():.5f}")
print(f"fraction of runs with FAR > alpha (0.01):    {np.mean(far > ALPHA):.4f}")
print(f"fraction of runs with FAR > 2*alpha (0.02):  {np.mean(far > 2*ALPHA):.4f}")
print(f"fraction of runs with FAR == 0:               {np.mean(far == 0):.4f}")
print(f"\nSaved {RDIR / 'far_distribution.csv'}")
print("\n" + "="*65)
print(f"FAR DISTRIBUTION COMPLETE (seed={SEED})")
print("="*65)
