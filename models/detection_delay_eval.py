"""
Round-2 peer review, point 6a: rigorous, multi-seed companion to Figure 7
(contribution_analysis.py's illustrative, reference-seed-42 CDF plot).
Computes, per fault tier (All/Easy/Hard) and overall, a censoring-aware
summary using NormMaxPV on FAULT_TEST_RUNS:
  - n_total: number of independent test runs in the tier
  - n_detected: number with at least one alarm during the fault period
  - detection_probability = n_detected / n_total (a run-level "ever
    detected" rate -- distinct from the window-level FDR already reported in
    Table 4)
  - conditional_median_delay / conditional_mean_delay: delay statistics
    among DETECTED runs only (undetected runs are never imputed a delay
    value, unlike the previous add=800 sentinel bug)

Multi-seed: set TCNAE_SEED=<int>; writes
results/seeds/seed_<TCNAE_SEED>/detection_delay_summary.csv. Aggregated
across MAIN_SEEDS by aggregate_multiseed.py into
results/agg/detection_delay_agg.csv.
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

print("="*65)
print("DETECTION DELAY EVALUATION (censoring-aware, multi-seed)")
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
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

tier_delays = {'All': [], 'Easy': [], 'Hard': []}
tier_totals = {'All': 0, 'Easy': 0, 'Hard': 0}
per_fault_records = []

for fid in fault_ids:
    tiers_for_fault = ['All'] + (['Easy'] if fid in EASY_FAULTS else []) + \
                       (['Hard'] if fid in HARD_FAULTS else [])
    fault_delays, fault_total = [], 0
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        Xs = scaler.transform(X_run).astype(np.float32)
        W_arr = make_run_windows(Xs, WINDOW)
        pv_w = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                b = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                recon, _ = model(b)
                pv_w.append(tensor_to_numpy(((b - recon) ** 2).mean(dim=2)))
        pv_w = np.concatenate(pv_w)
        normmaxpv_w = (pv_w / std_pv).max(axis=1)
        alarm_w = normmaxpv_w > tau_normmaxpv
        fault_w_start = FAULT_START - WINDOW
        first = np.where(alarm_w[fault_w_start:])[0]
        delay = int(first[0]) + 1 if len(first) > 0 else None

        fault_total += 1
        if delay is not None:
            fault_delays.append(delay)
        for tier in tiers_for_fault:
            tier_totals[tier] += 1
            if delay is not None:
                tier_delays[tier].append(delay)

    n_det = len(fault_delays)
    per_fault_records.append({
        'fault_id': fid, 'n_total': fault_total, 'n_detected': n_det,
        'detection_probability': n_det / fault_total if fault_total else float('nan'),
        'conditional_median_delay': float(np.median(fault_delays)) if n_det else float('nan'),
        'conditional_mean_delay': float(np.mean(fault_delays)) if n_det else float('nan'),
    })
    print(f"  F{fid:2d}: n_total={fault_total}  n_detected={n_det}  "
          f"det_prob={n_det/fault_total*100:.1f}%")

del faulty_cache
gc.collect()

tier_records = []
for tier in ['All', 'Easy', 'Hard']:
    delays = tier_delays[tier]
    total = tier_totals[tier]
    n_det = len(delays)
    tier_records.append({
        'tier': tier, 'n_total': total, 'n_detected': n_det,
        'detection_probability': n_det / total if total else float('nan'),
        'conditional_median_delay': float(np.median(delays)) if n_det else float('nan'),
        'conditional_mean_delay': float(np.mean(delays)) if n_det else float('nan'),
    })
    print(f"Tier {tier:5s}: n_total={total}  n_detected={n_det}  "
          f"det_prob={n_det/total*100:.1f}%  median_delay={np.median(delays) if n_det else float('nan'):.1f}")

pd.DataFrame(per_fault_records).to_csv(RDIR / "detection_delay_per_fault.csv", index=False)
pd.DataFrame(tier_records).to_csv(RDIR / "detection_delay_summary.csv", index=False)
print(f"\nSaved {RDIR / 'detection_delay_summary.csv'}, {RDIR / 'detection_delay_per_fault.csv'}")
print("\n" + "="*65)
print(f"DETECTION DELAY EVALUATION COMPLETE (seed={SEED})")
print("="*65)
