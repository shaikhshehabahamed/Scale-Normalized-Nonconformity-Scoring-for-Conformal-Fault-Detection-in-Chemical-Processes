"""
Round-2 peer review, point 2: overlapping calibration windows share W-1=19
samples and are strongly serially correlated, so treating ~24,050 calibration
windows as exchangeable units is questionable even though inter-run
independence holds exactly (Appendix A). Rather than argue this "by analogy"
(the manuscript's previous position), this script actually recomputes the
NormMaxPV conformal threshold three ways from the SAME cal-set runs (the
train/norm/cal split of tcnae_utils.split_faultfree_runs is untouched; only
the CAL windows are re-sliced):

  1. overlapping    (stride=1,      ~24,050 windows) -- the paper's main scheme
  2. non-overlapping (stride=W=20,  ~1,250 windows)  -- windows share no samples
  3. run-level block conformal (one score/run, 50 runs) -- simulation runs are
     the exchangeable unit, the most conservative alternative

and reports empirical FAR under each scheme, evaluated the matching way
(per-window for 1/2, per-run block-max for 3) on FAULT_TEST_RUNS' pre-fault
segments -- the same independent test set used for Table 5. This reuses the
already-trained model/scaler/norm_stats/cal_pv_scores for the given seed; no
retraining.

Multi-seed: set TCNAE_SEED=<int>; writes
results/seeds/seed_<TCNAE_SEED>/overlap_robustness.csv.
"""
import os
import sys
import gc
import numpy as np
import pandas as pd
import pyreadr
import joblib
import torch

sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae import TCNAE
from tcnae_utils import (
    ARCHIVE,
    FEAT,
    MODDIR,
    configured_batch_size,
    configured_seed,
    cp_threshold,
    load_faulty_test_cache,
    make_run_windows_strided,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    seed_results_dir,
    select_device,
    split_faultfree_runs,
    split_fault_test_runs,
    tensor_to_numpy,
)
SEED = configured_seed()
seed_everything(SEED)
SDIR = seed_model_dir(SEED)
RDIR = seed_results_dir(SEED)

WINDOW = 20
ALPHA = 0.01
DEVICE = select_device()
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
FAULT_START = 161

print("="*65)
print("OVERLAPPING-WINDOW DEPENDENCE ROBUSTNESS CHECK (point 2)")
print("="*65)
print_device(DEVICE)

# ── Load model + fixed nuisance parameters ────────────────────────────────────
scaler = joblib.load(MODDIR / "scaler.pkl")
model = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
std_pv = joblib.load(SDIR / "norm_stats.pkl")['std_pv']
print("Model + norm-estimation std_pv loaded.")

# ── Scheme 1: overlapping (reuse the already-saved cal_pv_scores.npy) ────────
cal_pv = np.load(SDIR / "cal_pv_scores.npy")            # (N_cal_windows, 52), stride=1
cal_normmaxpv_overlap = (cal_pv / std_pv).max(axis=1)
windows_per_cal_run = 500 - WINDOW + 1                    # 481, RunWindowDataset stride=1
n_cal_runs = len(cal_pv) // windows_per_cal_run
assert len(cal_pv) == n_cal_runs * windows_per_cal_run, "cal_pv_scores.npy not run-aligned"
tau_overlap = cp_threshold(cal_normmaxpv_overlap, ALPHA)
print(f"\nScheme 1 (overlapping, stride=1): n_cal_windows={len(cal_pv)}  tau={tau_overlap:.6f}")

# ── Scheme 3: run-level block conformal (reuse the same cal_pv_scores.npy) ───
# One score per cal run = max NormMaxPV over that run's FIRST n_windows_pre
# stride=1 windows (samples 1-160), not the full 481-window run. This has to
# match the test-time block score exactly: a test run only ever contributes
# 141 fault-free windows (its pre-fault segment, samples 1-160), so scoring
# the cal run's max over all 481 windows would make the cal block-score
# stochastically larger than the test block-score purely from being a max
# over more samples -- not a fault-free-vs-faulty difference, just a biased
# comparison that would understate the empirical test FAR under this scheme.
n_windows_pre = (FAULT_START - 1) - WINDOW + 1   # 141: windows fully inside samples 1-160
cal_pv_by_run = cal_normmaxpv_overlap.reshape(n_cal_runs, windows_per_cal_run)
cal_block = cal_pv_by_run[:, :n_windows_pre].max(axis=1)
tau_block = cp_threshold(cal_block, ALPHA)
print(f"Scheme 3 (run-level block, {n_cal_runs} runs, first {n_windows_pre} windows/run "
      f"to match the test pre-fault segment length): tau={tau_block:.6f}")

# ── Scheme 2: non-overlapping (stride=W) -- needs a fresh forward pass ───────
print("\nScheme 2 (non-overlapping, stride=W=20): re-scoring the cal-set runs ...")
r = pyreadr.read_r(str(ARCHIVE / "TEP_FaultFree_Training.RData"))
df_ff = list(r.values())[0]
del r
df_ff[FEAT] = df_ff[FEAT].astype(np.float32)
all_runs = sorted(df_ff['simulationRun'].unique())
_, _, cal_runs = split_faultfree_runs(all_runs)
assert len(cal_runs) == n_cal_runs, "cal run count mismatch vs cal_pv_scores.npy"

cal_normmaxpv_nonoverlap = []
with torch.no_grad():
    for run_id in cal_runs:
        X_run_raw = df_ff[df_ff['simulationRun'] == run_id][FEAT].to_numpy(copy=True)
        Xs = scaler.transform(X_run_raw).astype(np.float32)
        W_arr = make_run_windows_strided(Xs, WINDOW, stride=WINDOW)   # non-overlapping
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, _ = model(batch)
            pv = tensor_to_numpy(((batch - recon) ** 2).mean(dim=2))
            cal_normmaxpv_nonoverlap.append((pv / std_pv).max(axis=1))
del df_ff
gc.collect()
cal_normmaxpv_nonoverlap = np.concatenate(cal_normmaxpv_nonoverlap)
tau_nonoverlap = cp_threshold(cal_normmaxpv_nonoverlap, ALPHA)
print(f"  n_cal_windows(non-overlap)={len(cal_normmaxpv_nonoverlap)}  tau={tau_nonoverlap:.6f}")

# ── Evaluate empirical test FAR under all three schemes ──────────────────────
# Independent test set: pre-fault segments of FAULT_TEST_RUNS (same set used
# for the paper's headline Table 5 FAR), scored once at stride=1; schemes 2/3
# subsample that same stride=1 score sequence rather than re-running the
# model a third time (a non-overlapping/max subsample of an already-computed
# stride=1 score sequence is identical to computing it from scratch at that
# stride, since the score is a deterministic function of the window content).
print("\nEvaluating empirical FAR on the independent FAULT_TEST_RUNS pre-fault segments ...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
n_pre = FAULT_START - 1   # 160 fault-free samples per test run

alarms_overlap, alarms_nonoverlap, alarms_block = [], [], []
n_windows_per_run_pre = n_pre - WINDOW + 1   # 141 stride-1 windows in the pre-fault segment

for fid in fault_ids:
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        X_pre = X_run[:n_pre]
        Xs = scaler.transform(X_pre).astype(np.float32)
        W_arr = make_run_windows_strided(Xs, WINDOW, stride=1)   # all 141 stride-1 windows
        pv_list = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                recon, _ = model(batch)
                pv_list.append(tensor_to_numpy(((batch - recon) ** 2).mean(dim=2)))
        pv = np.concatenate(pv_list, axis=0)                      # (141, 52)
        s_overlap = (pv / std_pv).max(axis=1)                     # (141,) one per stride-1 window
        s_nonoverlap = s_overlap[::WINDOW]                        # non-overlapping subsample
        s_block = s_overlap.max()                                 # one score for this run

        alarms_overlap.extend((s_overlap > tau_overlap).tolist())
        alarms_nonoverlap.extend((s_nonoverlap > tau_nonoverlap).tolist())
        alarms_block.append(bool(s_block > tau_block))

del faulty_cache
gc.collect()

far_overlap = float(np.mean(alarms_overlap))
far_nonoverlap = float(np.mean(alarms_nonoverlap))
far_block = float(np.mean(alarms_block))   # fraction of TEST RUNS (not windows) that alarm

print(f"\nEmpirical test FAR (target alpha={ALPHA}):")
print(f"  Scheme 1 overlapping    : n={len(alarms_overlap)} windows   FAR={far_overlap:.5f}")
print(f"  Scheme 2 non-overlapping: n={len(alarms_nonoverlap)} windows   FAR={far_nonoverlap:.5f}")
print(f"  Scheme 3 run-level block: n={len(alarms_block)} runs      FAR={far_block:.5f}")

pd.DataFrame([
    {'scheme': 'overlapping_stride1', 'exchangeable_unit': 'window', 'n_cal': len(cal_pv),
     'tau': tau_overlap, 'n_test': len(alarms_overlap), 'far_test': far_overlap},
    {'scheme': 'non_overlapping_strideW', 'exchangeable_unit': 'window', 'n_cal': len(cal_normmaxpv_nonoverlap),
     'tau': tau_nonoverlap, 'n_test': len(alarms_nonoverlap), 'far_test': far_nonoverlap},
    {'scheme': 'run_level_block', 'exchangeable_unit': 'run', 'n_cal': n_cal_runs,
     'tau': tau_block, 'n_test': len(alarms_block), 'far_test': far_block},
]).to_csv(RDIR / "overlap_robustness.csv", index=False)
print(f"\nSaved {RDIR / 'overlap_robustness.csv'}")

print("\n" + "="*65)
print(f"OVERLAP ROBUSTNESS CHECK COMPLETE (seed={SEED})")
print("="*65)
