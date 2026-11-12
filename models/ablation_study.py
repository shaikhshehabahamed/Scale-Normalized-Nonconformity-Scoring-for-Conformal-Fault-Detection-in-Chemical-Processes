"""
Ablation study for TCN-AE + Conformal Prediction on TEP dataset.
Tests: window size, latent dim, nonconformity score type, conformal alpha.
Requires: tcnae_best.pt already trained (run train_tcnae.py first).
Usage: uv run python models/ablation_study.py
Multi-seed: set TCNAE_SEED=<int> (one of ABLATION_SEEDS); writes
results/ablation_seeds/seed_<TCNAE_SEED>/ablation_*.csv. The alpha-sensitivity
ablation reuses that seed's trained main model/calibration from
models/seeds/seed_<TCNAE_SEED>/. See models/run_multiseed.py.

Every sweep here evaluates on FAULT_VAL_RUNS (the first
tcnae_utils.FAULT_VAL_RUN_COUNT runs of each fault type), disjoint from
FAULT_TEST_RUNS used by every final headline result (evaluate_tcnae.py,
pvbc_eval.py, latent_mahal_eval.py, dense_ae_baseline.py, the new factorial
architectures). Model selection (window, latent dim, alpha) must never be
tuned on the same fault runs used to report final performance (round-2 peer
review, point 5). Each retrained ablation config also uses its own
norm-estimation/cal split (tcnae_utils.split_faultfree_runs) so std_pv is
never estimated on the same windows used for that config's conformal
threshold (round-2 peer review, point 1).
"""
import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import pyreadr
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae import TCNAE
from tcnae_utils import (
    ARCHIVE,
    FEAT,
    MODDIR,
    OUTDIR,
    RunWindowDataset,
    clear_cuda_cache,
    configured_batch_size,
    configured_seed,
    cp_threshold,
    load_faulty_test_cache,
    make_run_windows,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    select_device,
    split_faultfree_runs,
    split_fault_test_runs,
    tensor_to_numpy,
)
from tcnae_utils import ablation_seed_results_dir
SEED = configured_seed()
seed_everything(SEED)
SDIR = seed_model_dir(SEED)
ADIR = ablation_seed_results_dir(SEED)

os.makedirs(OUTDIR, exist_ok=True)

FAULT_START = 161
ALPHA  = 0.01
DEVICE = select_device()
PIN_MEMORY = DEVICE.type == "cuda"
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
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

def score_run_with_model(model, scaler, X_run_raw, window, threshold_mse, threshold_maxpv,
                          std_pv, threshold_normmaxpv):
    """Return per-sample MSE, MaxPV, and NormMaxPV scores for a single run (960, V)."""
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, window)
    mse_w, mpv_w, normmaxpv_w = [], [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i : i + INFER_BS], DEVICE)
            recon, _ = model(batch)
            err2 = (batch - recon)**2
            pv   = err2.mean(dim=2)               # (B, V) per-variable error
            mse  = err2.mean(dim=(1,2))
            mpv  = pv.max(dim=1).values
            pv_np = tensor_to_numpy(pv)
            normmaxpv = (pv_np / std_pv).max(axis=1)
            mse_w.append(tensor_to_numpy(mse))
            mpv_w.append(tensor_to_numpy(mpv))
            normmaxpv_w.append(normmaxpv)
    mse_w = np.concatenate(mse_w)
    mpv_w = np.concatenate(mpv_w)
    normmaxpv_w = np.concatenate(normmaxpv_w)
    # Assign to last sample index of each window
    s_mse = np.full(n, mse_w[0])
    s_mpv = np.full(n, mpv_w[0])
    s_normmaxpv = np.full(n, normmaxpv_w[0])
    for i, w_idx in enumerate(range(window - 1, n)):
        s_mse[w_idx] = mse_w[i]
        s_mpv[w_idx] = mpv_w[i]
        s_normmaxpv[w_idx] = normmaxpv_w[i]
    return (eval_run_metrics(s_mse, threshold_mse),
            eval_run_metrics(s_mpv, threshold_maxpv),
            eval_run_metrics(s_normmaxpv, threshold_normmaxpv))

def train_model_ablation(X_tr, X_norm, X_cal, n_runs_tr, n_runs_norm, n_runs_cal,
                          window, latent_dim, n_filters=64, epochs=40,
                          batch_size=None, lr=1e-3):
    """Train a TCN-AE for one ablation configuration.
    Returns model, tau_mse, tau_maxpv, tau_normmaxpv, std_pv. std_pv comes
    from X_norm; every threshold comes from X_cal (disjoint from X_norm) --
    same train/norm/cal split discipline as train_tcnae.py."""
    print(f"   Training W={window}, lat={latent_dim}, filters={n_filters} ...")
    t0 = time.time()
    batch_size = configured_batch_size("TCNAE_BATCH_SIZE", 2048) if batch_size is None else batch_size
    ds_tr = RunWindowDataset(X_tr, n_runs_tr, window)
    ds_norm = RunWindowDataset(X_norm, n_runs_norm, window)
    ds_cal = RunWindowDataset(X_cal, n_runs_cal, window)

    model = TCNAE(n_vars=52, window=window, n_filters=n_filters, latent_dim=latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0,
                       pin_memory=PIN_MEMORY)

    best_loss = float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        el = 0.0
        for (xb,) in dl_tr:
            xb = move_to_device(xb, DEVICE)
            recon, _ = model(xb)
            loss = nn.functional.mse_loss(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            el += loss.item() * len(xb)
        el /= len(ds_tr)
        scheduler.step()
        if el < best_loss:
            best_loss = el
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"     epoch {epoch:3d} loss={el:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # Norm-estimation pass: std_pv only, no thresholds.
    dl_norm = DataLoader(ds_norm, batch_size=INFER_BS, shuffle=False, num_workers=0,
                         pin_memory=PIN_MEMORY)
    norm_pv = []
    with torch.no_grad():
        for (xb,) in dl_norm:
            xb = move_to_device(xb, DEVICE)
            recon, _ = model(xb)
            norm_pv.append(tensor_to_numpy(((xb - recon)**2).mean(dim=2)))
    norm_pv = np.concatenate(norm_pv, axis=0)
    std_pv = norm_pv.std(axis=0) + 1e-9

    # Cal pass: every conformal threshold.
    dl_cal = DataLoader(ds_cal, batch_size=INFER_BS, shuffle=False, num_workers=0,
                        pin_memory=PIN_MEMORY)
    cal_mse, cal_mpv, cal_pv = [], [], []
    with torch.no_grad():
        for (xb,) in dl_cal:
            xb = move_to_device(xb, DEVICE)
            recon, _ = model(xb)
            err2 = (xb - recon)**2
            pv   = err2.mean(dim=2)
            mse  = err2.mean(dim=(1,2))
            mpv  = pv.max(dim=1).values
            cal_mse.append(tensor_to_numpy(mse))
            cal_mpv.append(tensor_to_numpy(mpv))
            cal_pv.append(tensor_to_numpy(pv))
    cal_mse = np.concatenate(cal_mse)
    cal_mpv = np.concatenate(cal_mpv)
    cal_pv  = np.concatenate(cal_pv, axis=0)          # (N_cal, V)
    cal_normmaxpv = (cal_pv / std_pv).max(axis=1)
    tau_mse       = cp_threshold(cal_mse, ALPHA)
    tau_maxpv     = cp_threshold(cal_mpv, ALPHA)
    tau_normmaxpv = cp_threshold(cal_normmaxpv, ALPHA)
    print(f"   Done in {(time.time()-t0)/60:.1f} min | "
          f"tau_mse={tau_mse:.5f} tau_maxpv={tau_maxpv:.5f} tau_normmaxpv={tau_normmaxpv:.5f}")
    return model, tau_mse, tau_maxpv, tau_normmaxpv, std_pv

def eval_model_on_test(model, scaler, faulty_cache, fault_ids, window, tau_mse, tau_maxpv,
                       tau_normmaxpv, std_pv, run_ids):
    """Evaluate model on faults, restricted to `run_ids` (the disjoint
    validation-run pool for ablation tuning -- never the final test runs)."""
    records = []
    for fid in fault_ids:
        mse_res, mpv_res, normmaxpv_res = [], [], []
        for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=run_ids):
            if len(X_run) < 960:
                continue
            r1, r2, r3 = score_run_with_model(model, scaler, X_run, window, tau_mse, tau_maxpv,
                                               std_pv, tau_normmaxpv)
            mse_res.append(r1)
            mpv_res.append(r2)
            normmaxpv_res.append(r3)
        m1 = np.mean(mse_res, axis=0)
        m2 = np.mean(mpv_res, axis=0)
        m3 = np.mean(normmaxpv_res, axis=0)
        records.append({'fault_id': fid, 'FAR_mse': m1[0], 'FDR_mse': m1[1], 'ADD_mse': m1[2],
                        'FAR_mpv': m2[0], 'FDR_mpv': m2[1], 'ADD_mpv': m2[2],
                        'FAR_normmaxpv': m3[0], 'FDR_normmaxpv': m3[1], 'ADD_normmaxpv': m3[2]})
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print("ABLATION STUDY: TCN-AE + CP on TEP")
print("="*65)
print_device(DEVICE)

# ── Load data ─────────────────────────────────────────────────────────────────
print("\n1. LOADING DATA")
r = pyreadr.read_r(str(ARCHIVE / "TEP_FaultFree_Training.RData"))
df_ff = list(r.values())[0]
del r
df_ff[FEAT] = df_ff[FEAT].astype(np.float32)
all_runs = sorted(df_ff['simulationRun'].unique())
tr_runs, norm_runs, cal_runs = split_faultfree_runs(all_runs)
X_tr_raw   = df_ff[df_ff['simulationRun'].isin(tr_runs)][FEAT].to_numpy(copy=True)
X_norm_raw = df_ff[df_ff['simulationRun'].isin(norm_runs)][FEAT].to_numpy(copy=True)
X_cal_raw  = df_ff[df_ff['simulationRun'].isin(cal_runs)][FEAT].to_numpy(copy=True)
del df_ff
gc.collect()

scaler = joblib.load(MODDIR / "scaler.pkl")
X_tr   = scaler.transform(X_tr_raw).astype(np.float32)
X_norm = scaler.transform(X_norm_raw).astype(np.float32)
X_cal  = scaler.transform(X_cal_raw).astype(np.float32)
print(f"Train: {X_tr.shape}  Norm: {X_norm.shape}  Cal: {X_cal.shape}")

print("\n2. OPENING FAULTY TEST DATA CACHE")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Ablation sweeps use FAULT_VAL_RUNS: {len(FAULT_VAL_RUNS)} runs/fault "
      f"(runs {FAULT_VAL_RUNS[0]}-{FAULT_VAL_RUNS[-1]}, disjoint from the "
      f"{len(FAULT_TEST_RUNS)} FAULT_TEST_RUNS used for final headline results)")

# ── Ablation 1: Window size ───────────────────────────────────────────────────
print("\n" + "="*65)
print("ABLATION 1: Window Size (W)")
print("="*65)
window_configs = [10, 20, 30, 40]
abl_window = []

for W in window_configs:
    print(f"\n -- W={W} --")
    m, t1, t2, t3, spv = train_model_ablation(X_tr, X_norm, X_cal, len(tr_runs), len(norm_runs),
                                               len(cal_runs), W, latent_dim=24)
    df_res = eval_model_on_test(m, scaler, faulty_cache, fault_ids, W, t1, t2, t3, spv,
                                 run_ids=FAULT_VAL_RUNS)
    df_res['config'] = f'W={W}'
    abl_window.append(df_res)
    print(f"   avg FDR_normmaxpv={df_res['FDR_normmaxpv'].mean():.4f}  "
          f"hard FDR={df_res[df_res['fault_id'].isin(HARD_FAULTS)]['FDR_normmaxpv'].mean():.4f}")
    gc.collect()
    clear_cuda_cache()

abl_w_df = pd.concat(abl_window, ignore_index=True)
abl_w_df.to_csv(f"{ADIR}/ablation_window.csv", index=False)
print(f"\nSaved {ADIR / 'ablation_window.csv'}")

# ── Ablation 2: Latent dimension ─────────────────────────────────────────────
print("\n" + "="*65)
print("ABLATION 2: Latent Dimension")
print("="*65)
latent_configs = [8, 16, 24, 32, 48]
abl_latent = []
WINDOW_BEST = 20

for ld in latent_configs:
    print(f"\n -- latent_dim={ld} --")
    m, t1, t2, t3, spv = train_model_ablation(X_tr, X_norm, X_cal, len(tr_runs), len(norm_runs),
                                               len(cal_runs), WINDOW_BEST, latent_dim=ld)
    df_res = eval_model_on_test(m, scaler, faulty_cache, fault_ids, WINDOW_BEST, t1, t2, t3, spv,
                                 run_ids=FAULT_VAL_RUNS)
    df_res['config'] = f'lat={ld}'
    abl_latent.append(df_res)
    print(f"   avg FDR_normmaxpv={df_res['FDR_normmaxpv'].mean():.4f}  "
          f"hard FDR={df_res[df_res['fault_id'].isin(HARD_FAULTS)]['FDR_normmaxpv'].mean():.4f}")
    gc.collect()

abl_l_df = pd.concat(abl_latent, ignore_index=True)
abl_l_df.to_csv(f"{ADIR}/ablation_latent.csv", index=False)
print(f"\nSaved {ADIR / 'ablation_latent.csv'}")

# ── Ablation 3: Data size (training run count) ────────────────────────────────
print("\n" + "="*65)
print("ABLATION 3: Training Data Size")
print("="*65)
data_configs = [100, 200, 300, 400]  # out of the 400 training runs
abl_data = []

for n_tr in data_configs:
    print(f"\n -- n_train={n_tr} (norm={len(norm_runs)}, cal={len(cal_runs)} unchanged) --")
    X_tr_sub = X_tr[:n_tr * 500]   # 500 samples per run
    m, t1, t2, t3, spv = train_model_ablation(X_tr_sub, X_norm, X_cal, n_tr, len(norm_runs),
                                               len(cal_runs), WINDOW_BEST, latent_dim=24)
    df_res = eval_model_on_test(m, scaler, faulty_cache, fault_ids, WINDOW_BEST, t1, t2, t3, spv,
                                 run_ids=FAULT_VAL_RUNS)
    df_res['config'] = f'n_tr={n_tr}'
    abl_data.append(df_res)
    print(f"   avg FDR_normmaxpv={df_res['FDR_normmaxpv'].mean():.4f}")
    gc.collect()

abl_d_df = pd.concat(abl_data, ignore_index=True)
abl_d_df.to_csv(f"{ADIR}/ablation_data_size.csv", index=False)
print(f"\nSaved {ADIR / 'ablation_data_size.csv'}")

# ── Alpha sensitivity (using pre-computed calibration scores) ─────────────────
# Uses NormMaxPV, the paper's proposed score (Section 5.9 studies "the
# NormMaxPV detector"), not raw MSE, so this sweep reflects the deployed
# method. Reuses the main model's std_pv from norm_stats.pkl (norm-estimation
# set, disjoint from cal_pv_scores.npy) -- NOT recomputed from cal_pv_scores.npy,
# which would repeat the point-1 leakage bug. Evaluated on FAULT_VAL_RUNS,
# since selecting a significance level from an FDR/FAR curve is itself a form
# of tuning (round-2 peer review, point 5).
print("\n" + "="*65)
print("ABLATION 4: Alpha (FAR Target) Sensitivity")
print("="*65)
norm_stats_main = joblib.load(SDIR / "norm_stats.pkl")
std_pv_main = norm_stats_main['std_pv']
cal_pv_main = np.load(SDIR / "cal_pv_scores.npy")          # (N_cal, V)
cal_normmaxpv_main = (cal_pv_main / std_pv_main).max(axis=1)
fault_ids2 = faulty_cache.fault_ids

model_main = TCNAE(n_vars=52, window=20, n_filters=64, latent_dim=24)
model_main.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model_main.to(DEVICE)
model_main.eval()
scaler_main = joblib.load(MODDIR / "scaler.pkl")

# Pre-compute per-variable scores for ALL validation runs ONCE, then apply
# multiple NormMaxPV thresholds (one inference pass, re-used across all alpha values)
print("Pre-computing run scores (one inference pass for all alpha values)...")
run_normmaxpv_cache = {}  # {fid: [s_normmaxpv_array, ...]}
for fid in fault_ids2:
    run_scores = []
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_VAL_RUNS):
        if len(X_run) < 960:
            continue
        Xs = scaler_main.transform(X_run).astype(np.float32)
        n = len(Xs)
        W_arr = make_run_windows(Xs, 20)
        normmaxpv_w = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                recon, _ = model_main(batch)
                pv = tensor_to_numpy(((batch - recon)**2).mean(dim=2))
                normmaxpv_w.append((pv / std_pv_main).max(axis=1))
        normmaxpv_w = np.concatenate(normmaxpv_w)
        s_normmaxpv = np.full(n, normmaxpv_w[0])
        for i, w_idx in enumerate(range(WINDOW_BEST - 1, n)):
            s_normmaxpv[w_idx] = normmaxpv_w[i]
        run_scores.append(s_normmaxpv)
    run_normmaxpv_cache[fid] = run_scores
    print(f"  F{fid:2d}: {len(run_scores)} runs cached")

alpha_configs = [0.001, 0.005, 0.01, 0.02, 0.05]
abl_alpha = []
for alpha in alpha_configs:
    tau_n = cp_threshold(cal_normmaxpv_main, alpha)
    records = []
    for fid in fault_ids2:
        res = []
        for s_normmaxpv in run_normmaxpv_cache[fid]:
            n_pre  = FAULT_START - 1
            n_post = len(s_normmaxpv) - n_pre
            alarms = s_normmaxpv > tau_n
            FAR = float(alarms[:n_pre].mean())
            FDR = float(alarms[n_pre:].mean())
            first = np.where(alarms[n_pre:])[0]
            ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
            res.append((FAR, FDR, ADD))
        m1 = np.mean(res, axis=0)
        records.append({'fault_id': fid, 'alpha': alpha,
                        'FAR': m1[0], 'FDR': m1[1], 'ADD': m1[2]})
    df_a = pd.DataFrame(records)
    df_a['config'] = f'alpha={alpha}'
    abl_alpha.append(df_a)
    avg_fdr = df_a['FDR'].mean()
    avg_far = df_a['FAR'].mean()
    print(f"  alpha={alpha:.3f}: avg_FDR={avg_fdr:.4f}, avg_FAR={avg_far:.4f}")

abl_a_df = pd.concat(abl_alpha, ignore_index=True)
abl_a_df.to_csv(f"{ADIR}/ablation_alpha.csv", index=False)
print(f"\nSaved {ADIR / 'ablation_alpha.csv'}")

# ── Limited calibration-data sensitivity (round-2 review, point 7) ────────────
# Does the FAR bound and NormMaxPV threshold stay stable with less conformal-
# calibration data? Subsets cal_pv_scores.npy by whole cal RUNS (not windows),
# since runs are the actual exchangeable/independent unit (Appendix A) --
# reuses run_normmaxpv_cache from Ablation 4 (same FAULT_VAL_RUNS scores, no
# new inference pass) and just recomputes tau from fewer calibration runs.
print("\n" + "="*65)
print("ABLATION 5: Limited Calibration-Data Sensitivity")
print("="*65)
windows_per_cal_run = 500 - 20 + 1   # 481, RunWindowDataset stride=1, W=20
n_cal_runs_total = len(cal_pv_main) // windows_per_cal_run
assert len(cal_pv_main) == n_cal_runs_total * windows_per_cal_run, \
    "cal_pv_scores.npy is not an exact multiple of windows_per_cal_run -- not run-aligned"
cal_normmaxpv_by_run = cal_normmaxpv_main.reshape(n_cal_runs_total, windows_per_cal_run)

cal_size_configs = [c for c in (10, 25, 50) if c <= n_cal_runs_total]
abl_calsize = []
for n_cal_runs in cal_size_configs:
    cal_subset = cal_normmaxpv_by_run[:n_cal_runs].reshape(-1)
    tau_n = cp_threshold(cal_subset, ALPHA)
    records = []
    for fid in fault_ids2:
        res = []
        for s_normmaxpv in run_normmaxpv_cache[fid]:
            n_pre  = FAULT_START - 1
            n_post = len(s_normmaxpv) - n_pre
            alarms = s_normmaxpv > tau_n
            FAR = float(alarms[:n_pre].mean())
            FDR = float(alarms[n_pre:].mean())
            first = np.where(alarms[n_pre:])[0]
            ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
            res.append((FAR, FDR, ADD))
        m1 = np.mean(res, axis=0)
        records.append({'fault_id': fid, 'n_cal_runs': n_cal_runs, 'tau': tau_n,
                        'FAR': m1[0], 'FDR': m1[1], 'ADD': m1[2]})
    df_c = pd.DataFrame(records)
    df_c['config'] = f'n_cal_runs={n_cal_runs}'
    abl_calsize.append(df_c)
    print(f"  n_cal_runs={n_cal_runs:3d} ({n_cal_runs*windows_per_cal_run} windows): "
          f"tau={tau_n:.5f}  avg_FDR={df_c['FDR'].mean():.4f}  avg_FAR={df_c['FAR'].mean():.4f}")

abl_c_df = pd.concat(abl_calsize, ignore_index=True)
abl_c_df.to_csv(f"{ADIR}/ablation_calsize.csv", index=False)
print(f"\nSaved {ADIR / 'ablation_calsize.csv'}")

del faulty_cache
gc.collect()
print("Run aggregate_multiseed.py + gen_ablation_figs.py once all ablation seeds are done.")
print("\n" + "="*65)
print(f"ABLATION STUDY COMPLETE (seed={SEED})")
print("="*65)
