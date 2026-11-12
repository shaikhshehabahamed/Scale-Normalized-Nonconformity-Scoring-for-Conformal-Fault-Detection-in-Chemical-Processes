"""
Train TCN-AE on TEP normal data + conformal calibration.
Saves: trained model, scaler, conformal thresholds, and every calibration
artifact downstream scripts need (cal_scores/cal_pv/cal_z for the CAL set,
norm_stats.pkl + norm_z.npy for the NORM set).

Fault-free runs are split three ways (tcnae_utils.split_faultfree_runs):
train (400) / norm-estimation (50) / conformal-calibration (50). The norm
set exists solely to fix nonconformity-score nuisance parameters --
NormMaxPV's per-variable std (this file) and the Mahalanobis mean/covariance
(latent_mahal.py, from this file's norm_z.npy) -- *before* any conformal
quantile is touched. Every conformal threshold (here and in pvbc_eval.py,
latent_mahal.py, ablation_study.py) is then computed only from the CAL set,
using those already-fixed nuisance parameters. This is the round-2 peer
review's point-1 fix: without this split, each calibration score would be
computed with normalization statistics that include that very observation,
while a fresh test score would not, breaking the exchangeability the
split-conformal FAR bound (Eq. 12) relies on.

Usage: uv run python models/train_tcnae.py
Multi-seed: set TCNAE_SEED=<int> to train one ensemble member; artifacts are
written to models/seeds/seed_<TCNAE_SEED>/ (scaler.pkl stays shared at
models/ since StandardScaler has no RNG). See models/run_multiseed.py.
"""
import os
import sys
import gc
import time
import numpy as np
import pyreadr
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# ─── make tcn_ae importable ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from tcn_ae import TCNAE, count_parameters
from tcnae_utils import (
    ARCHIVE,
    CAL_RUNS,
    FEAT,
    MODDIR,
    NORM_RUNS,
    RunWindowDataset,
    TRAIN_RUNS,
    configured_batch_size,
    configured_seed,
    cp_threshold,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    select_device,
    split_faultfree_runs,
    tensor_to_numpy,
)

# ─── config ─────────────────────────────────────────────────────────────────
os.makedirs(MODDIR, exist_ok=True)

WINDOW      = 20        # samples = 60 min at 3-min sampling
N_FILTERS   = 64
LATENT_DIM  = 24
DILATIONS   = (1, 2, 4, 8)
KERNEL      = 3
BATCH_SIZE  = configured_batch_size("TCNAE_BATCH_SIZE", 2048)
LR          = 1e-3
EPOCHS      = 100
ALPHA       = 0.01      # target FAR
SEED        = configured_seed()
seed_everything(SEED)
DEVICE = select_device()
PIN_MEMORY = DEVICE.type == "cuda"
SDIR = seed_model_dir(SEED)   # models/seeds/seed_{SEED}/ -- one ensemble member

# ─── 1. Load fault-free training data ───────────────────────────────────────
print("="*65)
print("1. LOADING FAULT-FREE TRAINING DATA")
print("="*65)
print_device(DEVICE)
r = pyreadr.read_r(str(ARCHIVE / "TEP_FaultFree_Training.RData"))
df = list(r.values())[0]
del r

df[FEAT] = df[FEAT].astype(np.float32)
all_runs = sorted(df['simulationRun'].unique())
tr_runs, norm_runs, cal_runs = split_faultfree_runs(all_runs)
print(f"Train runs: {len(tr_runs)} | Norm-estimation runs: {len(norm_runs)} | "
      f"Cal runs: {len(cal_runs)}")
assert not (set(norm_runs) & set(cal_runs)), "norm/cal run sets must be disjoint"

# ─── 2. Standardize ─────────────────────────────────────────────────────────
print("\n2. FITTING SCALER")
X_tr_raw   = df[df['simulationRun'].isin(tr_runs)][FEAT].to_numpy(copy=True)
X_norm_raw = df[df['simulationRun'].isin(norm_runs)][FEAT].to_numpy(copy=True)
X_cal_raw  = df[df['simulationRun'].isin(cal_runs)][FEAT].to_numpy(copy=True)
del df
gc.collect()

scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
X_norm = scaler.transform(X_norm_raw).astype(np.float32)
X_cal  = scaler.transform(X_cal_raw).astype(np.float32)
joblib.dump(scaler, MODDIR / "scaler.pkl")   # deterministic (no RNG) -> shared across seeds
print(f"X_tr: {X_tr.shape} | X_norm: {X_norm.shape} | X_cal: {X_cal.shape}")

print("\n3. PREPARING LAZY SLIDING-WINDOW DATASETS (W={})".format(WINDOW))
t0 = time.time()
ds_tr   = RunWindowDataset(X_tr, len(tr_runs), WINDOW)
ds_norm = RunWindowDataset(X_norm, len(norm_runs), WINDOW)
ds_cal  = RunWindowDataset(X_cal, len(cal_runs), WINDOW)
print(f"Train windows : {len(ds_tr):,}  ({time.time()-t0:.1f}s, lazy)")
print(f"Norm windows  : {len(ds_norm):,}  (lazy)")
print(f"Cal windows   : {len(ds_cal):,}  (lazy)")

# ─── 4. Initialise model ────────────────────────────────────────────────────
print("\n4. INITIALISING TCN-AE")
model = TCNAE(
    n_vars=len(FEAT), window=WINDOW,
    n_filters=N_FILTERS, latent_dim=LATENT_DIM,
    dilations=DILATIONS, kernel=KERNEL
).to(DEVICE)
print(f"Parameters: {count_parameters(model):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                   num_workers=0, pin_memory=PIN_MEMORY)

# ─── 5. Training loop ───────────────────────────────────────────────────────
print("\n5. TRAINING")
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for (xb,) in dl_tr:
        xb = move_to_device(xb, DEVICE)
        recon, _ = model(xb)
        loss = nn.functional.mse_loss(recon, xb)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item() * len(xb)
    epoch_loss /= len(ds_tr)
    scheduler.step()

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), SDIR / "tcnae_best.pt")

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={epoch_loss:.6f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  best={best_loss:.6f}")

print(f"\nBest training loss: {best_loss:.6f}")
print(f"Model saved: {SDIR / 'tcnae_best.pt'}")

# ─── 6. Normalization-set pass (nuisance parameters only, no thresholds) ────
print("\n6. NORMALIZATION-SET PASS (std_pv for NormMaxPV, latent codes for Mahalanobis)")
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.eval()

INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
dl_norm = DataLoader(ds_norm, batch_size=INFER_BS, shuffle=False,
                     num_workers=0, pin_memory=PIN_MEMORY)

norm_pv_scores, norm_z = [], []
with torch.no_grad():
    for (xb,) in dl_norm:
        xb = move_to_device(xb, DEVICE)
        recon, z = model(xb)
        pv = ((xb - recon) ** 2).mean(dim=2)   # (B, V)
        norm_pv_scores.append(tensor_to_numpy(pv))
        norm_z.append(tensor_to_numpy(z))

norm_pv = np.concatenate(norm_pv_scores, axis=0)   # (N_norm, 52)
norm_z = np.concatenate(norm_z, axis=0)            # (N_norm, latent_dim)
std_pv = norm_pv.std(axis=0) + 1e-9                # (52,) -- NormMaxPV's per-variable scale
print(f"Norm per-variable errors: {norm_pv.shape}  |  Norm latent codes: {norm_z.shape}")
print(f"std_pv: mean={std_pv.mean():.6f}  min={std_pv.min():.6f}  max={std_pv.max():.6f}")

np.save(SDIR / "norm_z.npy", norm_z)
joblib.dump({'std_pv': std_pv, 'n_norm_runs': len(norm_runs)}, SDIR / "norm_stats.pkl")
print(f"Saved {SDIR / 'norm_z.npy'}, {SDIR / 'norm_stats.pkl'}")
del norm_pv_scores, norm_z, dl_norm, ds_norm, X_norm
gc.collect()

# ─── 7. Calibration-set pass (every conformal quantile) ─────────────────────
print("\n7. CALIBRATION-SET PASS (conformal quantiles; std_pv fixed from step 6)")
dl_cal = DataLoader(ds_cal, batch_size=INFER_BS, shuffle=False,
                    num_workers=0, pin_memory=PIN_MEMORY)

cal_scores, cal_pv_scores, cal_z = [], [], []
with torch.no_grad():
    for (xb,) in dl_cal:
        xb = move_to_device(xb, DEVICE)
        recon, z = model(xb)
        err2 = (xb - recon) ** 2
        cal_scores.append(tensor_to_numpy(err2.mean(dim=(1, 2))))
        cal_pv_scores.append(tensor_to_numpy(err2.mean(dim=2)))
        cal_z.append(tensor_to_numpy(z))

cal_scores = np.concatenate(cal_scores)            # (N_cal,)
cal_pv = np.concatenate(cal_pv_scores, axis=0)      # (N_cal, 52)
cal_z = np.concatenate(cal_z, axis=0)               # (N_cal, latent_dim)
print(f"Cal scores: n={len(cal_scores)}, mean={cal_scores.mean():.6f}, "
      f"std={cal_scores.std():.6f}, p99={np.percentile(cal_scores,99):.6f}")

tau = cp_threshold(cal_scores, ALPHA)
far_cal = float((cal_scores > tau).mean())
print(f"Conformal threshold tau ({ALPHA*100:.0f}% FAR): {tau:.6f}")
print(f"Empirical FAR on calibration: {far_cal:.5f}  (target={ALPHA}; this is the "
      f"self-consistency check on the fitting sample, not the independent-test FAR "
      f"reported in the paper's Section 5.1/Table 5)")

np.save(SDIR / "cal_scores.npy", cal_scores)
np.save(SDIR / "cal_pv_scores.npy", cal_pv)
np.save(SDIR / "cal_z.npy", cal_z)
print(f"Saved {SDIR / 'cal_scores.npy'}, {SDIR / 'cal_pv_scores.npy'}, {SDIR / 'cal_z.npy'}")

# Raw (unnormalized) max per-variable score -- no nuisance parameter involved,
# so it needs no norm/cal split; calibrated directly on the CAL set like MSE.
cal_max_pv = cal_pv.max(axis=1)
q_mpv = cp_threshold(cal_max_pv, ALPHA)
far_mpv = float((cal_max_pv > q_mpv).mean())
print(f"Max-PV threshold: {q_mpv:.6f}  FAR={far_mpv:.5f}")
joblib.dump({'tau_mse': tau, 'tau_maxpv': q_mpv, 'alpha': ALPHA},
            SDIR / "conformal_thresholds.pkl")

print("\n" + "="*65)
print(f"TRAINING + CALIBRATION COMPLETE (seed={SEED})")
print("="*65)
