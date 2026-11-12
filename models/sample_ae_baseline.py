"""
Sample-wise autoencoder baseline -- the true temporal-blind control in the
architecture x score factorial (round-2 peer review, point 4). Input/output
is a single standardized sample x_t in R^52; unlike the Dense AE (which
flattens a (52 x 20) window and so still has full temporal information
available to its weights, a point the reviewer specifically raised), this
architecture never sees more than one time sample, so it cannot use temporal
structure even implicitly.

Together with dense_ae_baseline.py (window dense AE), conv_ae_baseline.py
(conventional, non-dilated non-residual CNN-AE), and the TCN-AE itself, this
gives 4 architectures. All four are crossed with MSE, MaxPV, and NormMaxPV
(12 cells) using the same train/norm-estimation/cal split discipline as
train_tcnae.py (tcnae_utils.split_faultfree_runs), so std_pv is fit only on
the norm-estimation set and never on the cal set used for the conformal
quantile.

Multi-seed: set TCNAE_SEED=<int>; results written to
results/seeds/seed_<TCNAE_SEED>/sample_ae_results.csv; checkpoint to
models/seeds/seed_<TCNAE_SEED>/sample_ae_best.pt. See models/run_multiseed.py.
"""
import gc
import os
import sys
import numpy as np
import pandas as pd
import pyreadr
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    ARCHIVE,
    FEAT,
    MODDIR,
    N_VARS,
    OUTDIR,
    configured_batch_size,
    configured_seed,
    cp_threshold,
    load_faulty_test_cache,
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

FAULT_START = 161
ALPHA  = 0.01
DEVICE = select_device()
PIN_MEMORY = DEVICE.type == "cuda"
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 4096)
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

class SampleAE(nn.Module):
    """Plain per-sample autoencoder. Input/output: (B, 52). No window."""
    def __init__(self, n_vars=52, hidden=64, latent_dim=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_vars, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_vars),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def reconstruction_error(self, x):
        recon, _ = self(x)
        return ((x - recon) ** 2).mean(dim=1)

    def per_variable_error(self, x):
        recon, _ = self(x)
        return (x - recon) ** 2

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

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
print("SAMPLE-WISE AE BASELINE (no window, true temporal-blind control)")
print("="*65)
print_device(DEVICE)

scaler = joblib.load(MODDIR / "scaler.pkl")

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

# No windowing: each standardized sample is directly a training example.
X_tr   = scaler.transform(X_tr_raw).astype(np.float32)
X_norm = scaler.transform(X_norm_raw).astype(np.float32)
X_cal  = scaler.transform(X_cal_raw).astype(np.float32)
print(f"Train samples: {X_tr.shape}  Norm samples: {X_norm.shape}  Cal samples: {X_cal.shape}")

model = SampleAE(N_VARS, hidden=64, latent_dim=24).to(DEVICE)
print(f"SampleAE params: {count_params(model):,}")

EPOCHS = 40
LR = 1e-3
BATCH = configured_batch_size("TCNAE_BATCH_SIZE", 4096)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
ds_tr = TensorDataset(torch.from_numpy(X_tr))
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=PIN_MEMORY)

print("\nTraining SampleAE...")
best_loss = float('inf')
best_state = None
for epoch in range(1, EPOCHS + 1):
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
    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={el:.6f}  best={best_loss:.6f}")

model.load_state_dict(best_state)
model.eval()
torch.save(model.state_dict(), SDIR / "sample_ae_best.pt")
print(f"Best training loss: {best_loss:.6f}  Saved: {SDIR / 'sample_ae_best.pt'}")

# ── Normalization-set pass (std_pv only) ──────────────────────────────────────
print("\nNormalization-set pass...")
with torch.no_grad():
    norm_pv = []
    xb_all = torch.from_numpy(X_norm)
    for i in range(0, len(xb_all), INFER_BS):
        xb = move_to_device(xb_all[i:i+INFER_BS], DEVICE)
        norm_pv.append(tensor_to_numpy(model.per_variable_error(xb)))
norm_pv = np.concatenate(norm_pv, axis=0)
std_pv = norm_pv.std(axis=0) + 1e-9
print(f"std_pv: mean={std_pv.mean():.6f}")
del norm_pv, X_norm
gc.collect()

# ── Calibration-set pass (every conformal quantile) ───────────────────────────
print("\nCalibration-set pass...")
with torch.no_grad():
    cal_scores, cal_pv = [], []
    xb_all = torch.from_numpy(X_cal)
    for i in range(0, len(xb_all), INFER_BS):
        xb = move_to_device(xb_all[i:i+INFER_BS], DEVICE)
        cal_scores.append(tensor_to_numpy(model.reconstruction_error(xb)))
        cal_pv.append(tensor_to_numpy(model.per_variable_error(xb)))
cal_scores = np.concatenate(cal_scores)
cal_pv = np.concatenate(cal_pv, axis=0)
cal_maxpv = cal_pv.max(axis=1)
cal_normmaxpv = (cal_pv / std_pv).max(axis=1)

tau_mse = cp_threshold(cal_scores, ALPHA)
tau_maxpv = cp_threshold(cal_maxpv, ALPHA)
tau_normmaxpv = cp_threshold(cal_normmaxpv, ALPHA)
print(f"tau_mse={tau_mse:.6f}  tau_maxpv={tau_maxpv:.6f}  tau_normmaxpv={tau_normmaxpv:.6f}")
print(f"FAR on cal: mse={np.mean(cal_scores>tau_mse):.5f}  maxpv={np.mean(cal_maxpv>tau_maxpv):.5f}  "
      f"normmaxpv={np.mean(cal_normmaxpv>tau_normmaxpv):.5f}")

joblib.dump({'tau_mse': float(tau_mse), 'tau_maxpv': float(tau_maxpv),
             'tau_normmaxpv': float(tau_normmaxpv), 'std_pv': std_pv, 'alpha': ALPHA},
            SDIR / "sample_ae_thresholds.pkl")
del X_tr, X_cal
gc.collect()

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\nOpening faulty test data cache...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

def score_run(X_run_raw):
    """No windowing: each sample is scored independently. Returns (960,)
    MSE/MaxPV/NormMaxPV arrays directly -- no window-to-sample assignment
    needed (unlike every windowed architecture in this codebase)."""
    Xs = torch.from_numpy(scaler.transform(X_run_raw).astype(np.float32))
    mse_s, maxpv_s, normmaxpv_s = [], [], []
    with torch.no_grad():
        for i in range(0, len(Xs), INFER_BS):
            xb = move_to_device(Xs[i:i+INFER_BS], DEVICE)
            mse_s.append(tensor_to_numpy(model.reconstruction_error(xb)))
            pv = tensor_to_numpy(model.per_variable_error(xb))
            maxpv_s.append(pv.max(axis=1))
            normmaxpv_s.append((pv / std_pv).max(axis=1))
    return np.concatenate(mse_s), np.concatenate(maxpv_s), np.concatenate(normmaxpv_s)

print("\nEvaluating SampleAE on all faults...")
records = []
for fid in fault_ids:
    res_mse, res_maxpv, res_normmaxpv = [], [], []
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        s_mse, s_maxpv, s_normmaxpv = score_run(X_run)
        res_mse.append(eval_run_metrics(s_mse, tau_mse))
        res_maxpv.append(eval_run_metrics(s_maxpv, tau_maxpv))
        res_normmaxpv.append(eval_run_metrics(s_normmaxpv, tau_normmaxpv))
    m1 = np.mean(res_mse, axis=0)
    m2 = np.mean(res_maxpv, axis=0)
    m3 = np.mean(res_normmaxpv, axis=0)
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'SampleAE-CP',
                    'FAR': round(m1[0], 5), 'FDR': round(m1[1], 5), 'ADD': round(m1[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'SampleAE-MaxPV',
                    'FAR': round(m2[0], 5), 'FDR': round(m2[1], 5), 'ADD': round(m2[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'SampleAE-NormMaxPV',
                    'FAR': round(m3[0], 5), 'FDR': round(m3[1], 5), 'ADD': round(m3[2], 2)})
    print(f"  F{fid:2d}: MSE FDR={m1[1]:.4f} | MaxPV FDR={m2[1]:.4f} | NormMaxPV FDR={m3[1]:.4f}")

del faulty_cache
gc.collect()
df = pd.DataFrame(records)
df.to_csv(RDIR / "sample_ae_results.csv", index=False)
print(f"\nSaved {RDIR / 'sample_ae_results.csv'}")
print("\n" + "="*65)
print(f"SAMPLE-WISE AE BASELINE COMPLETE (seed={SEED})")
print("="*65)
