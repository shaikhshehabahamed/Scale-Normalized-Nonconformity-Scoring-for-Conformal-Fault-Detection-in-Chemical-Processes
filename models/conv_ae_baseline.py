"""
Conventional CNN autoencoder baseline -- isolates what the TCN-AE's dilated
receptive field and residual connections specifically contribute, as opposed
to "any convolutional processing of the window" (round-2 peer review, point
4). Architecturally identical to tcn_ae.TCNAE (same input projection, same
pooling-to-latent tail, same decoder) except the four DilatedBlock stages
(dilation in {1,2,4,8}, residual skip connections) are replaced by four
PlainConvBlock stages (dilation fixed at 1, no residual skip) -- same kernel
size and channel counts, so the parameter count is essentially unchanged;
only the dilation/residual design is removed.

Together with sample_ae_baseline.py (no window), dense_ae_baseline.py
(flattened window, dense), and the TCN-AE itself, this gives 4 architectures
crossed with MSE/MaxPV/NormMaxPV (12 cells). Same train/norm-estimation/cal
split discipline as train_tcnae.py (tcnae_utils.split_faultfree_runs).

Multi-seed: set TCNAE_SEED=<int>; results written to
results/seeds/seed_<TCNAE_SEED>/conv_ae_results.csv; checkpoint to
models/seeds/seed_<TCNAE_SEED>/conv_ae_best.pt. See models/run_multiseed.py.
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    ARCHIVE,
    FEAT,
    MODDIR,
    N_VARS,
    OUTDIR,
    RunWindowDataset,
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
WINDOW = 20
N_FILTERS = 64
LATENT_DIM = 24
KERNEL = 3
N_BLOCKS = 4   # matches TCN-AE's four dilated blocks
DEVICE = select_device()
PIN_MEMORY = DEVICE.type == "cuda"
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
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

class PlainConvBlock(nn.Module):
    """Conventional (non-dilated, non-residual) conv block: same two-layer
    conv-BN-GELU structure as tcn_ae.DilatedBlock, dilation fixed at 1, no
    skip connection."""
    def __init__(self, n_ch: int, kernel: int = 3):
        super().__init__()
        pad = (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, n_ch, kernel, padding=pad),
            nn.BatchNorm1d(n_ch),
            nn.GELU(),
            nn.Conv1d(n_ch, n_ch, kernel, padding=pad),
            nn.BatchNorm1d(n_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)   # no residual add -- the point of this baseline

class ConvAE(nn.Module):
    """Conventional CNN-AE. Same shape budget as TCNAE (tcn_ae.py): pointwise
    input projection -> N_BLOCKS conv stages -> pool -> latent -> decoder.
    Differs from TCNAE only in the conv stages (plain vs. dilated+residual)."""
    def __init__(self, n_vars=52, window=20, n_filters=64, latent_dim=24, kernel=3, n_blocks=4):
        super().__init__()
        self.window = window
        enc_blocks = [nn.Conv1d(n_vars, n_filters, 1)]
        for _ in range(n_blocks):
            enc_blocks.append(PlainConvBlock(n_filters, kernel))
        enc_blocks += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, latent_dim),
            nn.GELU(),
        ]
        self.encoder = nn.Sequential(*enc_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, n_filters * window),
            nn.Unflatten(1, (n_filters, window)),
            nn.Conv1d(n_filters, n_filters // 2, kernel, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(n_filters // 2, n_vars, kernel, padding=kernel // 2),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon[..., :self.window]
        return recon, z

    def reconstruction_error(self, x):
        recon, _ = self(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))

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
print("CONVENTIONAL CNN-AE BASELINE (no dilation, no residual connections)")
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

X_tr   = scaler.transform(X_tr_raw).astype(np.float32)
X_norm = scaler.transform(X_norm_raw).astype(np.float32)
X_cal  = scaler.transform(X_cal_raw).astype(np.float32)

print(f"Preparing lazy windows (W={WINDOW})...")
ds_tr   = RunWindowDataset(X_tr, len(tr_runs), WINDOW)
ds_norm = RunWindowDataset(X_norm, len(norm_runs), WINDOW)
ds_cal  = RunWindowDataset(X_cal, len(cal_runs), WINDOW)
print(f"Train windows: {len(ds_tr):,}  Norm windows: {len(ds_norm):,}  Cal windows: {len(ds_cal):,}")

model = ConvAE(N_VARS, WINDOW, N_FILTERS, LATENT_DIM, KERNEL, N_BLOCKS).to(DEVICE)
print(f"ConvAE params: {count_params(model):,}  (TCN-AE reference: ~148,000)")

EPOCHS = 100   # match train_tcnae.py's epoch budget (same encoder depth/size)
LR = 1e-3
BATCH = configured_batch_size("TCNAE_BATCH_SIZE", 2048)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=PIN_MEMORY)

print("\nTraining ConvAE...")
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
    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={el:.6f}  best={best_loss:.6f}")

model.load_state_dict(best_state)
model.eval()
torch.save(model.state_dict(), SDIR / "conv_ae_best.pt")
print(f"Best training loss: {best_loss:.6f}  Saved: {SDIR / 'conv_ae_best.pt'}")

# ── Normalization-set pass (std_pv only) ──────────────────────────────────────
print("\nNormalization-set pass...")
dl_norm = DataLoader(ds_norm, batch_size=INFER_BS, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
norm_pv = []
with torch.no_grad():
    for (xb,) in dl_norm:
        xb = move_to_device(xb, DEVICE)
        recon, _ = model(xb)
        norm_pv.append(tensor_to_numpy(((xb - recon) ** 2).mean(dim=2)))
norm_pv = np.concatenate(norm_pv, axis=0)
std_pv = norm_pv.std(axis=0) + 1e-9
print(f"std_pv: mean={std_pv.mean():.6f}")
del norm_pv, dl_norm, ds_norm, X_norm
gc.collect()

# ── Calibration-set pass ──────────────────────────────────────────────────────
print("\nCalibration-set pass...")
dl_cal = DataLoader(ds_cal, batch_size=INFER_BS, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
cal_scores, cal_pv = [], []
with torch.no_grad():
    for (xb,) in dl_cal:
        xb = move_to_device(xb, DEVICE)
        recon, _ = model(xb)
        err2 = (xb - recon) ** 2
        cal_scores.append(tensor_to_numpy(err2.mean(dim=(1, 2))))
        cal_pv.append(tensor_to_numpy(err2.mean(dim=2)))
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
            SDIR / "conv_ae_thresholds.pkl")
del X_tr, X_cal, ds_tr, ds_cal
gc.collect()

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\nOpening faulty test data cache...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

def score_run(X_run_raw):
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)
    mse_w, maxpv_w, normmaxpv_w = [], [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            b = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, _ = model(b)
            err2 = (b - recon) ** 2
            pv = tensor_to_numpy(err2.mean(dim=2))
            mse_w.append(tensor_to_numpy(err2.mean(dim=(1, 2))))
            maxpv_w.append(pv.max(axis=1))
            normmaxpv_w.append((pv / std_pv).max(axis=1))
    mse_w = np.concatenate(mse_w)
    maxpv_w = np.concatenate(maxpv_w)
    normmaxpv_w = np.concatenate(normmaxpv_w)
    s_mse = np.full(n, mse_w[0])
    s_maxpv = np.full(n, maxpv_w[0])
    s_normmaxpv = np.full(n, normmaxpv_w[0])
    for i, w_idx in enumerate(range(WINDOW - 1, n)):
        s_mse[w_idx] = mse_w[i]
        s_maxpv[w_idx] = maxpv_w[i]
        s_normmaxpv[w_idx] = normmaxpv_w[i]
    return s_mse, s_maxpv, s_normmaxpv

print("\nEvaluating ConvAE on all faults...")
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
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'ConvAE-CP',
                    'FAR': round(m1[0], 5), 'FDR': round(m1[1], 5), 'ADD': round(m1[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'ConvAE-MaxPV',
                    'FAR': round(m2[0], 5), 'FDR': round(m2[1], 5), 'ADD': round(m2[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid], 'method': 'ConvAE-NormMaxPV',
                    'FAR': round(m3[0], 5), 'FDR': round(m3[1], 5), 'ADD': round(m3[2], 2)})
    print(f"  F{fid:2d}: MSE FDR={m1[1]:.4f} | MaxPV FDR={m2[1]:.4f} | NormMaxPV FDR={m3[1]:.4f}")

del faulty_cache
gc.collect()
df = pd.DataFrame(records)
df.to_csv(RDIR / "conv_ae_results.csv", index=False)
print(f"\nSaved {RDIR / 'conv_ae_results.csv'}")
print("\n" + "="*65)
print(f"CONVENTIONAL CNN-AE BASELINE COMPLETE (seed={SEED})")
print("="*65)
