"""
Dense Autoencoder baseline (no temporal structure).
Isolates the contribution of TCN temporal modeling vs. plain AE, and is one
of the four architectures in the architecture x score factorial (see also
sample_ae_baseline.py, conv_ae_baseline.py, and the TCN-AE itself).
Input: flattened window (52 x 20 = 1040) -> encode -> decode -> reconstruct.
Same three-way run split and conformal calibration as the TCN-AE (train /
norm-estimation / cal; see tcnae_utils.split_faultfree_runs and
train_tcnae.py's module docstring) for a fair comparison: std_pv (for
NormMaxPV) is fit on the norm-estimation set, and every conformal threshold
(MSE, MaxPV, NormMaxPV) comes only from the disjoint cal set.
Multi-seed: set TCNAE_SEED=<int>; results written to
results/seeds/seed_<TCNAE_SEED>/dense_ae_results.csv; model checkpoint to
models/seeds/seed_<TCNAE_SEED>/dense_ae_best.pt. See models/run_multiseed.py.
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
SDIR = seed_model_dir(SEED)     # models/seeds/seed_{SEED}/
RDIR = seed_results_dir(SEED)   # results/seeds/seed_{SEED}/

FAULT_START = 161
ALPHA  = 0.01
WINDOW = 20
FLAT   = N_VARS * WINDOW   # 1040
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

# ── Dense AE architecture ─────────────────────────────────────────────────────
class DenseAE(nn.Module):
    """Plain dense autoencoder. Input: (B, 1040), Output: (B, 1040)."""
    def __init__(self, input_dim=1040, hidden=[512, 256, 128], latent_dim=24):
        super().__init__()
        enc_layers = []
        in_d = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(in_d, h), nn.BatchNorm1d(h), nn.GELU()]
            in_d = h
        enc_layers += [nn.Linear(in_d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden[-1]), nn.GELU(),
            nn.Linear(hidden[-1], hidden[-2]), nn.GELU(),
            nn.Linear(hidden[-2], hidden[-3]), nn.GELU(),
            nn.Linear(hidden[-3], input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def reconstruction_error(self, x):
        recon, _ = self(x)
        return ((x - recon)**2).mean(dim=1)

    def per_variable_error(self, x, n_vars, window):
        """Per-variable squared error, reshaping the flattened (B, V*W)
        reconstruction back to (B, V, W) -- RunWindowDataset(flatten=True)
        flattens each (V, W) window in row-major order, so this reshape
        recovers the original (variable, time) axes exactly."""
        recon, _ = self(x)
        err2 = (x - recon).view(-1, n_vars, window)**2
        return err2.mean(dim=2)

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

# ── Train ─────────────────────────────────────────────────────────────────────
print("="*65)
print("DENSE AE BASELINE (no temporal structure)")
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
ds_tr   = RunWindowDataset(X_tr, len(tr_runs), WINDOW, flatten=True)
ds_norm = RunWindowDataset(X_norm, len(norm_runs), WINDOW, flatten=True)
ds_cal  = RunWindowDataset(X_cal, len(cal_runs), WINDOW, flatten=True)
print(f"Train windows: {len(ds_tr):,}  Norm windows: {len(ds_norm):,}  Cal windows: {len(ds_cal):,}")

model = DenseAE(FLAT, [512, 256, 128], latent_dim=24).to(DEVICE)
print(f"DenseAE params: {count_params(model):,}")

EPOCHS = 80
LR = 1e-3
BATCH = configured_batch_size("TCNAE_BATCH_SIZE", 2048)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0,
                   pin_memory=PIN_MEMORY)

print("\nTraining DenseAE...")
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
    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={el:.6f}  best={best_loss:.6f}")

model.load_state_dict(best_state)
model.eval()
torch.save(model.state_dict(), SDIR / "dense_ae_best.pt")
print(f"Best training loss: {best_loss:.6f}")
print(f"Model saved: {SDIR / 'dense_ae_best.pt'}")

# ── Normalization-set pass (std_pv only; NOT used for any threshold) ─────────
print("\nNormalization-set pass (std_pv for NormMaxPV)...")
dl_norm = DataLoader(ds_norm, batch_size=INFER_BS, shuffle=False, num_workers=0,
                     pin_memory=PIN_MEMORY)
norm_pv = []
with torch.no_grad():
    for (xb,) in dl_norm:
        xb = move_to_device(xb, DEVICE)
        pv = model.per_variable_error(xb, N_VARS, WINDOW)
        norm_pv.append(tensor_to_numpy(pv))
norm_pv = np.concatenate(norm_pv, axis=0)
std_pv = norm_pv.std(axis=0) + 1e-9
print(f"Norm windows: {norm_pv.shape}  std_pv: mean={std_pv.mean():.6f}")
del norm_pv, dl_norm, ds_norm, X_norm
gc.collect()

# ── Calibration-set pass (every conformal quantile) ───────────────────────────
print("\nCalibration-set pass...")
dl_cal = DataLoader(ds_cal, batch_size=INFER_BS, shuffle=False, num_workers=0,
                    pin_memory=PIN_MEMORY)
cal_scores, cal_pv = [], []
with torch.no_grad():
    for (xb,) in dl_cal:
        xb = move_to_device(xb, DEVICE)
        s = model.reconstruction_error(xb)
        pv = model.per_variable_error(xb, N_VARS, WINDOW)
        cal_scores.append(tensor_to_numpy(s))
        cal_pv.append(tensor_to_numpy(pv))
cal_scores = np.concatenate(cal_scores)
cal_pv = np.concatenate(cal_pv, axis=0)          # (N_cal, V)
cal_maxpv = cal_pv.max(axis=1)

tau = cp_threshold(cal_scores, ALPHA)
tau_maxpv = cp_threshold(cal_maxpv, ALPHA)
far_cal = (cal_scores > tau).mean()
far_maxpv_cal = (cal_maxpv > tau_maxpv).mean()
cal_normmaxpv = (cal_pv / std_pv).max(axis=1)
tau_normmaxpv = cp_threshold(cal_normmaxpv, ALPHA)
far_normmaxpv_cal = (cal_normmaxpv > tau_normmaxpv).mean()
print(f"tau_mse={tau:.6f}  FAR_cal={far_cal:.5f}")
print(f"tau_maxpv={tau_maxpv:.6f}  FAR_maxpv_cal={far_maxpv_cal:.5f}")
print(f"tau_normmaxpv={tau_normmaxpv:.6f}  FAR_normmaxpv_cal={far_normmaxpv_cal:.5f}")

joblib.dump({'tau_mse': float(tau), 'tau_maxpv': float(tau_maxpv),
             'tau_normmaxpv': float(tau_normmaxpv), 'std_pv': std_pv, 'alpha': ALPHA},
            SDIR / "dense_ae_thresholds.pkl")
del X_tr, X_cal, ds_tr, ds_cal
gc.collect()

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\nOpening faulty test data cache...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Faults: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault")

def score_run_dae(X_run_raw):
    """Returns (n,) MSE, (n,) MaxPV, (n,) NormMaxPV per-sample scores for a run."""
    Xs = scaler.transform(X_run_raw).astype(np.float32)
    n  = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW, flatten=True)
    mse_w, maxpv_w, normmaxpv_w = [], [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            b = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            s = model.reconstruction_error(b)
            pv = tensor_to_numpy(model.per_variable_error(b, N_VARS, WINDOW))
            mse_w.append(tensor_to_numpy(s))
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

print("\nEvaluating DenseAE on all faults...")
records = []
for fid in fault_ids:
    results_mse, results_maxpv, results_normmaxpv = [], [], []
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue
        s_mse, s_maxpv, s_normmaxpv = score_run_dae(X_run)
        results_mse.append(eval_run_metrics(s_mse, tau))
        results_maxpv.append(eval_run_metrics(s_maxpv, tau_maxpv))
        results_normmaxpv.append(eval_run_metrics(s_normmaxpv, tau_normmaxpv))
    m = np.mean(results_mse, axis=0)
    mm = np.mean(results_maxpv, axis=0)
    mn = np.mean(results_normmaxpv, axis=0)
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid],
                    'method': 'DenseAE-CP',
                    'FAR': round(m[0], 5), 'FDR': round(m[1], 5), 'ADD': round(m[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid],
                    'method': 'DenseAE-MaxPV',
                    'FAR': round(mm[0], 5), 'FDR': round(mm[1], 5), 'ADD': round(mm[2], 2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid],
                    'method': 'DenseAE-NormMaxPV',
                    'FAR': round(mn[0], 5), 'FDR': round(mn[1], 5), 'ADD': round(mn[2], 2)})
    print(f"  F{fid:2d}: MSE FDR={m[1]:.4f} FAR={m[0]:.4f} ADD={m[2]:.1f} | "
          f"MaxPV FDR={mm[1]:.4f} FAR={mm[0]:.4f} ADD={mm[2]:.1f} | "
          f"NormMaxPV FDR={mn[1]:.4f} FAR={mn[0]:.4f} ADD={mn[2]:.1f}")

del faulty_cache
gc.collect()
dae_df = pd.DataFrame(records)
dae_df.to_csv(f"{RDIR}/dense_ae_results.csv", index=False)

# ── Comparison summary ────────────────────────────────────────────────────────
pca_res  = pd.read_csv(OUTDIR / "pca_baseline_results.csv")
pca_best = pca_res[pca_res['method'] == 'PCA-Q(n=7)'].set_index('fault_id')

print("\n" + "="*65)
print("DenseAE vs PCA-Q SUMMARY")
print("="*65)
for method in ['DenseAE-CP', 'DenseAE-MaxPV', 'DenseAE-NormMaxPV']:
    sub = dae_df[dae_df['method'] == method]
    hard = sub[sub['fault_id'].isin(HARD_FAULTS)]
    easy = sub[~sub['fault_id'].isin(HARD_FAULTS)]
    print(f"  {method} avg FDR: {sub['FDR'].mean()*100:.2f}%  "
          f"avg FAR: {sub['FAR'].mean()*100:.4f}%  "
          f"hard FDR: {hard['FDR'].mean()*100:.2f}%  easy FDR: {easy['FDR'].mean()*100:.2f}%")
print(f"  PCA-Q   avg FDR: {pca_best['FDR'].mean()*100:.2f}%")
print(f"\nSaved {RDIR / 'dense_ae_results.csv'}")
print("\n" + "="*65)
print(f"DENSE AE BASELINE COMPLETE (seed={SEED})")
print("="*65)
