"""
Standalone script to generate fig21_latent_space.png.
Requires: tcnae_best.pt, scaler.pkl, cal_z.npy (latent cal embeddings).
Reads a small subset of cached faulty-test runs (3 runs × 5 faults).
"""
import os
import sys
import gc
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA

sys.path.insert(0, os.path.dirname(__file__))
import paper_style
from tcn_ae import TCNAE
from tcnae_utils import (
    MODDIR,
    OUTDIR,
    configured_batch_size,
    load_faulty_test_cache,
    make_run_windows,
    move_to_device,
    print_device,
    seed_everything,
    seed_model_dir,
    select_device,
    split_fault_test_runs,
    tensor_to_numpy,
)

paper_style.apply()
seed_everything(42)
REF_SEED = 42
SDIR = seed_model_dir(REF_SEED)

FAULT_START = 161
WINDOW   = 20
DEVICE   = select_device()
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)

print(f"Loading model (reference seed={REF_SEED}) and calibration latents...")
print_device(DEVICE)
scaler = joblib.load(MODDIR / "scaler.pkl")   # shared across seeds
model  = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

cal_z = np.load(SDIR / "cal_z.npy")
print(f"  cal_z loaded: {cal_z.shape}")

print("Opening faulty test data cache (subset access)...")
faulty_cache = load_faulty_test_cache()
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"  {faulty_cache.describe()}")

SHOW_FAULTS = [1, 3, 9, 15, 16]
N_RUNS_EACH = 3

z_groups   = []
label_list = []

n_normal = min(5000, len(cal_z))
z_groups.append(cal_z[:n_normal])
label_list.extend(['Normal (cal)'] * n_normal)

for fid in SHOW_FAULTS:
    z_f = []
    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS[:N_RUNS_EACH]):
        if len(X_run) < 960:
            continue
        Xs = scaler.transform(X_run).astype(np.float32)
        W_arr = make_run_windows(Xs[FAULT_START - 1:], WINDOW)
        zl = []
        with torch.no_grad():
            for i in range(0, len(W_arr), INFER_BS):
                batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
                _, z = model(batch)
                zl.append(tensor_to_numpy(z))
        z_f.append(np.concatenate(zl))
    if z_f:
        zz = np.concatenate(z_f)
        z_groups.append(zz)
        label_list.extend([f'Fault {fid}'] * len(zz))
        print(f"  Fault {fid}: {len(zz)} fault-period windows")

del faulty_cache
gc.collect()

all_z  = np.vstack(z_groups)
pca2   = SKPCA(n_components=2)
all_2d = pca2.fit_transform(all_z)
print(f"PCA explained variance: PC1={pca2.explained_variance_ratio_[0]*100:.1f}%, PC2={pca2.explained_variance_ratio_[1]*100:.1f}%")

colors_map = {
    'Normal (cal)': '#AAAAAA',
    'Fault 1':  '#2196F3',
    'Fault 3':  '#E85D4A',
    'Fault 9':  '#FF9800',
    'Fault 15': '#9C27B0',
    'Fault 16': '#4CAF50',
}
_, ax = plt.subplots(figsize=(5.5, 4.5))
for grp, col in colors_map.items():
    idx = [i for i, label in enumerate(label_list) if label == grp]
    if idx:
        ax.scatter(all_2d[idx, 0], all_2d[idx, 1], c=col, s=4, alpha=0.35, label=grp)
ax.set_xlabel(f'PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var.)', fontsize=9)
ax.set_ylabel(f'PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var.)', fontsize=9)
ax.set_title('TCN-AE Latent Space (2D PCA projection)\nFault-period vs Normal calibration windows', fontsize=9)
ax.legend(fontsize=8, markerscale=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig21_latent_space.png")
plt.close()
print("Saved fig21_latent_space.png")
print("DONE")
