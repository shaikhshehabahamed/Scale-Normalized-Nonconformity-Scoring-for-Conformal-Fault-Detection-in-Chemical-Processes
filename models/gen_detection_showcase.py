"""
Generate fig25_detection_showcase.png:
  Detection alarm comparison on Faults 5, 10, 16, and 19 (run 1 each).
  Shows score(t) vs threshold for NormMaxPV, TCN-MSE, PCA-Comb.
  These are the four faults discussed in Section 5.5 where NormMaxPV
  dramatically outperforms PCA-Comb and (for F10) TCN-MaxPV.
"""
import os
import sys
import numpy as np
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import paper_style
from tcn_ae import TCNAE
from tcnae_utils import (
    MODDIR,
    OUTDIR,
    configured_batch_size,
    cp_threshold,
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
ALPHA = 0.01
INFER_BS = configured_batch_size("TCNAE_INFER_BS", 2048)
DEVICE = select_device()

# ── Load main TCN-AE model (reference seed) ────────────────────────────────────
print(f"Loading main TCN-AE model (reference seed={REF_SEED}) ...")
print_device(DEVICE)
model = TCNAE(n_vars=52, window=20, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
scaler = joblib.load(MODDIR / "scaler.pkl")   # shared across seeds

# Load calibration scores + norm-estimation std_pv for NormMaxPV threshold.
# std_pv comes from norm_stats.pkl (the norm-estimation set), never from
# cal_pv itself (round-2 review, point 1).
print("Loading calibration scores ...")
cal_pv     = np.load(SDIR / "cal_pv_scores.npy")   # (n_cal_windows, 52)
cal_mse    = np.load(SDIR / "cal_scores.npy")       # (n_cal_windows,)
tau_mse    = cp_threshold(cal_mse, ALPHA)
pv_std = joblib.load(SDIR / "norm_stats.pkl")['std_pv']
tau_normmaxpv = cp_threshold((cal_pv / pv_std).max(axis=1), ALPHA)
print(f"  tau_mse={tau_mse:.5f}  tau_normmaxpv={tau_normmaxpv:.5f}")

# ── Load PCA-Comb(n=29) model ──────────────────────────────────────────────────
print("Loading PCA-Comb(n=29) model ...")
pca_path = MODDIR / "pca_nc29.pkl"
pca_obj   = joblib.load(pca_path)
pca_model = pca_obj['pca']
tau_t2b   = pca_obj['tau_t2b']
tau_qb    = pca_obj['tau_qb']
pca_evar  = pca_model.explained_variance_  # eigenvalues
print(f"  PCA n={pca_obj['nc']} tau_t2b={tau_t2b:.3f} tau_qb={tau_qb:.3f}")

def compute_pca_scores(X_raw):
    """Per-sample combined T² (Hotelling) and Q (SPE) scores for PCA-Comb."""
    Xs = scaler.transform(X_raw).astype(np.float32)
    scores = pca_model.transform(Xs)              # (n, 29)
    t2     = (scores**2 / pca_evar).sum(axis=1)   # Hotelling T²
    recon  = pca_model.inverse_transform(scores)
    q      = ((Xs - recon)**2).sum(axis=1)         # SPE / Q
    return t2, q

# ── Load faulty test data ──────────────────────────────────────────────────────
print("Opening faulty test data cache ...")
faulty_cache = load_faulty_test_cache()
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
SHOWCASE_RUN = FAULT_TEST_RUNS[0]   # first FAULT_TEST_RUNS id -- illustrative run
                                     # must come from the test pool, not the
                                     # ablation_study.py validation pool
print(f"  {faulty_cache.describe()}")
print(f"  Showcase run id: {SHOWCASE_RUN} (from FAULT_TEST_RUNS)")

def get_run(fid, run_id=SHOWCASE_RUN):
    return faulty_cache.get_run(fid, run_id)

def compute_tcn_scores(X_raw):
    """Returns s_mse, s_normmaxpv (shape: n_samples)."""
    Xs = scaler.transform(X_raw).astype(np.float32)
    n = len(Xs)
    W = 20
    W_arr = make_run_windows(Xs, W)
    mse_w, pv_w = [], []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i:i+INFER_BS], DEVICE)
            recon, _ = model(batch)
            err2 = (batch - recon)**2
            mse_w.append(tensor_to_numpy(err2.mean(dim=(1,2))))
            pv_w.append(tensor_to_numpy(err2.mean(dim=2)))   # (B, 52)
    mse_w = np.concatenate(mse_w)
    pv_w  = np.concatenate(pv_w)                    # (N_windows, 52)
    normpv_w = (pv_w / pv_std).max(axis=1)
    # Assign window score to last sample in window
    s_mse = np.full(n, mse_w[0])
    s_np  = np.full(n, normpv_w[0])
    for i, t in enumerate(range(W-1, n)):
        s_mse[t] = mse_w[i]
        s_np[t]  = normpv_w[i]
    return s_mse, s_np

# ── Plot for F5, F10, F16, F19 (Section 5.5's four illustrative cases) ────────
FAULTS_TO_SHOW = [
    (5, SHOWCASE_RUN, "Fault 5 (Condenser CW temp, step): PCA 25% → NormMaxPV ~100%"),
    (10, SHOWCASE_RUN, "Fault 10 (C-feed temp, random): PCA 34% → NormMaxPV 95%"),
    (16, SHOWCASE_RUN, "Fault 16 (Unknown-16): PCA 19% → NormMaxPV 98%"),
    (19, SHOWCASE_RUN, "Fault 19 (Unknown-19): PCA 12% → NormMaxPV ~100%"),
]

def stats_str(arr, name):
    fdr = arr.mean() * 100
    first = np.where(arr > 0)[0]
    add = first[0]+1 if len(first) else 999
    return f"{name}: FDR={fdr:.0f}% ADD={add}"

_, axes = plt.subplots(len(FAULTS_TO_SHOW), 2, figsize=(paper_style.W2 + 4.5, 4.5 * len(FAULTS_TO_SHOW)))
if len(FAULTS_TO_SHOW) == 1:
    axes = [axes]

for row_idx, (fid, run_id, title) in enumerate(FAULTS_TO_SHOW):
    print(f"\nProcessing F{fid} run {run_id} ...")
    X = get_run(fid, run_id)
    if len(X) < 960:
        print(f"  WARNING: only {len(X)} samples, skipping")
        continue
    s_mse, s_normpv = compute_tcn_scores(X)
    pca_t2, pca_q   = compute_pca_scores(X)
    t_axis = np.arange(len(s_mse))

    # Alarm indicators
    alarm_np  = (s_normpv > tau_normmaxpv).astype(float)
    alarm_mse = (s_mse    > tau_mse).astype(float)
    alarm_pca = ((pca_t2 > tau_t2b) | (pca_q > tau_qb)).astype(float)

    # Left panel: normalised score vs threshold
    ax_l = axes[row_idx][0]
    ax_r = axes[row_idx][1]

    mse_norm  = s_mse    / tau_mse
    np_norm   = s_normpv / tau_normmaxpv
    pca_norm  = np.maximum(pca_t2 / tau_t2b, pca_q / tau_qb)  # combined norm

    ax_l.plot(t_axis, np_norm,  color='#D4380D', lw=1.8, label='NormMaxPV score (norm.)', zorder=4)
    ax_l.plot(t_axis, mse_norm, color='#4CAF50', lw=1.2, label='TCN-AE MSE score (norm.)', alpha=0.85, zorder=3)
    ax_l.plot(t_axis, pca_norm, color='#3A7EBD', lw=1.2, label='PCA-Comb score (norm.)', alpha=0.85,
              linestyle='--', zorder=2)
    ax_l.axhline(1.0, color='black', lw=1.5, linestyle=':', label='Alarm threshold')
    ax_l.axvline(FAULT_START, color='navy', lw=1.5, linestyle='-', alpha=0.5)
    ymax = max(np_norm.max(), mse_norm.max(), pca_norm.max()) * 1.05
    ax_l.fill_betweenx([0, ymax], FAULT_START, len(s_mse), alpha=0.06, color='red')
    ax_l.set_xlim(0, len(s_mse))
    ax_l.set_ylim(bottom=0)
    ax_l.set_xlabel('Sample index', fontsize=10)
    ax_l.set_ylabel('Score / threshold  (>1 = alarm)', fontsize=10)
    ax_l.set_title(f'F{fid}: Detection scores (normalised by CP threshold)', fontsize=9)
    ax_l.legend(fontsize=9, loc='upper left')
    ax_l.grid(alpha=0.3)
    ax_l.text(FAULT_START+5, ymax * 0.95, 'Fault\ninjected', fontsize=8, color='navy', va='top')

    # Right panel: binary alarm indicator
    offset_np  = 1.08
    offset_mse = 1.00
    offset_pca = 0.92
    ax_r.step(t_axis, alarm_np  * offset_np,  color='#D4380D', lw=1.8,
              label='NormMaxPV', where='post', zorder=4)
    ax_r.step(t_axis, alarm_mse * offset_mse, color='#4CAF50', lw=1.2,
              label='TCN-AE MSE', where='post', alpha=0.9, zorder=3)
    ax_r.step(t_axis, alarm_pca * offset_pca, color='#3A7EBD', lw=1.2,
              label='PCA-Comb', where='post', alpha=0.9, linestyle='--', zorder=2)
    ax_r.axvline(FAULT_START, color='navy', lw=1.5, linestyle='-', alpha=0.5)
    ax_r.fill_betweenx([0, 1.2], FAULT_START, len(s_mse), alpha=0.06, color='red')
    ax_r.set_xlim(0, len(s_mse))
    ax_r.set_ylim(-0.05, 1.25)
    ax_r.set_xlabel('Sample index', fontsize=10)
    ax_r.set_ylabel('Alarm (1=triggered)', fontsize=10)
    ax_r.set_title(f'F{fid}: Alarm status over time', fontsize=9)
    ax_r.legend(fontsize=9, loc='upper left')
    ax_r.grid(alpha=0.3)
    ax_r.text(FAULT_START+5, 1.18, 'Fault injected', fontsize=8, color='navy', va='top')

    # Compute per-run stats (single run); fault period starts at index FAULT_START-1 (0-indexed)
    post_np  = alarm_np [FAULT_START - 1:]
    post_mse = alarm_mse[FAULT_START - 1:]
    post_pca = alarm_pca[FAULT_START - 1:]
    info = "\n".join([stats_str(post_np,'NormMaxPV'),
                      stats_str(post_mse,'TCN MSE '),
                      stats_str(post_pca,'PCA-Comb')])
    ax_r.text(0.02, 0.02, info, transform=ax_r.transAxes,
              fontsize=8.5, va='bottom', family='monospace',
              bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.85))

    axes[row_idx][0].annotate(
        title, xy=(1.0, 1.05), xycoords='axes fraction',
        ha='center', va='bottom', fontsize=11, fontweight='bold',
        annotation_clip=False
    )

plt.suptitle('Detection Showcase: NormMaxPV vs MSE Nonconformity Scores\n'
             '(α=0.01; fault injection at sample 161)', fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig25_detection_showcase.png")
plt.close()
print("\nSaved fig25_detection_showcase.png")
