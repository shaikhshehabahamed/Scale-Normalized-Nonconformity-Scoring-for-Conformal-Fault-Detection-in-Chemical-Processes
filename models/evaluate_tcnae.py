"""
Evaluate TCN-AE + Conformal Prediction on all 20 TEP faults.
Produces per-seed results/seeds/seed_<TCNAE_SEED>/tcnae_results.csv only;
comparison figures (fig08-11) and all_results.csv are built once from the
aggregated multi-seed data by gen_main_comparison_figs.py (run after
aggregate_multiseed.py). See models/run_multiseed.py.
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
    OUTDIR,
    configured_batch_size,
    configured_seed,
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

os.makedirs(OUTDIR, exist_ok=True)

FAULT_START = 161
ALPHA       = 0.01
WINDOW      = 20
INFER_BS    = configured_batch_size("TCNAE_INFER_BS", 2048)
DEVICE      = select_device()

FAULT_NAMES = {
    1:"A/C feed ratio", 2:"B composition", 3:"D feed temp step",
    4:"Reactor CW temp", 5:"Condenser CW temp", 6:"A feed loss",
    7:"C pressure loss", 8:"A,B,C feed comp", 9:"D feed temp random",
    10:"C feed temp random", 11:"Reactor CW random", 12:"Condenser CW random",
    13:"Reaction kinetics", 14:"Reactor CW valve", 15:"Condenser CW valve",
    16:"Unknown-16", 17:"Unknown-17", 18:"Unknown-18",
    19:"Unknown-19", 20:"Unknown-20"
}
HARD_FAULTS = {3, 9, 15, 16, 17, 18, 19, 20}

# ── Load trained model and config ────────────────────────────────────────────
print("="*65)
print("1. LOADING MODEL AND CONFIG")
print("="*65)
print_device(DEVICE)
print(f"Seed = {SEED}")
scaler    = joblib.load(MODDIR / "scaler.pkl")
cp_thresh = joblib.load(SDIR / "conformal_thresholds.pkl")

tau_mse   = cp_thresh['tau_mse']
tau_maxpv = cp_thresh['tau_maxpv']
print(f"tau_MSE   = {tau_mse:.6f}")
print(f"tau_MaxPV = {tau_maxpv:.6f}")
print(f"Window    = {WINDOW}, Alpha = {ALPHA}")

model = TCNAE(n_vars=52, window=WINDOW, n_filters=64, latent_dim=24)
model.load_state_dict(torch.load(SDIR / "tcnae_best.pt", map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
print("Model loaded.")

# Also load PCA baseline results for comparison (deterministic, shared across seeds)
pca_df = pd.read_csv(OUTDIR / "pca_baseline_results.csv")
# Best PCA: use n=7 (PCA-Q) as established
pca_best = pca_df[pca_df['method'] == 'PCA-Q(n=7)'].set_index('fault_id')
print(f"PCA-Q(n=7) loaded: {len(pca_best)} faults")

# ── Helper: score a single 960-sample run ────────────────────────────────────
def score_run(X_run_raw: np.ndarray):
    """
    X_run_raw: (960, 52) raw
    Returns per-sample MSE and MaxPV scores using causal windows.
    Samples 1..W-1 get score from first full window.
    """
    Xs = scaler.transform(X_run_raw).astype(np.float32)  # (960, 52)
    n = len(Xs)
    W_arr = make_run_windows(Xs, WINDOW)                  # (N_w, 52, W)

    mse_scores  = []
    maxpv_scores = []
    with torch.no_grad():
        for i in range(0, len(W_arr), INFER_BS):
            batch = move_to_device(W_arr[i : i + INFER_BS], DEVICE)
            recon, _ = model(batch)
            mse  = ((batch - recon)**2).mean(dim=(1,2))    # (B,)
            mpv  = ((batch - recon)**2).mean(dim=2).max(dim=1).values  # (B,)
            mse_scores.append(tensor_to_numpy(mse))
            maxpv_scores.append(tensor_to_numpy(mpv))

    mse_w   = np.concatenate(mse_scores)   # (n-W+1,)
    maxpv_w = np.concatenate(maxpv_scores)

    # Assign window score to LAST sample of each window
    # Window i -> samples [i, i+W-1] -> last sample index = i+W-1
    # Create per-sample score arrays (length 960)
    # Sample indices W-1 .. n-1 get a score; samples 0..W-2 use first score
    sample_mse   = np.full(n, mse_w[0])
    sample_maxpv = np.full(n, maxpv_w[0])
    for i, w_idx in enumerate(range(WINDOW - 1, n)):
        sample_mse[w_idx]   = mse_w[i]
        sample_maxpv[w_idx] = maxpv_w[i]

    return sample_mse, sample_maxpv   # each (960,)

def eval_run_metrics(scores, threshold):
    n_pre  = FAULT_START - 1   # 160 normal samples
    n_post = len(scores) - n_pre
    alarms = (scores > threshold).astype(int)
    FAR = float(alarms[:n_pre].mean())
    FDR = float(alarms[n_pre:].mean())
    first = np.where(alarms[n_pre:])[0]
    ADD = int(first[0]) + 1 if len(first) > 0 else n_post + 1
    return FAR, FDR, ADD

# ── Load faulty test data ─────────────────────────────────────────────────────
print("\n2. OPENING FAULTY TESTING DATA CACHE...")
faulty_cache = load_faulty_test_cache()
fault_ids = faulty_cache.fault_ids
FAULT_VAL_RUNS, FAULT_TEST_RUNS = split_fault_test_runs(faulty_cache.run_ids)
print(f"Fault IDs: {fault_ids}  |  {faulty_cache.describe()}")
print(f"Using FAULT_TEST_RUNS: {len(FAULT_TEST_RUNS)} runs/fault "
      f"(runs {FAULT_TEST_RUNS[0]}-{FAULT_TEST_RUNS[-1]}; the first "
      f"{len(FAULT_VAL_RUNS)} are reserved for ablation_study.py tuning)")

# ── Main evaluation loop ──────────────────────────────────────────────────────
def _agg(lst):
    return np.array(lst).mean(axis=0).tolist()

print("\n3. EVALUATING TCN-AE + CP ON ALL FAULTS")
print("="*65)

records = []
for fid in fault_ids:
    mse_results   = []
    maxpv_results = []

    for _, X_run in faulty_cache.iter_fault_runs(fid, run_ids=FAULT_TEST_RUNS):
        if len(X_run) < 960:
            continue

        s_mse, s_mpv = score_run(X_run)
        mse_results.append(eval_run_metrics(s_mse, tau_mse))
        maxpv_results.append(eval_run_metrics(s_mpv, tau_maxpv))

    m1 = _agg(mse_results)
    m2 = _agg(maxpv_results)
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid],
                    'method': 'TCNAE-MSE-CP',
                    'FAR': round(m1[0],5), 'FDR': round(m1[1],5),
                    'ADD': round(m1[2],2)})
    records.append({'fault_id': fid, 'fault_name': FAULT_NAMES[fid],
                    'method': 'TCNAE-MaxPV-CP',
                    'FAR': round(m2[0],5), 'FDR': round(m2[1],5),
                    'ADD': round(m2[2],2)})

    print(f"  F{fid:2d}: MSE FDR={m1[1]:.4f} FAR={m1[0]:.4f} ADD={m1[2]:.1f} | "
          f"MaxPV FDR={m2[1]:.4f} FAR={m2[0]:.4f}")

del faulty_cache
gc.collect()

# ── Save results ──────────────────────────────────────────────────────────────
res_df = pd.DataFrame(records)
res_df.to_csv(RDIR / "tcnae_results.csv", index=False)
print(f"\nSaved {RDIR / 'tcnae_results.csv'}")

# ── Comparison summary (this seed only; aggregate stats come later) ──────────
print("\n" + "="*65)
print(f"SUMMARY (seed={SEED}): TCNAE-MSE-CP vs PCA-Q(n=7)")
print("="*65)
tcn_mse = res_df[res_df['method'] == 'TCNAE-MSE-CP'].set_index('fault_id')

print(f"{'FaultID':>7}  {'Fault Name':25}  {'PCA-Q FDR':>9}  "
      f"{'TCN FDR':>8}  {'Delta':>7}  {'TCN ADD':>8}")
print("-"*75)
deltas_all  = []
deltas_hard = []
for fid in fault_ids:
    pca_fdr = pca_best.loc[fid, 'FDR'] if fid in pca_best.index else 0.0
    tcn_fdr = tcn_mse.loc[fid, 'FDR'] if fid in tcn_mse.index else 0.0
    tcn_add = tcn_mse.loc[fid, 'ADD'] if fid in tcn_mse.index else 0.0
    delta   = tcn_fdr - pca_fdr
    mark    = " ***" if fid in HARD_FAULTS else ""
    print(f"  {fid:3d}    {FAULT_NAMES[fid]:25}  {pca_fdr*100:8.2f}%  "
          f"{tcn_fdr*100:7.2f}%  {delta*100:+7.2f}%  {tcn_add:7.1f}{mark}")
    deltas_all.append(delta)
    if fid in HARD_FAULTS:
        deltas_hard.append(delta)

print("-"*75)
print(f"  Mean FDR improvement (all):  {np.mean(deltas_all)*100:+.2f}%")
print(f"  Mean FDR improvement (hard): {np.mean(deltas_hard)*100:+.2f}%")
print(f"\n  PCA-Q avg FDR: {pca_best['FDR'].mean()*100:.2f}%")
print(f"  TCN-AE avg FDR: {tcn_mse['FDR'].mean()*100:.2f}%")
print(f"  TCN-AE avg FAR: {tcn_mse['FAR'].mean()*100:.4f}%")
print("\n(Comparison figures are generated once from the aggregated multi-seed")
print(" results by gen_main_comparison_figs.py, not per seed.)")

print("\n" + "="*65)
print(f"EVALUATION COMPLETE (seed={SEED})")
print("="*65)
