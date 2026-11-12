"""
Latent-space Mahalanobis distance detector for TCN-AE.
Combines: (a) reconstruction error (MSE), (b) Mahalanobis distance in latent space.
OR-rule with Bonferroni correction: alarm if MSE > tau_MSE/2 OR Mahal > tau_Mahal/2.

Mean and covariance are fit on norm_z.npy (the norm-estimation set), never on
the cal set used for the conformal quantile -- otherwise each calibration
Mahalanobis score would be computed from a Gaussian fit that includes that
very point, while a fresh test score would not (round-2 peer review, point
1). The conformal threshold itself comes only from cal_z.npy. Both files are
produced by train_tcnae.py, so this script does no data loading or model
forward passes of its own.

Requires: models/seeds/seed_<S>/{norm_z.npy, cal_z.npy, cal_scores.npy,
conformal_thresholds.pkl} (all written by train_tcnae.py).
Multi-seed: set TCNAE_SEED=<int>; reads/writes models/seeds/seed_<TCNAE_SEED>/.
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    configured_seed,
    cp_threshold,
    seed_everything,
    seed_model_dir,
)
SEED = configured_seed()
seed_everything(SEED)
SDIR = seed_model_dir(SEED)

ALPHA = 0.01

print("="*65)
print("LATENT MAHALANOBIS DISTANCE DETECTOR")
print("="*65)

# ── Step 1: Fit Gaussian on the norm-estimation set's latent codes ───────────
print("\n1. FITTING LATENT GAUSSIAN (norm-estimation set)")
norm_z = np.load(SDIR / "norm_z.npy")     # (N_norm, latent_dim)
latent_dim = norm_z.shape[1]
print(f"Norm latent codes: {norm_z.shape}")

mu_z = norm_z.mean(axis=0)
cov_z = np.cov(norm_z.T) + np.eye(latent_dim) * 1e-6
cov_inv = np.linalg.inv(cov_z)

def mahal(Z, mu, cov_inv):
    diffs = Z - mu               # (N, latent_dim)
    return np.sqrt((diffs @ cov_inv * diffs).sum(axis=1))  # (N,)

# ── Step 2: Calibrate the conformal threshold on the (disjoint) cal set ──────
print("\n2. CALIBRATING ON THE CAL SET (disjoint from the norm-estimation set)")
cal_z = np.load(SDIR / "cal_z.npy")
cal_mse = np.load(SDIR / "cal_scores.npy")
cal_mahal = mahal(cal_z, mu_z, cov_inv)
print(f"Cal Mahal: n={len(cal_mahal)} mean={cal_mahal.mean():.4f}  std={cal_mahal.std():.4f}  "
      f"p99={np.percentile(cal_mahal,99):.4f}")

tau_mahal = cp_threshold(cal_mahal, ALPHA)
tau_mse_b = cp_threshold(cal_mse, ALPHA / 2)     # Bonferroni for OR-rule
tau_mahal_b = cp_threshold(cal_mahal, ALPHA / 2)

print(f"tau_MSE   (solo, alpha={ALPHA}) loaded from conformal_thresholds.pkl")
print(f"tau_Mahal (solo, alpha={ALPHA})         = {tau_mahal:.4f}")
print(f"tau_MSE   (Bonferroni, alpha/2={ALPHA/2}) = {tau_mse_b:.6f}")
print(f"tau_Mahal (Bonferroni, alpha/2={ALPHA/2}) = {tau_mahal_b:.4f}")

# Verify calibration FAR
cp_thresh = joblib.load(SDIR / "conformal_thresholds.pkl")
tau_mse_solo = cp_thresh['tau_mse']
far_mse = (cal_mse > tau_mse_solo).mean()
far_mahal = (cal_mahal > tau_mahal).mean()
far_or = ((cal_mse > tau_mse_b) | (cal_mahal > tau_mahal_b)).mean()
print("\nVerification on calibration:")
print(f"  FAR MSE:   {far_mse:.4f}")
print(f"  FAR Mahal: {far_mahal:.4f}")
print(f"  FAR OR:    {far_or:.4f}  (target={ALPHA})")

joblib.dump({'mu_z': mu_z, 'cov_inv': cov_inv,
             'tau_mahal': tau_mahal, 'tau_mahal_b': tau_mahal_b,
             'tau_mse_b': tau_mse_b},
            SDIR / "mahal_config.pkl")
print(f"Saved {SDIR / 'mahal_config.pkl'}")

print("\n" + "="*65)
print("MAHALANOBIS CALIBRATION COMPLETE")
print("Now run latent_mahal_eval.py for full 5-method evaluation.")
print("="*65)
