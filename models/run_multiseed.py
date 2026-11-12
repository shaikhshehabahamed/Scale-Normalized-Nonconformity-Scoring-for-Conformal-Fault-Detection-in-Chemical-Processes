"""
Multi-seed pipeline driver for the TCN-AE Q1 submission.

Runs the full train -> calibrate -> evaluate chain once per seed in
MAIN_SEEDS (skipping 42, whose artifacts were migrated from the original
single-run submission by migrate_seed42_artifacts.py), then the ablation
study once per seed in ABLATION_SEEDS (same skip rule).

Idempotent: each step is skipped if its output file already exists, so this
can be safely re-run after an interruption.

Usage: uv run python models/run_multiseed.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from tcnae_utils import (
    ABLATION_SEEDS,
    MAIN_SEEDS,
    MODDIR,
    OUTDIR,
    ablation_seed_results_dir,
    seed_model_dir,
    seed_results_dir,
)

MODELS_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run_step(script: str, seed: int, marker: Path, label: str) -> None:
    if marker.exists():
        print(f"  [skip] {label} (seed={seed}): {marker.name} already exists")
        return
    print(f"  [run ] {label} (seed={seed}) -> {script}")
    t0 = time.time()
    env = dict(os.environ, TCNAE_SEED=str(seed))
    result = subprocess.run(
        [PY, str(MODELS_DIR / script)],
        env=env, cwd=str(MODELS_DIR.parent),
        capture_output=True, text=True,
    )
    dt = time.time() - t0
    log_path = OUTDIR / "logs" / f"{script.replace('.py','')}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + "\n----STDERR----\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        print(f"  [FAIL] {label} (seed={seed}) exited {result.returncode} after {dt/60:.1f} min")
        print(f"         see {log_path}")
        raise RuntimeError(f"{script} failed for seed={seed} (see {log_path})")
    if not marker.exists():
        raise RuntimeError(
            f"{script} for seed={seed} exited 0 but expected output {marker} is missing"
        )
    print(f"  [ok  ] {label} (seed={seed}) in {dt/60:.1f} min -> {marker}")


def main() -> None:
    print("=" * 70)
    print(f"MULTI-SEED PIPELINE  |  MAIN_SEEDS={MAIN_SEEDS}  ABLATION_SEEDS={ABLATION_SEEDS}")
    print("=" * 70)

    for seed in MAIN_SEEDS:
        print(f"\n--- MAIN PIPELINE: seed={seed} ---")
        sdir = seed_model_dir(seed)
        rdir = seed_results_dir(seed)
        # marker = conformal_thresholds.pkl (last file train_tcnae.py writes),
        # not tcnae_best.pt (written mid-script, before calibration) -- using
        # the mid-script file as the marker would wrongly skip retraining on
        # resume if a prior run crashed during calibration.
        run_step("train_tcnae.py",       seed, sdir / "conformal_thresholds.pkl", "train TCN-AE (+norm/cal split)")
        run_step("dense_ae_baseline.py",  seed, rdir / "dense_ae_results.csv",  "train+eval DenseAE")
        run_step("sample_ae_baseline.py", seed, rdir / "sample_ae_results.csv", "train+eval SampleAE")
        run_step("conv_ae_baseline.py",   seed, rdir / "conv_ae_results.csv",  "train+eval ConvAE")
        run_step("evaluate_tcnae.py",     seed, rdir / "tcnae_results.csv",     "evaluate TCN-AE (MSE/MaxPV)")
        run_step("latent_mahal.py",       seed, sdir / "mahal_config.pkl",      "fit latent Mahalanobis")
        run_step("latent_mahal_eval.py",  seed, rdir / "tcnae_mahal_results.csv", "evaluate Mahalanobis/OR")
        run_step("pvbc_eval.py",          seed, rdir / "pvbc_results.csv",      "evaluate PVBC/NormMaxPV")
        run_step("overlap_robustness.py", seed, rdir / "overlap_robustness.csv", "point-2 overlap robustness")
        run_step("noise_robustness.py",   seed, rdir / "noise_robustness.csv",  "point-7 noise robustness")
        run_step("detection_delay_eval.py", seed, rdir / "detection_delay_summary.csv", "point-6a censored delay eval")

    for seed in ABLATION_SEEDS:
        print(f"\n--- ABLATION: seed={seed} ---")
        adir = ablation_seed_results_dir(seed)
        run_step("ablation_study.py", seed, adir / "ablation_calsize.csv", "ablation study (+ calsize sweep)")

    print("\n" + "=" * 70)
    print("MULTI-SEED PIPELINE COMPLETE")
    print("Next: uv run python models/aggregate_multiseed.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
