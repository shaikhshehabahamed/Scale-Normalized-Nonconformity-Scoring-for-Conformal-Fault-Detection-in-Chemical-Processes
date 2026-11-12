"""Shared runtime utilities for the TCN-AE experiments."""
from __future__ import annotations

import ctypes
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Tennessee Eastman Process Simulation Dataset (Rieth 2017)"
MODDIR = ROOT / "models"
OUTDIR = ROOT / "results"
CACHE_DIR = ARCHIVE / "_cache"
FAULTY_TEST_RDATA = ARCHIVE / "TEP_Faulty_Testing.RData"
FAULTY_TEST_CACHE_META = CACHE_DIR / "faulty_testing_cache.json"
FAULTY_TEST_CACHE_ARRAY = CACHE_DIR / "faulty_testing_features.npy"
CACHE_VERSION = 1

XMEAS = [f"xmeas_{i}" for i in range(1, 42)]
XMV = [f"xmv_{i}" for i in range(1, 12)]
FEAT = XMEAS + XMV
N_VARS = len(FEAT)
DEFAULT_SEED = 42

# Multi-seed evaluation protocol (Q1 statistical-robustness requirement).
# Seed 42 is the original single-run seed; its artifacts are reused as one
# ensemble member (see migrate_seed42_artifacts.py) rather than retrained.
MAIN_SEEDS = [42, 0, 1, 2, 3]
# Ablation configs are trained from scratch per config (13 configs), so the
# seed count is reduced to keep total compute tractable; still >1 for a
# genuine variance estimate.
ABLATION_SEEDS = [42, 0, 1]

# Fault-free run split (round-2 peer review, point 1): normalization
# statistics (std_pv for NormMaxPV; mu_z/cov_inv for Mahalanobis) must be
# estimated on data disjoint from the conformal-calibration set. Estimating
# them on the same windows used for the conformal quantile means each
# calibration score is computed with statistics that include that very
# observation, while a fresh test score is not -- breaking the exchangeability
# the split-conformal FAR bound (Eq. 12) relies on. norm_runs supplies the
# nuisance parameters; cal_runs supplies every conformal quantile (MSE,
# MaxPV, NormMaxPV, Mahalanobis, PVBC).
TRAIN_RUNS = 400
NORM_RUNS = 50
CAL_RUNS = 50

# Faulty-test run split (round-2 peer review, point 5): hyperparameter and
# alpha selection (ablation_study.py) must use a run pool disjoint from the
# runs used for every final headline result, or model selection is tuned on
# the same data used to report performance.
FAULT_VAL_RUN_COUNT = 100

SEED_MODEL_DIR = MODDIR / "seeds"
SEED_RESULTS_DIR = OUTDIR / "seeds"
ABLATION_SEED_RESULTS_DIR = OUTDIR / "ablation_seeds"
AGG_DIR = OUTDIR / "agg"


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_faultfree_runs(all_runs: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Split the 500 fault-free simulation runs into train / normalization-
    estimation / conformal-calibration sets (see TRAIN_RUNS/NORM_RUNS/CAL_RUNS).
    norm and cal are disjoint by construction (contiguous, non-overlapping
    slices), which is the property that fixes the round-2 review's point 1."""
    n = len(all_runs)
    expected = TRAIN_RUNS + NORM_RUNS + CAL_RUNS
    if n != expected:
        raise ValueError(f"Expected {expected} fault-free runs (train+norm+cal), got {n}")
    train_runs = all_runs[:TRAIN_RUNS]
    norm_runs = all_runs[TRAIN_RUNS:TRAIN_RUNS + NORM_RUNS]
    cal_runs = all_runs[TRAIN_RUNS + NORM_RUNS:]
    return train_runs, norm_runs, cal_runs


def split_fault_test_runs(run_ids: list[int]) -> tuple[list[int], list[int]]:
    """Split one fault type's simulation runs into a validation pool (used
    only for hyperparameter/alpha selection in ablation_study.py) and a test
    pool (used for every final headline result), so tuning and final
    reporting never draw on the same runs (round-2 review, point 5)."""
    val_runs = list(run_ids[:FAULT_VAL_RUN_COUNT])
    test_runs = list(run_ids[FAULT_VAL_RUN_COUNT:])
    return val_runs, test_runs


def configured_seed(default: int = DEFAULT_SEED) -> int:
    """Read the random seed from TCNAE_SEED, falling back to `default`.

    Lets the multi-seed driver run each script once per seed via subprocess
    env vars while keeping `uv run python models/X.py` single-seed-friendly.
    """
    raw = os.getenv("TCNAE_SEED", "").strip()
    return int(raw) if raw else default


def seed_model_dir(seed: int) -> Path:
    d = SEED_MODEL_DIR / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def seed_results_dir(seed: int) -> Path:
    d = SEED_RESULTS_DIR / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ablation_seed_results_dir(seed: int) -> Path:
    d = ABLATION_SEED_RESULTS_DIR / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def select_device() -> torch.device:
    """Select the compute device.

    Default keeps the original CUDA path when available. Set TCNAE_DEVICE=cpu
    for low-VRAM machines, or TCNAE_DEVICE=cuda to require CUDA explicitly.
    """
    requested = os.getenv("TCNAE_DEVICE", "auto").strip().lower()
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TCNAE_DEVICE=cuda was requested, but CUDA is not available to PyTorch."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in {"", "auto"}:
        raise ValueError("TCNAE_DEVICE must be one of: auto, cuda, cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device(device: torch.device) -> None:
    if device.type != "cuda":
        print(f"Using device: {device.type}")
        return
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024 ** 3)
    print(f"Using device: cuda ({props.name}, {total_gb:.1f} GB VRAM)")


def configured_batch_size(env_name: str, default: int) -> int:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer, got {value}")
    return value


def move_to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=(device.type == "cuda"))


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def cp_threshold(scores: np.ndarray, alpha: float) -> float:
    """Conformal prediction threshold: smallest tau s.t. empirical FAR <= alpha."""
    n = len(scores)
    q = min(np.ceil((1 - alpha) * (n + 1)) / n, 1.0)
    return float(np.quantile(scores, q))


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _source_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _total_memory_gb() -> float | None:
    """Best-effort total physical memory, avoiding an extra psutil dependency."""
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullTotalPhys / (1024 ** 3)


def _metadata_is_current(meta: dict) -> bool:
    if meta.get("cache_version") != CACHE_VERSION:
        return False
    if meta.get("source") != _source_signature(FAULTY_TEST_RDATA):
        return False
    if not FAULTY_TEST_CACHE_ARRAY.exists():
        return False
    return True


def _read_faulty_cache_meta() -> dict | None:
    if not FAULTY_TEST_CACHE_META.exists():
        return None
    with FAULTY_TEST_CACHE_META.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_faulty_testing_cache(force: bool = False) -> dict:
    """Build a disk-backed cache for the large faulty testing RData.

    The original 837 MB RData expands to multiple GB in pandas. This one-time
    conversion writes a float32 memmap ordered as
    (fault, simulationRun, sample, variable), letting evaluation scripts process
    one run at a time without keeping the full pandas frame in RAM.
    """
    if not FAULTY_TEST_RDATA.exists():
        raise FileNotFoundError(f"Missing dataset file: {FAULTY_TEST_RDATA}")

    existing = _read_faulty_cache_meta()
    if existing and _metadata_is_current(existing) and not force:
        return existing

    total_ram = _total_memory_gb()
    if total_ram is not None and total_ram < 12 and not _env_flag("TCNAE_ALLOW_LOW_RAM_CACHE"):
        raise RuntimeError(
            "Building the faulty-test cache requires loading the source RData once. "
            f"This machine reports only {total_ram:.1f} GB RAM. Build the cache on a "
            "larger machine, or set TCNAE_ALLOW_LOW_RAM_CACHE=1 if you accept the risk."
        )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(CACHE_DIR).free < 3 * (1024 ** 3):
        raise RuntimeError(
            f"Not enough free disk space for cache in {CACHE_DIR}; at least 3 GB is required."
        )

    print("Building one-time faulty-test memmap cache.")
    print(f"  Source: {FAULTY_TEST_RDATA.name}")
    print(f"  Cache : {FAULTY_TEST_CACHE_ARRAY}")
    print("  This avoids reloading the large RData file in each evaluation script.")

    import pyreadr

    rdata = pyreadr.read_r(str(FAULTY_TEST_RDATA), use_objects=["faulty_testing"])
    if "faulty_testing" in rdata:
        df = rdata["faulty_testing"]
    else:
        df = list(rdata.values())[0]
    del rdata

    required = ["faultNumber", "simulationRun", "sample"] + FEAT
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faulty testing data is missing required columns: {missing}")

    fault_col = np.asarray(df["faultNumber"].to_numpy(), dtype=np.int16)
    run_col = np.asarray(df["simulationRun"].to_numpy(), dtype=np.int16)
    sample_col = np.asarray(df["sample"].to_numpy(), dtype=np.int16)
    if np.any(fault_col <= 0):
        raise ValueError("Faulty testing cache expects only faultNumber > 0 rows.")

    fault_ids = [int(x) for x in np.unique(fault_col)]
    run_ids = [int(x) for x in np.unique(run_col[fault_col == fault_ids[0]])]
    sample_ids = [int(x) for x in np.unique(sample_col[(fault_col == fault_ids[0]) & (run_col == run_ids[0])])]
    n_faults = len(fault_ids)
    n_runs = len(run_ids)
    n_samples = len(sample_ids)
    expected_rows = n_faults * n_runs * n_samples
    if len(df) != expected_rows:
        raise ValueError(
            f"Unexpected faulty-test shape: {len(df):,} rows, expected "
            f"{expected_rows:,} = {n_faults} faults x {n_runs} runs x {n_samples} samples."
        )

    for fid in fault_ids:
        runs = [int(x) for x in np.unique(run_col[fault_col == fid])]
        if runs != run_ids:
            raise ValueError(f"Fault {fid} has non-uniform simulationRun values.")

    base_shape = (n_faults, n_runs, n_samples)
    full_shape = base_shape + (N_VARS,)
    tmp_array = FAULTY_TEST_CACHE_ARRAY.with_suffix(".tmp.npy")
    tmp_meta = FAULTY_TEST_CACHE_META.with_suffix(".tmp.json")
    for path in (tmp_array, tmp_meta):
        if path.exists():
            path.unlink()

    expected_faults = np.repeat(np.asarray(fault_ids, dtype=np.int16), n_runs * n_samples)
    expected_runs = np.tile(np.repeat(np.asarray(run_ids, dtype=np.int16), n_samples), n_faults)
    expected_samples = np.tile(np.asarray(sample_ids, dtype=np.int16), n_faults * n_runs)
    canonical = (
        np.array_equal(fault_col, expected_faults)
        and np.array_equal(run_col, expected_runs)
        and np.array_equal(sample_col, expected_samples)
    )
    del expected_faults, expected_runs, expected_samples
    gc.collect()

    features = np.lib.format.open_memmap(tmp_array, mode="w+", dtype=np.float32, shape=full_shape)
    if canonical:
        del fault_col, run_col, sample_col
        gc.collect()
        for j, feat in enumerate(FEAT):
            col = df[feat].to_numpy(dtype=np.float32, copy=False)
            features[..., j] = col.reshape(base_shape)
            if (j + 1) % 10 == 0 or j + 1 == N_VARS:
                print(f"  Cached {j + 1:2d}/{N_VARS} variables")
    else:
        print("  Source rows are not in canonical order; using indexed cache build.")
        feature_cols = [df[feat].to_numpy(dtype=np.float32, copy=False) for feat in FEAT]
        for fi, fid in enumerate(fault_ids):
            fault_mask = fault_col == fid
            for ri, run_id in enumerate(run_ids):
                idx = np.flatnonzero(fault_mask & (run_col == run_id))
                if len(idx) != n_samples:
                    raise ValueError(f"Fault {fid}, run {run_id} has {len(idx)} samples.")
                idx = idx[np.argsort(sample_col[idx])]
                for j, col in enumerate(feature_cols):
                    features[fi, ri, :, j] = col[idx]
            print(f"  Cached fault {fid:2d}/{fault_ids[-1]}")

    features.flush()
    del features, df
    gc.collect()

    metadata = {
        "cache_version": CACHE_VERSION,
        "source": _source_signature(FAULTY_TEST_RDATA),
        "array_file": FAULTY_TEST_CACHE_ARRAY.name,
        "dtype": "float32",
        "shape": list(full_shape),
        "fault_ids": fault_ids,
        "run_ids": run_ids,
        "sample_ids": sample_ids,
        "feature_names": FEAT,
    }
    with tmp_meta.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if FAULTY_TEST_CACHE_ARRAY.exists():
        FAULTY_TEST_CACHE_ARRAY.unlink()
    if FAULTY_TEST_CACHE_META.exists():
        FAULTY_TEST_CACHE_META.unlink()
    os.replace(tmp_array, FAULTY_TEST_CACHE_ARRAY)
    os.replace(tmp_meta, FAULTY_TEST_CACHE_META)
    print("Faulty-test cache ready.")
    return metadata


class FaultyTestingCache:
    """Read-only run-level view over the faulty testing memmap."""

    def __init__(self, metadata: dict):
        self.metadata = metadata
        self.path = CACHE_DIR / metadata["array_file"]
        self.features = np.load(self.path, mmap_mode="r")
        expected_shape = tuple(metadata["shape"])
        if self.features.shape != expected_shape:
            raise ValueError(f"Cache shape mismatch: {self.features.shape} != {expected_shape}")
        self.fault_ids = [int(x) for x in metadata["fault_ids"]]
        self.run_ids = [int(x) for x in metadata["run_ids"]]
        self.sample_ids = [int(x) for x in metadata["sample_ids"]]
        self._fault_index = {fid: i for i, fid in enumerate(self.fault_ids)}
        self._run_index = {run: i for i, run in enumerate(self.run_ids)}

    def describe(self) -> str:
        return (
            f"{len(self.fault_ids)} faults x {len(self.run_ids)} runs x "
            f"{len(self.sample_ids)} samples, float32 memmap"
        )

    def get_run(self, fault_id: int, run_id: int) -> np.ndarray:
        return self.features[self._fault_index[int(fault_id)], self._run_index[int(run_id)]]

    def iter_fault_runs(
        self,
        fault_id: int,
        max_runs: int | None = None,
        run_ids: list[int] | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate (run_id, x_run) for one fault.

        `run_ids`, an explicit subset (e.g. the disjoint validation/test
        pools from split_fault_test_runs), takes precedence over `max_runs`
        (legacy: first N runs by id) when both are given; with neither, all
        runs are used.
        """
        if run_ids is not None:
            runs = [r for r in run_ids if r in self._run_index]
        elif max_runs is not None:
            runs = self.run_ids[:max_runs]
        else:
            runs = self.run_ids
        for run_id in runs:
            yield run_id, self.get_run(fault_id, run_id)

    def iter_runs(
        self,
        fault_ids: list[int] | None = None,
        max_runs: int | None = None,
        run_ids: list[int] | None = None,
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        selected_faults = self.fault_ids if fault_ids is None else fault_ids
        for fault_id in selected_faults:
            for run_id, x_run in self.iter_fault_runs(fault_id, max_runs=max_runs, run_ids=run_ids):
                yield fault_id, run_id, x_run


def load_faulty_test_cache(force_rebuild: bool | None = None) -> FaultyTestingCache:
    """Return a run-level memmap cache for TEP_Faulty_Testing.RData."""
    if force_rebuild is None:
        force_rebuild = _env_flag("TCNAE_REBUILD_FAULTY_CACHE")
    metadata = _read_faulty_cache_meta()
    if metadata is None or force_rebuild or not _metadata_is_current(metadata):
        metadata = build_faulty_testing_cache(force=force_rebuild)
    return FaultyTestingCache(metadata)


class RunWindowDataset(Dataset):
    """Lazy within-run sliding windows over stacked run data.

    Materializing all training windows for 400 TEP runs at W=20 is about
    800 MB before Python/NumPy overhead. This dataset
    keeps only the standardized flat array and creates each window on demand.
    """

    def __init__(self, x_flat: np.ndarray, n_runs: int, window: int, flatten: bool = False):
        self.x_flat = np.asarray(x_flat, dtype=np.float32)
        self.n_runs = int(n_runs)
        self.window = int(window)
        self.flatten = bool(flatten)

        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if len(self.x_flat) % self.n_runs != 0:
            raise ValueError(
                f"x_flat length {len(self.x_flat)} is not divisible by n_runs={self.n_runs}"
            )

        self.samples_per_run = len(self.x_flat) // self.n_runs
        self.windows_per_run = self.samples_per_run - self.window + 1
        if self.windows_per_run <= 0:
            raise ValueError(
                f"window={self.window} is too large for {self.samples_per_run} samples/run"
            )

    def __len__(self) -> int:
        return self.n_runs * self.windows_per_run

    def __getitem__(self, index: int):
        if hasattr(index, "item"):
            index = int(index.item())
        run_idx, offset = divmod(int(index), self.windows_per_run)
        start = run_idx * self.samples_per_run + offset
        window = self.x_flat[start : start + self.window].T
        if self.flatten:
            window = window.reshape(-1)
        return (torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)),)


def make_run_windows(x_run: np.ndarray, window: int, flatten: bool = False) -> torch.Tensor:
    """Create windows for one run; small enough to materialize safely."""
    x_run = np.asarray(x_run, dtype=np.float32)
    if len(x_run) < window:
        raise ValueError(f"run has {len(x_run)} samples, shorter than window={window}")
    wins = np.lib.stride_tricks.sliding_window_view(x_run, window_shape=window, axis=0)
    if flatten:
        wins = wins.reshape(wins.shape[0], -1)
    return torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))


def make_run_windows_strided(x_run: np.ndarray, window: int, stride: int) -> torch.Tensor:
    """Like make_run_windows, but with a configurable stride between window
    start positions. stride=1 reproduces the default overlapping windows;
    stride=window gives non-overlapping windows. Used for the round-2 review's
    point-2 overlapping-window-dependence robustness check (see
    overlap_robustness.py), which recomputes conformal calibration at
    stride=window as a more conservative alternative to the main stride=1
    calibration."""
    x_run = np.asarray(x_run, dtype=np.float32)
    if len(x_run) < window:
        raise ValueError(f"run has {len(x_run)} samples, shorter than window={window}")
    wins = np.lib.stride_tricks.sliding_window_view(x_run, window_shape=window, axis=0)[::stride]
    return torch.from_numpy(np.ascontiguousarray(wins, dtype=np.float32))
