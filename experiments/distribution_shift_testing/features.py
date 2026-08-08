"""Load the ENA split and turn each back-trajectory into a summary feature vector.

The raw covariate is a (241, 22) back-trajectory per sample -- too high-dimensional
(and too autocorrelated across timesteps) to feed a two-sample test directly. We
collapse each trajectory to a fixed set of per-channel summary statistics so the
distribution-shift tests operate on interpretable, moderate-dimensional vectors.

Every helper here is shared by both the C2ST and the kNN overlap test so they see
exactly the same features, split, and (optional) decorrelation subsample.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    load_ena_supervised_dataset,
    select_real_split,
)
from experiments.pipeline import set_reproducible_seeds

# Per-channel statistics computed over the 241-step trajectory. Each maps a
# (n, 241, 22) array to (n, 22); stacking all of them gives (n, 22 * len(STATS)).
_STATS = ("mean", "std", "min", "max", "t0", "tend")


def summarize_trajectories(x: np.ndarray) -> np.ndarray:
    """(n, seq_len, n_feat) back-trajectories -> (n, n_feat * len(_STATS)) summaries.

    Column order matches ``summary_feature_names``: all channels for stat 0, then
    all channels for stat 1, ... (i.e. grouped by statistic).
    """
    x = np.asarray(x, dtype=np.float64)
    parts = [
        x.mean(axis=1),
        x.std(axis=1),
        x.min(axis=1),
        x.max(axis=1),
        x[:, 0, :],
        x[:, -1, :],
    ]
    return np.concatenate(parts, axis=1).astype(np.float64)


def summary_feature_names(channel_names: tuple[str, ...]) -> list[str]:
    """Names aligned with ``summarize_trajectories`` columns: ``<channel>::<stat>``."""
    return [f"{ch}::{stat}" for stat in _STATS for ch in channel_names]


def load_split(
    *,
    mat_path: str = DEFAULT_MAT_PATH,
    target: str = "ccn",
    log_ccn: bool = True,
    split: str = "paper",
    seq_stride: int = 1,
    max_samples: int | None = None,
    train_size: int | None = None,
    test_size: int | None = None,
    seed: int = 2026,
):
    """Load the dataset and the train/test split indices.

    Uses the SAME loader call and split seed convention as the t-learner
    (``seed`` for the data, ``seed + 17`` for the split) so "train" and "test"
    here are byte-for-byte the same rows the models were trained/evaluated on.
    """
    set_reproducible_seeds(seed)
    dataset = load_ena_supervised_dataset(
        mat_path=mat_path,
        target=target,
        max_samples=max_samples,
        seq_stride=seq_stride,
        seed=seed,
        log_ccn=log_ccn,
    )
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=split,
        train_size=train_size,
        test_size=test_size,
        seed=seed + 17,
    )
    return dataset, train_idx, test_idx


def decorrelate(idx: np.ndarray, stride: int) -> np.ndarray:
    """Thin autocorrelated rows: keep every ``stride``-th index (in time order).

    Adjacent hours share ~99.7%-identical back-trajectories, so the ~10^4 rows are
    really ~10^2 independent episodes. Subsampling before a two-sample test stops
    pseudo-replication from inflating both the classifier AUC and its confidence.
    ``idx`` is assumed sorted (``select_real_split`` returns sorted indices, which
    are in acquisition-time order).
    """
    if stride <= 1:
        return np.asarray(idx)
    return np.asarray(idx)[::stride]


def standardize_by_train(train: np.ndarray, test: np.ndarray):
    """Z-score both arrays using TRAIN column moments (test never leaks in)."""
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return ((train - mean) / std, (test - mean) / std)
