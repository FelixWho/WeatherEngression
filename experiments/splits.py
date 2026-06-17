"""Train/test split utilities for synthetic weather experiments."""

from __future__ import annotations

import numpy as np

from .constants import SPLIT_MODES


def phi_support(phi: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """Return the min/max training support in scalar ``phi(X)`` space."""

    values = np.asarray(phi)[rows]
    return float(np.min(values)), float(np.max(values))


def select_split(
    phi: np.ndarray,
    train_size: int,
    test_size: int,
    seed: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Choose train/test rows for in-support or extrapolation evaluation.

    ``phi`` is only a scalar summary of the high-dimensional lag window, so this
    is a pragmatic support proxy rather than a full support test in flattened
    ``X`` space.
    """

    if split not in SPLIT_MODES:
        valid = ", ".join(SPLIT_MODES)
        raise ValueError(f"unknown split mode {split!r}; choose one of: {valid}")
    if train_size + test_size > len(phi):
        raise ValueError("train_size + test_size must be no larger than num_samples")

    if split == "in-support":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(phi))
        train_idx = shuffled[:train_size]
        candidates = shuffled[train_size:]

        support_lo = float(np.quantile(phi[train_idx], 0.02))
        support_hi = float(np.quantile(phi[train_idx], 0.98))
        in_support = candidates[(phi[candidates] >= support_lo) & (phi[candidates] <= support_hi)]
        if len(in_support) < test_size:
            raise ValueError(
                "not enough held-out rows inside the train phi support; "
                "increase num_samples or lower test_size"
            )
        return train_idx, in_support[:test_size], (support_lo, support_hi)

    order = np.argsort(phi)
    if split == "right-extrapolation":
        train_idx = order[:train_size]
        test_idx = order[train_size : train_size + test_size]
        return train_idx, test_idx, phi_support(phi, train_idx)

    if split == "left-extrapolation":
        train_idx = order[-train_size:]
        test_idx = order[-train_size - test_size : -train_size]
        return train_idx, test_idx, phi_support(phi, train_idx)

    train_start = (len(phi) - train_size) // 2
    train_end = train_start + train_size
    lower_test_size = test_size // 2
    upper_test_size = test_size - lower_test_size
    lower_pool = order[:train_start]
    upper_pool = order[train_end:]
    if len(lower_pool) < lower_test_size or len(upper_pool) < upper_test_size:
        raise ValueError(
            "not enough rows in both phi tails for two-sided extrapolation; "
            "increase num_samples, lower train_size, or lower test_size"
        )
    train_idx = order[train_start:train_end]
    test_idx = np.concatenate([lower_pool[-lower_test_size:], upper_pool[:upper_test_size]])
    return train_idx, test_idx, phi_support(phi, train_idx)
