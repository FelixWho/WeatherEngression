"""Out-of-support diagnostics for generated-data experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class OOSConfig:
    """Configuration for out-of-support diagnostics.

    The kNN method uses a sampled training reference set by default so the
    diagnostic stays comfortable on laptop CPU runs.
    """

    marginal_quantile_low: float = 0.01
    marginal_quantile_high: float = 0.99
    knn_threshold_quantile: float = 0.95
    knn_reference_size: int = 2_000
    knn_batch_size: int = 512
    seed_offset: int = 211


@dataclass(frozen=True)
class OOSDiagnostics:
    """Row-level OOS flags and compact JSON-friendly summaries."""

    flags: dict[str, np.ndarray]
    summary: dict[str, object]


def as_numpy_2d(values: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a two-dimensional float32 NumPy array."""

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    return array


def summarize_flags(flags: np.ndarray) -> dict[str, object]:
    """Return count and fraction summaries for boolean OOS flags."""

    flags = np.asarray(flags, dtype=bool)
    return {
        "count": int(np.sum(flags)),
        "fraction": float(np.mean(flags)) if len(flags) else 0.0,
    }


def scalar_projection_oos(
    train_phi: np.ndarray,
    test_phi: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    r"""Flag rows outside the training range of scalar \(\phi(X)\)."""

    train_phi = np.asarray(train_phi, dtype=np.float32)
    test_phi = np.asarray(test_phi, dtype=np.float32)
    support = (float(np.min(train_phi)), float(np.max(train_phi)))
    flags = (test_phi < support[0]) | (test_phi > support[1])
    summary = summarize_flags(flags)
    summary.update(
        {
            "support": [support[0], support[1]],
            "test_range": [float(np.min(test_phi)), float(np.max(test_phi))],
        }
    )
    return flags, summary


def marginal_range_oos(
    x_train: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, dict[str, object]]:
    """Flag rows where at least one feature is outside train min/max range."""

    train = as_numpy_2d(x_train)
    test = as_numpy_2d(x_test)
    lower = np.min(train, axis=0)
    upper = np.max(train, axis=0)
    violations = (test < lower) | (test > upper)
    flags = np.any(violations, axis=1)
    violated_feature_counts = np.sum(violations, axis=1)
    summary = summarize_flags(flags)
    summary.update(
        {
            "feature_count": int(train.shape[1]),
            "mean_violated_features": float(np.mean(violated_feature_counts)),
            "max_violated_features": int(np.max(violated_feature_counts)) if len(test) else 0,
        }
    )
    return flags, summary


def marginal_quantile_oos(
    x_train: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    low: float,
    high: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Flag rows where at least one feature is outside train quantile ranges."""

    if not 0.0 <= low < high <= 1.0:
        raise ValueError("expected 0 <= low < high <= 1")
    train = as_numpy_2d(x_train)
    test = as_numpy_2d(x_test)
    lower = np.quantile(train, low, axis=0)
    upper = np.quantile(train, high, axis=0)
    violations = (test < lower) | (test > upper)
    flags = np.any(violations, axis=1)
    violated_feature_counts = np.sum(violations, axis=1)
    summary = summarize_flags(flags)
    summary.update(
        {
            "low_quantile": float(low),
            "high_quantile": float(high),
            "feature_count": int(train.shape[1]),
            "mean_violated_features": float(np.mean(violated_feature_counts)),
            "max_violated_features": int(np.max(violated_feature_counts)) if len(test) else 0,
        }
    )
    return flags, summary


def standardize_by_train(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize train and test arrays using train column moments."""

    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def choose_reference_indices(n_rows: int, max_size: int, seed: int) -> np.ndarray:
    """Choose a stable training reference subset for kNN diagnostics."""

    if max_size <= 0 or max_size >= n_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=max_size, replace=False))


def nearest_neighbor_distances(
    queries: np.ndarray,
    reference: np.ndarray,
    batch_size: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Return Euclidean nearest-neighbor distances from queries to reference."""

    queries = np.asarray(queries, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if len(reference) == 0:
        raise ValueError("reference must contain at least one row")
    if exclude_self and len(reference) < 2:
        raise ValueError("at least two reference rows are needed when exclude_self=True")

    ref_norm = np.sum(reference * reference, axis=1, keepdims=True).T
    distances = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = queries[start:end]
        batch_norm = np.sum(batch * batch, axis=1, keepdims=True)
        dist2 = batch_norm + ref_norm - 2.0 * (batch @ reference.T)
        np.maximum(dist2, 0.0, out=dist2)
        if exclude_self:
            rows = np.arange(end - start)
            cols = np.arange(start, end)
            dist2[rows, cols] = np.inf
        distances[start:end] = np.sqrt(np.min(dist2, axis=1))
    return distances


def knn_distance_oos(
    x_train: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    threshold_quantile: float,
    reference_size: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Flag rows whose standardized kNN distance exceeds a train threshold."""

    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must lie between 0 and 1")
    train = as_numpy_2d(x_train)
    test = as_numpy_2d(x_test)
    train_z, test_z = standardize_by_train(train, test)
    reference_idx = choose_reference_indices(len(train_z), reference_size, seed)
    reference = train_z[reference_idx]

    train_reference_distances = nearest_neighbor_distances(
        queries=reference,
        reference=reference,
        batch_size=batch_size,
        exclude_self=True,
    )
    threshold = float(np.quantile(train_reference_distances, threshold_quantile))
    test_distances = nearest_neighbor_distances(
        queries=test_z,
        reference=reference,
        batch_size=batch_size,
        exclude_self=False,
    )
    flags = test_distances > threshold
    summary = summarize_flags(flags)
    summary.update(
        {
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
            "reference_size": int(len(reference)),
            "mean_test_distance": float(np.mean(test_distances)),
            "max_test_distance": float(np.max(test_distances)) if len(test_distances) else 0.0,
            "mean_train_reference_distance": float(np.mean(train_reference_distances)),
        }
    )
    return flags, summary


def mahalanobis_distances(
    reference: np.ndarray | torch.Tensor,
    queries: np.ndarray | torch.Tensor,
    reg: float = 1e-6,
) -> np.ndarray:
    """Per-row Mahalanobis distance from ``queries`` to the ``reference`` distribution.

    Distance is ``sqrt((x - mu) Sigma^-1 (x - mu))`` using the reference mean and
    covariance. The covariance is ridge-regularized and inverted with a pseudo-
    inverse, so it stays stable in high-dimensional (e.g. LSTM-embedding) spaces.
    """

    ref = as_numpy_2d(reference).astype(np.float64)
    query = as_numpy_2d(queries).astype(np.float64)
    mean = ref.mean(axis=0, keepdims=True)
    cov = np.atleast_2d(np.cov(ref, rowvar=False))
    cov += reg * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)
    delta = query - mean
    squared = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared).astype(np.float32)


def mahalanobis_distance_oos(
    x_train: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    threshold_quantile: float,
    reference_size: int,
    seed: int,
    reg: float = 1e-6,
) -> tuple[np.ndarray, dict[str, object]]:
    """Flag rows whose Mahalanobis distance exceeds a train-distribution threshold."""

    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must lie between 0 and 1")
    train = as_numpy_2d(x_train)
    test = as_numpy_2d(x_test)
    reference_idx = choose_reference_indices(len(train), reference_size, seed)
    reference = train[reference_idx]

    train_reference_distances = mahalanobis_distances(reference, reference, reg=reg)
    threshold = float(np.quantile(train_reference_distances, threshold_quantile))
    test_distances = mahalanobis_distances(reference, test, reg=reg)
    flags = test_distances > threshold
    summary = summarize_flags(flags)
    summary.update(
        {
            "threshold_quantile": float(threshold_quantile),
            "threshold": threshold,
            "reference_size": int(len(reference)),
            "mean_test_distance": float(np.mean(test_distances)),
            "max_test_distance": float(np.max(test_distances)) if len(test_distances) else 0.0,
            "mean_train_reference_distance": float(np.mean(train_reference_distances)),
        }
    )
    return flags, summary


def compute_oos_diagnostics(
    x_train: np.ndarray | torch.Tensor,
    x_test: np.ndarray | torch.Tensor,
    train_phi: np.ndarray,
    test_phi: np.ndarray,
    config: OOSConfig | None = None,
    seed: int = 0,
) -> OOSDiagnostics:
    """Compute the first four OOS diagnostics used in this project."""

    if config is None:
        config = OOSConfig()
    flags: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {}

    flags["scalar_projection"], summary["scalar_projection"] = scalar_projection_oos(
        train_phi=train_phi,
        test_phi=test_phi,
    )
    flags["marginal_range"], summary["marginal_range"] = marginal_range_oos(
        x_train=x_train,
        x_test=x_test,
    )
    flags["marginal_quantile"], summary["marginal_quantile"] = marginal_quantile_oos(
        x_train=x_train,
        x_test=x_test,
        low=config.marginal_quantile_low,
        high=config.marginal_quantile_high,
    )
    flags["knn_distance"], summary["knn_distance"] = knn_distance_oos(
        x_train=x_train,
        x_test=x_test,
        threshold_quantile=config.knn_threshold_quantile,
        reference_size=config.knn_reference_size,
        batch_size=config.knn_batch_size,
        seed=seed + config.seed_offset,
    )
    return OOSDiagnostics(flags=flags, summary=summary)
