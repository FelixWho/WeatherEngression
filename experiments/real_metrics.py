"""Metrics for engression on real data with no known conditional law.

The synthetic diagnostics in ``metrics.py`` compare predicted quantiles against
generator-truth quantiles. Real data has no truth, so here we score the
predictive distribution against held-out realized targets using:

- interval coverage and width at the 50% and 90% levels (calibration);
- median absolute error of the predictive median (sharpness of the center);
- the energy score / CRPS, the proper score engression's loss optimizes.
"""

from __future__ import annotations

import numpy as np


def empirical_quantile_metrics(
    predicted_quantiles: np.ndarray,
    y_test: np.ndarray,
    levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> dict[str, object]:
    """Coverage, interval width, and central error from predicted quantiles.

    ``predicted_quantiles`` has shape ``(n, len(levels))`` with columns ordered as
    ``levels``. The 0.05/0.95 and 0.25/0.75 pairs and the 0.50 median must be
    present.
    """

    cols = {level: i for i, level in enumerate(levels)}
    for needed in (0.05, 0.25, 0.50, 0.75, 0.95):
        if needed not in cols:
            raise ValueError(f"levels must include {needed}; got {levels}")

    y = np.asarray(y_test).reshape(-1)
    q05 = predicted_quantiles[:, cols[0.05]]
    q25 = predicted_quantiles[:, cols[0.25]]
    q50 = predicted_quantiles[:, cols[0.50]]
    q75 = predicted_quantiles[:, cols[0.75]]
    q95 = predicted_quantiles[:, cols[0.95]]

    return {
        "coverage_90": float(np.mean((y >= q05) & (y <= q95))),
        "coverage_50": float(np.mean((y >= q25) & (y <= q75))),
        "mean_width_90": float(np.mean(q95 - q05)),
        "mean_width_50": float(np.mean(q75 - q25)),
        "median_abs_error": float(np.mean(np.abs(y - q50))),
    }


def pit_values(samples: np.ndarray, y_test: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomized probability-integral-transform (PIT) value per test point.

    ``samples`` has shape ``(n, sample_size)``; ``y_test`` is the realized target.
    For point ``i`` the PIT is where ``y_i`` falls in its own predictive sample
    distribution. Using the randomized-rank form

        u_i = (a_i + U_i * (b_i - a_i + 1)) / (S + 1),

    with ``a = #{s < y}``, ``b = #{s <= y}`` and ``U_i ~ Uniform(0, 1)``, makes the
    ``u_i`` EXACTLY Uniform(0, 1) when the predictive shape is correct (the ``U_i``
    term breaks ties and removes the grid/boundary bias of the plain rank). A flat
    histogram of these values is the shape-calibration check the 50%/90% coverage
    numbers cannot provide.
    """

    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y_test, dtype=np.float64).reshape(-1, 1)
    sample_size = samples.shape[1]
    below = np.sum(samples < y, axis=1)              # a = #{s < y}
    at_or_below = np.sum(samples <= y, axis=1)       # b = #{s <= y}
    u = np.random.default_rng(seed).random(samples.shape[0])
    return (below + u * (at_or_below - below + 1)) / (sample_size + 1)


def pit_calibration_metrics(pit: np.ndarray, n_bins: int = 20) -> dict[str, object]:
    """Scalar summaries of PIT uniformity: bias, dispersion, and overall shape.

    Under correct predictive shape the PIT is Uniform(0, 1), so mean = 0.5 and
    variance = 1/12. The signs are diagnostic:

    - ``pit_mean`` != 0.5   -> location bias (predictions shifted low/high);
    - ``pit_var`` > 1/12    -> UNDERdispersed (U-shaped PIT, intervals too narrow);
    - ``pit_var`` < 1/12    -> OVERdispersed (dome-shaped PIT, intervals too wide);
    - ``pit_ks``            -> Kolmogorov-Smirnov distance to uniform (overall).
    - ``pit_l1``            -> mean |bin freq - uniform| over ``n_bins`` (overall).
    """

    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    n = pit.size
    uniform_var = 1.0 / 12.0
    ecdf = np.sort(pit)
    grid = (np.arange(1, n + 1)) / n
    ks = float(np.max(np.abs(ecdf - grid))) if n else float("nan")
    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    l1 = float(np.mean(np.abs(counts / max(n, 1) - 1.0 / n_bins)))
    var = float(np.var(pit))
    if var > uniform_var * 1.15:
        shape = "underdispersed (intervals too narrow)"
    elif var < uniform_var * 0.85:
        shape = "overdispersed (intervals too wide)"
    else:
        shape = "well-dispersed"
    return {
        "pit_mean": float(np.mean(pit)),
        "pit_var": var,
        "pit_var_ideal": uniform_var,
        "pit_ks": ks,
        "pit_l1": l1,
        "pit_shape": shape,
        "pit_n_bins": int(n_bins),
    }


def coverage_by_distance_bins(
    distances: np.ndarray,
    covered: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, object]]:
    """Stratify 90% interval coverage into equal-count bins of an OOD distance.

    Test points are sorted by ``distances`` and split into ``n_bins`` equal-count
    groups (bin 0 = closest to training, last bin = farthest). For each bin the
    mean distance and the realized coverage are reported, so calibration can be
    read off as a function of distance from the training feature space.
    """

    distances = np.asarray(distances, dtype=np.float64)
    covered = np.asarray(covered, dtype=np.float64)
    order = np.argsort(distances, kind="stable")
    out: list[dict[str, object]] = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        out.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "mean_distance": float(distances[idx].mean()),
                "coverage_90": float(covered[idx].mean()),
            }
        )
    return out


def energy_score_samples(samples: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    """Mean univariate energy score (CRPS) of conditional samples vs realized y.

    ``samples`` has shape ``(n, sample_size)``. For each row the score is

        CRPS = E|X - y| - 0.5 * E|X - X'|,

    where ``X, X'`` are independent draws from the predictive distribution. The
    second term uses the exact sorted-sample identity
    ``E|X - X'| = (2 / S^2) * sum_i (2i - S - 1) * x_(i)`` so the cost is
    ``O(n S log S)`` instead of ``O(n S^2)``. Lower is better.
    """

    samples = np.asarray(samples, dtype=np.float64)
    y = np.asarray(y_test, dtype=np.float64).reshape(-1, 1)
    n, sample_size = samples.shape
    if sample_size < 2:
        raise ValueError("energy score needs at least 2 samples per point")

    term_fit = np.mean(np.abs(samples - y), axis=1)  # E|X - y|

    ordered = np.sort(samples, axis=1)
    rank = np.arange(1, sample_size + 1)
    weights = (2 * rank - sample_size - 1).astype(np.float64)
    e_spread = (2.0 / (sample_size * sample_size)) * (ordered @ weights)  # E|X - X'|

    crps = term_fit - 0.5 * e_spread
    return {
        "energy_score": float(np.mean(crps)),
        "energy_score_fit_term": float(np.mean(term_fit)),
        "energy_score_spread_term": float(np.mean(e_spread)),
    }
