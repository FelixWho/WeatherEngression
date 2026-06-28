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
