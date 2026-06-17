"""Metric helpers for synthetic engression diagnostics."""

from __future__ import annotations

import numpy as np


def quantile_diagnostic_metrics(
    predicted_quantiles: np.ndarray,
    true_quantiles: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    """Compute scalar diagnostics for predicted conditional quantiles."""

    quantile_mae = np.mean(np.abs(predicted_quantiles - true_quantiles), axis=0)
    predicted_interval_coverage = float(
        np.mean((y_test >= predicted_quantiles[:, 0]) & (y_test <= predicted_quantiles[:, 2]))
    )
    true_interval_coverage = float(
        np.mean((y_test >= true_quantiles[:, 0]) & (y_test <= true_quantiles[:, 2]))
    )
    predicted_width = float(np.mean(predicted_quantiles[:, 2] - predicted_quantiles[:, 0]))
    true_width = float(np.mean(true_quantiles[:, 2] - true_quantiles[:, 0]))
    return {
        "quantile_mae": {
            "q05": float(quantile_mae[0]),
            "q50": float(quantile_mae[1]),
            "q95": float(quantile_mae[2]),
        },
        "mean_quantile_mae": float(np.mean(quantile_mae)),
        "median_mae": float(quantile_mae[1]),
        "predicted_interval_coverage": predicted_interval_coverage,
        "true_interval_coverage": true_interval_coverage,
        "mean_predicted_interval_width": predicted_width,
        "mean_true_interval_width": true_width,
        "width_ratio": predicted_width / true_width,
    }
