"""Prediction helpers for fitted engression models."""

from __future__ import annotations

import numpy as np
import torch

from .constants import QUANTILE_LEVELS


def predict_quantiles(
    engressor: object,
    x_test: torch.Tensor,
    sample_size: int,
) -> np.ndarray:
    """Return predicted quantiles as an ``(n_test, 3)`` NumPy array."""

    predictions = engressor.predict(
        x_test,
        target=list(QUANTILE_LEVELS),
        sample_size=sample_size,
    )
    if not isinstance(predictions, list):
        predictions = [predictions]
    return np.column_stack(
        [prediction.detach().cpu().numpy().reshape(-1) for prediction in predictions]
    )
