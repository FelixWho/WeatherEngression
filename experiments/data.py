"""Dataset conversion helpers for engression experiments."""

from __future__ import annotations

import numpy as np
import torch

from generate_data import flatten_x_windows


def tensors_from_dataset(
    dataset: dict[str, np.ndarray],
    rows: np.ndarray,
    input_kind: str = "flat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``X`` and column-vector ``y`` tensors for engression.

    ``input_kind="flat"`` returns the package-style MLP input with shape
    ``(n, lags * features)``. ``input_kind="sequence"`` keeps the lag-window
    structure with shape ``(n, lags, features)`` for sequence-native models.
    """

    x = np.asarray(dataset["X"], dtype=np.float32)
    if input_kind == "flat":
        x_model = flatten_x_windows(x)
    elif input_kind == "sequence":
        x_model = x
    else:
        raise ValueError("input_kind must be 'flat' or 'sequence'")
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(x_model[rows]), torch.from_numpy(y[rows])


def flattened_tensors_from_dataset(
    dataset: dict[str, np.ndarray],
    rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened ``X`` plus ``y`` regardless of model input kind."""

    return tensors_from_dataset(dataset, rows, input_kind="flat")
