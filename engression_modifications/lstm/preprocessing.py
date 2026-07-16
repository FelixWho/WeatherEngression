"""Input validation and standardization for sequence engression.

Pure tensor plumbing shared by the fitter and the fitted engressor: shape
checks and train-set moment computation. No model or training logic here.
"""

from __future__ import annotations

import torch


def validate_sequence_x(x: torch.Tensor) -> torch.Tensor:
    """Return a float tensor shaped ``(n, sequence_length, feature_dim)``."""

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    x = x.float()
    if x.ndim != 3:
        raise ValueError(
            "LSTM engression expects X with shape "
            "(n, sequence_length, feature_dim); use input_kind='sequence'"
        )
    return x


def validate_y(y: torch.Tensor) -> torch.Tensor:
    """Return a float target tensor shaped ``(n, out_dim)``."""

    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    y = y.float()
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        raise ValueError("y must have shape (n,) or (n, out_dim)")
    return y


def standardization_stats(
    x: torch.Tensor,
    y: torch.Tensor,
    standardize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute train-set moments for sequence inputs and scalar outputs."""

    if not standardize:
        x_mean = torch.zeros(1, 1, x.shape[2], dtype=x.dtype)
        x_std = torch.ones(1, 1, x.shape[2], dtype=x.dtype)
        y_mean = torch.zeros(1, y.shape[1], dtype=y.dtype)
        y_std = torch.ones(1, y.shape[1], dtype=y.dtype)
        return x_mean, x_std, y_mean, y_std

    x_mean = x.mean(dim=(0, 1), keepdim=True)
    x_std = x.std(dim=(0, 1), keepdim=True)
    x_std[x_std == 0.0] = 1.0
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True)
    y_std[y_std == 0.0] = 1.0
    return x_mean, x_std, y_mean, y_std
