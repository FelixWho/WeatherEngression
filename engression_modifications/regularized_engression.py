"""Fit public-package engression with optimizer regularization.

The upstream Python package exposes useful architecture knobs, but the
high-level ``engression(...)`` helper does not expose optimizer options such as
``weight_decay``. This module keeps the upstream stochastic MLP model unchanged
and only replaces the optimizer before training.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import torch

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from engression.engression import Engressor


@dataclass(frozen=True)
class RegularizedEngressionConfig:
    """Configuration for an optimizer-regularized upstream ``Engressor``."""

    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    add_bn: bool = False
    resblock: bool = False
    out_act: str | None = None
    beta: float = 1.0
    lr: float = 0.003
    weight_decay: float = 1e-4
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    device: str | torch.device = "cpu"
    verbose: bool = False
    print_every_nepoch: int | None = None
    print_times_per_epoch: int = 1


def fit_regularized_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: RegularizedEngressionConfig | None = None,
) -> Engressor:
    r"""Fit engression with Adam weight decay.

    Parameters
    ----------
    x:
        Predictor matrix of shape ``(n, p)``. For the current weather
        experiments, lag-window tensors should be flattened before calling this.
    y:
        Target matrix of shape ``(n, k)``. A scalar target should be shaped
        ``(n, 1)``.
    config:
        Model and optimizer settings. The default keeps the same basic scale as
        the large CPU-friendly smoke-test run.

    Returns
    -------
    Engressor
        A fitted upstream ``Engressor`` instance.

    Notes
    -----
    This adds L2-style optimizer regularization through Adam's
    ``weight_decay`` argument:

    \[
    \min_\theta \mathcal{L}_{\mathrm{energy}}(\theta)
    + \lambda \lVert \theta \rVert_2^2.
    \]

    It does not add dropout, gradient clipping, or an LSTM architecture. Those
    require changing the upstream model or writing a local generator.
    """

    if config is None:
        config = RegularizedEngressionConfig()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    engressor = Engressor(
        in_dim=x.shape[1],
        out_dim=y.shape[1],
        num_layer=config.num_layer,
        hidden_dim=config.hidden_dim,
        noise_dim=config.noise_dim,
        add_bn=config.add_bn,
        resblock=config.resblock,
        out_act=config.out_act,
        beta=config.beta,
        lr=config.lr,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        standardize=config.standardize,
        device=config.device,
        check_device=config.verbose,
        verbose=config.verbose,
    )
    engressor.optimizer = torch.optim.Adam(
        engressor.model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    engressor.train(
        x,
        y,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr=None,
        print_every_nepoch=(
            config.print_every_nepoch
            if config.print_every_nepoch is not None
            else config.num_epochs + 1
        ),
        print_times_per_epoch=config.print_times_per_epoch,
        standardize=config.standardize,
        verbose=config.verbose,
    )
    return engressor
