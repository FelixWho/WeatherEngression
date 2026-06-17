"""Fit the public-package engression baseline.

This module wraps the upstream high-level ``engression(...)`` function so the
unregularized baseline has the same import shape as the regularized variants.
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

from engression import engression as upstream_engression
from engression.engression import Engressor


@dataclass(frozen=True)
class VanillaEngressionConfig:
    """Configuration for the unmodified upstream ``engression(...)`` helper."""

    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    add_bn: bool = False
    resblock: bool = False
    out_act: str | None = None
    beta: float = 1.0
    lr: float = 0.003
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    device: str | torch.device = "cpu"
    verbose: bool = False
    print_every_nepoch: int | None = None
    print_times_per_epoch: int = 1


def fit_vanilla_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: VanillaEngressionConfig | None = None,
) -> Engressor:
    """Fit the unmodified public-package engression baseline."""

    if config is None:
        config = VanillaEngressionConfig()
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    return upstream_engression(
        x,
        y,
        num_layer=config.num_layer,
        hidden_dim=config.hidden_dim,
        noise_dim=config.noise_dim,
        out_act=config.out_act,
        add_bn=config.add_bn,
        resblock=config.resblock,
        beta=config.beta,
        lr=config.lr,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        print_every_nepoch=(
            config.print_every_nepoch
            if config.print_every_nepoch is not None
            else config.num_epochs + 1
        ),
        print_times_per_epoch=config.print_times_per_epoch,
        device=config.device,
        standardize=config.standardize,
        verbose=config.verbose,
    )
