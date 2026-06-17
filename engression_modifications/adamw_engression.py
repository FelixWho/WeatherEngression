"""Fit public-package engression with decoupled AdamW weight decay.

This keeps the upstream stochastic MLP generator unchanged, but replaces the
optimizer with ``torch.optim.AdamW`` before training. AdamW applies weight decay
as a separate shrinkage step instead of mixing it into Adam's adaptive gradient
statistics.
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
class AdamWEngressionConfig:
    """Configuration for an upstream ``Engressor`` trained with AdamW."""

    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    add_bn: bool = False
    resblock: bool = False
    out_act: str | None = None
    beta: float = 1.0
    lr: float = 0.003
    weight_decay: float = 1e-3
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    device: str | torch.device = "cpu"
    verbose: bool = False
    print_every_nepoch: int | None = None
    print_times_per_epoch: int = 1


def fit_adamw_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: AdamWEngressionConfig | None = None,
) -> Engressor:
    r"""Fit engression with decoupled AdamW weight decay.

    AdamW separates the energy-loss gradient step from parameter shrinkage:

    \[
    \theta \leftarrow \theta - \eta\,\operatorname{AdamStep}(\nabla_\theta \mathcal{L}),
    \qquad
    \theta \leftarrow (1-\eta\lambda)\theta.
    \]

    This makes \(\lambda\) easier to interpret than classic Adam's coupled
    ``weight_decay`` behavior.
    """

    if config is None:
        config = AdamWEngressionConfig()
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
    engressor.optimizer = torch.optim.AdamW(
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
