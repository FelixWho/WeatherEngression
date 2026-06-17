"""Sequence-native LSTM engression model.

The public ``engression`` package flattens every predictor tensor before it
reaches the stochastic MLP. This module keeps the energy-loss idea, but replaces
the flat generator with an LSTM encoder followed by a stochastic MLP head.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from engression.engression import energy_loss_two_sample


@dataclass(frozen=True)
class LSTMEngressionConfig:
    """Configuration for a sequence-native engression model."""

    num_layer: int = 3
    hidden_dim: int = 192
    noise_dim: int = 96
    lstm_hidden_dim: int | None = None
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    add_bn: bool = False
    beta: float = 1.0
    lr: float = 0.003
    weight_decay: float = 0.0
    num_epochs: int = 250
    batch_size: int | None = 512
    standardize: bool = True
    grad_clip: float | None = 5.0
    device: str | torch.device = "cpu"
    verbose: bool = False
    print_every_nepoch: int | None = None


class LSTMStochasticGenerator(nn.Module):
    """Map a lag-window sequence and fresh noise to one generated response."""

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        config: LSTMEngressionConfig,
    ) -> None:
        super().__init__()
        lstm_hidden_dim = config.lstm_hidden_dim or config.hidden_dim
        dropout = config.lstm_dropout if config.lstm_num_layers > 1 else 0.0
        self.noise_dim = config.noise_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=dropout,
            batch_first=True,
        )

        layers: list[nn.Module] = []
        in_dim = lstm_hidden_dim + config.noise_dim
        hidden_layers = max(1, config.num_layer)
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            if config.add_bn:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = config.hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate one stochastic response for each input sequence."""

        _, (hidden, _) = self.lstm(x)
        encoded = hidden[-1]
        noise = torch.randn(x.shape[0], self.noise_dim, device=x.device, dtype=x.dtype)
        return self.head(torch.cat([encoded, noise], dim=1))


class LSTMEngressor:
    """Small fitted-model wrapper matching the upstream ``predict`` API."""

    def __init__(
        self,
        model: LSTMStochasticGenerator,
        config: LSTMEngressionConfig,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device) if isinstance(config.device, str) else config.device
        self.standardize = config.standardize
        self.x_mean = x_mean.to(self.device)
        self.x_std = x_std.to(self.device)
        self.y_mean = y_mean.to(self.device)
        self.y_std = y_std.to(self.device)

    def _standardize_x(self, x: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return x
        return (x - self.x_mean) / self.x_std

    def _unstandardize_y(self, y: torch.Tensor) -> torch.Tensor:
        if not self.standardize:
            return y
        return y * self.y_std + self.y_mean

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        sample_size: int = 100,
        expand_dim: bool = True,
    ) -> torch.Tensor:
        """Draw conditional samples with shape ``(n, out_dim, sample_size)``."""

        self.model.eval()
        x = validate_sequence_x(x).to(self.device)
        x = self._standardize_x(x)
        samples = [self._unstandardize_y(self.model(x)) for _ in range(sample_size)]
        stacked = torch.stack(samples, dim=2)
        if sample_size == 1 and not expand_dim:
            return stacked.squeeze(2)
        return stacked

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        target: str | float | Iterable[float] = "mean",
        sample_size: int = 100,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Predict means or conditional quantiles from generated samples."""

        samples = self.sample(x, sample_size=sample_size, expand_dim=True)
        if target == "mean":
            return samples.mean(dim=2)
        if isinstance(target, (float, int)):
            return torch.quantile(samples, float(target), dim=2)
        return [torch.quantile(samples, float(level), dim=2) for level in target]


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


def fit_lstm_engression(
    x: torch.Tensor,
    y: torch.Tensor,
    config: LSTMEngressionConfig | None = None,
) -> LSTMEngressor:
    """Fit LSTM engression on unflattened lag-window inputs."""

    if config is None:
        config = LSTMEngressionConfig()
    x = validate_sequence_x(x)
    y = validate_y(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")

    device = torch.device(config.device) if isinstance(config.device, str) else config.device
    x_mean, x_std, y_mean, y_std = standardization_stats(x, y, config.standardize)
    x_train = ((x - x_mean) / x_std if config.standardize else x).to(device)
    y_train = ((y - y_mean) / y_std if config.standardize else y).to(device)

    model = LSTMStochasticGenerator(
        input_dim=x.shape[2],
        out_dim=y.shape[1],
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    batch_size = config.batch_size or len(x_train)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    print_every = config.print_every_nepoch or config.num_epochs + 1

    model.train()
    for epoch_idx in range(config.num_epochs):
        total_loss = 0.0
        total_rows = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_sample1 = model(x_batch)
            y_sample2 = model(x_batch)
            loss = energy_loss_two_sample(
                y_batch,
                y_sample1,
                y_sample2,
                beta=config.beta,
                verbose=False,
            )
            loss.backward()
            if config.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(x_batch)
            total_rows += len(x_batch)
        if config.verbose and (epoch_idx == 0 or (epoch_idx + 1) % print_every == 0):
            mean_loss = total_loss / max(1, total_rows)
            print(f"[Epoch {epoch_idx + 1}] energy-loss: {mean_loss:.4f}")

    return LSTMEngressor(
        model=model,
        config=config,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )
