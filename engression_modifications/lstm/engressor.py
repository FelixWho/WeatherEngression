"""The fitted-model wrapper, separate from the stochastic model.

``LSTMEngressor`` wraps a trained generator (one of the ``LSTMGenerator``
subclasses, or any custom ``nn.Module``) with the bookkeeping needed to *use*
it: standardization,
repeated sampling, quantile prediction, and the embedding hook for OOS
diagnostics. It matches the upstream ``engression`` package's ``predict`` API so
the rest of the pipeline is model-agnostic. The network itself lives in
``generator.py``.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .config import LSTMEngressionConfig
from .preprocessing import validate_sequence_x


class LSTMEngressor:
    """Small fitted-model wrapper matching the upstream ``predict`` API."""

    def __init__(
        self,
        model: nn.Module,
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
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the LSTM trajectory encoding ``h(x)``, shape ``(n, lstm_hidden_dim)``.

        This is the deterministic representation the stochastic head conditions on
        (fresh noise is concatenated only afterward). Distances measured in this
        space reflect the model's own notion of input similarity, which is the
        relevant geometry for post-hoc out-of-support analysis.
        """

        self.model.eval()
        x = validate_sequence_x(x).to(self.device)
        x = self._standardize_x(x)
        # A custom model can expose its own representation via ``encode(x)``; prefer
        # that. Otherwise fall back to the built-in LSTM's final hidden state.
        encode_fn = getattr(self.model, "encode", None)
        if callable(encode_fn):
            return encode_fn(x)
        lstm = getattr(self.model, "lstm", None)
        if lstm is not None:
            _, (hidden, _) = lstm(x)
            return hidden[-1]
        raise AttributeError(
            "encode() needs the underlying model to define `encode(x)` or expose an "
            "`lstm` attribute; the provided custom model has neither."
        )

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
