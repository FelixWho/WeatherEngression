"""Pooled quantile recalibration around a fitted conditional sampler.

Only quantiles are recalibrated. The underlying sampler and its joint predictive
distribution are unchanged; access them explicitly through ``engressor``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class RecalibratedEngressor:
    """Store nominal -> base-model quantile levels shared by all predictions."""

    def __init__(self, engressor, quantile_levels=None):
        self.engressor = engressor
        self.quantile_levels = {}
        if quantile_levels is not None:
            self.quantile_levels = {
                self._level(q): self._level(p) for q, p in quantile_levels.items()
            }
            pairs = sorted(self.quantile_levels.items())
            if any(left[1] > right[1] for left, right in zip(pairs, pairs[1:])):
                raise ValueError("adjusted quantile levels must be nondecreasing")

    @staticmethod
    def _level(value):
        value = float(value)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("quantile levels must be finite and between 0 and 1")
        return round(value, 12)

    @property
    def device(self):
        return self.engressor.device

    def fit_from_pit(self, pit, levels):
        """Invert the pooled empirical PIT CDF at the requested nominal levels.

        ``pit`` must come from separate clean recalibration rows, not the rows
        used to fit the network, select its checkpoint, or evaluate coverage.
        This is empirical quantile mapping, not a fitted isotonic regressor.
        """
        pit = np.asarray(pit, dtype=float).reshape(-1)
        levels = [self._level(q) for q in levels]
        if not pit.size or not np.isfinite(pit).all() or np.any((pit < 0) | (pit > 1)):
            raise ValueError("PIT values must be nonempty, finite, and between 0 and 1")
        if not levels:
            raise ValueError("at least one nominal quantile level is required")
        adjusted = np.quantile(pit, levels)
        self.quantile_levels = dict(zip(levels, map(float, adjusted)))
        return self

    def quantiles_from_samples(self, samples, levels):
        """Return quantiles shaped (levels, rows, outputs), computed on CPU.

        An empty mapping means identity (raw quantiles). A fitted mapping must
        contain every requested level; missing levels never silently fall back.
        """
        levels = [self._level(q) for q in levels]
        adjusted = [self.quantile_levels[q] for q in levels] if self.quantile_levels else levels
        samples = samples.detach().cpu()
        probabilities = torch.tensor(adjusted, dtype=samples.dtype)
        return torch.quantile(samples, probabilities, dim=-1)

    def predict(self, x, target=0.5, sample_size=400):
        """Predict one quantile or a list, in the base engressor's output units."""
        scalar = np.isscalar(target)
        levels = [target] if scalar else list(target)
        samples = self.engressor.sample(x, sample_size=sample_size)
        quantiles = self.quantiles_from_samples(samples, levels)
        return quantiles[0] if scalar else list(quantiles)

    def save(self, path):
        """Save the mapping beside its separately saved base-model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.quantile_levels, indent=2) + "\n")

    @classmethod
    def load(cls, engressor, path):
        return cls(engressor, json.loads(Path(path).read_text()))
