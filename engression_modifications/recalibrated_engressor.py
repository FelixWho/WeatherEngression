"""Pooled quantile recalibration around a fitted conditional sampler.

Quantiles and marginal CDF/PIT evaluations are recalibrated. The underlying
sampler and joint distribution are unchanged; access them through ``engressor``.
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
        self.calibration_pit = None
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
        self.calibration_pit = np.sort(pit)
        return self

    @property
    def is_recalibrated(self):
        return self.calibration_pit is not None or bool(self.quantile_levels)

    def require_full_mapping(self):
        if self.calibration_pit is None and self.quantile_levels:
            raise ValueError("AD/PIT evaluation requires the full mapping; refit recalibration "
                             "for this legacy quantile-only model")

    def transform_cdf(self, probabilities, *, side="right"):
        """R(p), the generalized inverse of the linear empirical quantile map.

        Values below/above its fitted range map to 0/1. ``side='left'``
        evaluates R(p-) at repeated knots (atoms); otherwise return R(p).
        """
        self.require_full_mapping()
        p = np.asarray(probabilities, dtype=float)
        if side not in ("left", "right"):
            raise ValueError("CDF side must be left or right")
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("CDF probabilities must be finite and between 0 and 1")
        if self.calibration_pit is None:
            return p.copy()
        knots = self.calibration_pit
        if len(knots) == 1:
            return (p > knots[0] if side == "left" else p >= knots[0]).astype(float)
        # Between distinct knots the inverse is linear. At repeated knots the
        # left/right search selects the lower/upper end of the probability jump.
        idx = np.searchsorted(knots, p, side=side)
        lo = np.clip(idx - 1, 0, len(knots) - 2)
        hi = lo + 1
        span = knots[hi] - knots[lo]
        frac = np.divide(p - knots[lo], span, out=np.zeros_like(p), where=span > 0)
        result = (lo + frac) / (len(knots) - 1)
        return np.where(idx == 0, 0., np.where(idx == len(knots), 1., result))

    @staticmethod
    def _sample_ranks(samples, y):
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        samples, y = np.asarray(samples), np.asarray(y)
        if samples.ndim < 2 or samples.shape[:-1] != y.shape or samples.shape[-1] < 2:
            raise ValueError("samples must have target shape plus a draw axis of size >= 2")
        if not y.size or not np.isfinite(samples).all() or not np.isfinite(y).all():
            raise ValueError("targets and predictive samples must be nonempty and finite")
        return ((samples < y[..., None]).sum(axis=-1),
                (samples <= y[..., None]).sum(axis=-1), samples.shape[-1])

    def cdf_from_samples(self, samples, y, *, side="right"):
        """Corrected empirical marginal CDF (or left limit), in target shape.

        This uses the discrete empirical forecast, unlike the smoothed rank
        bounds used by pit_from_samples. The empirical CDF's left limit is
        attained on an interval, so both sides use the right-continuous R.
        """
        if side not in ("left", "right"):
            raise ValueError("CDF side must be left or right")
        a, b, size = self._sample_ranks(samples, y)
        count = a if side == "left" else b
        result = self.transform_cdf(count / size)
        # Retain proper CDF limits even if the correction has an atom at 0/1.
        return np.where(count == 0, 0., np.where(count == size, 1., result))

    def pit_from_samples(self, samples, y, *, seed=0):
        """Finite-sample randomized rank PIT, in target shape.

        Transform the rank interval BEFORE randomizing, including ties. The
        identity case exactly matches real_metrics.pit_values. Corrected PIT
        uses Monte Carlo CDF bounds and is an approximation at finite draw count.
        """
        self.require_full_mapping()
        a, b, size = self._sample_ranks(samples, y)
        random = np.random.default_rng(seed).uniform(size=a.shape)
        if self.calibration_pit is None:
            return (a + random * (b - a + 1)) / (size + 1)
        lower = self.transform_cdf(a / (size + 1), side="left")
        upper = self.transform_cdf((b + 1) / (size + 1), side="right")
        return lower + random * (upper - lower)

    def quantiles_from_samples(self, samples, levels):
        """Return quantiles shaped (levels, rows, outputs), computed on CPU.

        An empty mapping means identity. Full mappings support arbitrary levels;
        legacy sparse mappings still require every requested level explicitly.
        """
        levels = [self._level(q) for q in levels]
        if self.calibration_pit is not None:
            adjusted = [self.quantile_levels[q] if q in self.quantile_levels else
                        float(np.quantile(self.calibration_pit, q)) for q in levels]
        else:
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
        path.write_text(json.dumps({
            "version": 1,
            "quantile_levels": self.quantile_levels,
            "calibration_pit": (None if self.calibration_pit is None else self.calibration_pit.tolist()),
        }, indent=2) + "\n")

    @classmethod
    def load(cls, engressor, path):
        payload = json.loads(Path(path).read_text())
        if "version" not in payload:
            return cls(engressor, payload)
        if payload["version"] != 1:
            raise ValueError("unsupported recalibration mapping version")
        model = cls(engressor, payload["quantile_levels"])
        if payload["calibration_pit"] is not None:
            # Recompute rather than trusting an inconsistent pair of mappings.
            model.fit_from_pit(payload["calibration_pit"], model.quantile_levels)
        return model
