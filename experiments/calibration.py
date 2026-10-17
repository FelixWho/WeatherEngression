"""Marginal PIT/AD evaluation without loading data or training a model.

The block bootstrap conditions on the fitted model and recalibration map. Its
p-value is approximate: arrival-time blocks must be approximately independent
and representative. It does not establish conditional or joint calibration.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.wildfire_sensitivity.control import time_blocks

_EPS = 1e-12
AD_ALPHA = 0.05
AD_BOOTSTRAP_REPLICATES = 2000
AD_BOOTSTRAP_SEED = 2430
AD_MIN_BLOCKS = 10


def _valid_pit(pit):
    values = np.asarray(pit, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("PIT values must be nonempty, finite, and between 0 and 1")
    return values


def ad_statistic_uniform(pit: np.ndarray) -> float:
    """Classical AD A^2 against Uniform(0,1); clip endpoints at 1e-12."""
    u = np.sort(np.clip(_valid_pit(pit).reshape(-1), _EPS, 1 - _EPS))
    weights = 2 * np.arange(1, len(u) + 1) - 1
    return float(max(0., -len(u) - np.sum(weights * (np.log(u) + np.log1p(-u[::-1]))) / len(u)))


def _ad_statistic_rows(sorted_rows: np.ndarray) -> np.ndarray:
    """Vectorized AD for already sorted rows, used by historical IID diagnostics."""
    values = _valid_pit(sorted_rows)
    if values.ndim != 2:
        raise ValueError("expected a matrix of sorted PIT rows")
    u = np.clip(values, _EPS, 1 - _EPS)
    n = u.shape[1]
    return np.maximum(0., -n - np.sum((2 * np.arange(1, n + 1) - 1) *
                                    (np.log(u) + np.log1p(-u[:, ::-1])), axis=1) / n)


def ad_mc_pvalue(a2_obs: float, n: int, n_sim: int, seed: int) -> float:
    """IID-uniform Monte Carlo reference, NOT the sensitivity acceptance gate."""
    if n < 1 or n_sim < 1 or not np.isfinite(a2_obs) or a2_obs < 0:
        raise ValueError("AD simulation requires positive counts and a finite nonnegative statistic")
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_sim, 128):
        draws = np.sort(rng.random((min(128, n_sim - start), n)), axis=1)
        exceed += int(np.sum(_ad_statistic_rows(draws) >= a2_obs))
    return (1 + exceed) / (n_sim + 1)


def validate_evaluation_targets(x, y, evaluation_scope):
    """Require scalar arrival targets or full trajectory targets, never slice."""
    if evaluation_scope not in ("arrival", "trajectory"):
        raise ValueError(f"unknown evaluation scope: {evaluation_scope!r}")
    if x.ndim != 3 or y.ndim != 2 or not len(x) or x.shape[1] == 0:
        raise ValueError("expected nonempty X (rows, timesteps, features) and y (rows, outputs)")
    expected_outputs = 1 if evaluation_scope == "arrival" else x.shape[1]
    if y.shape != (len(x), expected_outputs):
        raise ValueError(f"{evaluation_scope} requires y shape {(len(x), expected_outputs)}, got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("observed targets must be finite")


def validate_evaluation_outputs(samples, y):
    if samples.ndim != 3 or samples.shape[:2] != y.shape or samples.shape[2] < 2:
        raise ValueError(f"sample shape {samples.shape} does not match target shape {y.shape}")
    values = samples.detach().cpu().numpy() if isinstance(samples, torch.Tensor) else samples
    if not np.isfinite(values).all():
        raise ValueError("predictive samples must be finite")
    return samples, y


def validate_ad_settings(alpha, n_bootstrap, seed, block_days=30., min_blocks=AD_MIN_BLOCKS):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("AD alpha must lie strictly between 0 and 1")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("AD bootstrap replicates must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("AD bootstrap seed must be a nonnegative integer")
    if not np.isfinite(block_days) or block_days <= 0:
        raise ValueError("AD block days must be positive and finite")
    if not isinstance(min_blocks, (int, np.integer)) or min_blocks < 2:
        raise ValueError("AD minimum blocks must be an integer >= 2")


def _block_residual_gram(pit, block_inverse, sizes, chunk_size=4096):
    """Integral of H_b H_c/[u(1-u)], where H_b=C_b-n_b*Fhat.

    Integrate exactly between sorted (clipped) PIT breakpoints. Chunked block
    cumulative counts avoid allocating either replicates x values or blocks x
    all values. Duplicate breakpoints simply contribute zero width.
    """
    order = np.argsort(pit)
    u = np.clip(pit[order], _EPS, 1 - _EPS)
    blocks = block_inverse[order]
    logit = np.log(u) - np.log1p(-u)
    widths = np.diff(logit)
    counts = np.zeros(len(sizes))
    gram = np.zeros((len(sizes), len(sizes)))
    for start in range(0, len(u) - 1, chunk_size):
        stop = min(start + chunk_size, len(u) - 1)
        increments = np.zeros((stop - start, len(sizes)))
        increments[np.arange(stop - start), blocks[start:stop]] = 1
        cumulative = increments.cumsum(axis=0) + counts
        counts = cumulative[-1].copy()
        pooled_cdf = np.arange(start + 1, stop + 1) / len(u)
        residuals = cumulative - pooled_cdf[:, None] * sizes
        weighted = residuals * np.sqrt(widths[start:stop, None])
        gram += weighted.T @ weighted
    return gram


def _bootstrap_statistics(pit, block_inverse, sizes, n_bootstrap, seed):
    gram = _block_residual_gram(pit, block_inverse, sizes)
    rng = np.random.default_rng(seed)
    result = np.empty(n_bootstrap)
    b = len(sizes)
    for start in range(0, n_bootstrap, 256):
        stop = min(start + 256, n_bootstrap)
        multiplicities = rng.multinomial(b, np.full(b, 1 / b), size=stop - start)
        # Fhat* - Fhat = sum(m_b H_b) / N*. Keep actual replicate sizes.
        numerators = np.einsum("ij,jk,ik->i", multiplicities, gram, multiplicities)
        result[start:stop] = np.maximum(0., numerators) / (multiplicities @ sizes)
    return result


def ad_block_bootstrap(pit, time_days, *, alpha=AD_ALPHA,
                       n_bootstrap=AD_BOOTSTRAP_REPLICATES, seed=AD_BOOTSTRAP_SEED,
                       block_days=30., min_blocks=AD_MIN_BLOCKS):
    """Null-centered whole-block AD check, preserving all outputs of each row.

    ``pit`` is (rows,) or (rows, outputs), times are one arrival day per row.
    Resample B blocks with replacement. Compare observed A^2 to
    N* integral (Fhat* - Fhat)^2/[u(1-u)] du; never bootstrap uncentered AD.
    """
    validate_ad_settings(alpha, n_bootstrap, seed, block_days, min_blocks)
    result = dict(statistic=None, p_value=None, valid=False, accepted=False,
                  reason=None, n_rows=0, n_values=0, n_blocks=0, block_sizes=[],
                  alpha=float(alpha), bootstrap_replicates=int(n_bootstrap),
                  bootstrap_seed=int(seed), block_days=float(block_days),
                  min_blocks=int(min_blocks), method="null_centered_block_bootstrap",
                  endpoint_epsilon=_EPS)
    try:
        values = _valid_pit(pit)
        if values.ndim not in (1, 2):
            raise ValueError("PIT must have shape (rows,) or (rows, outputs)")
        times = np.asarray(time_days, dtype=float)
        if times.shape != (len(values),) or not np.isfinite(times).all():
            raise ValueError("AD requires one finite arrival time per row")
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    values = values.reshape(len(values), -1)
    _, row_blocks, row_counts = np.unique(time_blocks(times, block_days),
                                         return_inverse=True, return_counts=True)
    sizes = row_counts * values.shape[1]
    result.update(statistic=ad_statistic_uniform(values), n_rows=len(values),
                  n_values=values.size, n_blocks=len(sizes), block_sizes=sizes.tolist())
    if len(sizes) < min_blocks:
        result["reason"] = f"insufficient blocks: {len(sizes)} < {min_blocks}"
        return result
    simulated = _bootstrap_statistics(values.ravel(), np.repeat(row_blocks, values.shape[1]),
                                      sizes, n_bootstrap, seed)
    if not np.isfinite(simulated).all():
        result["reason"] = "nonfinite bootstrap statistic"
        return result
    p_value = float((1 + np.sum(simulated >= result["statistic"])) / (n_bootstrap + 1))
    accepted = p_value >= alpha
    result.update(p_value=p_value, valid=True, accepted=accepted,
                  reason="AD did not reject calibration" if accepted else "AD rejected calibration")
    return result


@contextmanager
def _sampling_rng(seed, device):
    """Optional isolated Torch stream; the sensitivity loop supplies existing draws."""
    if seed is None:
        yield
        return
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("sampling seed must be a nonnegative integer")
    mps_state = torch.mps.get_rng_state() if torch.device(device).type == "mps" else None
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(seed)
            yield
    finally:
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def evaluate_ad(model, x, y, time_days, *, evaluation_scope="arrival", sample_size=400,
                batch_size=1024, sampling_seed=2026, pit_seed=2026, **ad_options):
    """Evaluate a raw or recalibrated sampler using independent evaluation rows.

    Predictive draws are batched. For precomputed draws call the wrapper's
    pit_from_samples followed by ad_block_bootstrap instead of sampling again.
    """
    x, y = np.asarray(x), np.asarray(y)
    validate_evaluation_targets(x, y, evaluation_scope)
    if batch_size < 1 or sample_size < 2:
        raise ValueError("require batch_size >= 1 and sample_size >= 2")
    wrapped = model if isinstance(model, RecalibratedEngressor) else RecalibratedEngressor(model)
    wrapped.require_full_mapping()
    batches = []
    with _sampling_rng(sampling_seed, wrapped.device), torch.no_grad():
        for start in range(0, len(x), batch_size):
            samples = wrapped.engressor.sample(torch.from_numpy(x[start:start + batch_size]),
                                               sample_size=sample_size).detach().cpu()
            observed = y[start:start + batch_size]
            validate_evaluation_outputs(samples, observed)
            batches.append(wrapped.pit_from_samples(samples, observed, seed=pit_seed + start))
    result = ad_block_bootstrap(np.concatenate(batches), time_days, **ad_options)
    result.update(evaluation_scope=evaluation_scope, pit_is_recalibrated=wrapped.is_recalibrated,
                  prediction_samples=sample_size, evaluation_batch_size=batch_size,
                  sampling_seed=sampling_seed, pit_batch_base=pit_seed)
    return result
