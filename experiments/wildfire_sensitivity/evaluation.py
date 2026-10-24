"""Check a fitted model on clean control data and measure wildfire coverage.

Recalibration is fitted in counterfactual.py, never here. Coverage uses corrected
quantiles. The optional AD acceptance gate uses corrected PIT from the same
predictive draws; the descriptive PIT histogram still shows the raw model.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.calibration import (
    AD_ALPHA, AD_BOOTSTRAP_REPLICATES, AD_BOOTSTRAP_SEED,
    ad_block_bootstrap, validate_ad_settings,
    validate_evaluation_targets, validate_evaluation_outputs,
)
from experiments.real_metrics import pit_calibration_metrics, pit_values
from experiments.wildfire_sensitivity.control import time_blocks
from experiments.wildfire_sensitivity.settings import (
    SWEEP_SEED, EVALUATION_BATCH_SIZE, PREDICTION_SAMPLES, HOLDOUT_BLOCK_DAYS,
    CALIBRATION_LEVELS, CALIBRATION_REQUIRED_LEVELS, QUANTILE_LEVELS,
    PIT_BINS, CALIBRATION_TOLERANCE_SE,
)


def interval_coverage(
    model: RecalibratedEngressor,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    alpha_threshold: float = 0.95,
    evaluation_scope: str = "trajectory",
) -> float:
    """Fraction of observed values falling inside the predicted central interval."""

    validate_evaluation_targets(x, y, evaluation_scope)
    lower_quantile = (1 - alpha_threshold) / 2
    upper_quantile = 1 - lower_quantile
    if not isinstance(model, RecalibratedEngressor):
        model = RecalibratedEngressor(model)
    if model.device.type == "mps":
        # Release cached backward activations from training before sampling.
        torch.mps.empty_cache()

    total_batches = (len(x) + EVALUATION_BATCH_SIZE - 1) // EVALUATION_BATCH_SIZE
    print(
        f"  [{label}] drawing {PREDICTION_SAMPLES} samples for {len(x)} trajectories "
        f"in {total_batches} batches of at most {EVALUATION_BATCH_SIZE}..."
    )
    inside_count = 0
    total_count = 0
    for batch_number, start in enumerate(range(0, len(x), EVALUATION_BATCH_SIZE), start=1):
        stop = min(start + EVALUATION_BATCH_SIZE, len(x))
        # Compute quantiles on CPU: torch.quantile may be unsupported on MPS.
        samples = model.engressor.sample(
            torch.from_numpy(x[start:stop]),
            sample_size=PREDICTION_SAMPLES,
        ).cpu()
        samples, observed = validate_evaluation_outputs(samples, y[start:stop])
        lower, upper = model.quantiles_from_samples(samples, [lower_quantile, upper_quantile])
        observed = torch.from_numpy(observed).to(dtype=samples.dtype)
        inside = (observed >= lower) & (observed <= upper)
        inside_count += inside.sum().item()
        total_count += inside.numel()

        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            print(f"    sampled batch {batch_number}/{total_batches}")

    coverage = inside_count / total_count
    print(f"  [{label}] {inside_count}/{total_count} inside ({coverage:.1%})")
    return coverage


def print_calibration(
    model: RecalibratedEngressor,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    time_days: np.ndarray,
    plot_path: Optional[Path] = None,
    evaluation_scope: str = "trajectory",
    calibration_metric: str = "coverage",
    ad_alpha: float = AD_ALPHA,
    ad_bootstrap_replicates: int = AD_BOOTSTRAP_REPLICATES,
    ad_bootstrap_seed: int = AD_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Check a model on clean control rows, print diagnostics, and return metrics.

    Sample once: reuse draws for corrected coverage and (when selected) corrected
    PIT/AD. Coverage mode requires central 90/95% coverage within three block SE;
    AD mode uses only the bootstrap acceptance decision. Both modes return clean
    95% coverage for the later sensitivity comparison. Raw PIT plots are descriptive.

    Targets must be (N, 1) for arrival or (N, T) for trajectory. The latter pools
    marginal coverage across timesteps, rather than testing joint path coverage.
    """

    validate_evaluation_targets(x, y, evaluation_scope)
    if calibration_metric not in ("coverage", "ad"):
        raise ValueError(f"unknown calibration metric: {calibration_metric!r}")
    validate_ad_settings(ad_alpha, ad_bootstrap_replicates, ad_bootstrap_seed)
    if np.asarray(time_days).shape != (len(x),) or not np.isfinite(time_days).all():
        raise ValueError("calibration requires one finite arrival time per row")
    if not isinstance(model, RecalibratedEngressor):
        model = RecalibratedEngressor(model)
    if model.device.type == "mps":
        torch.mps.empty_cache()

    pit_batches = []
    corrected_pit_batches = []
    ad_mapping_error = None
    if calibration_metric == "ad":
        try:
            model.require_full_mapping()
        except ValueError as exc:
            ad_mapping_error = str(exc)
    inside_batches = {level: [] for level in CALIBRATION_LEVELS}
    below_counts = {level: 0 for level in CALIBRATION_LEVELS}
    for start in range(0, len(x), EVALUATION_BATCH_SIZE):
        stop = min(start + EVALUATION_BATCH_SIZE, len(x))
        samples = model.engressor.sample(
            torch.from_numpy(x[start:stop]), sample_size=PREDICTION_SAMPLES
        ).cpu()
        samples, observed = validate_evaluation_outputs(samples, y[start:stop])
        quantiles = model.quantiles_from_samples(samples, QUANTILE_LEVELS).numpy()
        boundaries = dict(zip(QUANTILE_LEVELS, quantiles))
        for level in CALIBRATION_LEVELS:
            lower = boundaries[round((1 - level) / 2, 12)]
            upper = boundaries[round((1 + level) / 2, 12)]
            inside_batches[level].append((observed >= lower) & (observed <= upper))
            below_counts[level] += int((observed <= boundaries[level]).sum())
        samples = samples.numpy()                         # (batch, timesteps, samples)
        if calibration_metric == "ad" and ad_mapping_error is None:
            corrected_pit_batches.append(model.pit_from_samples(
                samples, observed, seed=SWEEP_SEED + start,
            ))
        n_rows, n_steps, n_samples = samples.shape
        pit_batches.append(
            pit_values(
                samples.reshape(n_rows * n_steps, n_samples),
                observed.reshape(n_rows * n_steps),
                seed=SWEEP_SEED + start,
            )
        )
    pit = np.concatenate(pit_batches)

    inside_by_level = {level: np.concatenate(batches) for level, batches in inside_batches.items()}
    below = {level: count / pit.size for level, count in below_counts.items()}

    coverage = _coverage_diagnostics(inside_by_level, time_days)
    sufficient_blocks = all(
        np.isfinite(coverage["standard_error"][level]) for level in CALIBRATION_REQUIRED_LEVELS
    )
    calibrated = coverage["coverage_calibrated"]
    ad_result = None
    if calibration_metric == "ad":
        ad_result = ad_block_bootstrap(
            np.concatenate(corrected_pit_batches) if corrected_pit_batches else np.array([]),
            time_days, alpha=ad_alpha, n_bootstrap=ad_bootstrap_replicates,
            seed=ad_bootstrap_seed, block_days=HOLDOUT_BLOCK_DAYS,
        )
        ad_result.update(evaluation_scope=evaluation_scope,
                         pit_is_recalibrated=model.is_recalibrated,
                         prediction_samples=PREDICTION_SAMPLES,
                         evaluation_batch_size=EVALUATION_BATCH_SIZE,
                         pit_batch_base=SWEEP_SEED,
                         sampling_seed=None)
        if ad_mapping_error is not None:
            ad_result.update(reason=ad_mapping_error, pit_is_recalibrated=False)
        calibrated = ad_result["accepted"]
    summary = pit_calibration_metrics(pit, n_bins=PIT_BINS)
    counts, _ = np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0))

    if calibration_metric == "ad":
        verdict = ad_result["reason"]
        if not calibrated:
            verdict += " -> undetermined"
    elif not sufficient_blocks:
        verdict = "insufficient blocks -> undetermined"
    elif calibrated:
        verdict = "accepted: central 90% and 95% coverage pass"
    else:
        verdict = "central 90% or 95% coverage failed -> undetermined"
    result = {
        "central": coverage["central"],
        "below": below,
        "standard_error": coverage["standard_error"],
        "tolerance": coverage["tolerance"],
        "passes": coverage["passes"],
        "calibrated": calibrated,
        "coverage_calibrated": coverage["coverage_calibrated"],
        "calibration_metric": calibration_metric,
        "acceptance_reason": verdict,
        "model_is_recalibrated": model.is_recalibrated,
        "ad": ad_result,
        "n_blocks": coverage["n_blocks"],
        "pit_counts": counts,
        "pit_is_recalibrated": False,
        "evaluation_scope": evaluation_scope,
        **summary,
    }
    _print_calibration_report(result, label, pit.size, plot_path,
                              ad_alpha, ad_bootstrap_replicates, ad_bootstrap_seed)
    return result


def _coverage_diagnostics(inside_by_level, time_days):
    """Pool observations, then estimate uncertainty using whole 30-day blocks.

    For block sizes n_b, coverages p_b and pooled p, the variance is
    B/(B-1) * sum(n_b**2 * (p_b-p)**2) / sum(n_b)**2.
    This weights observations equally, including unequal-sized blocks. A single
    block yields an invalid standard error and cannot pass the coverage gate.
    """
    central = {level: float(inside.mean()) for level, inside in inside_by_level.items()}
    # Cluster standard error for the pooled coverage, including every block.
    block_of_row = time_blocks(np.asarray(time_days), HOLDOUT_BLOCK_DAYS)
    block_ids, block_inverse, block_counts = np.unique(
        block_of_row, return_inverse=True, return_counts=True
    )
    n_blocks = len(block_ids)
    block_sizes = block_counts.astype(np.float64) * next(iter(inside_by_level.values())).shape[1]
    standard_error: dict[float, float] = {}
    tolerance: dict[float, float] = {}
    passes: dict[float, bool] = {}
    for level in CALIBRATION_LEVELS:
        inside = inside_by_level[level]
        if n_blocks >= 2:
            block_covered = np.bincount(
                block_inverse, weights=inside.sum(axis=1), minlength=n_blocks
            )
            # n_b * (p_b - p): each block's contribution to the pooled error.
            block_residuals = block_covered - block_sizes * central[level]
            spread = float(np.sqrt(
                n_blocks / (n_blocks - 1) * np.sum(block_residuals**2)
            ) / inside.size)
        else:
            spread = float("nan")
        standard_error[level] = spread
        tolerance[level] = CALIBRATION_TOLERANCE_SE * spread
        passes[level] = bool(abs(central[level] - level) <= tolerance[level])
    sufficient_blocks = all(
        np.isfinite(standard_error[level]) for level in CALIBRATION_REQUIRED_LEVELS
    )
    calibrated = sufficient_blocks and all(
        passes[level] for level in CALIBRATION_REQUIRED_LEVELS
    )
    return dict(central=central, standard_error=standard_error, tolerance=tolerance,
                passes=passes, coverage_calibrated=calibrated, n_blocks=n_blocks)


def _print_calibration_report(result, label, n_values, plot_path,
                              ad_alpha, ad_bootstrap_replicates, ad_bootstrap_seed):
    """Render already-computed diagnostics; no sampling or acceptance decisions."""
    central, below = result["central"], result["below"]
    tolerance, passes = result["tolerance"], result["passes"]
    calibration_metric, ad_result = result["calibration_metric"], result["ad"]
    n_blocks, verdict = result["n_blocks"], result["acceptance_reason"]
    counts = result["pit_counts"]
    edges = np.linspace(0., 1., PIT_BINS + 1)
    shares = counts / n_values
    print(
        f"  [{label}] calibration over {n_values:,} values "
        f"in {n_blocks} blocks of {HOLDOUT_BLOCK_DAYS:g} days"
    )
    print("    level   central coverage   below quantile   allowed band      pass")
    for level in CALIBRATION_LEVELS:
        band = tolerance[level]
        band_text = (
            "         n/a"
            if not np.isfinite(band)
            else f"{level - band:>5.1%} to {level + band:<5.1%}"
        )
        print(
            f"    {level:>5.0%}   {central[level]:>16.1%}   {below[level]:>14.1%}   "
            f"{band_text}   {'pass' if passes[level] else 'FAIL':>7}"
        )
    if calibration_metric == "ad":
        pit_label = "corrected" if ad_result["pit_is_recalibrated"] else "raw/unavailable"
        print(f"    AD on {pit_label} PIT: A^2={ad_result['statistic']}, "
              f"p={ad_result['p_value']}, alpha={ad_alpha:g}; "
              f"{ad_bootstrap_replicates} bootstrap replicates, seed={ad_bootstrap_seed}")
    print(f"    acceptance metric: {calibration_metric}")
    print(f"    verdict: {verdict}")
    print(
        f"    raw-model PIT mean {result['pit_mean']:.3f} (ideal 0.500)   "
        f"var {result['pit_var']:.4f} (ideal {result['pit_var_ideal']:.4f})   "
        f"-> {result['pit_shape']}"
    )

    # Text histogram. The | marks the height a calibrated model would reach.
    uniform_share = 1.0 / PIT_BINS
    uniform_width = 20  # characters a calibrated bin fills
    print(f"    raw-model PIT histogram (| = uniform, {uniform_share:.0%} per bin)")
    for i, share in enumerate(shares):
        n_hash = int(round(share / uniform_share * uniform_width))
        cells = ["#" if j < n_hash else " " for j in range(max(n_hash, uniform_width + 1))]
        cells[uniform_width] = "|"
        print(f"    {edges[i]:.1f}-{edges[i + 1]:.1f} {share:>6.1%} {''.join(cells).rstrip()}")

    if plot_path is not None:
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(edges[:-1], shares / uniform_share, width=1 / PIT_BINS, align="edge",
               edgecolor="white", color="#4c72b0")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("density (1 = calibrated)")
        ax.set_title(f"{label}: raw-model PIT", fontsize=10)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"    PIT histogram saved: {plot_path}")

