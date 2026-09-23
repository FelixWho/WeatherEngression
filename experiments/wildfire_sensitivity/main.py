"""
Start with a set of wildfire-insensitive covariates, and one by one test the sensitive covariates
to see if they should be brought into the insensitive covariate set.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import DEFAULT_MAT_PATH, slugify
from engression_modifications.lstm import (
    LSTMEngressionConfig,
    LSTMEngressor,
    fit_lstm_engression,
    load_lstm_engressor_checkpoint,
)
from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.pipeline import set_reproducible_seeds
from experiments.real_metrics import pit_calibration_metrics, pit_values
from experiments.t_learner.train import DEFAULT_CKPT_ROOT
from experiments.wildfire_arms import load_wildfire_arms
from experiments.wildfire_sensitivity.control import (
    month_matched_subsample,
    split_clean_holdout,
    time_blocks,
)


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SWEEP_SEED = 2026
PREDICTION_SAMPLES = 400
# The PIT sweep used 256.  MPS retains LSTM backward activations in unified
# memory, so its 20 GB limit cannot accommodate this global-latent model at
# that batch size.
TRAINING_BATCH_SIZE = 64 if DEVICE == "mps" else 256
# Sampling all 6,786 wildfire trajectories at once creates 400 full
# conditional trajectory draws in MPS memory.  Evaluate in small batches.
EVALUATION_BATCH_SIZE = 64 if DEVICE == "mps" else 1_024

SEQ_STRIDE = 1
# Fraction of clean arrival-time blocks held back as the control group.
HOLDOUT_FRACTION = 0.25
HOLDOUT_BLOCK_DAYS = 30.0
# Fraction of the remaining clean training blocks held back for early stopping.
# Separate from the control and from the rows used to fit quantile corrections.
VALIDATION_FRACTION = 0.15
# Separate clean rows for fitting the pooled quantile correction.
RECALIBRATION_FRACTION = 0.20
# A candidate counts as sensitive when its wildfire coverage falls this far
# below the clean control's. The margin is a judgment call rather than a
# calibrated significance test; uncertainty in the coverage difference is not tested.
SENSITIVITY_MARGIN = 0.02

# Central-interval levels reported by print_calibration, and PIT histogram bins.
CALIBRATION_LEVELS = (0.50, 0.75, 0.90, 0.95)
CALIBRATION_REQUIRED_LEVELS = (0.90, 0.95)
QUANTILE_LEVELS = tuple(sorted({
    round(q, 12)
    for level in CALIBRATION_LEVELS
    for q in ((1 - level) / 2, (1 + level) / 2, level)
}))
PIT_BINS = 10
# A level passes when its coverage sits within this many standard errors of the
# level itself. Use 3-SE bands; only the central 90% and 95%
# levels determine acceptance. The other levels remain descriptive diagnostics.
CALIBRATION_TOLERANCE_SE = 3.0

# Checkpoints go to storage3 alongside every other run; /home has a small quota.
DEFAULT_CHECKPOINT_DIR = DEFAULT_CKPT_ROOT / "wildfire_sensitivity"


ALL_COVARIATE_NAMES = (
    "P",
    "LAND",
    "PBLH^(1/5)",
    "CFLOW",
    "CFMID",
    "SWGDN",
    "SLP",
    "LWP^(1/5)",
    "WS",
    "LTS",
    "TS",
    "DUST^(1/5)",
    "OMEGA",
    "LWC^(1/5)",
    "RH",
    "T",
    "EPV",
    "CO^(1/5)",
    "PREC^(1/5)",
    "CHL^(1/2)",
    "DMS^(1/2)",
    "SO2EM^(1/5)",
)


def timestamp() -> str:
    """Return a concise local timestamp for the run log."""

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def train_counterfactual(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidate: str,
    checkpoint_dir: Path,
    val_x: np.ndarray,
    val_y: np.ndarray,
) -> LSTMEngressor:
    # Give every candidate the same initialization and minibatch-shuffle seed.
    set_reproducible_seeds(SWEEP_SEED)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_slug = slugify(candidate)

    config = LSTMEngressionConfig(
        num_layer=5,
        global_latent_noise=True,  # Selects the global-latent generator.
        lr=0.01,
        num_epochs=100,
        batch_size=TRAINING_BATCH_SIZE,
        device=DEVICE,
        early_stop_patience=12,
        checkpoint_every_nepoch=25,
        checkpoint_path=str(checkpoint_dir / f"{candidate_slug}_latest.pt"),
        checkpoint_best_path=str(checkpoint_dir / f"{candidate_slug}_best.pt"),
    )
    # Early stopping watches the held-out energy loss. Stopping on the training
    # loss rewards sharpness and leaves the model under-dispersed, which is what
    # made most of the previous run's insensitive calls uninformative.
    fit_lstm_engression(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        config,
        x_val=torch.from_numpy(val_x),
        y_val=torch.from_numpy(val_y),
    )
    # The fit returns the final epoch; evaluation must use the epoch with the
    # lowest validation energy loss, saved separately during training.
    print(f"  loading best validation checkpoint: {config.checkpoint_best_path}")
    return load_lstm_engressor_checkpoint(config.checkpoint_best_path, device=DEVICE)


def filter_dataset_columns(
    data: np.ndarray,
    feature_idxs: list[int],
    target_idx: int,
    evaluation_scope: str = "trajectory",
) -> tuple[np.ndarray, np.ndarray]:
    """Keep full input histories; arrival mode has one scalar target per row."""
    if target_idx in feature_idxs:
        raise ValueError("the candidate target must not be included in its conditioning inputs")
    if evaluation_scope == "arrival":
        return data[:, :, feature_idxs], data[:, -1:, target_idx]
    if evaluation_scope != "trajectory":
        raise ValueError(f"unknown evaluation scope: {evaluation_scope!r}")
    return data[:, :, feature_idxs], data[:, :, target_idx]


def validate_evaluation_targets(x, y, evaluation_scope):
    """Reject scope/shape mismatches before sampling, including accidental broadcasting."""
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
    """Require one predictive distribution per target; never slice a wrong model to fit."""
    if samples.ndim != 3 or samples.shape[:2] != y.shape or samples.shape[2] < 2:
        raise ValueError(f"sample shape {samples.shape} does not match target shape {y.shape}")
    # All callers sample on CPU. Reject invalid forecasts rather than counting
    # NaN comparisons as ordinary coverage misses or converting them into PITs.
    values = samples.numpy() if isinstance(samples, torch.Tensor) else samples
    if not np.isfinite(values).all():
        raise ValueError("predictive samples must be finite")
    return samples, y


def recalibrate_counterfactual(model, x, y, mapping_path, evaluation_scope="trajectory"):
    """Fit one pooled quantile mapping using only clean recalibration rows."""
    if not len(x):
        raise ValueError("recalibration needs nonempty clean data")
    validate_evaluation_targets(x, y, evaluation_scope)
    print(f"  fitting pooled quantile correction on {len(x)} clean trajectories...")
    pit_batches = []
    for start in range(0, len(x), EVALUATION_BATCH_SIZE):
        stop = min(start + EVALUATION_BATCH_SIZE, len(x))
        samples = model.sample(
            torch.from_numpy(x[start:stop]), sample_size=PREDICTION_SAMPLES
        ).cpu().numpy()
        samples, observed = validate_evaluation_outputs(samples, y[start:stop])
        pit_batches.append(pit_values(
            samples.reshape(-1, samples.shape[-1]), observed.reshape(-1),
            seed=SWEEP_SEED + start,
        ))
    wrapped = RecalibratedEngressor(model).fit_from_pit(
        np.concatenate(pit_batches), QUANTILE_LEVELS
    )
    wrapped.save(mapping_path)
    print("  nominal quantile -> adjusted base-model quantile")
    for q, adjusted in sorted(wrapped.quantile_levels.items()):
        print(f"    {q:7.3%} -> {adjusted:7.3%}")
    print(f"  quantile mapping saved: {mapping_path}")
    return wrapped


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
) -> dict[str, object]:
    """Print calibration metrics for a trained counterfactual model on (x, y).

    Central and one-sided coverage use the wrapper's adjusted quantile boundaries,
    exactly as interval_coverage does. The PIT histogram and moments describe the
    RAW base-model samples, since the wrapper recalibrates quantiles only.

    Whether a level passes is judged against how much its coverage moves by chance,
    not against a hand-picked tolerance. ``time_days`` gives each row's arrival
    time; rows are grouped into month-long blocks. For B blocks with sizes n_b,
    block coverages p_b, and pooled coverage p, the variance estimate is
    B / (B - 1) * sum(n_b**2 * (p_b - p)**2) / sum(n_b)**2.
    Sizes count evaluated values (one per row for arrival), and every block is included, matching
    the pooled coverage. This allows dependence within blocks and assumes blocks
    are approximately independent. A level passes when its coverage
    lands within ``CALIBRATION_TOLERANCE_SE`` standard errors of the level. Blocks
    rather than rows are the unit because trajectories an hour apart share almost
    all of their history and carry nowhere near a full row of information.

    A model whose coverage swings between blocks widens its own tolerance and so
    passes more easily. The printed standard error makes that visible: a tolerance
    far wider than the other candidates' is itself a reason to distrust the model.

    Like the screening's coverage, this pools the selected timesteps. The
    histogram is printed as text so it shows up in the SLURM log, and also saved as
    a PNG when ``plot_path`` is given. Acceptance requires passing both levels in
    ``CALIBRATION_REQUIRED_LEVELS`` with finite uncertainty estimates; other
    levels are diagnostic only. Returns the numbers for reuse.
    """

    validate_evaluation_targets(x, y, evaluation_scope)
    if np.asarray(time_days).shape != (len(x),) or not np.isfinite(time_days).all():
        raise ValueError("calibration requires one finite arrival time per row")
    if not isinstance(model, RecalibratedEngressor):
        model = RecalibratedEngressor(model)
    if model.device.type == "mps":
        torch.mps.empty_cache()

    pit_batches = []
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
        n_rows, n_steps, n_samples = samples.shape
        pit_batches.append(
            pit_values(
                samples.reshape(n_rows * n_steps, n_samples),
                observed.reshape(n_rows * n_steps),
                seed=SWEEP_SEED + start,
            )
        )
    pit = np.concatenate(pit_batches)
    pit_by_row = pit.reshape(len(x), -1)          # (trajectories, timesteps)

    inside_by_level = {level: np.concatenate(batches) for level, batches in inside_batches.items()}
    central = {level: float(inside.mean()) for level, inside in inside_by_level.items()}
    below = {level: count / pit.size for level, count in below_counts.items()}

    # Cluster standard error for the pooled coverage, including every block.
    block_of_row = time_blocks(np.asarray(time_days), HOLDOUT_BLOCK_DAYS)
    block_ids, block_inverse, block_counts = np.unique(
        block_of_row, return_inverse=True, return_counts=True
    )
    n_blocks = len(block_ids)
    block_sizes = block_counts.astype(np.float64) * pit_by_row.shape[1]
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
    summary = pit_calibration_metrics(pit, n_bins=PIT_BINS)
    counts, edges = np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0))
    shares = counts / pit.size

    print(
        f"  [{label}] calibration over {pit.size:,} values "
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
    if not sufficient_blocks:
        verdict = "insufficient blocks -> undetermined"
    elif calibrated:
        verdict = "accepted: central 90% and 95% coverage pass"
    else:
        verdict = "central 90% or 95% coverage failed -> undetermined"
    print(f"    verdict: {verdict}")
    print(
        f"    raw-model PIT mean {summary['pit_mean']:.3f} (ideal 0.500)   "
        f"var {summary['pit_var']:.4f} (ideal {summary['pit_var_ideal']:.4f})   "
        f"-> {summary['pit_shape']}"
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

    return {
        "central": central,
        "below": below,
        "standard_error": standard_error,
        "tolerance": tolerance,
        "passes": passes,
        "calibrated": calibrated,
        "n_blocks": n_blocks,
        "pit_counts": counts,
        "pit_is_recalibrated": False,
        "evaluation_scope": evaluation_scope,
        **summary,
    }


def main(screening_mode: str, mat_path: str, checkpoint_dir: Path,
         evaluation_scope: str = "trajectory", candidate_order_seed: Optional[int] = None) -> None:
    if candidate_order_seed is not None and candidate_order_seed < 0:
        raise ValueError("candidate order seed must be nonnegative")
    if evaluation_scope not in ("trajectory", "arrival"):
        raise ValueError(f"unknown evaluation scope: {evaluation_scope!r}")
    seeds = {
        "sweep": SWEEP_SEED,
        "dataset": SWEEP_SEED,
        # load_wildfire_arms forwards seed + 17 to select_real_split.
        "paper_split": SWEEP_SEED + 17,
        "training_each_candidate": SWEEP_SEED,
        "control_split": SWEEP_SEED,
        "control_month_matching": SWEEP_SEED,
        "recalibration_split": SWEEP_SEED + 202,
        "recalibration_month_matching": SWEEP_SEED + 202,
        "validation_split": SWEEP_SEED + 101,
        "pit_batch_base": SWEEP_SEED,
        "candidate_order": candidate_order_seed,
    }
    seed_usage = {
        "candidate_order": "null means original order; otherwise shuffle once with a local NumPy generator",
        "training": "NumPy and Torch are reset to training_each_candidate before each fit in every round",
        "prediction_draws": "continue the Torch RNG stream from training; no separate sampling seed is set",
        "pit_batches": "pit_batch_base + zero-based batch start row, separately for recalibration and control",
    }
    # Print before data loading so even an interrupted run records its seeds.
    print("Random seeds:", flush=True)
    for name, seed in seeds.items():
        print(f"  {name}: {seed}")
    for name, usage in seed_usage.items():
        print(f"  {name} seed usage: {usage}")
    if evaluation_scope == "arrival":
        # Keep arrival mappings and results separate from trajectory artifacts.
        checkpoint_dir = checkpoint_dir / "arrival"
    dataset, arms = load_wildfire_arms(
        mat_path=mat_path,
        target="ccn",
        log_ccn=True,
        split="paper",
        seq_stride=SEQ_STRIDE,
        max_samples=None,
        train_size=None,
        test_size=None,
        seed=seeds["dataset"],
        wildfire_flag="BB_criterion1",
    )
    all_features = dataset.feature_names
    assert set(ALL_COVARIATE_NAMES) == set(all_features)

    # Every candidate is screened against the same wildfire rows.
    wildfire_index = np.concatenate([arms.train_wildfire, arms.test_wildfire])

    # The paper split belongs to the CCN forecasting experiment. Screening uses
    # all clean rows, then creates training, validation, recalibration, and control sets.
    clean_index = np.sort(np.concatenate([arms.train_clean, arms.test_clean]))

    # Hold back part of the clean data as a control. Coverage on it is what
    # wildfire coverage is judged against, so model under-dispersion and the
    # Jun-Sep seasonal shift move both numbers together instead of masquerading
    # as a wildfire effect.
    window_hours = dataset.x.shape[1] * SEQ_STRIDE
    clean_fit_index, clean_holdout_index = split_clean_holdout(
        clean_index,
        dataset.time,
        window_hours=window_hours,
        holdout_fraction=HOLDOUT_FRACTION,
        block_days=HOLDOUT_BLOCK_DAYS,
        seed=seeds["control_split"],
    )
    control_index = month_matched_subsample(
        clean_holdout_index,
        wildfire_index,
        dataset.time,
        seed=seeds["control_month_matching"],
    )

    # Fit quantile corrections on their own clean holdout, never on the control
    # used to accept models or on the validation rows that select checkpoints.
    clean_fit_index, clean_recalibration_pool = split_clean_holdout(
        clean_fit_index,
        dataset.time,
        window_hours=window_hours,
        holdout_fraction=RECALIBRATION_FRACTION,
        block_days=HOLDOUT_BLOCK_DAYS,
        seed=seeds["recalibration_split"],
    )
    recalibration_index = month_matched_subsample(
        clean_recalibration_pool, wildfire_index, dataset.time, seed=seeds["recalibration_month_matching"]
    )

    # Split the remaining clean rows again into train and validation. Same
    # time-blocked splitter, different seed, so validation is separated in time
    # from training rather than sampled at random out of overlapping trajectories.
    clean_train_index, clean_val_index = split_clean_holdout(
        clean_fit_index,
        dataset.time,
        window_hours=window_hours,
        holdout_fraction=VALIDATION_FRACTION,
        block_days=HOLDOUT_BLOCK_DAYS,
        seed=seeds["validation_split"],
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_dir / "screening_split.npz",
        train=clean_train_index, validation=clean_val_index,
        recalibration=recalibration_index, control=control_index, wildfire=wildfire_index,
        evaluation_scope=evaluation_scope,
        training_scope=evaluation_scope,
    )

    print("Wildfire-sensitivity counterfactual screening")
    print(f"  device: {DEVICE}")
    print(f"  training batch size: {TRAINING_BATCH_SIZE}")
    print(f"  evaluation batch size: {EVALUATION_BATCH_SIZE}")
    print(f"  screening mode: {screening_mode}")
    print(f"  evaluation scope: {evaluation_scope} (training, validation, recalibration, control, wildfire)")
    print(f"  data: {mat_path}")
    print(f"  checkpoints: {checkpoint_dir}")
    print(
        f"  clean pool: {len(arms.train_clean)} paper-train + "
        f"{len(arms.test_clean)} paper-test = {len(clean_index)} rows"
    )
    print(
        f"  screening split: "
        f"{len(clean_train_index)} train / {len(clean_val_index)} validation / "
        f"{len(recalibration_index)} recalibration / {len(control_index)} control "
        f"({HOLDOUT_BLOCK_DAYS:g}-day blocks, {window_hours:g}h guard gap)"
    )
    print("  early stopping monitors: held-out validation energy loss")
    print(f"  month-matched clean recalibration rows: {len(recalibration_index)}")
    print(f"  month-matched clean control rows: {len(control_index)}")
    print(f"  wildfire evaluation rows: {len(wildfire_index)}")
    print(f"  predictive samples per interval: {PREDICTION_SAMPLES}")
    print(f"  sensitivity margin: {SENSITIVITY_MARGIN:.1%} below control coverage")

    feature_to_index = {
        name: index for index, name in enumerate(dataset.feature_names)
    }

    # Start with insensitive covariates, defer to Shengqian
    insensitive_covariates = [
        "P",
        "LAND",
        "SLP",
        "WS",
        "TS",
        "DUST^(1/5)",
        "EPV",
        "CHL^(1/2)",
        "DMS^(1/2)",
        "SO2EM^(1/5)",
    ]
    remaining_covariates = [
        c for c in ALL_COVARIATE_NAMES if c not in insensitive_covariates
    ]
    # A separate local RNG changes only candidate order. Shuffle once; later
    # rounds retain the relative order of the candidates still under consideration.
    if candidate_order_seed is not None:
        np.random.default_rng(candidate_order_seed).shuffle(remaining_covariates)
    print(f"  candidate order seed: {candidate_order_seed} (None = original order)")
    print(f"  initial candidate order: {remaining_covariates}")
    (checkpoint_dir / "screening_order.json").write_text(json.dumps({
        "sweep_seed": SWEEP_SEED,
        "candidate_order_seed": candidate_order_seed,
        "evaluation_scope": evaluation_scope,
        "initial_candidate_order": remaining_covariates,
        "seeds": seeds,
        "seed_usage": seed_usage,
        "screening_mode": screening_mode,
        "training_batch_size": TRAINING_BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "prediction_samples": PREDICTION_SAMPLES,
    }, indent=2) + "\n")

    screening_round = 1
    classifications = {}
    while remaining_covariates:
        print(f"\nScreening round {screening_round}")
        next_remaining_covariates = []
        moved_this_round = 0

        for candidate_number, c in enumerate(remaining_covariates, start=1):
            candidate_started_at = timestamp()
            candidate_started_after = perf_counter()
            print(
                f"\n[{candidate_number}/{len(remaining_covariates)}] "
                f"candidate: {c}"
            )
            print(f"  started: {candidate_started_at}")
            input_column_idxs = [
                feature_to_index[name] for name in insensitive_covariates
            ]
            candidate_column_idx = feature_to_index[c]
            train_x, train_y = filter_dataset_columns(
                dataset.x[clean_train_index],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            val_x, val_y = filter_dataset_columns(
                dataset.x[clean_val_index],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            validate_evaluation_targets(train_x, train_y, evaluation_scope)
            validate_evaluation_targets(val_x, val_y, evaluation_scope)
            print(f"  inputs: {insensitive_covariates}")
            print(f"  train X: {train_x.shape}  train y: {train_y.shape}")
            print(f"  val X: {val_x.shape}  val y: {val_y.shape}")

            # Train a counterfactual model M' = f(M, epsilon)
            print("  training on clean trajectories...")
            # Keep each round's checkpoint and quantile mapping together.
            round_dir = checkpoint_dir / f"round_{screening_round}"
            model = train_counterfactual(
                train_x, train_y, c, round_dir, val_x, val_y
            )
            recalibration_x, recalibration_y = filter_dataset_columns(
                dataset.x[recalibration_index], input_column_idxs, candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            model = recalibrate_counterfactual(
                model, recalibration_x, recalibration_y,
                round_dir / f"{slugify(c)}_quantile_levels.json",
                evaluation_scope=evaluation_scope,
            )

            # First require acceptable calibration on held-out clean trajectories.
            control_x, control_y = filter_dataset_columns(
                dataset.x[control_index],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            calibration = print_calibration(
                model,
                control_x,
                control_y,
                f"{c} control",
                dataset.time[control_index],
                plot_path=round_dir / f"{slugify(c)}_raw_pit.png",
                evaluation_scope=evaluation_scope,
            )
            if not calibration["calibrated"]:
                classifications[c] = "undetermined"
                next_remaining_covariates.append(c)
                print("  classification: undetermined; kept out of the input set")
                elapsed_minutes = (perf_counter() - candidate_started_after) / 60
                print(f"  finished: {timestamp()} ({elapsed_minutes:.1f} min)")
                continue

            # Use the exact control coverage that passed the check, avoiding a
            # second Monte-Carlo draw with slightly different interval coverage.
            control_coverage = calibration["central"][0.95]

            # Coverage on wildfire trajectories: the same measurement under smoke.
            wildfire_x, wildfire_y = filter_dataset_columns(
                dataset.x[wildfire_index],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            wildfire_coverage = interval_coverage(
                model, wildfire_x, wildfire_y, "wildfire", evaluation_scope=evaluation_scope
            )

            # Sensitive only if wildfire values sit outside the counterfactual
            # noticeably more often than clean held-out values do.
            coverage_drop = control_coverage - wildfire_coverage
            print(
                f"  coverage drop vs control: {coverage_drop:+.1%} "
                f"(threshold {SENSITIVITY_MARGIN:.1%})"
            )
            # Do not turn an exactly 2-point drop into >2 through float roundoff.
            sensitive = coverage_drop > SENSITIVITY_MARGIN + 1e-12
            if not sensitive:
                classifications[c] = "insensitive"
                insensitive_covariates.append(c)
                moved_this_round += 1
                print("  classification: insensitive; added to the input set")
            else:
                classifications[c] = "sensitive"
                next_remaining_covariates.append(c)
                print("  classification: sensitive; kept out of the input set")
            elapsed_minutes = (perf_counter() - candidate_started_after) / 60
            print(f"  finished: {timestamp()} ({elapsed_minutes:.1f} min)")

        if screening_mode == "single_pass" or moved_this_round == 0:
            break
        remaining_covariates = next_remaining_covariates
        screening_round += 1

    print(insensitive_covariates)
    for classification in ("sensitive", "undetermined"):
        candidates = [c for c, result in classifications.items() if result == classification]
        print(f"  {classification}: {candidates}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-order-seed", type=int, default=None,
        help="Shuffle candidates once using a separate seed; omit to retain original order. "
        "Does not change training or data-split seeds.",
    )
    parser.add_argument(
        "--evaluation-scope", choices=("trajectory", "arrival"), default="trajectory",
        help="Target timesteps used for training, validation, recalibration, and coverage. "
        "arrival predicts only the final value from the full input trajectory.",
    )
    parser.add_argument(
        "--screening-mode",
        choices=("single_pass", "until_stable"),
        default="single_pass",
        help="single_pass screens each candidate once; until_stable re-tests "
        "sensitive and undetermined covariates until none move into the input set.",
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        default=DEFAULT_MAT_PATH,
        help="ENA weather .mat file (default: the copy on storage3).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Where per-candidate checkpoints are written.",
    )
    main(**vars(parser.parse_args()))
