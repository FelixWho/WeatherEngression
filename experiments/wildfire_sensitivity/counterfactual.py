"""Build candidate targets, fit the best checkpoint, then fit its quantile map.

Inputs always contain the full trajectory. The target is either its final value
(arrival) or all timesteps (trajectory). Validation selects the checkpoint;
separate recalibration observations fit the pooled probability correction.
"""
from pathlib import Path

import numpy as np
import torch

from data_generation.ena_weather import slugify
from engression_modifications.lstm import (
    LSTMEngressionConfig, LSTMEngressor, fit_lstm_engression,
    load_lstm_engressor_checkpoint,
)
from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.calibration import validate_evaluation_targets, validate_evaluation_outputs
from experiments.pipeline import set_reproducible_seeds
from experiments.real_metrics import pit_values
from experiments.wildfire_sensitivity.settings import (
    DEVICE, SWEEP_SEED, TRAINING_BATCH_SIZE, EVALUATION_BATCH_SIZE,
    PREDICTION_SAMPLES, QUANTILE_LEVELS,
)


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
    # Validation energy loss controls early stopping and checkpoint selection.
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
