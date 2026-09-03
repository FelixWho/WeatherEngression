"""
Start with a set of wildfire-insensitive covariates, and one by one test the sensitive covariates
to see if they should be brought into the insensitive covariate set.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import slugify
from engression_modifications.lstm import (
    LSTMEngressionConfig,
    LSTMEngressor,
    fit_lstm_engression,
)
from experiments.pipeline import set_reproducible_seeds
from experiments.wildfire_arms import load_wildfire_arms


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
) -> LSTMEngressor:
    # Give every candidate the same initialization and minibatch-shuffle seed.
    set_reproducible_seeds(SWEEP_SEED)

    checkpoint_dir = Path("weather_checkpoints/wildfire_sensitivity")
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
    return fit_lstm_engression(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
        config,
    )


def filter_dataset_columns(
    data: np.ndarray,
    feature_idxs: list[int],
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    return data[:, :, feature_idxs], data[:, :, target_idx]


def is_sensitive(
    model: LSTMEngressor,
    wildfire_x: np.ndarray,
    wildfire_y: np.ndarray,
    alpha_threshold: float = 0.95,
) -> bool:
    """Return whether wildfire values under-cover the predicted central interval."""

    lower_quantile = (1 - alpha_threshold) / 2
    upper_quantile = 1 - lower_quantile
    if model.device.type == "mps":
        # Release cached backward activations from training before sampling.
        torch.mps.empty_cache()

    total_batches = (len(wildfire_x) + EVALUATION_BATCH_SIZE - 1) // EVALUATION_BATCH_SIZE
    print(
        f"  drawing {PREDICTION_SAMPLES} samples in {total_batches} "
        f"batches of at most {EVALUATION_BATCH_SIZE} trajectories..."
    )
    inside_count = 0
    total_count = 0
    for batch_number, start in enumerate(
        range(0, len(wildfire_x), EVALUATION_BATCH_SIZE), start=1
    ):
        stop = min(start + EVALUATION_BATCH_SIZE, len(wildfire_x))
        # Compute quantiles on CPU: torch.quantile may be unsupported on MPS.
        samples = model.sample(
            torch.from_numpy(wildfire_x[start:stop]),
            sample_size=PREDICTION_SAMPLES,
        ).cpu()
        lower = torch.quantile(samples, lower_quantile, dim=2)
        upper = torch.quantile(samples, upper_quantile, dim=2)
        observed = torch.from_numpy(wildfire_y[start:stop]).to(dtype=samples.dtype)
        inside = (observed >= lower) & (observed <= upper)
        inside_count += inside.sum().item()
        total_count += inside.numel()

        if batch_number == 1 or batch_number % 10 == 0 or batch_number == total_batches:
            print(f"    sampled batch {batch_number}/{total_batches}")

    coverage = inside_count / total_count

    print(f"{inside_count}/{total_count} inside ({coverage:.1%})")

    return coverage < alpha_threshold


def main(screening_mode: str) -> None:
    dataset, arms = load_wildfire_arms(
        mat_path="data_generation/weather_data.mat",
        target="ccn",
        log_ccn=True,
        split="paper",
        seq_stride=1,
        max_samples=None,
        train_size=None,
        test_size=None,
        seed=SWEEP_SEED,
        wildfire_flag="BB_criterion1",
    )
    all_features = dataset.feature_names
    assert set(ALL_COVARIATE_NAMES) == set(all_features)

    print("Wildfire-sensitivity counterfactual screening")
    print(f"  device: {DEVICE}")
    print(f"  training batch size: {TRAINING_BATCH_SIZE}")
    print(f"  evaluation batch size: {EVALUATION_BATCH_SIZE}")
    print(f"  screening mode: {screening_mode}")
    print(
        f"  clean training rows: {len(arms.train_clean)}  "
        f"wildfire evaluation rows: {len(arms.train_wildfire) + len(arms.test_wildfire)}"
    )
    print(f"  predictive samples per interval: {PREDICTION_SAMPLES}")

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
    # TODO: test whether results depend on covariate consideration order.
    remaining_covariates = [
        c for c in ALL_COVARIATE_NAMES if c not in insensitive_covariates
    ]

    screening_round = 1
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
                dataset.x[arms.train_clean],
                input_column_idxs,
                candidate_column_idx,
            )
            print(f"  inputs: {insensitive_covariates}")
            print(f"  train X: {train_x.shape}  train y: {train_y.shape}")

            # Train a counterfactual model M' = f(M, epsilon)
            print("  training on clean trajectories...")
            model = train_counterfactual(train_x, train_y, c)

            # Evaluate c versus counterfactual distribution
            wildfire_train_x, wildfire_train_y = filter_dataset_columns(
                dataset.x[arms.train_wildfire],
                input_column_idxs,
                candidate_column_idx,
            )
            wildfire_test_x, wildfire_test_y = filter_dataset_columns(
                dataset.x[arms.test_wildfire],
                input_column_idxs,
                candidate_column_idx,
            )
            wildfire_x = np.concatenate([wildfire_train_x, wildfire_test_x], axis=0)
            wildfire_y = np.concatenate([wildfire_train_y, wildfire_test_y], axis=0)

            # Reclassify only candidates whose wildfire values remain well-covered.
            print(f"  evaluating {wildfire_x.shape[0]} wildfire trajectories...")
            sensitive = is_sensitive(model, wildfire_x, wildfire_y)
            if not sensitive:
                insensitive_covariates.append(c)
                moved_this_round += 1
                print("  classification: insensitive; added to the input set")
            else:
                next_remaining_covariates.append(c)
                print("  classification: sensitive; kept out of the input set")
            elapsed_minutes = (perf_counter() - candidate_started_after) / 60
            print(f"  finished: {timestamp()} ({elapsed_minutes:.1f} min)")

        if screening_mode == "single_pass" or moved_this_round == 0:
            break
        remaining_covariates = next_remaining_covariates
        screening_round += 1

    print(insensitive_covariates)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screening-mode",
        choices=("single_pass", "until_stable"),
        default="single_pass",
        help="single_pass preserves the original behavior; until_stable re-tests "
        "remaining sensitive covariates until none move into the input set.",
    )
    main(**vars(parser.parse_args()))
