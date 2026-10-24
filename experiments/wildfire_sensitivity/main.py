"""Run counterfactual sensitivity screening, from clean-data split to classification.

For each candidate: fit on training data, select a validation checkpoint, fit
recalibration on its own rows, check calibration on clean control, then compare
wildfire coverage. Until-stable mode retries remaining candidates after the
conditioning set grows. See README.md for the code map and interpretation.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import DEFAULT_MAT_PATH, slugify
from experiments.calibration import (
    AD_ALPHA, AD_BOOTSTRAP_REPLICATES, AD_BOOTSTRAP_SEED, AD_MIN_BLOCKS,
    validate_ad_settings, validate_evaluation_targets, validate_evaluation_outputs,
)
from experiments.wildfire_arms import load_wildfire_arms
from experiments.wildfire_sensitivity.control import split_screening_data
# Keep the established helper imports available through this entry point too.
from experiments.wildfire_sensitivity.counterfactual import (
    LSTMEngressionConfig, load_lstm_engressor_checkpoint,
    train_counterfactual, filter_dataset_columns, recalibrate_counterfactual,
)
from experiments.wildfire_sensitivity.evaluation import interval_coverage, print_calibration
from experiments.wildfire_sensitivity.settings import (
    DEVICE, SWEEP_SEED, PREDICTION_SAMPLES, TRAINING_BATCH_SIZE, EVALUATION_BATCH_SIZE,
    SEQ_STRIDE, HOLDOUT_FRACTION, HOLDOUT_BLOCK_DAYS, VALIDATION_FRACTION,
    RECALIBRATION_FRACTION, SENSITIVITY_MARGIN, CALIBRATION_LEVELS,
    CALIBRATION_REQUIRED_LEVELS, QUANTILE_LEVELS, PIT_BINS, CALIBRATION_TOLERANCE_SE,
    DEFAULT_CHECKPOINT_DIR, ALL_COVARIATE_NAMES, INITIAL_INSENSITIVE_COVARIATES,
)


def timestamp() -> str:
    """Return a concise local timestamp for the run log."""

    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _print_candidate_finished(started_after):
    elapsed_minutes = (perf_counter() - started_after) / 60
    print(f"  finished: {timestamp()} ({elapsed_minutes:.1f} min)")


def _json_ready(value):
    """Strict JSON for diagnostics, including invalid checks (null, never NaN)."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(screening_mode: str, mat_path: str, checkpoint_dir: Path,
         evaluation_scope: str = "trajectory", candidate_order_seed: Optional[int] = None,
         calibration_metric: str = "coverage", ad_alpha: float = AD_ALPHA,
         ad_bootstrap_replicates: int = AD_BOOTSTRAP_REPLICATES,
         ad_bootstrap_seed: int = AD_BOOTSTRAP_SEED) -> None:
    if candidate_order_seed is not None and candidate_order_seed < 0:
        raise ValueError("candidate order seed must be nonnegative")
    if evaluation_scope not in ("trajectory", "arrival"):
        raise ValueError(f"unknown evaluation scope: {evaluation_scope!r}")
    if calibration_metric not in ("coverage", "ad"):
        raise ValueError(f"unknown calibration metric: {calibration_metric!r}")
    validate_ad_settings(ad_alpha, ad_bootstrap_replicates, ad_bootstrap_seed)
    calibration_settings = {
        "metric": calibration_metric,
        "coverage_required_levels": CALIBRATION_REQUIRED_LEVELS,
        "coverage_tolerance_se": CALIBRATION_TOLERANCE_SE,
        "block_days": HOLDOUT_BLOCK_DAYS,
        "ad_alpha": ad_alpha,
        "ad_bootstrap_replicates": ad_bootstrap_replicates,
        "ad_bootstrap_seed": ad_bootstrap_seed,
        "ad_min_blocks": AD_MIN_BLOCKS,
    }
    print(f"Calibration settings: {json.dumps(calibration_settings)}", flush=True)
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
    if calibration_metric == "ad":
        seeds["ad_bootstrap"] = ad_bootstrap_seed
        seed_usage["ad_bootstrap"] = "separate local NumPy generator; same seed for each check"
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

    # Production split: control first, recalibration second, validation third.
    # Experimental spaced-point splits live in split_feasibility.py only.
    window_hours = dataset.x.shape[1] * SEQ_STRIDE
    splits = split_screening_data(
        clean_index, wildfire_index, dataset.time, window_hours=window_hours,
        control_fraction=HOLDOUT_FRACTION, recalibration_fraction=RECALIBRATION_FRACTION,
        validation_fraction=VALIDATION_FRACTION, block_days=HOLDOUT_BLOCK_DAYS,
        seeds=seeds,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        checkpoint_dir / "screening_split.npz",
        **splits,
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
        f"{len(splits['train'])} train / {len(splits['validation'])} validation / "
        f"{len(splits['recalibration'])} recalibration / {len(splits['control'])} control "
        f"({HOLDOUT_BLOCK_DAYS:g}-day blocks, {window_hours:g}h guard gap)"
    )
    print("  early stopping monitors: held-out validation energy loss")
    print(f"  month-matched clean recalibration rows: {len(splits['recalibration'])}")
    print(f"  month-matched clean control rows: {len(splits['control'])}")
    print(f"  wildfire evaluation rows: {len(wildfire_index)}")
    print(f"  predictive samples per interval: {PREDICTION_SAMPLES}")
    print(f"  sensitivity margin: {SENSITIVITY_MARGIN:.1%} below control coverage")

    feature_to_index = {
        name: index for index, name in enumerate(dataset.feature_names)
    }

    # Newly insensitive candidates join these expert-provided inputs in order.
    insensitive_covariates = list(INITIAL_INSENSITIVE_COVARIATES)
    remaining_covariates = [
        candidate for candidate in ALL_COVARIATE_NAMES if candidate not in insensitive_covariates
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
        "calibration_settings": calibration_settings,
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

        for candidate_number, candidate in enumerate(remaining_covariates, start=1):
            candidate_started_at = timestamp()
            candidate_started_after = perf_counter()
            print(
                f"\n[{candidate_number}/{len(remaining_covariates)}] "
                f"candidate: {candidate}"
            )
            print(f"  started: {candidate_started_at}")
            input_column_idxs = [
                feature_to_index[name] for name in insensitive_covariates
            ]
            candidate_column_idx = feature_to_index[candidate]
            train_x, train_y = filter_dataset_columns(
                dataset.x[splits['train']],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            val_x, val_y = filter_dataset_columns(
                dataset.x[splits['validation']],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            validate_evaluation_targets(train_x, train_y, evaluation_scope)
            validate_evaluation_targets(val_x, val_y, evaluation_scope)
            print(f"  inputs: {insensitive_covariates}")
            print(f"  train X: {train_x.shape}  train y: {train_y.shape}")
            print(f"  val X: {val_x.shape}  val y: {val_y.shape}")

            # 1. Fit and reload the best validation checkpoint.
            print("  training on clean trajectories...")
            # Keep each round's checkpoint and quantile mapping together.
            round_dir = checkpoint_dir / f"round_{screening_round}"
            model = train_counterfactual(
                train_x, train_y, candidate, round_dir, val_x, val_y
            )
            # 2. Fit the probability correction on separate recalibration rows.
            recalibration_x, recalibration_y = filter_dataset_columns(
                dataset.x[splits['recalibration']], input_column_idxs, candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            model = recalibrate_counterfactual(
                model, recalibration_x, recalibration_y,
                round_dir / f"{slugify(candidate)}_quantile_levels.json",
                evaluation_scope=evaluation_scope,
            )

            # 3. Accept/reject the corrected model on clean control rows.
            control_x, control_y = filter_dataset_columns(
                dataset.x[splits['control']],
                input_column_idxs,
                candidate_column_idx,
                evaluation_scope=evaluation_scope,
            )
            calibration = print_calibration(
                model,
                control_x,
                control_y,
                f"{candidate} control",
                dataset.time[splits['control']],
                plot_path=round_dir / f"{slugify(candidate)}_raw_pit.png",
                evaluation_scope=evaluation_scope,
                calibration_metric=calibration_metric,
                ad_alpha=ad_alpha,
                ad_bootstrap_replicates=ad_bootstrap_replicates,
                ad_bootstrap_seed=ad_bootstrap_seed,
            )
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / f"{slugify(candidate)}_calibration.json").write_text(json.dumps(_json_ready({
                "candidate": candidate, "screening_round": screening_round,
                "conditioning_covariates": list(insensitive_covariates),
                "evaluation_scope": evaluation_scope,
                "calibration_settings": calibration_settings,
                "seeds": seeds, "seed_usage": seed_usage,
                "prediction_samples": PREDICTION_SAMPLES,
                "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                "metrics": calibration,
            }), indent=2, allow_nan=False) + "\n")
            if not calibration["calibrated"]:
                classifications[candidate] = "undetermined"
                next_remaining_covariates.append(candidate)
                print("  classification: undetermined; kept out of the input set")
                _print_candidate_finished(candidate_started_after)
                continue

            # Reuse the control coverage measured alongside the selected gate, avoiding a
            # second Monte-Carlo draw with slightly different interval coverage.
            control_coverage = calibration["central"][0.95]

            # 4. For accepted models, compare wildfire and clean 95% coverage.
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
                classifications[candidate] = "insensitive"
                insensitive_covariates.append(candidate)
                moved_this_round += 1
                print("  classification: insensitive; added to the input set")
            else:
                classifications[candidate] = "sensitive"
                next_remaining_covariates.append(candidate)
                print("  classification: sensitive; kept out of the input set")
            _print_candidate_finished(candidate_started_after)

        if screening_mode == "single_pass" or moved_this_round == 0:
            break
        remaining_covariates = next_remaining_covariates
        screening_round += 1

    print(insensitive_covariates)
    for classification in ("sensitive", "undetermined"):
        candidates = [candidate for candidate, result in classifications.items() if result == classification]
        print(f"  {classification}: {candidates}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-metric", choices=("coverage", "ad"), default="coverage",
                        help="Clean-control acceptance gate: 90/95 coverage or corrected-PIT AD.")
    parser.add_argument("--ad-alpha", type=float, default=AD_ALPHA,
                        help="Reject calibration when the AD bootstrap p-value is below this level.")
    parser.add_argument("--ad-bootstrap-replicates", type=int, default=AD_BOOTSTRAP_REPLICATES)
    parser.add_argument("--ad-bootstrap-seed", type=int, default=AD_BOOTSTRAP_SEED)
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
    return parser


if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))
