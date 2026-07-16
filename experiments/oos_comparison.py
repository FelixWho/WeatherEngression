r"""Compare vanilla engression performance across OOS definitions.

This script uses one generated dataset, one shared training set, and one fitted
engression model. It then evaluates the fitted model on held-out subsets
selected by each OOS definition:

1. scalar projection \(\phi(X)\);
2. marginal feature min/max range;
3. marginal feature quantile range;
4. standardized kNN distance.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile

import numpy as np
import torch


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engression_modifications import MODEL_REGISTRY, fit_engression_model
from experiments.constants import QUANTILE_KEYS, format_model_name
from experiments.data import flattened_tensors_from_dataset, tensors_from_dataset
from experiments.metrics import quantile_diagnostic_metrics
from experiments.oos import OOSConfig, compute_oos_diagnostics
from experiments.predictions import predict_quantiles
from experiments.pipeline import set_reproducible_seeds
from data_generation.generate_data import (
    BASE_X_DIMENSION,
    DEFAULT_SEASONAL_PERIOD,
    MODEL_NAMES,
    generate_supervised_dataset,
)


OOS_METHODS = ("scalar_projection", "marginal_range", "marginal_quantile", "knn_distance")
RUN_ROOT = REPO_ROOT / "runs" / "oos_comparisons"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="narx_student_t")
    parser.add_argument(
        "--engression-model",
        choices=tuple(name for name in MODEL_REGISTRY if name in {"vanilla", "regularized", "adamw", "lstm"}),
        default="vanilla",
    )
    parser.add_argument("-d", "--dimension", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--max-test-size", type=int, default=500)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--seasonal-period", type=int, default=DEFAULT_SEASONAL_PERIOD)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prediction-samples", type=int, default=800)
    parser.add_argument("--oos-quantile-low", type=float, default=0.01)
    parser.add_argument("--oos-quantile-high", type=float, default=0.99)
    parser.add_argument("--oos-knn-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--oos-knn-reference-size", type=int, default=2000)
    parser.add_argument("--oos-knn-batch-size", type=int, default=512)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to runs/oos_comparisons/<model>_<engression-model>_d<d>.",
    )
    return parser


def shell_command() -> str:
    """Return the current command in a copy-pasteable form."""

    parts = ["python", "experiments/oos_comparison.py", *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def central_phi_train_candidates(
    phi: np.ndarray,
    train_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Use the central \(\phi(X)\) block for training and tails as candidates."""

    if not 0 < train_size < len(phi):
        raise ValueError("train_size must be positive and smaller than num_samples")
    order = np.argsort(phi)
    train_start = (len(phi) - train_size) // 2
    train_end = train_start + train_size
    train_idx = order[train_start:train_end]
    candidate_idx = np.concatenate([order[:train_start], order[train_end:]])
    return train_idx, candidate_idx


def select_eval_positions(
    candidate_idx: np.ndarray,
    flags: np.ndarray,
    max_test_size: int,
) -> np.ndarray:
    """Select stable candidate positions for one OOS method."""

    positions = np.flatnonzero(flags)
    if len(positions) <= max_test_size:
        return positions[np.argsort(candidate_idx[positions])]
    sorted_positions = positions[np.argsort(candidate_idx[positions])]
    chosen = np.linspace(0, len(sorted_positions) - 1, num=max_test_size, dtype=int)
    return sorted_positions[chosen]


def fit_model(args: argparse.Namespace, x_train: torch.Tensor, y_train: torch.Tensor) -> object:
    """Fit the requested importable engression model."""

    overrides = {
        "num_layer": args.num_layer,
        "hidden_dim": args.hidden_dim,
        "noise_dim": args.noise_dim,
        "add_bn": False,
        "lr": args.lr,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "standardize": True,
        "device": "cpu",
        "verbose": False,
    }
    if args.engression_model in {"regularized", "adamw", "lstm"}:
        overrides["weight_decay"] = args.weight_decay
    return fit_engression_model(args.engression_model, x_train, y_train, **overrides)


def evaluate_oos_subsets(
    args: argparse.Namespace,
    candidate_idx: np.ndarray,
    oos_flags: dict[str, np.ndarray],
    predicted_quantiles: np.ndarray,
    true_quantiles: np.ndarray,
    y_candidates: np.ndarray,
) -> list[dict[str, object]]:
    """Compute metrics on each OOS-selected held-out subset."""

    rows: list[dict[str, object]] = []
    for method in OOS_METHODS:
        positions = select_eval_positions(
            candidate_idx=candidate_idx,
            flags=oos_flags[method],
            max_test_size=args.max_test_size,
        )
        row: dict[str, object] = {
            "method": method,
            "oos_candidates": int(np.sum(oos_flags[method])),
            "evaluated_rows": int(len(positions)),
        }
        if len(positions) == 0:
            row.update(
                {
                    "mean_quantile_mae": np.nan,
                    "q05_mae": np.nan,
                    "q50_mae": np.nan,
                    "q95_mae": np.nan,
                    "predicted_interval_coverage": np.nan,
                    "true_interval_coverage": np.nan,
                    "mean_predicted_interval_width": np.nan,
                    "mean_true_interval_width": np.nan,
                    "width_ratio": np.nan,
                }
            )
        else:
            metrics = quantile_diagnostic_metrics(
                predicted_quantiles=predicted_quantiles[positions],
                true_quantiles=true_quantiles[positions],
                y_test=y_candidates[positions],
            )
            row.update(
                {
                    "mean_quantile_mae": metrics["mean_quantile_mae"],
                    "q05_mae": metrics["quantile_mae"]["q05"],
                    "q50_mae": metrics["quantile_mae"]["q50"],
                    "q95_mae": metrics["quantile_mae"]["q95"],
                    "predicted_interval_coverage": metrics["predicted_interval_coverage"],
                    "true_interval_coverage": metrics["true_interval_coverage"],
                    "mean_predicted_interval_width": metrics["mean_predicted_interval_width"],
                    "mean_true_interval_width": metrics["mean_true_interval_width"],
                    "width_ratio": metrics["width_ratio"],
                }
            )
        rows.append(row)
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    """Print a compact metrics table."""

    print(
        "method                oos_n  eval_n  mean_q_mae  q05_mae  q50_mae  "
        "q95_mae  pred_cov  true_cov  pred_width  true_width  width_ratio"
    )
    for row in rows:
        print(
            f"{str(row['method']):<20} "
            f"{int(row['oos_candidates']):>5} "
            f"{int(row['evaluated_rows']):>7} "
            f"{float(row['mean_quantile_mae']):>11.3f} "
            f"{float(row['q05_mae']):>8.3f} "
            f"{float(row['q50_mae']):>8.3f} "
            f"{float(row['q95_mae']):>8.3f} "
            f"{float(row['predicted_interval_coverage']):>9.3f} "
            f"{float(row['true_interval_coverage']):>9.3f} "
            f"{float(row['mean_predicted_interval_width']):>11.3f} "
            f"{float(row['mean_true_interval_width']):>10.3f} "
            f"{float(row['width_ratio']):>11.3f}"
        )


def write_outputs(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    """Write CSV, JSON, and README outputs for the comparison run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "oos_candidates",
        "evaluated_rows",
        "mean_quantile_mae",
        "q05_mae",
        "q50_mae",
        "q95_mae",
        "predicted_interval_coverage",
        "true_interval_coverage",
        "mean_predicted_interval_width",
        "mean_true_interval_width",
        "width_ratio",
    ]
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "command": shell_command(),
        "data_model": args.model,
        "engression_model": args.engression_model,
        "dimension": args.dimension,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
        "max_test_size": args.max_test_size,
        "window": args.window,
        "seed": args.seed,
        "epochs": args.epochs,
        "prediction_samples": args.prediction_samples,
        "results": rows,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    readme_lines = [
        f"# OOS Comparison: {args.model} / {args.engression_model} / d={args.dimension}",
        "",
        "This run fits one model on one shared training set, then evaluates it on",
        "held-out subsets chosen by different out-of-support definitions.",
        "",
        "## Command",
        "",
        "```bash",
        shell_command(),
        "```",
        "",
        "## Outputs",
        "",
        "- `results.csv`: summary metrics by OOS definition.",
        "- `results.json`: run metadata and the same metrics.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def main() -> None:
    """Run the comparison."""

    args = build_arg_parser().parse_args()
    if args.dimension < BASE_X_DIMENSION:
        raise ValueError(f"dimension must be at least {BASE_X_DIMENSION}")
    out_dir = args.out_dir or RUN_ROOT / f"{args.model}_{args.engression_model}_d{args.dimension}"

    set_reproducible_seeds(args.seed)
    dataset = generate_supervised_dataset(
        model=args.model,
        num_samples=args.num_samples,
        x_dimension=args.dimension,
        window=args.window,
        seed=args.seed,
        noise_scale=args.noise_scale,
        seasonal_period=args.seasonal_period,
    )
    phi = np.asarray(dataset["phi"], dtype=np.float32)
    train_idx, candidate_idx = central_phi_train_candidates(phi=phi, train_size=args.train_size)
    model_spec = MODEL_REGISTRY[args.engression_model]
    x_train, y_train = tensors_from_dataset(
        dataset,
        train_idx,
        input_kind=model_spec.input_kind,
    )
    x_candidates, y_candidates_t = tensors_from_dataset(
        dataset,
        candidate_idx,
        input_kind=model_spec.input_kind,
    )
    x_train_flat, _ = flattened_tensors_from_dataset(dataset, train_idx)
    x_candidates_flat, _ = flattened_tensors_from_dataset(dataset, candidate_idx)
    y_candidates = y_candidates_t.detach().cpu().numpy().reshape(-1)
    true_quantiles = np.column_stack(
        [np.asarray(dataset[key], dtype=np.float32)[candidate_idx] for key in QUANTILE_KEYS]
    )

    oos_config = OOSConfig(
        marginal_quantile_low=args.oos_quantile_low,
        marginal_quantile_high=args.oos_quantile_high,
        knn_threshold_quantile=args.oos_knn_threshold_quantile,
        knn_reference_size=args.oos_knn_reference_size,
        knn_batch_size=args.oos_knn_batch_size,
    )
    oos = compute_oos_diagnostics(
        x_train=x_train_flat,
        x_test=x_candidates_flat,
        train_phi=phi[train_idx],
        test_phi=phi[candidate_idx],
        config=oos_config,
        seed=args.seed,
    )

    print(f"OOS comparison: {format_model_name(args.model)} / {args.engression_model}")
    print(f"  train rows: {len(train_idx)}")
    print(f"  candidate rows: {len(candidate_idx)}")
    print(f"  model X shape: train={tuple(x_train.shape)}, candidates={tuple(x_candidates.shape)}")
    print(
        "  flattened X shape for OOS diagnostics: "
        f"train={tuple(x_train_flat.shape)}, candidates={tuple(x_candidates_flat.shape)}"
    )

    engressor = fit_model(args, x_train, y_train)
    torch.manual_seed(args.seed + 101)
    predicted_quantiles = predict_quantiles(
        engressor=engressor,
        x_test=x_candidates,
        sample_size=args.prediction_samples,
    )
    rows = evaluate_oos_subsets(
        args=args,
        candidate_idx=candidate_idx,
        oos_flags=oos.flags,
        predicted_quantiles=predicted_quantiles,
        true_quantiles=true_quantiles,
        y_candidates=y_candidates,
    )
    print_summary(rows)
    write_outputs(out_dir=out_dir, args=args, rows=rows)
    print(f"wrote OOS comparison results to {out_dir}")


if __name__ == "__main__":
    main()
