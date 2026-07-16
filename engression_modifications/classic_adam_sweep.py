"""Sweep classic Adam weight decay for public-package engression.

This script varies optimizer parameters while keeping the upstream stochastic
MLP architecture fixed. The goal is to find classic Adam settings that reduce
the too-narrow posterior interval behavior seen in the synthetic weather
diagnostics.
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
import time

import numpy as np
import torch

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engression_modifications import EngressionModelSpec, regularized
from experiments.constants import SPLIT_MODES
from experiments.pipeline import (
    SplitConfig,
    SyntheticDataConfig,
    prepare_experiment_data,
    set_reproducible_seeds,
)
from experiments.predictions import predict_quantiles
from data_generation.generate_data import (
    BASE_X_DIMENSION,
    DEFAULT_SEASONAL_PERIOD,
    MODEL_NAMES,
)


SWEEP_ROOT = REPO_ROOT / "runs" / "optimizer_sweeps" / "classic_adam"


def parse_float_list(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one float")
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list: {value}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="narx_student_t")
    parser.add_argument("--split", choices=SPLIT_MODES, default="in-support")
    parser.add_argument("-d", "--dimension", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--seasonal-period", type=int, default=DEFAULT_SEASONAL_PERIOD)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--prediction-samples", type=int, default=800)
    parser.add_argument(
        "--lrs",
        type=parse_float_list,
        default=parse_float_list("0.001,0.003,0.006"),
        help="Comma-separated learning rates to sweep.",
    )
    parser.add_argument(
        "--weight-decays",
        type=parse_float_list,
        default=parse_float_list("0,0.0003,0.001,0.003,0.01"),
        help="Comma-separated classic Adam weight_decay values to sweep.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to runs/optimizer_sweeps/classic_adam/<model>_<split>_d<d>.",
    )
    return parser


def shell_command() -> str:
    """Return the current command in a copy-pasteable form."""

    parts = ["python", "engression_modifications/classic_adam_sweep.py", *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def default_out_dir(model: str, split: str, dimension: int) -> Path:
    """Return a stable default output folder for the sweep."""

    return SWEEP_ROOT / f"{model}_{split}_d{dimension}"


def evaluate_fit(
    name: str,
    model_spec: EngressionModelSpec,
    config: object,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test_np: np.ndarray,
    true_quantiles: np.ndarray,
    true_width: float,
    true_coverage: float,
    seed: int,
    prediction_samples: int,
) -> dict[str, float | str]:
    """Fit one classic Adam configuration and return scalar diagnostics."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    start = time.perf_counter()
    engressor = model_spec.fit(x_train, y_train, config=config)
    seconds = time.perf_counter() - start

    torch.manual_seed(seed + 101)
    predicted_quantiles = predict_quantiles(
        engressor=engressor,
        x_test=x_test,
        sample_size=prediction_samples,
    )
    quantile_mae = np.mean(np.abs(predicted_quantiles - true_quantiles), axis=0)
    predicted_width = float(np.mean(predicted_quantiles[:, 2] - predicted_quantiles[:, 0]))
    predicted_coverage = float(
        np.mean(
            (y_test_np >= predicted_quantiles[:, 0])
            & (y_test_np <= predicted_quantiles[:, 2])
        )
    )
    width_ratio = predicted_width / true_width
    calibration_score = (
        float(np.mean(quantile_mae))
        + abs(predicted_coverage - true_coverage)
        + 0.5 * abs(width_ratio - 1.0)
    )

    return {
        "name": name,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "seconds": seconds,
        "q05_mae": float(quantile_mae[0]),
        "q50_mae": float(quantile_mae[1]),
        "q95_mae": float(quantile_mae[2]),
        "mean_quantile_mae": float(np.mean(quantile_mae)),
        "predicted_coverage": predicted_coverage,
        "true_coverage": true_coverage,
        "predicted_width": predicted_width,
        "true_width": true_width,
        "width_ratio": width_ratio,
        "coverage_error": abs(predicted_coverage - true_coverage),
        "width_ratio_error": abs(width_ratio - 1.0),
        "calibration_score": calibration_score,
    }


def write_results(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, float | str]]) -> None:
    """Write CSV, JSON, and README outputs for the sweep."""

    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["calibration_score"]))
    csv_fields = [
        "name",
        "lr",
        "weight_decay",
        "seconds",
        "mean_quantile_mae",
        "q05_mae",
        "q50_mae",
        "q95_mae",
        "predicted_coverage",
        "true_coverage",
        "predicted_width",
        "true_width",
        "width_ratio",
        "coverage_error",
        "width_ratio_error",
        "calibration_score",
    ]
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(ordered)

    metadata = {
        "command": shell_command(),
        "model": args.model,
        "split": args.split,
        "dimension": args.dimension,
        "num_samples": args.num_samples,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "window": args.window,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "noise_dim": args.noise_dim,
        "num_layer": args.num_layer,
        "prediction_samples": args.prediction_samples,
        "lrs": args.lrs,
        "weight_decays": args.weight_decays,
        "results": ordered,
    }
    (out_dir / "results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best = ordered[0]
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Classic Adam Sweep: {args.model} / {args.split} / d={args.dimension}",
                "",
                "This run sweeps classic `torch.optim.Adam(..., weight_decay=...)`",
                "while keeping the engression stochastic MLP architecture fixed.",
                "",
                "## Command",
                "",
                "```bash",
                shell_command(),
                "```",
                "",
                "## Outputs",
                "",
                "- `results.csv`: sweep table sorted by the heuristic calibration score.",
                "- `results.json`: command metadata and all scalar results.",
                "",
                "## Best By Heuristic Calibration Score",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| learning rate | `{best['lr']}` |",
                f"| weight decay | `{best['weight_decay']}` |",
                f"| mean quantile MAE | `{best['mean_quantile_mae']:.6f}` |",
                f"| predicted 90% coverage | `{best['predicted_coverage']:.6f}` |",
                f"| true 90% coverage | `{best['true_coverage']:.6f}` |",
                f"| predicted 90% width | `{best['predicted_width']:.6f}` |",
                f"| true 90% width | `{best['true_width']:.6f}` |",
                f"| width ratio | `{best['width_ratio']:.6f}` |",
                f"| calibration score | `{best['calibration_score']:.6f}` |",
                "",
                "The heuristic score is:",
                "",
                "\\[",
                "\\text{mean quantile MAE}",
                "+ |\\widehat{c}_{90}-c_{90}|",
                "+ 0.5\\,|\\widehat{w}_{90}/w_{90}-1|.",
                "\\]",
                "",
                "It is only a selection aid. The raw columns in `results.csv` are the",
                "important record.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def print_table(rows: list[dict[str, float | str]]) -> None:
    """Print a compact sweep table."""

    print(
        "name                   lr      wd       mean_q_mae  coverage  "
        "pred_width  width_ratio  score"
    )
    for row in sorted(rows, key=lambda item: float(item["calibration_score"])):
        print(
            f"{str(row['name']):<21} "
            f"{float(row['lr']):<7.4g} "
            f"{float(row['weight_decay']):<8.4g} "
            f"{float(row['mean_quantile_mae']):>10.3f} "
            f"{float(row['predicted_coverage']):>9.3f} "
            f"{float(row['predicted_width']):>11.3f} "
            f"{float(row['width_ratio']):>12.3f} "
            f"{float(row['calibration_score']):>7.3f}"
        )


def main() -> None:
    """Run the sweep."""

    args = build_arg_parser().parse_args()
    if args.dimension < BASE_X_DIMENSION:
        raise ValueError(f"dimension must be at least {BASE_X_DIMENSION}")

    out_dir = args.out_dir or default_out_dir(args.model, args.split, args.dimension)

    set_reproducible_seeds(args.seed)
    data_config = SyntheticDataConfig(
        data_model=args.model,
        num_samples=args.num_samples,
        x_dimension=args.dimension,
        window=args.window,
        seed=args.seed,
        noise_scale=args.noise_scale,
        seasonal_period=args.seasonal_period,
    )
    split_config = SplitConfig(
        split=args.split,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    data = prepare_experiment_data(data_config, split_config)
    x_train, y_train = data.x_train, data.y_train
    x_test = data.x_test
    y_test_np = data.y_test_np
    true_quantiles = data.true_quantiles
    true_width = float(np.mean(true_quantiles[:, 2] - true_quantiles[:, 0]))
    true_coverage = float(
        np.mean((y_test_np >= true_quantiles[:, 0]) & (y_test_np <= true_quantiles[:, 2]))
    )

    print(
        f"Classic Adam sweep: model={args.model}, split={args.split}, "
        f"d={args.dimension}, X_train={tuple(x_train.shape)}, X_test={tuple(x_test.shape)}"
    )
    print(f"train phi support: [{data.phi_support[0]:.3f}, {data.phi_support[1]:.3f}]")
    print(f"true width={true_width:.3f}, realized true coverage={true_coverage:.3f}")

    rows: list[dict[str, float | str]] = []
    total = len(args.lrs) * len(args.weight_decays)
    run_idx = 0
    for lr in args.lrs:
        for weight_decay in args.weight_decays:
            run_idx += 1
            name = f"lr={lr:g}, wd={weight_decay:g}"
            print(f"[{run_idx}/{total}] fitting {name}")
            config = regularized.config(
                num_layer=args.num_layer,
                hidden_dim=args.hidden_dim,
                noise_dim=args.noise_dim,
                add_bn=False,
                lr=lr,
                weight_decay=weight_decay,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                standardize=True,
                device="cpu",
                verbose=False,
            )
            rows.append(
                evaluate_fit(
                    name=name,
                    model_spec=regularized,
                    config=config,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test_np=y_test_np,
                    true_quantiles=true_quantiles,
                    true_width=true_width,
                    true_coverage=true_coverage,
                    seed=args.seed,
                    prediction_samples=args.prediction_samples,
                )
            )

    print_table(rows)
    write_results(out_dir=out_dir, args=args, rows=rows)
    print(f"wrote sweep results to {out_dir}")


if __name__ == "__main__":
    main()
