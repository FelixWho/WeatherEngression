"""Sweep ordinary tuning knobs for LSTM engression.

This script intentionally keeps the LSTM engression objective unchanged. It
only varies training epochs and classic Adam ``weight_decay`` so we can see
whether the under-dispersed full-sized LSTM run is mostly an optimization or
regularization issue.
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

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.constants import SPLIT_MODES
from experiments.pipeline import (
    EngressionFitConfig,
    OOSConfig,
    PredictionConfig,
    SplitConfig,
    SyntheticDataConfig,
    run_engression_experiment,
)
from data_generation.generate_data import BASE_X_DIMENSION, DEFAULT_SEASONAL_PERIOD, MODEL_NAMES


SWEEP_ROOT = REPO_ROOT / "runs" / "optimizer_sweeps" / "lstm"


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc


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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--prediction-samples", type=int, default=300)
    parser.add_argument(
        "--epochs-list",
        type=parse_int_list,
        default=parse_int_list("40,60,80,120"),
        help="Comma-separated epoch counts to sweep.",
    )
    parser.add_argument(
        "--weight-decays",
        type=parse_float_list,
        default=parse_float_list("0,0.003"),
        help="Comma-separated classic Adam weight_decay values to sweep.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder. Defaults to runs/optimizer_sweeps/lstm/<model>_<split>_d<d>.",
    )
    return parser


def shell_command() -> str:
    """Return the current command in a copy-pasteable form."""

    parts = ["python", "engression_modifications/lstm_sweep.py", *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def default_out_dir(model: str, split: str, dimension: int) -> Path:
    """Return a stable default output folder for the sweep."""

    return SWEEP_ROOT / f"{model}_{split}_d{dimension}"


def calibration_score(metrics: dict[str, object]) -> float:
    """Rank candidates by quantile error, coverage, and width calibration."""

    return (
        float(metrics["mean_quantile_mae"])
        + abs(float(metrics["predicted_interval_coverage"]) - float(metrics["true_interval_coverage"]))
        + 0.5 * abs(float(metrics["width_ratio"]) - 1.0)
    )


def run_one(args: argparse.Namespace, epochs: int, weight_decay: float) -> dict[str, object]:
    """Run one LSTM configuration and return scalar diagnostics."""

    start = time.perf_counter()
    result = run_engression_experiment(
        data_config=SyntheticDataConfig(
            data_model=args.model,
            num_samples=args.num_samples,
            x_dimension=args.dimension,
            window=args.window,
            seed=args.seed,
            noise_scale=args.noise_scale,
            seasonal_period=args.seasonal_period,
        ),
        split_config=SplitConfig(
            split=args.split,
            train_size=args.train_size,
            test_size=args.test_size,
        ),
        fit_config=EngressionFitConfig(
            engression_model="lstm",
            num_layer=args.num_layer,
            hidden_dim=args.hidden_dim,
            noise_dim=args.noise_dim,
            add_bn=False,
            lr=args.lr,
            weight_decay=weight_decay,
            num_epochs=epochs,
            batch_size=args.batch_size,
            standardize=True,
            device="cpu",
            verbose=False,
        ),
        prediction_config=PredictionConfig(sample_size=args.prediction_samples),
        oos_config=OOSConfig(),
    )
    seconds = time.perf_counter() - start
    metrics = result.metrics
    score = calibration_score(metrics)
    return {
        "epochs": epochs,
        "weight_decay": weight_decay,
        "seconds": seconds,
        "mean_quantile_mae": metrics["mean_quantile_mae"],
        "q05_mae": metrics["quantile_mae"]["q05"],
        "q50_mae": metrics["quantile_mae"]["q50"],
        "q95_mae": metrics["quantile_mae"]["q95"],
        "predicted_coverage": metrics["predicted_interval_coverage"],
        "true_coverage": metrics["true_interval_coverage"],
        "predicted_width": metrics["mean_predicted_interval_width"],
        "true_width": metrics["mean_true_interval_width"],
        "width_ratio": metrics["width_ratio"],
        "coverage_error": abs(
            float(metrics["predicted_interval_coverage"]) - float(metrics["true_interval_coverage"])
        ),
        "width_ratio_error": abs(float(metrics["width_ratio"]) - 1.0),
        "calibration_score": score,
        "x_train_shape": metrics["x_train_shape"],
        "x_test_shape": metrics["x_test_shape"],
    }


def write_results(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    """Write CSV, JSON, and README outputs for the sweep."""

    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: float(row["calibration_score"]))
    csv_fields = [
        "epochs",
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
        "x_train_shape",
        "x_test_shape",
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
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "noise_dim": args.noise_dim,
        "num_layer": args.num_layer,
        "lr": args.lr,
        "prediction_samples": args.prediction_samples,
        "epochs_list": args.epochs_list,
        "weight_decays": args.weight_decays,
        "results": ordered,
    }
    (out_dir / "results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    best = ordered[0]
    readme_lines = [
        f"# LSTM Engression Sweep: {args.model} / {args.split} / d={args.dimension}",
        "",
        "This run sweeps only ordinary optimization knobs for the local LSTM",
        "engression model: epoch count and classic Adam `weight_decay`.",
        "",
        "## Command",
        "",
        "```bash",
        shell_command(),
        "```",
        "",
        "## Best By Heuristic Calibration Score",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| epochs | `{best['epochs']}` |",
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
        "Use the raw `results.csv` columns for final judgment.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def print_row(row: dict[str, object]) -> None:
    """Print one completed sweep row."""

    print(
        "epochs={epochs:>3} wd={weight_decay:<7g} "
        "qMAE={mean_quantile_mae:.3f} cov={predicted_coverage:.3f} "
        "width_ratio={width_ratio:.3f} score={calibration_score:.3f} "
        "seconds={seconds:.1f}".format(**row),
        flush=True,
    )


def main() -> None:
    """Run the LSTM sweep."""

    args = build_arg_parser().parse_args()
    if args.dimension < BASE_X_DIMENSION:
        raise ValueError(f"dimension must be at least {BASE_X_DIMENSION}")

    out_dir = args.out_dir or default_out_dir(args.model, args.split, args.dimension)
    rows: list[dict[str, object]] = []
    total = len(args.epochs_list) * len(args.weight_decays)
    run_idx = 0
    print(f"Running {total} LSTM sweep configurations...", flush=True)
    for epochs in args.epochs_list:
        for weight_decay in args.weight_decays:
            run_idx += 1
            print(f"[{run_idx}/{total}] starting epochs={epochs}, wd={weight_decay:g}", flush=True)
            row = run_one(args, epochs=epochs, weight_decay=weight_decay)
            rows.append(row)
            print_row(row)
            write_results(out_dir, args, rows)

    best = min(rows, key=lambda row: float(row["calibration_score"]))
    print("\nBest by heuristic calibration score:", flush=True)
    print_row(best)
    print(f"wrote LSTM sweep results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
