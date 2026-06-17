"""Run artifact helpers for synthetic engression experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

from .constants import format_model_name


REPO_ROOT = Path(__file__).resolve().parents[1]
ENGRESSION_RUNS_ROOT = REPO_ROOT / "runs" / "engression_diagnostics"


def default_engression_run_dir(model: str, split: str, model_variant: str = "vanilla") -> Path:
    """Return the default folder for one engression diagnostic run."""

    return ENGRESSION_RUNS_ROOT / model / split / model_variant


def shell_command(script_path: str) -> str:
    """Return the current command in a copy-pasteable form."""

    parts = ["python", script_path, *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def write_run_artifacts(
    run_dir: Path,
    plot_out: Path,
    args: argparse.Namespace,
    metrics: dict[str, object],
    command: str,
) -> None:
    """Write README.md and metrics.json for one diagnostic run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    oos_lines = []
    if "oos" in metrics:
        oos_lines = [
            "",
            "## Out-Of-Support Diagnostics",
            "",
            "| Method | Count | Fraction |",
            "|---|---:|---:|",
        ]
        for method, summary in metrics["oos"].items():
            oos_lines.append(
                f"| `{method}` | `{summary['count']}` | `{summary['fraction']:.6f}` |"
            )

    split_note = (
        "The default in-support split keeps held-out rows whose scalar summary "
        "\\(\\phi(X)\\) lies inside the central training range. This checks "
        "conditional distribution learning, not chronological forecasting."
        if args.split == "in-support"
        else (
            "This extrapolation split holds out rows outside the training range "
            "in scalar \\(\\phi(X)\\)-space. It is a controlled covariate-support "
            "test, not proof of full high-dimensional out-of-support behavior."
        )
    )

    readme = run_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                (
                    f"# Engression Diagnostic Run: {format_model_name(args.model)} / "
                    f"{args.split} / {args.engression_model}"
                ),
                "",
                "This folder contains one engression diagnostic run on synthetic",
                "weather-like data with known conditional quantiles.",
                "",
                "## Command",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Outputs",
                "",
                f"- `{plot_out.name}`: true versus engression-predicted posterior bands.",
                "- `metrics.json`: scalar metrics printed by the run.",
                "",
                "## Main Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| mean quantile MAE | `{metrics['mean_quantile_mae']:.6f}` |",
                f"| q05 MAE | `{metrics['quantile_mae']['q05']:.6f}` |",
                f"| q50 MAE | `{metrics['quantile_mae']['q50']:.6f}` |",
                f"| q95 MAE | `{metrics['quantile_mae']['q95']:.6f}` |",
                f"| predicted 90% interval coverage | `{metrics['predicted_interval_coverage']:.6f}` |",
                f"| true 90% interval coverage on realized y | `{metrics['true_interval_coverage']:.6f}` |",
                f"| mean predicted 90% interval width | `{metrics['mean_predicted_interval_width']:.6f}` |",
                f"| mean true 90% interval width | `{metrics['mean_true_interval_width']:.6f}` |",
                f"| width ratio | `{metrics['width_ratio']:.6f}` |",
                *oos_lines,
                "",
                "## Run Parameters",
                "",
                "| Parameter | Value |",
                "|---|---:|",
                f"| data model | `{args.model}` |",
                f"| engression model | `{args.engression_model}` |",
                f"| split | `{args.split}` |",
                f"| dimension | `{args.dimension}` |",
                f"| window | `{args.window}` |",
                f"| num_samples | `{args.num_samples}` |",
                f"| train_size | `{args.train_size}` |",
                f"| test_size | `{args.test_size}` |",
                f"| epochs | `{args.epochs}` |",
                f"| batch_size | `{args.batch_size}` |",
                f"| hidden_dim | `{args.hidden_dim}` |",
                f"| noise_dim | `{args.noise_dim}` |",
                f"| num_layer | `{args.num_layer}` |",
                f"| learning_rate | `{args.lr}` |",
                f"| weight_decay | `{args.weight_decay}` |",
                f"| prediction_samples | `{args.prediction_samples}` |",
                f"| seed | `{args.seed}` |",
                "",
                "## Split Interpretation",
                "",
                split_note,
                "",
                "The model sees flattened lag windows, so the training matrix has shape",
                f"`{tuple(metrics['x_train_shape'])}` and the test matrix has shape",
                f"`{tuple(metrics['x_test_shape'])}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
