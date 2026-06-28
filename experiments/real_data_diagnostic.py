"""Fit LSTM engression on the real ENA weather data and score the forecast.

This is the real-data counterpart to ``experiments/engression_diagnostic.py``.
The synthetic diagnostic compares predicted quantiles against a known
conditional law; real data has no such truth, so this script instead scores the
predictive distribution against held-out realized targets (calibration of 50%
and 90% intervals, central error, and the energy score / CRPS).

It fits the sequence-native ``lstm`` engression variant on airmass-trajectory
inputs and writes a chart, metrics, and a run README under
``runs/real_data_diagnostics/`` by default.

Example
-------
```bash
python experiments/real_data_diagnostic.py --target ccn --split random \
    --max-samples 6000 --seq-stride 4 --epochs 40
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    load_ena_supervised_dataset,
    select_real_split,
)
from engression_modifications import get_engression_model
from experiments.pipeline import set_reproducible_seeds
from experiments.real_metrics import empirical_quantile_metrics, energy_score_samples
from experiments.real_plotting import write_real_posterior_band_chart

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
SPLIT_MODES = ("random", "chronological", "event")
REAL_RUNS_ROOT = REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    parser.add_argument(
        "--target",
        type=str,
        default="ccn",
        help="'ccn' (scalar target) or a trajectory variable name/index (one step ahead).",
    )
    parser.add_argument("--split", choices=SPLIT_MODES, default="random")
    parser.add_argument(
        "--event-flag",
        type=str,
        default="dust",
        help="Event mask held out as the test set when --split event (e.g. dust, BB_criterion1).",
    )
    parser.add_argument("--max-samples", type=int, default=6_000)
    parser.add_argument("--seq-stride", type=int, default=4)
    parser.add_argument("--train-size", type=int, default=4_000)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prediction-samples", type=int, default=800)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch energy loss.")
    parser.add_argument(
        "--skip-assertions",
        action="store_true",
        help="Write artifacts without failing on loose sanity thresholds.",
    )
    parser.add_argument("--min-coverage-90", type=float, default=0.60)
    parser.add_argument("--max-coverage-90", type=float, default=0.99)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def default_run_dir(target_slug: str, split: str) -> Path:
    """Return the default folder for one real-data LSTM diagnostic run."""

    return REAL_RUNS_ROOT / target_slug / split / "lstm"


def shell_command() -> str:
    """Return the current command in copy-pasteable form."""

    parts = ["python", "experiments/real_data_diagnostic.py", *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def predict_quantiles_and_samples(
    engressor: object,
    x_test: torch.Tensor,
    sample_size: int,
    levels: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw conditional samples once, then derive quantiles and a sample matrix.

    Sampling once and reusing the draws keeps the quantiles and the energy score
    consistent and avoids generating the (expensive) sample set twice.
    """

    samples = engressor.sample(x_test, sample_size=sample_size, expand_dim=True)
    samples = samples.detach().cpu().numpy()[:, 0, :]  # (n, sample_size), out_dim == 1
    quantiles = np.quantile(samples, np.asarray(levels), axis=1).T  # (n, len(levels))
    return quantiles, samples


def write_run_artifacts(
    run_dir: Path,
    plot_name: str,
    args: argparse.Namespace,
    metrics: dict[str, object],
    command: str,
) -> None:
    """Write README.md and metrics.json for one real-data run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_note = {
        "random": (
            "A shuffled in-distribution split: held-out rows are drawn from the "
            "same airmass population as training. This checks conditional "
            "distribution learning, not forecasting or shift."
        ),
        "chronological": (
            "Earliest trajectories train, later ones test. This is a genuine "
            "forecast-the-future split over the ENA record."
        ),
        "event": (
            f"The test set is the held-out `{args.event_flag}` event period and "
            "training uses the remaining clean periods. This is a real "
            "distribution-shift stress test, the data analogue of the synthetic "
            "out-of-support splits."
        ),
    }[args.split]

    readme = run_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Real-Data Engression Run: ENA {metrics['target_name']} / {args.split} / lstm",
                "",
                "This folder contains one LSTM-engression run on the **real** ENA",
                "weather data (`weather_data.mat`). The true conditional law is",
                "unknown, so the predictive distribution is scored against held-out",
                "realized targets rather than against generator truth.",
                "",
                "## Command",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Outputs",
                "",
                f"- `{plot_name}`: predicted conditional bands with realized targets overlaid.",
                "- `metrics.json`: scalar metrics printed by the run.",
                "",
                "## Main Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| energy score (CRPS, lower better) | `{metrics['energy_score']:.6f}` |",
                f"| median abs error | `{metrics['median_abs_error']:.6f}` |",
                f"| 90% interval coverage | `{metrics['coverage_90']:.6f}` |",
                f"| 50% interval coverage | `{metrics['coverage_50']:.6f}` |",
                f"| mean 90% interval width | `{metrics['mean_width_90']:.6f}` |",
                f"| mean 50% interval width | `{metrics['mean_width_50']:.6f}` |",
                "",
                "Nominal coverage is 0.90 and 0.50; closer is better-calibrated.",
                "",
                "## Run Parameters",
                "",
                "| Parameter | Value |",
                "|---|---:|",
                f"| data | `{metrics['mat_path']}` |",
                f"| target | `{metrics['target_name']}` |",
                f"| split | `{args.split}` |",
                f"| event_flag | `{args.event_flag if args.split == 'event' else 'n/a'}` |",
                f"| max_samples | `{args.max_samples}` |",
                f"| seq_stride | `{args.seq_stride}` |",
                f"| sequence shape (train) | `{tuple(metrics['x_train_shape'])}` |",
                f"| train_size | `{metrics['train_rows']}` |",
                f"| test_size | `{metrics['test_rows']}` |",
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
                "The `lstm` variant keeps lag windows unflattened, so the training",
                f"tensor has shape `{tuple(metrics['x_train_shape'])}`",
                "`(n, sequence_length, n_features)`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run the real-data LSTM-engression diagnostic."""

    args = build_arg_parser().parse_args()
    set_reproducible_seeds(args.seed)

    dataset = load_ena_supervised_dataset(
        mat_path=args.mat_path,
        target=args.target,
        max_samples=args.max_samples,
        seq_stride=args.seq_stride,
        seed=args.seed,
    )
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=args.split,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed + 17,
        event_flag=args.event_flag,
    )

    x_train = torch.from_numpy(dataset.x[train_idx])
    y_train = torch.from_numpy(dataset.y[train_idx].reshape(-1, 1))
    x_test = torch.from_numpy(dataset.x[test_idx])
    y_test = dataset.y[test_idx].astype(np.float64)

    engressor = get_engression_model("lstm").fit(
        x_train,
        y_train,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        add_bn=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        standardize=True,
        device=args.device,
        verbose=args.verbose,
    )

    torch.manual_seed(args.seed + 101)
    predicted_quantiles, samples = predict_quantiles_and_samples(
        engressor=engressor,
        x_test=x_test,
        sample_size=args.prediction_samples,
        levels=QUANTILE_LEVELS,
    )

    metrics: dict[str, object] = {
        "dataset": "ena_weather",
        "mat_path": args.mat_path,
        "target_name": dataset.target_name,
        "target_slug": dataset.target_slug,
        "split": args.split,
        "event_flag": args.event_flag if args.split == "event" else None,
        "feature_names": list(dataset.feature_names),
        "sequence_length": dataset.sequence_length,
        "n_features": dataset.n_features,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "quantile_levels": list(QUANTILE_LEVELS),
    }
    metrics.update(empirical_quantile_metrics(predicted_quantiles, y_test, levels=QUANTILE_LEVELS))
    metrics.update(energy_score_samples(samples, y_test))

    print(f"Real-data LSTM engression: ENA {dataset.target_name} / {args.split}")
    print(f"  usable samples loaded: {len(dataset.y)}  sequence shape: {tuple(x_train.shape[1:])}")
    print(f"  train rows: {len(train_idx)}   test rows: {len(test_idx)}")
    print(f"  energy score (CRPS): {metrics['energy_score']:.4f}")
    print(f"  median abs error:    {metrics['median_abs_error']:.4f}")
    print(f"  90% coverage: {metrics['coverage_90']:.3f}  (width {metrics['mean_width_90']:.3f})")
    print(f"  50% coverage: {metrics['coverage_50']:.3f}  (width {metrics['mean_width_50']:.3f})")

    run_dir = default_run_dir(dataset.target_slug, args.split)
    plot_name = "posterior_bands.png"
    if not args.no_plot:
        write_real_posterior_band_chart(
            out=run_dir / plot_name,
            target_name=dataset.target_name,
            split=args.split,
            y_test=y_test,
            predicted_quantiles=predicted_quantiles,
            levels=QUANTILE_LEVELS,
            show=args.show,
        )
    if not args.no_readme:
        write_run_artifacts(run_dir, plot_name, args, metrics, shell_command())
    print(f"  wrote run to: {run_dir}")

    if args.skip_assertions:
        return
    if not np.isfinite(predicted_quantiles).all():
        raise AssertionError("engression produced non-finite quantile predictions")
    if not np.all(np.diff(predicted_quantiles, axis=1) >= -1e-4):
        raise AssertionError("predicted quantiles are not monotonically non-decreasing")
    if not (args.min_coverage_90 <= metrics["coverage_90"] <= args.max_coverage_90):
        raise AssertionError(
            f"90% interval coverage {metrics['coverage_90']:.3f} outside "
            f"[{args.min_coverage_90:.3f}, {args.max_coverage_90:.3f}]"
        )


if __name__ == "__main__":
    main()
