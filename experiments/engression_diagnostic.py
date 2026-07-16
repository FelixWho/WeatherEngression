"""Run an engression diagnostic on synthetic weather data.

The diagnostic:

1. generates weather-like lag-window inputs ``X`` and scalar targets ``y``;
2. splits rows using an in-support or scalar ``phi(X)`` extrapolation rule;
3. fits a selected engression model variant;
4. compares predicted conditional quantiles against the generator truth.

This is experiment infrastructure, not a unit test. It writes charts and
metrics under ``runs/engression_diagnostics/`` by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engression_modifications import MODEL_REGISTRY
from experiments.artifacts import default_engression_run_dir, shell_command, write_run_artifacts
from experiments.constants import SPLIT_MODES, format_model_name
from experiments.pipeline import (
    EngressionFitConfig,
    OOSConfig,
    PredictionConfig,
    SplitConfig,
    SyntheticDataConfig,
    run_engression_experiment,
)
from experiments.plotting import write_posterior_band_chart
from data_generation.generate_data import (
    BASE_X_DIMENSION,
    DEFAULT_SEASONAL_PERIOD,
    MODEL_NAMES,
)


ENGRESSION_MODEL_NAMES = ("vanilla", "regularized", "adamw", "lstm")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="narx_student_t",
        help="Synthetic target law to generate and evaluate.",
    )
    parser.add_argument(
        "--engression-model",
        choices=ENGRESSION_MODEL_NAMES,
        default="vanilla",
        help="Importable engression model variant to fit.",
    )
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--train-size", type=int, default=8_000)
    parser.add_argument("--test-size", type=int, default=1_000)
    parser.add_argument(
        "--split",
        choices=SPLIT_MODES,
        default="in-support",
        help=(
            "Evaluation split. Extrapolation modes hold out rows outside the "
            "training support as measured by phi(X)."
        ),
    )
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("-d", "--dimension", type=int, default=BASE_X_DIMENSION)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--seasonal-period", type=int, default=DEFAULT_SEASONAL_PERIOD)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--stonet-head", action="store_true", help="LSTM: use package StoNet head (loose).")
    parser.add_argument("--pre-additive", action="store_true", help="LSTM: monotone pre-ANM head Y=g(phi(X)+eta).")
    parser.add_argument("--index-dim", type=int, default=None, help="LSTM: latent index width for --pre-additive.")
    parser.add_argument("--resblock", action="store_true", help="LSTM: residual StoNet blocks with --stonet-head.")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prediction-samples", type=int, default=2000)
    parser.add_argument("--oos-quantile-low", type=float, default=0.01)
    parser.add_argument("--oos-quantile-high", type=float, default=0.99)
    parser.add_argument("--oos-knn-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--oos-knn-reference-size", type=int, default=2000)
    parser.add_argument("--max-median-mae", type=float, default=0.80)
    parser.add_argument("--max-mean-quantile-mae", type=float, default=1.00)
    parser.add_argument("--min-interval-coverage", type=float, default=0.35)
    parser.add_argument("--max-interval-coverage", type=float, default=1.00)
    parser.add_argument(
        "--skip-assertions",
        action="store_true",
        help="Print metrics and write charts without failing on loose diagnostic thresholds.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help=(
            "Path for the posterior quantile-band diagnostic chart. "
            "Defaults to runs/engression_diagnostics/<data-model>/<split>/<engression-model>/posterior_bands.png."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the diagnostic chart.")
    parser.add_argument("--no-readme", action="store_true", help="Skip writing README.md and metrics.json.")
    parser.add_argument("--show", action="store_true", help="Display the chart interactively after saving it.")
    return parser


def main() -> None:
    """Run the diagnostic."""

    args = build_arg_parser().parse_args()
    if args.engression_model not in MODEL_REGISTRY:
        raise ValueError(f"unknown engression model: {args.engression_model}")

    run_dir = default_engression_run_dir(args.model, args.split, args.engression_model)
    plot_out = args.plot_out or run_dir / "posterior_bands.png"
    if args.plot_out is not None:
        run_dir = args.plot_out.parent

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
    fit_config = EngressionFitConfig(
        engression_model=args.engression_model,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        stonet_head=args.stonet_head,
        pre_additive=args.pre_additive,
        index_dim=args.index_dim,
        resblock=args.resblock,
        add_bn=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        standardize=True,
        device="cpu",
        verbose=False,
    )
    prediction_config = PredictionConfig(sample_size=args.prediction_samples)
    oos_config = OOSConfig(
        marginal_quantile_low=args.oos_quantile_low,
        marginal_quantile_high=args.oos_quantile_high,
        knn_threshold_quantile=args.oos_knn_threshold_quantile,
        knn_reference_size=args.oos_knn_reference_size,
    )
    result = run_engression_experiment(
        data_config=data_config,
        split_config=split_config,
        fit_config=fit_config,
        prediction_config=prediction_config,
        oos_config=oos_config,
    )
    data = result.data
    metrics = result.metrics
    support = data.phi_support
    test_phi = data.test_phi

    print(f"Engression diagnostic: {format_model_name(args.model)} / {args.engression_model}")
    print(f"  split: {args.split}")
    print(f"  train rows: {len(data.train_idx)}")
    print(f"  test rows: {len(data.test_idx)}")
    print(f"  test rows outside train phi support: {metrics['test_rows_outside_train_phi_support']}")
    print(f"  train phi support: [{support[0]:.3f}, {support[1]:.3f}]")
    print(f"  test phi range: [{float(np.min(test_phi)):.3f}, {float(np.max(test_phi)):.3f}]")
    print(f"  model X shape: train={tuple(data.x_train.shape)}, test={tuple(data.x_test.shape)}")
    print(
        "  flattened X shape for OOS diagnostics: "
        f"train={tuple(data.x_train_flat.shape)}, test={tuple(data.x_test_flat.shape)}"
    )
    print(
        "  quantile MAE: "
        f"q05={metrics['quantile_mae']['q05']:.3f}, "
        f"q50={metrics['quantile_mae']['q50']:.3f}, "
        f"q95={metrics['quantile_mae']['q95']:.3f}"
    )
    print(f"  mean quantile MAE: {metrics['mean_quantile_mae']:.3f}")
    print(f"  predicted 90% interval coverage: {metrics['predicted_interval_coverage']:.3f}")
    print(f"  true 90% interval coverage on realized y: {metrics['true_interval_coverage']:.3f}")
    print(f"  mean predicted 90% interval width: {metrics['mean_predicted_interval_width']:.3f}")
    print(f"  mean true 90% interval width: {metrics['mean_true_interval_width']:.3f}")
    print("  OOS rates:")
    for method, summary in metrics["oos"].items():
        print(f"    {method}: {summary['count']}/{len(data.test_idx)} ({summary['fraction']:.3f})")

    if not args.no_plot:
        write_posterior_band_chart(
            out=plot_out,
            model=args.model,
            split=args.split,
            test_idx=data.test_idx,
            train_phi_support=support,
            test_phi=test_phi,
            y_test=data.y_test_np,
            true_quantiles=data.true_quantiles,
            predicted_quantiles=result.predicted_quantiles,
            show=args.show,
        )

    if not args.no_readme:
        write_run_artifacts(
            run_dir=run_dir,
            plot_out=plot_out,
            args=args,
            metrics=metrics,
            command=shell_command("experiments/engression_diagnostic.py"),
        )

    if args.skip_assertions:
        return

    if not np.isfinite(result.predicted_quantiles).all():
        raise AssertionError("engression produced non-finite quantile predictions")
    if not np.all(result.predicted_quantiles[:, 0] <= result.predicted_quantiles[:, 1] + 1e-6):
        raise AssertionError("predicted q05 exceeds predicted q50")
    if not np.all(result.predicted_quantiles[:, 1] <= result.predicted_quantiles[:, 2] + 1e-6):
        raise AssertionError("predicted q50 exceeds predicted q95")
    if metrics["mean_predicted_interval_width"] <= 0.0:
        raise AssertionError("predicted conditional interval has non-positive width")
    if metrics["median_mae"] > args.max_median_mae:
        raise AssertionError(
            f"median quantile MAE {metrics['median_mae']:.3f} exceeds {args.max_median_mae:.3f}"
        )
    if metrics["mean_quantile_mae"] > args.max_mean_quantile_mae:
        raise AssertionError(
            "mean quantile MAE "
            f"{metrics['mean_quantile_mae']:.3f} exceeds {args.max_mean_quantile_mae:.3f}"
        )
    if not (
        args.min_interval_coverage
        <= metrics["predicted_interval_coverage"]
        <= args.max_interval_coverage
    ):
        raise AssertionError(
            "predicted interval coverage "
            f"{metrics['predicted_interval_coverage']:.3f} outside "
            f"[{args.min_interval_coverage:.3f}, {args.max_interval_coverage:.3f}]"
        )


if __name__ == "__main__":
    main()
