"""Smoke-test the public engression package on synthetic weather data.

This script checks the integration path we care about for the research project:

1. generate weather-like lag-window inputs ``X`` and a scalar target ``y`` from
   a synthetic model with saved reference conditional quantiles;
2. flatten each lag window so the published ``engression`` package receives a
   standard two-dimensional regression matrix;
3. train engression on either an in-support or extrapolation split;
4. compare predicted conditional quantiles against the generator's saved true
   quantiles for held-out ``x`` values.

It is a smoke test, not a benchmark. The default thresholds are deliberately
loose enough for a small CPU run while still catching broken data plumbing,
shape mismatches, non-finite predictions, and total failure to produce a
reasonable conditional distribution.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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

from engression import engression as fit_engression

from generate_data import (
    BASE_X_DIMENSION,
    DEFAULT_SEASONAL_PERIOD,
    MODEL_NAMES,
    flatten_x_windows,
    generate_supervised_dataset,
)


QUANTILE_KEYS = ("q05", "q50", "q95")
QUANTILE_LEVELS = (0.05, 0.50, 0.95)
SPLIT_MODES = (
    "in-support",
    "right-extrapolation",
    "left-extrapolation",
    "two-sided-extrapolation",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="narx_student_t",
        help="Synthetic target law to generate and test.",
    )
    parser.add_argument("--num-samples", type=int, default=640)
    parser.add_argument("--train-size", type=int, default=480)
    parser.add_argument("--test-size", type=int, default=96)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--noise-dim", type=int, default=32)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--prediction-samples", type=int, default=400)
    parser.add_argument("--max-median-mae", type=float, default=0.80)
    parser.add_argument("--max-mean-quantile-mae", type=float, default=1.00)
    parser.add_argument("--min-interval-coverage", type=float, default=0.35)
    parser.add_argument("--max-interval-coverage", type=float, default=1.00)
    parser.add_argument(
        "--skip-assertions",
        action="store_true",
        help="Print metrics and write charts without failing on loose smoke-test thresholds.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help=(
            "Path for the posterior quantile-band diagnostic chart. "
            "Defaults to figures/engression_<model>_<split>_smoke.png."
        ),
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the diagnostic chart.")
    parser.add_argument("--show", action="store_true", help="Display the chart interactively after saving it.")
    return parser


def format_model_name(model: str) -> str:
    """Return a compact display label for a synthetic model name."""

    special_names = {
        "narx_gaussian": "NARX Gaussian",
        "narx_student_t": "NARX Student-t",
        "narx_garch": "NARX GARCH",
        "hurdle_lognormal": "Hurdle Lognormal",
        "regime_mixture": "Regime Mixture",
        "preadditive": "Preadditive",
    }
    return special_names.get(model, model.replace("_", " ").title())


def select_split(
    phi: np.ndarray,
    train_size: int,
    test_size: int,
    seed: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Choose train/test rows for in-support or extrapolation evaluation.

    ``phi`` is only a scalar summary of the high-dimensional lag window, so this
    is a pragmatic in-support guard rather than a full support test in flattened
    ``X`` space.
    """

    if train_size + test_size > len(phi):
        raise ValueError("train_size + test_size must be no larger than num_samples")

    if split == "in-support":
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(len(phi))
        train_idx = shuffled[:train_size]
        candidates = shuffled[train_size:]

        support_lo = float(np.quantile(phi[train_idx], 0.02))
        support_hi = float(np.quantile(phi[train_idx], 0.98))
        in_support = candidates[(phi[candidates] >= support_lo) & (phi[candidates] <= support_hi)]
        if len(in_support) < test_size:
            raise ValueError(
                "not enough held-out rows inside the train phi support; "
                "increase num_samples or lower test_size"
            )
        return train_idx, in_support[:test_size], (support_lo, support_hi)

    order = np.argsort(phi)
    if split == "right-extrapolation":
        train_idx = order[:train_size]
        test_idx = order[train_size : train_size + test_size]
        return train_idx, test_idx, phi_support(phi, train_idx)

    if split == "left-extrapolation":
        train_idx = order[-train_size:]
        test_idx = order[-train_size - test_size : -train_size]
        return train_idx, test_idx, phi_support(phi, train_idx)

    if split == "two-sided-extrapolation":
        train_start = (len(phi) - train_size) // 2
        train_end = train_start + train_size
        lower_test_size = test_size // 2
        upper_test_size = test_size - lower_test_size
        lower_pool = order[:train_start]
        upper_pool = order[train_end:]
        if len(lower_pool) < lower_test_size or len(upper_pool) < upper_test_size:
            raise ValueError(
                "not enough rows in both phi tails for two-sided extrapolation; "
                "increase num_samples, lower train_size, or lower test_size"
            )
        train_idx = order[train_start:train_end]
        test_idx = np.concatenate([lower_pool[-lower_test_size:], upper_pool[:upper_test_size]])
        return train_idx, test_idx, phi_support(phi, train_idx)

    raise ValueError(f"unknown split mode: {split}")


def phi_support(phi: np.ndarray, rows: np.ndarray) -> tuple[float, float]:
    """Return the min/max training support in ``phi(X)`` space."""

    values = np.asarray(phi)[rows]
    return float(np.min(values)), float(np.max(values))


def tensors_from_dataset(
    dataset: dict[str, np.ndarray],
    rows: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flattened ``X`` and column-vector ``y`` tensors for engression."""

    x_flat = flatten_x_windows(np.asarray(dataset["X"], dtype=np.float32))
    y = np.asarray(dataset["y"], dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(x_flat[rows]), torch.from_numpy(y[rows])


def predict_quantiles(
    engressor: object,
    x_test: torch.Tensor,
    sample_size: int,
) -> np.ndarray:
    """Return predicted quantiles as an ``(n_test, 3)`` NumPy array."""

    predictions = engressor.predict(
        x_test,
        target=list(QUANTILE_LEVELS),
        sample_size=sample_size,
    )
    if not isinstance(predictions, list):
        predictions = [predictions]
    return np.column_stack(
        [prediction.detach().cpu().numpy().reshape(-1) for prediction in predictions]
    )


def load_plotting_libraries():
    """Import plotting libraries lazily so ``--help`` stays lightweight."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Plotting requires matplotlib and seaborn. Run `pip install -r requirements.txt` first."
        ) from exc
    return plt, sns


def write_posterior_band_chart(
    out: Path,
    model: str,
    split: str,
    test_idx: np.ndarray,
    train_phi_support: tuple[float, float],
    test_phi: np.ndarray,
    y_test: np.ndarray,
    true_quantiles: np.ndarray,
    predicted_quantiles: np.ndarray,
    show: bool,
) -> None:
    """Chart true and engression-predicted conditional quantile bands."""

    plt, sns = load_plotting_libraries()
    if split == "in-support":
        order = np.argsort(test_idx)
        x_axis = np.arange(len(order))
        x_label = "held-out original sample index, sorted by synthetic time"
        ordered_index = test_idx[order]
        tick_positions = np.linspace(0, len(ordered_index) - 1, num=min(6, len(ordered_index)), dtype=int)
        tick_labels = [str(int(ordered_index[i])) for i in tick_positions]
        x_limits = None
    else:
        order = np.argsort(test_phi)
        x_axis = test_phi[order]
        x_label = "held-out phi(X), sorted from low to high"
        tick_positions = None
        tick_labels = None
        x_limits = (
            min(float(np.min(test_phi)), train_phi_support[0]),
            max(float(np.max(test_phi)), train_phi_support[1]),
        )
    y_ordered = y_test[order]
    true_ordered = true_quantiles[order]
    predicted_ordered = predicted_quantiles[order]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    target_ax, width_ax = axes

    target_ax.fill_between(
        x_axis,
        true_ordered[:, 0],
        true_ordered[:, 2],
        color="#7aa6c2",
        alpha=0.26,
        label="true 90% interval",
    )
    if split != "in-support":
        target_ax.axvspan(
            train_phi_support[0],
            train_phi_support[1],
            color="#64748b",
            alpha=0.08,
            label="train phi support",
        )
    target_ax.fill_between(
        x_axis,
        predicted_ordered[:, 0],
        predicted_ordered[:, 2],
        color="#f59e0b",
        alpha=0.24,
        label="engression predicted 90% interval",
    )
    sns.lineplot(
        x=x_axis,
        y=true_ordered[:, 1],
        ax=target_ax,
        color="#2563eb",
        linewidth=1.8,
        label="true conditional median",
    )
    sns.lineplot(
        x=x_axis,
        y=predicted_ordered[:, 1],
        ax=target_ax,
        color="#c2410c",
        linewidth=1.8,
        label="engression predicted median",
    )
    sns.scatterplot(
        x=x_axis,
        y=y_ordered,
        ax=target_ax,
        color="#111827",
        s=22,
        alpha=0.68,
        linewidth=0,
        label="held-out realized y",
    )
    target_ax.set_title(f"{format_model_name(model)} Conditional Distribution: True vs Engression", pad=18)
    target_ax.set_ylabel("target y")
    target_ax.margins(x=0)
    if x_limits is not None:
        target_ax.set_xlim(*x_limits)
    target_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        ncol=3,
    )

    true_width = true_ordered[:, 2] - true_ordered[:, 0]
    predicted_width = predicted_ordered[:, 2] - predicted_ordered[:, 0]
    sns.lineplot(
        x=x_axis,
        y=true_width,
        ax=width_ax,
        color="#2563eb",
        linewidth=1.7,
        label="true interval width",
    )
    if split != "in-support":
        width_ax.axvspan(
            train_phi_support[0],
            train_phi_support[1],
            color="#64748b",
            alpha=0.08,
            label="train phi support",
        )
    sns.lineplot(
        x=x_axis,
        y=predicted_width,
        ax=width_ax,
        color="#c2410c",
        linewidth=1.7,
        label="engression interval width",
    )
    width_ax.set_ylabel("q95 - q05")
    width_ax.set_xlabel(x_label)
    width_ax.margins(x=0)
    if x_limits is not None:
        width_ax.set_xlim(*x_limits)
    width_ax.legend(loc="upper left", frameon=True, ncol=2)

    if tick_positions is not None and len(tick_positions) >= 2:
        width_ax.set_xticks(tick_positions)
        width_ax.set_xticklabels(tick_labels)
        width_ax.set_xlabel(x_label)

    fig.suptitle(f"Engression Smoke Test Posterior Bands ({split})", y=0.99)
    sns.despine(fig=fig)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  wrote posterior chart: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_out = args.plot_out
    if plot_out is None:
        safe_split = args.split.replace("-", "_")
        plot_out = REPO_ROOT / "figures" / f"engression_{args.model}_{safe_split}_smoke.png"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

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
    train_idx, test_idx, support = select_split(
        phi=phi,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed + 17,
        split=args.split,
    )
    x_train, y_train = tensors_from_dataset(dataset, train_idx)
    x_test, y_test = tensors_from_dataset(dataset, test_idx)

    engressor = fit_engression(
        x_train,
        y_train,
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        add_bn=False,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        print_every_nepoch=max(args.epochs + 1, 1),
        device="cpu",
        standardize=True,
        verbose=False,
    )

    predicted_quantiles = predict_quantiles(
        engressor=engressor,
        x_test=x_test,
        sample_size=args.prediction_samples,
    )
    true_quantiles = np.column_stack(
        [np.asarray(dataset[key], dtype=np.float32)[test_idx] for key in QUANTILE_KEYS]
    )
    y_test_np = y_test.detach().cpu().numpy().reshape(-1)

    quantile_mae = np.mean(np.abs(predicted_quantiles - true_quantiles), axis=0)
    mean_quantile_mae = float(np.mean(quantile_mae))
    median_mae = float(quantile_mae[1])
    interval_coverage = float(
        np.mean(
            (y_test_np >= predicted_quantiles[:, 0])
            & (y_test_np <= predicted_quantiles[:, 2])
        )
    )
    true_interval_coverage = float(
        np.mean((y_test_np >= true_quantiles[:, 0]) & (y_test_np <= true_quantiles[:, 2]))
    )
    interval_width = float(np.mean(predicted_quantiles[:, 2] - predicted_quantiles[:, 0]))
    test_phi = phi[test_idx]
    outside_support = int(np.sum((test_phi < support[0]) | (test_phi > support[1])))

    print(f"Engression {format_model_name(args.model)} smoke test")
    print(f"  split: {args.split}")
    print(f"  train rows: {len(train_idx)}")
    print(f"  test rows: {len(test_idx)}")
    print(f"  test rows outside train phi support: {outside_support}")
    print(f"  train phi support: [{support[0]:.3f}, {support[1]:.3f}]")
    print(f"  test phi range: [{float(np.min(test_phi)):.3f}, {float(np.max(test_phi)):.3f}]")
    print(f"  flattened X shape: train={tuple(x_train.shape)}, test={tuple(x_test.shape)}")
    print(
        "  quantile MAE: "
        f"q05={quantile_mae[0]:.3f}, q50={quantile_mae[1]:.3f}, q95={quantile_mae[2]:.3f}"
    )
    print(f"  mean quantile MAE: {mean_quantile_mae:.3f}")
    print(f"  predicted 90% interval coverage: {interval_coverage:.3f}")
    print(f"  true 90% interval coverage on realized y: {true_interval_coverage:.3f}")
    print(f"  mean predicted 90% interval width: {interval_width:.3f}")

    if not args.no_plot:
        write_posterior_band_chart(
            out=plot_out,
            model=args.model,
            split=args.split,
            test_idx=test_idx,
            train_phi_support=support,
            test_phi=test_phi,
            y_test=y_test_np,
            true_quantiles=true_quantiles,
            predicted_quantiles=predicted_quantiles,
            show=args.show,
        )

    if args.skip_assertions:
        return

    if not np.isfinite(predicted_quantiles).all():
        raise AssertionError("engression produced non-finite quantile predictions")
    if not np.all(predicted_quantiles[:, 0] <= predicted_quantiles[:, 1] + 1e-6):
        raise AssertionError("predicted q05 exceeds predicted q50")
    if not np.all(predicted_quantiles[:, 1] <= predicted_quantiles[:, 2] + 1e-6):
        raise AssertionError("predicted q50 exceeds predicted q95")
    if interval_width <= 0.0:
        raise AssertionError("predicted conditional interval has non-positive width")
    if median_mae > args.max_median_mae:
        raise AssertionError(
            f"median quantile MAE {median_mae:.3f} exceeds {args.max_median_mae:.3f}"
        )
    if mean_quantile_mae > args.max_mean_quantile_mae:
        raise AssertionError(
            "mean quantile MAE "
            f"{mean_quantile_mae:.3f} exceeds {args.max_mean_quantile_mae:.3f}"
        )
    if not (args.min_interval_coverage <= interval_coverage <= args.max_interval_coverage):
        raise AssertionError(
            "predicted interval coverage "
            f"{interval_coverage:.3f} outside "
            f"[{args.min_interval_coverage:.3f}, {args.max_interval_coverage:.3f}]"
        )


if __name__ == "__main__":
    main()
