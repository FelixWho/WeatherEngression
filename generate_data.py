"""Generate supervised synthetic weather-like data.

The CLI creates one target column ``y`` and one lag-window input tensor ``X``.
The ``-d/--dimension`` flag controls the number of feature columns in ``X``.
The first six columns are the weather variables used by the synthetic target
models. Extra columns are additional autocorrelated covariates that affect the
target through a known positive scale transform, so the full ``d``-dimensional
``X`` matters while the conditional law remains inspectable.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from data_generation.common import (
    FEATURE_NAMES,
    SyntheticWeatherConfig,
    make_lag_windows,
)
from data_generation.synthetic_weather import MODEL_NAMES, generate_dataset as _generate_dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
DEFAULT_SEASONAL_PERIOD = 24 * 365
BASE_X_DIMENSION = len(FEATURE_NAMES)


def build_generation_config(
    n_steps: int,
    window: int,
    seed: int,
    noise_scale: float,
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
) -> SyntheticWeatherConfig:
    """Build a validated config for the underlying synthetic generators."""

    if n_steps <= window:
        raise ValueError("n_steps must be greater than window")
    if window < 1:
        raise ValueError("window must be positive")
    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")

    return SyntheticWeatherConfig(
        n_steps=n_steps,
        window=window,
        seed=seed,
        noise_scale=noise_scale,
        seasonal_period=seasonal_period,
    )


def generate_extra_lag_windows(
    n_steps: int,
    window: int,
    n_features: int,
    seed: int,
    seasonal_period: int,
) -> np.ndarray:
    """Generate extra covariates for increasing ``X`` dimension.

    These columns are autocorrelated and mildly seasonal, so they look like
    plausible exogenous time-series features. They influence the final target
    through ``extra_target_scale(...)`` below.
    """

    if n_features <= 0:
        return np.empty((n_steps - window, window + 1, 0), dtype=np.float32)

    rng = np.random.default_rng(seed)
    raw = np.zeros((n_steps, n_features), dtype=np.float64)
    ar = rng.uniform(0.55, 0.92, size=n_features)
    innovation_scale = rng.uniform(0.12, 0.38, size=n_features)
    daily_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    seasonal_phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    seasonal_weight = rng.uniform(0.05, 0.25, size=n_features)

    for t in range(1, n_steps):
        daily = np.sin(2.0 * np.pi * t / 24.0 + daily_phase)
        seasonal = np.sin(2.0 * np.pi * t / seasonal_period + seasonal_phase)
        raw[t] = (
            ar * raw[t - 1]
            + 0.08 * daily
            + seasonal_weight * seasonal
            + rng.normal(0.0, innovation_scale)
        )

    return make_lag_windows(raw, window).astype(np.float32)


def extra_target_scale(extra_x: np.ndarray) -> np.ndarray:
    """Compute a known positive target scale from extra ``X`` columns.

    The transform is deliberately modest:

    ``Y_final = extra_target_scale(X_extra) * Y_base``.

    Because the scale is always positive, reference quantiles and conditional
    samples can be transformed by simple multiplication.
    """

    if extra_x.shape[-1] == 0:
        return np.ones(extra_x.shape[0], dtype=np.float32)

    current = extra_x[:, -1, :]
    recent = extra_x.mean(axis=1)
    tendency = current - extra_x[:, 0, :]
    n_features = extra_x.shape[-1]
    weights = np.linspace(0.6, 1.2, n_features, dtype=np.float64)
    weights *= np.where(np.arange(n_features) % 2 == 0, 1.0, -1.0)
    weights /= np.linalg.norm(weights)

    score = (
        0.55 * (recent @ weights)
        + 0.30 * np.tanh(current @ weights[::-1])
        + 0.15 * (tendency @ weights)
    )
    return np.exp(0.25 * np.tanh(score)).astype(np.float32)


def _multiply_rowwise(values: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Multiply row-aligned arrays by a one-dimensional factor."""

    array = np.asarray(values)
    reshape = (len(factor),) + (1,) * (array.ndim - 1)
    return (array * factor.reshape(reshape)).astype(np.float32)


def apply_extra_target_scale(dataset: dict[str, np.ndarray], scale: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the extra-feature scale to ``y`` and saved truth fields."""

    transformed = dict(dataset)
    transformed["extra_target_scale"] = scale.astype(np.float32)

    for key in ("y", "q05", "q50", "q95", "mu", "sigma", "scale", "base_sigma", "garch_residual"):
        if key in transformed:
            transformed[key] = _multiply_rowwise(np.asarray(transformed[key]), scale)

    for key in ("mixture_means", "mixture_sigmas"):
        if key in transformed:
            transformed[key] = _multiply_rowwise(np.asarray(transformed[key]), scale)

    if "garch_variance" in transformed:
        transformed["garch_variance"] = _multiply_rowwise(
            np.asarray(transformed["garch_variance"]),
            scale**2,
        )

    if "log_mu" in transformed:
        transformed["log_mu"] = (
            np.asarray(transformed["log_mu"], dtype=np.float32) + np.log(scale).astype(np.float32)
        )

    return transformed


def add_extra_x_columns(
    dataset: dict[str, np.ndarray],
    x_dimension: int,
    n_steps: int,
    window: int,
    seed: int,
    seasonal_period: int,
) -> dict[str, np.ndarray]:
    """Return a dataset whose ``X`` has exactly ``x_dimension`` columns."""

    if x_dimension < BASE_X_DIMENSION:
        raise ValueError(
            f"dimension must be at least {BASE_X_DIMENSION}; "
            "the target formulas depend on the six base weather features"
        )

    x = np.asarray(dataset["X"], dtype=np.float32)
    if x.shape[-1] != BASE_X_DIMENSION:
        raise ValueError(f"expected base X dimension {BASE_X_DIMENSION}, got {x.shape[-1]}")

    extra_count = x_dimension - BASE_X_DIMENSION
    feature_names = list(np.asarray(dataset["feature_names"]).astype(str))
    if extra_count:
        extra_x = generate_extra_lag_windows(
            n_steps=n_steps,
            window=window,
            n_features=extra_count,
            seed=seed + 100_003,
            seasonal_period=seasonal_period,
        )
        x = np.concatenate([x, extra_x], axis=-1)
        feature_names.extend(f"extra_{i}" for i in range(extra_count))
        scale = extra_target_scale(extra_x)
    else:
        scale = np.ones(x.shape[0], dtype=np.float32)

    expanded = dict(dataset)
    expanded["X"] = x.astype(np.float32)
    expanded["feature_names"] = np.array(feature_names)
    expanded["base_feature_count"] = np.array(BASE_X_DIMENSION, dtype=np.int64)
    expanded["extra_feature_count"] = np.array(extra_count, dtype=np.int64)
    return apply_extra_target_scale(expanded, scale)


def generate_model_dataset(
    model: str,
    n_steps: int,
    window: int,
    seed: int,
    noise_scale: float,
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
    x_dimension: int = BASE_X_DIMENSION,
) -> dict[str, np.ndarray]:
    """Generate one full dataset dictionary for a named model."""

    config = build_generation_config(
        n_steps=n_steps,
        window=window,
        seed=seed,
        noise_scale=noise_scale,
        seasonal_period=seasonal_period,
    )
    dataset = _generate_dataset(model, config)
    return add_extra_x_columns(
        dataset=dataset,
        x_dimension=x_dimension,
        n_steps=n_steps,
        window=window,
        seed=seed,
        seasonal_period=seasonal_period,
    )


def generate_model_datasets(
    models: Sequence[str],
    n_steps: int,
    window: int,
    seed: int,
    noise_scale: float,
    seasonal_period: int = DEFAULT_SEASONAL_PERIOD,
    x_dimension: int = BASE_X_DIMENSION,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate full dataset dictionaries for several named models."""

    return {
        model: generate_model_dataset(
            model=model,
            n_steps=n_steps,
            window=window,
            seed=seed,
            noise_scale=noise_scale,
            seasonal_period=seasonal_period,
            x_dimension=x_dimension,
        )
        for model in models
    }


def generate_supervised_dataset(
    model: str,
    num_samples: int,
    x_dimension: int,
    window: int,
    seed: int,
    noise_scale: float,
    seasonal_period: int,
) -> dict[str, np.ndarray]:
    """Generate exactly ``num_samples`` supervised pairs ``(X, y)``."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    return generate_model_dataset(
        model=model,
        n_steps=num_samples + window,
        window=window,
        seed=seed,
        noise_scale=noise_scale,
        seasonal_period=seasonal_period,
        x_dimension=x_dimension,
    )


def flatten_x_windows(x: np.ndarray) -> np.ndarray:
    """Flatten ``X`` from ``(n, lags, d)`` to ``(n, lags * d)``."""

    return np.asarray(x, dtype=np.float32).reshape(x.shape[0], -1)


def flattened_x_column_names(feature_names: Sequence[str], n_lags: int) -> list[str]:
    """Build stable column names for flattened lag-window output."""

    names: list[str] = []
    max_lag = n_lags - 1
    for lag_index in range(n_lags):
        lag_offset = lag_index - max_lag
        lag_name = "t" if lag_offset == 0 else f"t{lag_offset}"
        for feature in feature_names:
            names.append(f"X_{lag_name}_{feature}")
    return names


def supervised_table(dataset: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    """Return flattened ``X`` columns plus one final ``y`` column."""

    x = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32)
    feature_names = list(np.asarray(dataset["feature_names"]).astype(str))
    headers = [*flattened_x_column_names(feature_names, x.shape[1]), "y"]
    table = np.column_stack([flatten_x_windows(x), y])
    return headers, table


def write_csv_stdout(dataset: dict[str, np.ndarray]) -> None:
    """Print flattened supervised data as CSV."""

    headers, table = supervised_table(dataset)
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    writer.writerows(table.tolist())


def load_plotting_libraries():
    """Import plotting libraries lazily so non-image outputs stay lightweight."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Image output requires matplotlib and seaborn. Run `pip install -r requirements.txt` first."
        ) from exc
    return plt, sns


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a one-dimensional series with a centered moving average."""

    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def write_image_output(
    out: Path,
    dataset: dict[str, np.ndarray],
    rolling_window: int,
    show_rolling_mean: bool,
) -> None:
    """Save a compact image of ``y`` and current-time ``X`` columns."""

    plt, sns = load_plotting_libraries()
    x = np.asarray(dataset["X"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32)
    feature_names = list(np.asarray(dataset["feature_names"]).astype(str))
    current_x = x[:, -1, :]
    time = np.arange(len(y))
    n_panels = current_x.shape[1] + 1

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(15, max(3.0 * n_panels, 5.0)),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]

    target_ax = axes_flat[0]
    if {"q05", "q95"}.issubset(dataset):
        target_ax.fill_between(
            time,
            np.asarray(dataset["q05"], dtype=np.float32),
            np.asarray(dataset["q95"], dtype=np.float32),
            color="#7aa6c2",
            alpha=0.24,
            label="true 90% interval",
        )
    sns.lineplot(x=time, y=y, ax=target_ax, color="#111827", linewidth=1.2, label="y")
    if show_rolling_mean:
        sns.lineplot(
            x=time,
            y=rolling_mean(y, rolling_window),
            ax=target_ax,
            color="#c2410c",
            linewidth=2.0,
            label=f"{rolling_window}-step rolling mean",
        )
    target_ax.set_title("target y", pad=10)
    target_ax.set_ylabel("y")
    target_ax.legend(loc="upper left", frameon=True)

    palette = sns.color_palette("tab10", n_colors=current_x.shape[1])
    for feature_index, ax in enumerate(axes_flat[1:]):
        sns.lineplot(
            x=time,
            y=current_x[:, feature_index],
            ax=ax,
            color=palette[feature_index % len(palette)],
            linewidth=1.1,
            label=feature_names[feature_index],
        )
        ax.set_title(f"current X: {feature_names[feature_index]}", pad=10)
        ax.set_ylabel("value")
        ax.legend(loc="upper left", frameon=True)

    axes_flat[-1].set_xlabel("synthetic time index")
    fig.suptitle(
        f"Synthetic Supervised Weather Data ({len(y)} samples, {current_x.shape[1]} X columns)",
        y=0.995,
    )
    sns.despine(fig=fig)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metadata_json(dataset: dict[str, np.ndarray], model: str) -> str:
    """Return JSON metadata describing a generated supervised dataset."""

    x = np.asarray(dataset["X"])
    y = np.asarray(dataset["y"])
    metadata = {
        "model": model,
        "X_shape": list(x.shape),
        "y_shape": list(y.shape),
        "feature_names": list(np.asarray(dataset["feature_names"]).astype(str)),
        "base_feature_count": int(np.asarray(dataset["base_feature_count"])),
        "extra_feature_count": int(np.asarray(dataset["extra_feature_count"])),
        "extra_features_affect_y": bool(np.any(np.asarray(dataset["extra_target_scale"]) != 1.0)),
    }
    return json.dumps(metadata)


def write_output(
    out: Path,
    dataset: dict[str, np.ndarray],
    model: str,
    rolling_window: int,
    show_rolling_mean: bool,
) -> None:
    """Save supervised data to data or image formats."""

    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    headers, table = supervised_table(dataset)

    if suffix == ".csv":
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(table.tolist())
        return

    if suffix == ".npy":
        np.save(out, table)
        return

    if suffix == ".npz":
        save_fields = dict(dataset)
        save_fields["X_flat"] = flatten_x_windows(np.asarray(dataset["X"]))
        save_fields["flat_column_names"] = np.array(headers[:-1])
        save_fields["metadata_json"] = np.array(metadata_json(dataset, model))
        np.savez_compressed(out, **save_fields)
        return

    if suffix in IMAGE_SUFFIXES:
        write_image_output(out, dataset, rolling_window, show_rolling_mean)
        return

    raise ValueError("out must end in .csv, .npy, .npz, .png, .jpg, .jpeg, .pdf, or .svg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="narx_gaussian")
    parser.add_argument(
        "-d",
        "--dimension",
        type=int,
        default=BASE_X_DIMENSION,
        help=(
            "Number of X feature columns. Must be at least 6 because the target "
            "depends on the six base weather features."
        ),
    )
    parser.add_argument("-n", "--num-samples", type=int, default=500, help="Number of supervised rows.")
    parser.add_argument("--window", type=int, default=24, help="Lag-window length L.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--seasonal-period", type=int, default=DEFAULT_SEASONAL_PERIOD)
    parser.add_argument("--rolling-window", type=int, default=18)
    parser.add_argument("--no-rolling-mean", action="store_true", help="Hide rolling means in image output.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional .csv, .npy, .npz, or image output path. Use .npz to preserve lag-window X.",
    )
    args = parser.parse_args()

    dataset = generate_supervised_dataset(
        model=args.model,
        num_samples=args.num_samples,
        x_dimension=args.dimension,
        window=args.window,
        seed=args.seed,
        noise_scale=args.noise_scale,
        seasonal_period=args.seasonal_period,
    )

    if args.out is None:
        write_csv_stdout(dataset)
        return

    write_output(
        out=args.out,
        dataset=dataset,
        model=args.model,
        rolling_window=args.rolling_window,
        show_rolling_mean=not args.no_rolling_mean,
    )
    print(
        f"wrote {args.out} with model={args.model}, "
        f"X={dataset['X'].shape}, y={dataset['y'].shape}"
    )


if __name__ == "__main__":
    main()
