"""Generate and plot a synthetic single-target weather time series.

The plotted variable is the generated target ``y``. The input ``X`` is still a
multivariate lag-window weather history, but this script visualizes the single
response series produced by one synthetic data-generating model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from generate_data import MODEL_NAMES, generate_model_dataset, generate_model_datasets


RUN_MODEL_NAMES = (*MODEL_NAMES, "all")


def load_plotting_libraries():
    """Import plotting libraries lazily so ``--help`` still works."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Plotting requires matplotlib and seaborn. Run `pip install -r requirements.txt` first."
        ) from exc
    return plt, sns


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Smooth a one-dimensional series with a centered moving average."""
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def truncate_dataset(dataset: dict[str, np.ndarray], plot_length: int | None) -> dict[str, np.ndarray]:
    """Return a shallow dataset copy with row-wise fields shortened for plotting."""
    if plot_length is None:
        return dataset

    n = len(dataset["y"])
    limit = min(plot_length, n)
    truncated: dict[str, np.ndarray] = {}
    for key, value in dataset.items():
        array = np.asarray(value)
        if array.shape[:1] == (n,):
            truncated[key] = array[:limit]
        else:
            truncated[key] = array
    return truncated


def draw_target_panel(
    sns,
    ax,
    dataset: dict[str, np.ndarray],
    model: str,
    rolling_window: int,
    show_rolling_mean: bool,
) -> None:
    """Draw the target sample, true quantile band, and optional trend line."""
    y = np.asarray(dataset["y"], dtype=np.float64)
    time = np.arange(len(y))
    q05 = np.asarray(dataset["q05"], dtype=np.float64)[: len(y)]
    q50 = np.asarray(dataset["q50"], dtype=np.float64)[: len(y)]
    q95 = np.asarray(dataset["q95"], dtype=np.float64)[: len(y)]

    ax.fill_between(
        time,
        q05,
        q95,
        color="#7aa6c2",
        alpha=0.24,
        label="true 90% interval",
    )
    sns.lineplot(
        x=time,
        y=q50,
        ax=ax,
        color="#2563eb",
        linewidth=1.7,
        alpha=0.85,
        label="true conditional median",
    )
    sns.lineplot(
        x=time,
        y=y,
        ax=ax,
        color="#111827",
        linewidth=0.9,
        alpha=0.58,
        label="sampled target y",
    )

    if show_rolling_mean:
        trend = rolling_mean(y, rolling_window)
        sns.lineplot(
            x=time,
            y=trend,
            ax=ax,
            color="#c2410c",
            linewidth=2.5,
            label=f"{rolling_window}-step rolling mean",
        )

    if "sampled_regime" in dataset:
        regimes = np.asarray(dataset["sampled_regime"])[: len(y)]
        sns.scatterplot(
            x=time,
            y=y,
            hue=regimes,
            palette="Set2",
            s=18,
            alpha=0.52,
            linewidth=0,
            ax=ax,
            legend=False,
        )

    ax.set_title(f"{model}", pad=12)
    ax.set_ylabel("target y")
    ax.margins(x=0)
    ax.legend(loc="upper left", frameon=True)


def draw_diagnostic_panel(sns, ax, dataset: dict[str, np.ndarray], model: str) -> bool:
    """Draw a model-specific panel. Return False if no diagnostic is available."""
    n = len(dataset["y"])
    time = np.arange(n)

    if model == "regime_mixture":
        weights = np.asarray(dataset["mixture_weights"], dtype=np.float64)[:n]
        ax.stackplot(
            time,
            weights[:, 0],
            weights[:, 1],
            weights[:, 2],
            labels=["bright/dry", "wet/humid", "transport"],
            colors=["#facc15", "#38bdf8", "#a78bfa"],
            alpha=0.75,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("regime prob.")
        ax.legend(loc="upper left", ncol=3, frameon=True)
        return True

    if model == "hurdle_lognormal":
        p_wet = np.asarray(dataset["p_wet"], dtype=np.float64)[:n]
        wet_event = np.asarray(dataset["wet_event"], dtype=bool)[:n]
        sns.lineplot(x=time, y=p_wet, ax=ax, color="#0284c7", linewidth=2.0, label="P(Y > 0 | X)")
        ax.scatter(
            time[wet_event],
            np.full(wet_event.sum(), 0.05),
            color="#111827",
            marker="|",
            s=80,
            alpha=0.6,
            label="sampled wet event",
        )
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("wet prob.")
        ax.legend(loc="upper left", frameon=True)
        return True

    if model == "narx_gaussian":
        sigma = np.asarray(dataset["sigma"], dtype=np.float64)[:n]
        sns.lineplot(x=time, y=sigma, ax=ax, color="#7c3aed", linewidth=2.0, label="conditional sigma")
        ax.set_ylabel("sigma")
        ax.legend(loc="upper left", frameon=True)
        return True

    if model == "narx_student_t":
        scale = np.asarray(dataset["scale"], dtype=np.float64)[:n]
        sns.lineplot(x=time, y=scale, ax=ax, color="#7c3aed", linewidth=2.0, label="Student-t scale")
        ax.set_ylabel("scale")
        ax.legend(loc="upper left", frameon=True)
        return True

    if model == "narx_garch":
        sigma = np.asarray(dataset["sigma"], dtype=np.float64)[:n]
        base_sigma = np.asarray(dataset["base_sigma"], dtype=np.float64)[:n]
        sns.lineplot(x=time, y=sigma, ax=ax, color="#7c3aed", linewidth=2.0, label="GARCH sigma")
        sns.lineplot(
            x=time,
            y=base_sigma,
            ax=ax,
            color="#64748b",
            linewidth=1.4,
            linestyle="--",
            label="weather base sigma",
        )
        ax.set_ylabel("sigma")
        ax.legend(loc="upper left", frameon=True)
        return True

    if model == "preadditive":
        phi = np.asarray(dataset["phi"], dtype=np.float64)[:n]
        sns.lineplot(x=time, y=phi, ax=ax, color="#16a34a", linewidth=2.0, label="latent phi(X)")
        ax.set_ylabel("phi")
        ax.legend(loc="upper left", frameon=True)
        return True

    return False


def plot_timeseries(
    dataset: dict[str, np.ndarray],
    model: str,
    out: Path,
    rolling_window: int,
    plot_length: int | None,
    show_rolling_mean: bool,
    show_diagnostics: bool,
    show: bool,
) -> None:
    """Create a seaborn-styled plot for one generated target series."""
    plt, sns = load_plotting_libraries()
    dataset = truncate_dataset(dataset, plot_length)

    sns.set_theme(style="whitegrid", context="talk")
    if show_diagnostics:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(15, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.15]},
        )
        target_ax, diagnostic_ax = axes
    else:
        fig, target_ax = plt.subplots(figsize=(15, 7))
        diagnostic_ax = None

    draw_target_panel(
        sns=sns,
        ax=target_ax,
        dataset=dataset,
        model=model,
        rolling_window=rolling_window,
        show_rolling_mean=show_rolling_mean,
    )

    if diagnostic_ax is not None:
        has_diagnostic = draw_diagnostic_panel(sns, diagnostic_ax, dataset, model)
        if has_diagnostic:
            diagnostic_ax.set_xlabel("synthetic time index")
            diagnostic_ax.margins(x=0)
        else:
            diagnostic_ax.remove()
            target_ax.set_xlabel("synthetic time index")
    else:
        target_ax.set_xlabel("synthetic time index")

    fig.suptitle("Synthetic Single-Target Weather Time Series", y=0.99)
    sns.despine(fig=fig)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote plot to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_models(
    datasets: dict[str, dict[str, np.ndarray]],
    out: Path,
    rolling_window: int,
    plot_length: int | None,
    show_rolling_mean: bool,
    show_diagnostics: bool,
    show: bool,
) -> None:
    """Create a compact gallery comparing all supported generators."""
    plt, sns = load_plotting_libraries()
    sns.set_theme(style="whitegrid", context="talk")

    models = list(MODEL_NAMES)
    n_cols = 2 if show_diagnostics else 1
    fig, axes = plt.subplots(
        len(models),
        n_cols,
        figsize=(18 if show_diagnostics else 15, 4.0 * len(models)),
        squeeze=False,
        sharex=False,
    )

    for row, model in enumerate(models):
        dataset = truncate_dataset(datasets[model], plot_length)
        draw_target_panel(
            sns=sns,
            ax=axes[row, 0],
            dataset=dataset,
            model=model,
            rolling_window=rolling_window,
            show_rolling_mean=show_rolling_mean,
        )
        axes[row, 0].set_xlabel("synthetic time index")

        if show_diagnostics:
            draw_diagnostic_panel(sns, axes[row, 1], dataset, model)
            axes[row, 1].set_xlabel("synthetic time index")
            axes[row, 1].margins(x=0)

    fig.suptitle("Synthetic Weather Generators", y=0.995)
    sns.despine(fig=fig)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote plot to {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RUN_MODEL_NAMES, default="regime_mixture")
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--noise-scale", type=float, default=0.35)
    parser.add_argument("--rolling-window", type=int, default=18)
    parser.add_argument("--no-rolling-mean", action="store_true", help="Hide the rolling mean trend line.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Hide model-specific diagnostic panels.")
    parser.add_argument(
        "--plot-length",
        type=int,
        default=220,
        help="Number of generated target points to show. Use 0 to show the full generated range.",
    )
    parser.add_argument("--out", type=Path, default=Path("figures/synthetic_weather_timeseries.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.model == "all":
        datasets = generate_model_datasets(
            models=MODEL_NAMES,
            n_steps=args.n_steps,
            window=args.window,
            seed=args.seed,
            noise_scale=args.noise_scale,
        )
        plot_all_models(
            datasets=datasets,
            out=args.out,
            rolling_window=args.rolling_window,
            plot_length=args.plot_length or None,
            show_rolling_mean=not args.no_rolling_mean,
            show_diagnostics=not args.no_diagnostics,
            show=args.show,
        )
    else:
        dataset = generate_model_dataset(
            model=args.model,
            n_steps=args.n_steps,
            window=args.window,
            seed=args.seed,
            noise_scale=args.noise_scale,
        )
        plot_timeseries(
            dataset=dataset,
            model=args.model,
            out=args.out,
            rolling_window=args.rolling_window,
            plot_length=args.plot_length or None,
            show_rolling_mean=not args.no_rolling_mean,
            show_diagnostics=not args.no_diagnostics,
            show=args.show,
        )


if __name__ == "__main__":
    main()
