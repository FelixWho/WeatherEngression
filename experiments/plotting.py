"""Plotting utilities for synthetic engression diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

from .constants import format_model_name


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "weatherengression_mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("MPLBACKEND", "Agg")


def load_plotting_libraries():
    """Import plotting libraries lazily so non-plot commands stay lightweight."""

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

    fig.suptitle(f"Engression Diagnostic Posterior Bands ({split})", y=0.99)
    sns.despine(fig=fig)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  wrote posterior chart: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)
