"""Plotting for real-data engression diagnostics.

Kept separate from ``plotting.py`` because real runs have no generator-truth
band to overlay: the chart shows the predicted conditional bands against the
held-out realized targets, sorted by the predictive median.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np

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
            "Plotting requires matplotlib and seaborn. Run "
            "`.venv/bin/pip install -r requirements.txt` first."
        ) from exc
    return plt, sns


def write_coverage_vs_distance_chart(
    out: Path,
    target_name: str,
    knn_bins: list[dict[str, object]],
    maha_bins: list[dict[str, object]],
    nominal: float = 0.90,
    show: bool = False,
    model_label: str = "engression",
) -> None:
    """Chart 90% coverage stratified by OOD distance, for both distance metrics.

    Each line is coverage per equal-count distance bin (left = closest to the
    training feature space, right = farthest). A drop toward the right is the
    expected signature of calibration degrading away from training support.
    """

    plt, sns = load_plotting_libraries()
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))
    for bins, label, color in (
        (knn_bins, "kNN distance (embedding)", "#4c72b0"),
        (maha_bins, "Mahalanobis distance (embedding)", "#c44e52"),
    ):
        if not bins:
            continue
        x_axis = [row["bin"] for row in bins]
        coverage = [row["coverage_90"] for row in bins]
        ax.plot(x_axis, coverage, marker="o", color=color, label=label)
    ax.axhline(nominal, ls="--", color="#2a2a2a", alpha=0.7, label=f"nominal {nominal:.2f}")
    ax.set_xlabel("distance bin (0 = closest to training, last = farthest)")
    ax.set_ylabel("empirical 90% coverage")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"ENA {target_name} [{model_label}] - coverage vs distance from training")
    ax.legend(loc="lower left", fontsize="small", framealpha=0.9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)


def write_pit_histogram_chart(
    out: Path,
    target_name: str,
    pit: np.ndarray,
    n_bins: int = 20,
    show: bool = False,
    model_label: str = "engression",
) -> None:
    """PIT / rank histogram: the shape-calibration check across ALL quantile levels.

    Each test point's realized target is mapped to where it falls in its own
    predictive sample distribution (PIT). Under correct predictive SHAPE these
    values are Uniform(0, 1), so the bars should sit on the flat line inside the
    consistency band. Read the departures:
      U-shaped -> intervals too narrow;  dome -> too wide;
      sloped   -> location bias;         asymmetric -> skew/tail mismatch.
    Unlike 50%/90% coverage (two points), this judges the whole distribution.
    """

    plt, sns = load_plotting_libraries()
    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    n = pit.size

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(11, 7))
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    ax.bar(
        edges[:-1], counts, width=np.diff(edges), align="edge",
        color="#4c72b0", alpha=0.75, edgecolor="white", label="PIT histogram",
    )
    # Uniform expectation and 95% binomial consistency band (per bin ~ Binom(n, 1/B)).
    expected = n / n_bins
    half = 1.96 * np.sqrt(n * (1.0 / n_bins) * (1.0 - 1.0 / n_bins))
    ax.axhline(expected, ls="--", color="#2a2a2a", alpha=0.8, label="uniform (calibrated)")
    ax.fill_between(
        [0.0, 1.0], expected - half, expected + half,
        color="#dd8452", alpha=0.20, label="95% consistency band",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(counts.max(), expected + half) * 1.12)
    ax.set_xlabel("PIT value  (rank of realized target within its predictive samples)")
    ax.set_ylabel("count")
    ax.set_title(f"ENA {target_name} [{model_label}] - PIT histogram (shape calibration)")
    ax.legend(loc="upper center", fontsize="small", framealpha=0.9, ncol=2)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)


def write_coverage_calibration_curve_chart(
    out: Path,
    target_name: str,
    pit: np.ndarray,
    show: bool = False,
    model_label: str = "engression",
) -> None:
    """Reliability curve: empirical vs nominal coverage across EVERY level.

    The 50%/90% coverage numbers are two points on this curve. Plotting the PIT
    empirical CDF against the diagonal shows, for each nominal quantile level tau,
    the fraction of targets actually falling below the predicted tau-quantile.
    On the diagonal = calibrated at that level; above = predicted quantile too
    high (over-covering below it), below = too low. Bows reveal WHICH part of the
    distribution (center vs tails) is mis-estimated.
    """

    plt, sns = load_plotting_libraries()
    pit = np.sort(np.asarray(pit, dtype=np.float64).reshape(-1))
    n = pit.size
    nominal = np.linspace(0.0, 1.0, 101)
    # empirical CDF of PIT at each nominal level = fraction of targets below that quantile
    empirical = np.searchsorted(pit, nominal, side="right") / max(n, 1)

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.plot([0, 1], [0, 1], ls="--", color="#2a2a2a", alpha=0.8, label="perfect calibration")
    ax.plot(nominal, empirical, color="#4c72b0", lw=2.2, label="engression")
    ax.fill_between(nominal, nominal, empirical, color="#4c72b0", alpha=0.12)
    for lvl in (0.05, 0.25, 0.50, 0.75, 0.95):  # mark the levels the band charts use
        ax.scatter([lvl], [np.searchsorted(pit, lvl, side="right") / max(n, 1)],
                   color="#c44e52", zorder=5, s=45)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("nominal quantile level  tau")
    ax.set_ylabel("empirical fraction below predicted quantile")
    ax.set_title(
        f"ENA {target_name} [{model_label}]\ncoverage calibration curve", fontsize="medium"
    )
    ax.legend(loc="upper left", fontsize="small", framealpha=0.9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)


def write_pit_by_distance_chart(
    out: Path,
    target_name: str,
    pit: np.ndarray,
    distances: np.ndarray,
    n_bins: int = 5,
    distance_label: str = "kNN distance (embedding)",
    show: bool = False,
    model_label: str = "engression",
) -> None:
    """Stratified PIT: does the predictive SHAPE degrade away from training support?

    Test points are split into equal-count bins of distance-from-training (the
    same OOS axis as ``coverage_vs_distance``), and a PIT histogram is drawn per
    bin (left = closest to training, right = farthest). Flat near the training
    data turning U-shaped / sloped in the far bins is the signature of a model
    whose conditional shape (not just its intervals) breaks down out of support --
    the conditional-calibration view the pooled histogram can hide.
    """

    plt, sns = load_plotting_libraries()
    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    order = np.argsort(distances, kind="stable")
    groups = [g for g in np.array_split(order, n_bins) if len(g)]

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 4.2), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    hist_bins = 10
    for i, (ax, idx) in enumerate(zip(axes, groups)):
        counts, edges = np.histogram(pit[idx], bins=hist_bins, range=(0.0, 1.0))
        expected = len(idx) / hist_bins
        half = 1.96 * np.sqrt(len(idx) * (1.0 / hist_bins) * (1.0 - 1.0 / hist_bins))
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
               color="#4c72b0", alpha=0.75, edgecolor="white")
        ax.axhline(expected, ls="--", color="#2a2a2a", alpha=0.8)
        ax.fill_between([0.0, 1.0], expected - half, expected + half, color="#dd8452", alpha=0.20)
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"bin {i}  (mean d={distances[idx].mean():.2f})", fontsize="small")
        ax.set_xlabel("PIT", fontsize="small")
    axes[0].set_ylabel("count")
    fig.suptitle(
        f"ENA {target_name} [{model_label}] - PIT by {distance_label} "
        f"(bin 0 = closest to training, bin {len(groups)-1} = farthest)",
        fontsize="medium",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)


def write_real_posterior_band_chart(
    out: Path,
    target_name: str,
    split: str,
    y_test: np.ndarray,
    predicted_quantiles: np.ndarray,
    levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    show: bool = False,
    model_label: str = "engression",
) -> None:
    """Chart predicted conditional bands and realized targets, sorted by median."""

    plt, sns = load_plotting_libraries()
    cols = {level: i for i, level in enumerate(levels)}
    q05 = predicted_quantiles[:, cols[0.05]]
    q25 = predicted_quantiles[:, cols[0.25]]
    q50 = predicted_quantiles[:, cols[0.50]]
    q75 = predicted_quantiles[:, cols[0.75]]
    q95 = predicted_quantiles[:, cols[0.95]]

    order = np.argsort(q50)
    x_axis = np.arange(len(order))
    y_ordered = np.asarray(y_test).reshape(-1)[order]

    covered = (y_ordered >= q05[order]) & (y_ordered <= q95[order])
    coverage = float(np.mean(covered)) if len(covered) else float("nan")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(
        x_axis, q05[order], q95[order], alpha=0.25, color="#4c72b0", label="predicted 90% band"
    )
    ax.fill_between(
        x_axis, q25[order], q75[order], alpha=0.40, color="#4c72b0", label="predicted 50% band"
    )
    ax.plot(x_axis, q50[order], color="#1f3b73", lw=1.5, label="predicted median")
    ax.scatter(
        x_axis[covered], y_ordered[covered], s=10, color="#2a2a2a", alpha=0.6, label="realized (in band)"
    )
    ax.scatter(
        x_axis[~covered], y_ordered[~covered], s=14, color="#c44e52", alpha=0.8, label="realized (outside)"
    )
    ax.set_xlabel("held-out sample, sorted by predicted median")
    ax.set_ylabel(target_name)
    ax.set_title(
        f"ENA {target_name} [{model_label}] ({split} split)\n"
        f"empirical 90% coverage = {coverage:.3f} (nominal 0.90)"
    )
    ax.legend(loc="upper left", fontsize="small", framealpha=0.9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)
