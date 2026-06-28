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


def write_real_posterior_band_chart(
    out: Path,
    target_name: str,
    split: str,
    y_test: np.ndarray,
    predicted_quantiles: np.ndarray,
    levels: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    show: bool = False,
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
        f"ENA {target_name} - LSTM engression ({split} split)\n"
        f"empirical 90% coverage = {coverage:.3f} (nominal 0.90)"
    )
    ax.legend(loc="upper left", fontsize="small", framealpha=0.9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    if show:  # pragma: no cover - interactive only
        plt.show()
    plt.close(fig)
