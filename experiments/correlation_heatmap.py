"""Correlation heatmap of every ENA variable (22 channels + target log10(CCN)).

Each sample is summarized by its per-channel trajectory MEAN (one scalar per sample
per variable), matching the mean-mode scatter matrix; the target is appended as a
23rd variable. We then show two correlation matrices side by side:

* Pearson  -- linear correlation.
* Spearman -- rank correlation (captures any monotonic relationship, incl. the
              nonlinear ones visible in the scatter matrix). Computed as Pearson on
              column ranks, so no SciPy dependency.

Cells are annotated and colored on a diverging scale (-1 red .. 0 white .. +1 blue).

Shell use
---------
```bash
python -m experiments.correlation_heatmap --max-samples 20000 \
    --out reports/correlation_heatmap/correlation_heatmap.png
```
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import DEFAULT_MAT_PATH, load_ena_supervised_dataset


def _rank_columns(a: np.ndarray) -> np.ndarray:
    """Average-rank each column (ties share their mean rank) for Spearman."""
    out = np.empty_like(a, dtype=np.float64)
    n = a.shape[0]
    for j in range(a.shape[1]):
        order = np.argsort(a[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1)
        # average tied ranks
        col = a[:, j]
        uniq, inv, counts = np.unique(col, return_inverse=True, return_counts=True)
        csum = np.cumsum(counts)
        avg = (csum - counts + csum + 1) / 2.0
        out[:, j] = avg[inv]
    return out


def _draw_heatmap(ax, C, names, title):
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    k = len(names)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title(title, fontsize=11, pad=8)
    for i in range(k):
        for j in range(k):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0").replace("-0", "-"),
                    ha="center", va="center", fontsize=4.2,
                    color="white" if abs(v) > 0.55 else "#222222")
    return im


def build_heatmap(values: np.ndarray, names: list[str], out_path: Path, dpi: int = 200) -> None:
    pearson = np.corrcoef(values, rowvar=False)
    spearman = np.corrcoef(_rank_columns(values), rowvar=False)
    k = len(names)
    fig, axes = plt.subplots(1, 2, figsize=(2 * (0.42 * k + 1.5), 0.42 * k + 1.5))
    _draw_heatmap(axes[0], pearson, names, "Pearson (linear)")
    im = _draw_heatmap(axes[1], spearman, names, "Spearman (rank / monotonic)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("correlation", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return pearson, spearman


def main() -> None:
    p = argparse.ArgumentParser(description="Correlation heatmap of ENA variables.")
    p.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    p.add_argument("--max-samples", type=int, default=20000,
                   help="Random subset for the correlation estimate (cheap; more = steadier).")
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", type=str, default="reports/correlation_heatmap/correlation_heatmap.png")
    args = p.parse_args()

    print(f"loading up to {args.max_samples} random samples ...", flush=True)
    ds = load_ena_supervised_dataset(
        mat_path=args.mat_path, target="ccn", log_ccn=True,
        max_samples=args.max_samples, seq_stride=args.seq_stride, seed=args.seed,
    )
    chan_means = ds.x.mean(axis=1)                                   # (n, 22)
    values = np.concatenate([chan_means, ds.y.reshape(-1, 1)], axis=1)  # (n, 23)
    names = list(ds.feature_names) + [ds.target_name]
    print(f"variables ({len(names)}): {names}", flush=True)
    print(f"correlation over {len(values)} samples", flush=True)

    out = Path(args.out)
    pearson, spearman = build_heatmap(values, names, out, dpi=args.dpi)

    # print the target's correlations (last row) so the log is self-contained
    tgt = names[-1]
    order_p = np.argsort(np.abs(pearson[-1, :-1]))[::-1]
    print(f"\nStrongest |Pearson| with {tgt}:", flush=True)
    for j in order_p[:8]:
        print(f"  {names[j]:14s} pearson {pearson[-1, j]:+.3f}   spearman {spearman[-1, j]:+.3f}",
              flush=True)
    print(f"\nwrote {out} and {out.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
