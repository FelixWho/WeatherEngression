"""Pairwise scatter matrix (SPLOM) of every variable in the ENA dataset.

Each sample is a (241, 22) back-trajectory; we summarize each channel by its
per-sample trajectory MEAN, giving one scalar per sample per variable. Together
with the target log10(CCN) that is 23 variables, and we draw a scatter for every
pair (diagonal = the variable's own histogram + name). Output is a single-page PDF.

Points are rasterized (small file) while axes/text stay vector (crisp when zoomed).

Shell use
---------
```bash
python -m experiments.pairwise_scatter --max-samples 6000 \
    --out reports/pairwise_scatter/pairwise_scatter.pdf
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


def build_scatter_matrix(values: np.ndarray, names: list[str], out_path: Path,
                         point_size: float = 1.2, alpha: float = 0.18,
                         panel_inches: float = 1.7, dpi: int = 150) -> None:
    """values: (n, k). Draw a k x k SPLOM to a single-page PDF.

    ``dpi`` sets the rasterization resolution of the scatter points (text stays
    vector); ``panel_inches`` sets each subplot's size. Raise both for a sharper,
    larger figure at the cost of file size.
    """
    k = values.shape[1]
    fig, axes = plt.subplots(k, k, figsize=(panel_inches * k, panel_inches * k))
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.03, right=0.99, top=0.99, bottom=0.03)

    for i in range(k):          # row -> y variable
        for j in range(k):      # col -> x variable
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            if i == j:
                ax.hist(values[:, i], bins=40, color="#34495e", alpha=0.85)
                ax.set_facecolor("#f4f6f7")
                ax.text(0.5, 0.5, names[i], transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, fontweight="bold",
                        color="#c0392b",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            else:
                ax.scatter(values[:, j], values[:, i], s=point_size, alpha=alpha,
                           c="#2c3e50", edgecolors="none", rasterized=True)
            # label only the outer edges so the grid stays clean
            if j == 0:
                ax.set_ylabel(names[i], fontsize=6, rotation=0, ha="right", va="center")
            if i == k - 1:
                ax.set_xlabel(names[j], fontsize=6, rotation=90, va="top")

    fig.suptitle("", y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Render the ENTIRE figure as ONE raster (PNG), then wrap it into a one-page PDF
    # with Pillow. Saving PDF directly invokes matplotlib's mixed-mode renderer, which
    # rasterizes each of the k*k scatter artists separately and OOMs at high dpi even
    # with 96G. A single PNG buffer is bounded (~(panel*k*dpi)^2 * 4 bytes) and safe.
    from PIL import Image, ImageFile
    Image.MAX_IMAGE_PIXELS = None           # our own render, not an untrusted upload
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # allow full decode of the large PNG
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    im = Image.open(png_path); im.load()
    im.convert("RGB").save(out_path, "PDF", resolution=float(dpi))


def main() -> None:
    p = argparse.ArgumentParser(description="Pairwise scatter matrix of ENA variables.")
    p.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    p.add_argument("--max-samples", type=int, default=6000,
                   help="Random subset of samples to plot (keeps the PDF light).")
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dpi", type=int, default=150,
                   help="Rasterization resolution of the scatter points (text stays vector).")
    p.add_argument("--panel-inches", type=float, default=1.7,
                   help="Size of each subplot in inches; larger => bigger, sharper panels.")
    p.add_argument("--per-timestep", action="store_true",
                   help="Use every (sample, timestep) as its own point instead of the "
                        "per-sample trajectory mean. Far more points -> use fewer samples "
                        "and lower --alpha.")
    p.add_argument("--point-size", type=float, default=1.2)
    p.add_argument("--alpha", type=float, default=0.18)
    p.add_argument("--out", type=str, default="reports/pairwise_scatter/pairwise_scatter.pdf")
    args = p.parse_args()

    print(f"loading up to {args.max_samples} random samples ...", flush=True)
    ds = load_ena_supervised_dataset(
        mat_path=args.mat_path, target="ccn", log_ccn=True,
        max_samples=args.max_samples, seq_stride=args.seq_stride, seed=args.seed,
    )
    print(f"loaded x={ds.x.shape}  y={ds.y.shape}  channels={len(ds.feature_names)}", flush=True)

    n, T, d = ds.x.shape
    if args.per_timestep:
        # Every (sample, timestep) becomes its own point: pool all timesteps of the
        # selected samples. The target log10(CCN) is per-sample (no time axis), so we
        # broadcast each sample's value across its T timesteps -> CCN panels show each
        # trajectory as a horizontal band at its outcome level.
        chan = ds.x.reshape(n * T, d)                     # (n*T, 22)
        y_col = np.repeat(ds.y, T).reshape(-1, 1)         # (n*T, 1), broadcast CCN
        values = np.concatenate([chan, y_col], axis=1)    # (n*T, 23)
        print(f"per-timestep: {n} samples x {T} timesteps = {len(values)} points/panel "
              f"(CCN column broadcast across timesteps)", flush=True)
    else:
        # per-sample trajectory mean for each channel (one point per sample)
        chan_means = ds.x.mean(axis=1)                    # (n, 22)
        values = np.concatenate([chan_means, ds.y.reshape(-1, 1)], axis=1)  # (n, 23)
        print(f"per-sample means: {len(values)} points/panel", flush=True)
    names = list(ds.feature_names) + [ds.target_name]    # 22 channels + log10(CCN)
    print(f"scatter-matrix variables ({len(names)}): {names}", flush=True)

    out = Path(args.out)
    build_scatter_matrix(values, names, out, point_size=args.point_size, alpha=args.alpha,
                         panel_inches=args.panel_inches, dpi=args.dpi)
    size_kb = out.stat().st_size // 1024
    print(f"wrote {out}  ({len(names)}x{len(names)} panels, {len(values)} points, {size_kb} KB)",
          flush=True)


if __name__ == "__main__":
    main()
