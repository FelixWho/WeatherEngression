#!/usr/bin/env python
"""Summarize the embedding-OOS diagnostic across the LSTM sweep.

For each run it reads ``metrics.json['embedding_oos']`` and reports, per config:
overall 90% coverage, and the coverage in the CLOSEST vs FARTHEST distance bin for
both kNN and Mahalanobis (so a drop closest->farthest = calibration degrading away
from training). Sorted by the kNN drop (largest degradation first).

    .venv/bin/python slurm/real_data_lstm_sweep_oos/aggregate_oos.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = (
    REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather"
    / "log10_ccn" / "paper" / "lstm_sweep_oos"
)


def _edges(bins: list[dict]) -> tuple[float, float]:
    """Return (closest-bin coverage, farthest-bin coverage)."""

    if not bins:
        return float("nan"), float("nan")
    return bins[0]["coverage_90"], bins[-1]["coverage_90"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.sweep_dir.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        oos = m.get("embedding_oos")
        if oos is None:
            continue
        knn_near, knn_far = _edges(oos["knn"]["coverage_by_distance_bin"])
        mah_near, mah_far = _edges(oos["mahalanobis"]["coverage_by_distance_bin"])
        rows.append(
            {
                "config": metrics_path.parent.name,
                "cov90": m.get("coverage_90", float("nan")),
                "knn_near": knn_near,
                "knn_far": knn_far,
                "knn_drop": knn_near - knn_far,
                "mah_near": mah_near,
                "mah_far": mah_far,
                "corr": oos.get("knn_mahalanobis_distance_corr", float("nan")),
            }
        )

    if not rows:
        raise SystemExit(
            f"no metrics.json with embedding_oos found under {args.sweep_dir}"
        )

    rows.sort(key=lambda r: -r["knn_drop"])
    header = (
        f"{'config':32s} {'cov90':>6s} | {'kNN near':>8s} {'kNN far':>8s} {'drop':>6s} | "
        f"{'Mah near':>8s} {'Mah far':>8s} | {'corr':>5s}"
    )
    print(f"sweep dir: {args.sweep_dir}")
    print(f"{len(rows)} runs; closest vs farthest distance bin (drop = near - far)\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:32s} {r['cov90']:6.3f} | {r['knn_near']:8.3f} {r['knn_far']:8.3f} "
            f"{r['knn_drop']:6.3f} | {r['mah_near']:8.3f} {r['mah_far']:8.3f} | {r['corr']:5.2f}"
        )


if __name__ == "__main__":
    main()
