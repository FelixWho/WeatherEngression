#!/usr/bin/env python
"""Aggregate the vanilla-engression sweep into a leaderboard.

Scans every ``metrics.json`` under the sweep folder and prints a table sorted by
calibration error (|coverage_90 - 0.90|), with the energy score (CRPS) and the
median absolute error alongside. Run after the array job finishes:

    .venv/bin/python slurm/real_data_vanilla_sweep/aggregate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = (
    REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather"
    / "log10_ccn" / "paper" / "vanilla_sweep"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--nominal", type=float, default=0.90)
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted(args.sweep_dir.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        rows.append(
            {
                "config": metrics_path.parent.name,
                "cov90": m.get("coverage_90", float("nan")),
                "cov50": m.get("coverage_50", float("nan")),
                "energy": m.get("energy_score", float("nan")),
                "mae": m.get("median_abs_error", float("nan")),
                "width90": m.get("mean_width_90", float("nan")),
            }
        )

    if not rows:
        raise SystemExit(f"no metrics.json found under {args.sweep_dir}")

    rows.sort(key=lambda r: abs(r["cov90"] - args.nominal))
    header = f"{'config':28s} {'cov90':>7s} {'cov50':>7s} {'energy':>8s} {'MAE':>7s} {'width90':>8s}"
    print(f"sweep dir: {args.sweep_dir}")
    print(f"{len(rows)} runs; sorted by |coverage_90 - {args.nominal:.2f}| (best calibrated first)\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:28s} {r['cov90']:7.3f} {r['cov50']:7.3f} "
            f"{r['energy']:8.4f} {r['mae']:7.3f} {r['width90']:8.3f}"
        )


if __name__ == "__main__":
    main()
