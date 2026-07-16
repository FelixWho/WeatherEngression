#!/usr/bin/env python
"""Aggregate the LSTM-engression head sweep into a leaderboard.

Scans every ``metrics.json`` under the sweep folder and prints a table sorted by
calibration error (|coverage_90 - 0.90|), with energy score (CRPS), median
absolute error, and interval width alongside. The config (head + lr + layers +
hidden) is read from each run's folder name. Run after the array job finishes:

    .venv/bin/python slurm/real_data_lstm_sweep/aggregate.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SWEEP_DIR = (
    REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather"
    / "log10_ccn" / "paper" / "lstm_sweep"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument("--nominal", type=float, default=0.90)
    parser.add_argument(
        "--sort-by",
        choices=("coverage", "energy"),
        default="coverage",
        help="coverage = |cov90 - nominal| (calibration); energy = CRPS (distributional fit).",
    )
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

    if args.sort_by == "coverage":
        rows.sort(key=lambda r: abs(r["cov90"] - args.nominal))
        sort_note = f"|coverage_90 - {args.nominal:.2f}| (best calibrated first)"
    else:
        rows.sort(key=lambda r: r["energy"])
        sort_note = "energy score / CRPS (best distributional fit first)"

    header = f"{'config':32s} {'cov90':>7s} {'cov50':>7s} {'energy':>8s} {'MAE':>7s} {'width90':>8s}"
    print(f"sweep dir: {args.sweep_dir}")
    print(f"{len(rows)} runs; sorted by {sort_note}\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['config']:32s} {r['cov90']:7.3f} {r['cov50']:7.3f} "
            f"{r['energy']:8.4f} {r['mae']:7.3f} {r['width90']:8.3f}"
        )


if __name__ == "__main__":
    main()
