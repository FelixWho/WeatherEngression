"""Evaluate the partial-learner gridsearch (arch x wildfire-oversample) and
aggregate into one metrics JSON + an attenuation chart + a couple of money plots.

For each grid cell it computes the effect ``all(x) - clean(x)`` at the wildfire
test covariates (the partial-learner's ATT-analogue) plus per-arm calibration,
and overlays the T-learner ATT (from reports/t_learner_effect/<arch>/metrics.json)
as the reference the partial estimate should climb toward as oversampling rises.

Run (needs GPU for fast sampling):
    python -m experiments.partial_learner.aggregate_grid --grid-run-id 2251623
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_plgrid"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.partial_learner.test import load_saved, plot_money
from experiments.t_learner.test import arm_calibration, energy_loss, estimand

ARCHS = ["recurrent", "per_timestep", "global_latent"]
OVERSAMPLES = [1, 4, 8]
GRID_ROOT = ("runs/real_data_diagnostics/ena_weather/log10_ccn/paper/partial_learner_grid")
CKPT_ROOT = Path("/storage3/fs1/myu/Active/felixhu/weather_checkpoints")


def _t_learner_att(arch: str) -> float | None:
    """T-learner ATT ratio for this arch, if its metrics.json exists."""
    p = REPO_ROOT / "reports" / "t_learner_effect" / arch / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["effects"]["ATT"]["ratio"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid-run-id", required=True, help="SLURM array job id of the grid (dir suffix)")
    ap.add_argument("--wildfire-flag", default="BB_criterion1")
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--out-dir", default="reports/partial_learner_effect")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict[str, dict]] = {a: {} for a in ARCHS}
    money_saved: dict[str, np.ndarray] = {}   # for the illustrative money plots

    for arch in ARCHS:
        for over in OVERSAMPLES:
            name = f"{args.wildfire_flag}_{arch}_os{over}x_{args.grid_run_id}"
            ckpt = CKPT_ROOT / GRID_ROOT / name
            if not (ckpt / "checkpoint_all_best.pt").exists():
                print(f"  SKIP {name} (missing)", flush=True); continue
            eng_all, eng_clean, ds, _, tr_clean, te_wf, te_clean = load_saved(ckpt, device=device)
            xw = ds.x[te_wf]
            eff = estimand(eng_all, eng_clean, xw, args.n_samples)   # all - clean at wildfire x
            # calibration of the all-data arm on the full test set
            te_all = np.union1d(te_wf, te_clean)
            cal_all = arm_calibration(eng_all, ds.x[te_all], ds.y[te_all], args.n_samples)
            crps_all = energy_loss(eng_all, ds.x[te_all], ds.y[te_all], n_samples_per_x=args.n_samples)
            cells[arch][str(over)] = {
                "mean_log_effect": eff["mean_log_effect"],
                "ratio": eff["ratio"],
                "mean_raw_ccn_effect": eff["mean_raw_ccn_effect"],
                "all_crps": crps_all,
                "all_cov90": cal_all["coverage_90"],
            }
            print(f"  {arch:14s} os{over}x: ratio {eff['ratio']:.3f} | log10 {eff['mean_log_effect']:+.3f} "
                  f"| raw {eff['mean_raw_ccn_effect']:+.1f} | all cov90 {cal_all['coverage_90']:.3f}", flush=True)
            # keep recurrent 1x and 8x pooled draws for the money plots
            if arch == "recurrent" and over in (1, 8):
                money_saved[f"os{over}x_factual"] = eff["factual"]
                money_saved[f"os{over}x_counterfactual"] = eff["counterfactual"]

    t_att = {a: _t_learner_att(a) for a in ARCHS}
    metrics = {"grid_run_id": args.grid_run_id, "n_samples": args.n_samples,
               "cells": cells, "t_learner_att_ratio": t_att}
    (out_dir / "grid_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # --- attenuation chart: effect ratio vs oversampling, one line per arch ---
    colors = {"recurrent": "#4c72b0", "per_timestep": "#dd8452", "global_latent": "#55a868"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for arch in ARCHS:
        xs = [o for o in OVERSAMPLES if str(o) in cells[arch]]
        ys = [cells[arch][str(o)]["ratio"] for o in xs]
        if xs:
            ax.plot(xs, ys, marker="o", color=colors[arch], label=f"{arch} (partial)")
        if t_att[arch] is not None:
            ax.axhline(t_att[arch], ls="--", color=colors[arch], alpha=0.6,
                       label=f"{arch} T-learner ATT")
    ax.axhline(1.0, color="#888", lw=0.8, alpha=0.6)
    ax.set_xticks(OVERSAMPLES)
    ax.set_xlabel("wildfire oversample factor (1x = natural ~11%, 8x ~= balanced)")
    ax.set_ylabel("effect: with-wildfire / no-wildfire CCN  (x)")
    ax.set_title("All-data vs clean effect grows toward the T-learner as wildfire is upweighted")
    ax.legend(fontsize="x-small", ncol=2)
    fig.tight_layout(); fig.savefig(out_dir / "attenuation_curve.png", dpi=130); plt.close(fig)

    # --- money plots: recurrent 1x vs 8x (gap widening) ---
    for over in (1, 8):
        fk, ck = f"os{over}x_factual", f"os{over}x_counterfactual"
        if fk in money_saved:
            plot_money(money_saved[fk], money_saved[ck],
                       f"All-data vs clean model (recurrent, oversample {over}x)",
                       out_dir / f"money_recurrent_os{over}x.png")

    print(f"\nwrote grid_metrics.json + charts to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
