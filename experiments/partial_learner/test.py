"""Partial-learner analysis: contrast the ALL-DATA arm against the CLEAN arm.

Mirror of ``experiments.t_learner.test`` but the two models are ``all`` (trained
on clean + wildfire) and ``clean``. Reuses the T-learner's metric/plot helpers;
only the checkpoint names, the pickled index names, and the money-plot labels
differ.

The effect here is ``all(x) - clean(x)``. Because the all-data arm is a blend of
the wildfire and clean conditionals, this is a KNOWN-attenuated estimate. Run it
alongside the T-learner to see how much smaller it comes out.

Run:
    python -m experiments.partial_learner.test --checkpoint-dir /storage3/.../<run>
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Headless plotting (compute nodes have no display / writable HOME cache).
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_partial"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/partial_learner/test.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import ENADataset  # noqa: F401 (needed to unpickle dataset_obj.pkl)
from engression_modifications.lstm import load_lstm_engressor_checkpoint
from experiments.t_learner.train import _resolve_save_dir
# Reuse the generic analysis helpers; they take (engressor, x, y) and don't care
# which arm they're pointed at.
from experiments.t_learner.test import (
    arm_calibration,
    energy_loss,
    estimand,
    plot_pit,
    sample_conditional,  # noqa: F401 (used indirectly via estimand/arm_calibration)
)


def load_saved(save_checkpoint_dir: str | Path, *, which: str = "best", device: str | None = None):
    """Reload the all + clean arms and the pickled dataset / indices (no retraining).

    Returns ``(eng_all, eng_clean, dataset, train_all_idx, train_clean_idx,
    test_wildfire_idx, test_no_wildfire_idx)``.
    """
    ckpt_dir = _resolve_save_dir(save_checkpoint_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engressor_all = load_lstm_engressor_checkpoint(ckpt_dir / f"checkpoint_all_{which}.pt", device=device)
    engressor_clean = load_lstm_engressor_checkpoint(ckpt_dir / f"checkpoint_clean_{which}.pt", device=device)

    ds_dir = ckpt_dir / "dataset"
    if not (ds_dir / "dataset_obj.pkl").exists():
        raise FileNotFoundError(f"no pickled dataset under {ds_dir}; retrain with partial_learner.train.")

    def _unpickle(name: str):
        with open(ds_dir / name, "rb") as f:
            return pickle.load(f)

    dataset: ENADataset = _unpickle("dataset_obj.pkl")
    return (
        engressor_all,
        engressor_clean,
        dataset,
        _unpickle("train_all_idx.pkl"),
        _unpickle("train_clean_idx.pkl"),
        _unpickle("test_wildfire_idx.pkl"),
        _unpickle("test_no_wildfire_idx.pkl"),
    )


def plot_money(all_samples: np.ndarray, clean_samples: np.ndarray, title: str, out_path: Path) -> None:
    """All-data model vs clean model, pooled distributions (labels for this setup)."""
    lo = float(min(all_samples.min(), clean_samples.min()))
    hi = float(max(all_samples.max(), clean_samples.max()))
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean_samples, bins=bins, density=True, alpha=0.55, color="#55a868", label="clean model")
    ax.hist(all_samples, bins=bins, density=True, alpha=0.55, color="#c44e52", label="all-data model")
    ax.axvline(clean_samples.mean(), color="#55a868", ls="--", lw=1.5)
    ax.axvline(all_samples.mean(), color="#c44e52", ls="--", lw=1.5)
    ax.set_xlabel("log10(CCN)"); ax.set_ylabel("density"); ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Contrast the all-data arm against the clean arm (partial-learner).")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Dir with checkpoint_{all,clean}_best.pt + dataset/ pickles (abs, or under storage3).")
    p.add_argument("--which", choices=("best", "latest"), default="best")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--n-samples", type=int, default=400, help="conditional draws per covariate")
    p.add_argument("--out-dir", type=str, default="reports/partial_learner_effect")
    p.add_argument("--seed", type=int, default=0, help="seed for the randomized PIT tie-break")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    print("Parsed arguments:", flush=True)
    for name, value in sorted(vars(args).items()):
        print(f"  {name}: {value}", flush=True)

    (
        eng_all,
        eng_clean,
        dataset,
        train_all_idx,
        train_clean_idx,
        test_wildfire_idx,
        test_no_wildfire_idx,
    ) = load_saved(args.checkpoint_dir, which=args.which, device=args.device)
    print("Loaded checkpointed arms and dataset", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test covariates. The clean arm's home is the clean test set; the all-data
    # arm's home is the FULL test set (it trained on everything).
    test_all_idx = np.union1d(test_wildfire_idx, test_no_wildfire_idx)
    xw, yw = dataset.x[test_wildfire_idx], dataset.y[test_wildfire_idx]
    xc, yc = dataset.x[test_no_wildfire_idx], dataset.y[test_no_wildfire_idx]
    xa, ya = dataset.x[test_all_idx], dataset.y[test_all_idx]

    # --- 1) Per-arm quality on each arm's own test set. ---
    crps_all = energy_loss(eng_all, xa, ya, n_samples_per_x=args.n_samples)
    crps_clean = energy_loss(eng_clean, xc, yc, n_samples_per_x=args.n_samples)
    cal_all = arm_calibration(eng_all, xa, ya, args.n_samples, seed=args.seed)
    cal_clean = arm_calibration(eng_clean, xc, yc, args.n_samples, seed=args.seed)
    plot_pit(cal_all["pit"], "PIT - all-data arm (full test set)", out_dir / "pit_all.png")
    plot_pit(cal_clean["pit"], "PIT - clean arm (clean test set)", out_dir / "pit_clean.png")

    # --- 2) Effect = all(x) - clean(x): at wildfire x, clean x, all x. ---
    eff_wildfire = estimand(eng_all, eng_clean, xw, args.n_samples)   # "ATT-like"
    eff_clean = estimand(eng_all, eng_clean, xc, args.n_samples)
    eff_all = estimand(eng_all, eng_clean, xa, args.n_samples)

    # --- 3) Money plot at wildfire covariates. ---
    plot_money(eff_wildfire["factual"], eff_wildfire["counterfactual"],
               "All-data model vs clean model (at wildfire covariates)", out_dir / "money_plot.png")

    # --- 4) metrics.json ---
    def _effect(d):
        return {k: d[k] for k in ("n_periods", "mean_log_effect", "ratio", "mean_raw_ccn_effect")}
    metrics = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_samples": args.n_samples,
        "note": "effect = all(x) - clean(x); known-attenuated vs the T-learner.",
        "arms": {
            "all": {"n_test": int(len(ya)), "crps": crps_all,
                    "coverage_90": cal_all["coverage_90"], "coverage_50": cal_all["coverage_50"],
                    "mean_width_90": cal_all["mean_width_90"]},
            "clean": {"n_test": int(len(yc)), "crps": crps_clean,
                      "coverage_90": cal_clean["coverage_90"], "coverage_50": cal_clean["coverage_50"],
                      "mean_width_90": cal_clean["mean_width_90"]},
        },
        "effects": {"at_wildfire": _effect(eff_wildfire), "at_clean": _effect(eff_clean), "at_all": _effect(eff_all)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\n=== per-arm quality ===", flush=True)
    print(f"  all   arm: CRPS {crps_all:.4f} | cov90 {cal_all['coverage_90']:.3f} | cov50 {cal_all['coverage_50']:.3f}")
    print(f"  clean arm: CRPS {crps_clean:.4f} | cov90 {cal_clean['coverage_90']:.3f} | cov50 {cal_clean['coverage_50']:.3f}")
    print("\n=== all-vs-clean effect (log10 Δ | ×ratio | raw Δ cm^-3) ===", flush=True)
    for name, d in (("at_wildfire", eff_wildfire), ("at_clean", eff_clean), ("at_all", eff_all)):
        print(f"  {name:11s} (n={d['n_periods']:5d}): {d['mean_log_effect']:+.3f} log10 "
              f"| {d['ratio']:.2f}x | {d['mean_raw_ccn_effect']:+.1f} cm^-3")
    print(f"\nwrote charts + metrics.json to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
