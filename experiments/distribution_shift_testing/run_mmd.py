"""MMD permutation test, run on the same decorrelated episodes as C2ST and kNN.

Same front half as ``run.py`` (load, summarize, decorrelate, standardize), then the
permutation test and the null-distribution chart.

    python -m experiments.distribution_shift_testing.run_mmd \
        --device cuda --decorrelate-stride 24 --n-perm 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.distribution_shift_testing.features import (
    decorrelate,
    load_split,
    standardize_by_train,
    summarize_trajectories,
)
from experiments.distribution_shift_testing.mmd import run_mmd_test


def _plot_null(res: dict, out_path: Path) -> None:
    null = res["_mmd2_null"]
    obs = res["_mmd2_obs"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(null, bins=60, color="#7f8c8d", alpha=0.8,
            label=f"permutation null (n={res['n_permutations']})")
    ax.axvline(obs, color="#c0392b", lw=2.0,
               label=f"observed MMD$^2$ = {obs:.4g}")
    ax.set_xlabel("MMD$^2$ (unbiased, Gaussian kernel)")
    ax.set_ylabel("count")
    ax.set_title(
        f"MMD permutation test: p = {res['mmd2_pvalue']:.4g}, "
        f"z = {res['mmd2_zscore']:.1f}  "
        f"({res['n_train_episodes']} train / {res['n_test_episodes']} test episodes)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="MMD permutation two-sample test (train vs test).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="paper")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--decorrelate-stride", type=int, default=24,
                   help="Keep every k-th time-ordered row (independence for a valid "
                        "permutation null). Must match the C2ST/kNN run.")
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default="reports/distribution_shift_testing")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Run parameters:", flush=True)
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}", flush=True)

    dataset, train_idx, test_idx = load_split(split=args.split, seed=args.seed)
    print(f"\nfull split: train={len(train_idx)} test={len(test_idx)} rows", flush=True)

    train_idx = decorrelate(train_idx, args.decorrelate_stride)
    test_idx = decorrelate(test_idx, args.decorrelate_stride)
    print(f"after stride {args.decorrelate_stride}: train={len(train_idx)} "
          f"test={len(test_idx)} (approx. independent episodes)", flush=True)

    feats_train = summarize_trajectories(dataset.x[train_idx])
    feats_test = summarize_trajectories(dataset.x[test_idx])
    feats_train_z, feats_test_z = standardize_by_train(feats_train, feats_test)

    print("\n=== MMD permutation test ===", flush=True)
    res = run_mmd_test(
        feats_train_z, feats_test_z,
        device=args.device, n_perm=args.n_perm, seed=args.seed,
    )
    print(f"RBF gamma (median heuristic) = {res['rbf_gamma_median_heuristic']:.4g}", flush=True)
    print(f"MMD^2 observed = {res['mmd2_observed']:.5g}   "
          f"null mean = {res['mmd2_null_mean']:.2e} +/- {res['mmd2_null_std']:.2e}", flush=True)
    print(f"MMD^2  p-value = {res['mmd2_pvalue']:.4g}   z = {res['mmd2_zscore']:.1f}", flush=True)
    print(f"energy p-value = {res['energy_pvalue']:.4g}   z = {res['energy_zscore']:.1f}  "
          f"(cross-check)", flush=True)

    _plot_null(res, out_dir / "mmd_null.png")
    payload = {"params": vars(args), **{k: v for k, v in res.items() if not k.startswith("_")}}
    with open(out_dir / "mmd_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/mmd_results.json, mmd_null.png", flush=True)


if __name__ == "__main__":
    main()
