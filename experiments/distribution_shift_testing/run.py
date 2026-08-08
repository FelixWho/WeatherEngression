"""Run both train-vs-test distribution-shift tests and write results + charts.

Pipeline:  load split -> summarize trajectories -> decorrelate -> standardize
           -> C2ST (MLP + linear AUC, permutation importance)
           -> kNN overlap (test->train vs train->train nearest-neighbor distance)
           -> JSON + two charts

Shell use
---------
```bash
python -m experiments.distribution_shift_testing.run \
    --device cuda --decorrelate-stride 24 --out-dir reports/distribution_shift_testing
```
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
    summary_feature_names,
)
from experiments.distribution_shift_testing.c2st import run_c2st
from experiments.distribution_shift_testing.knn_overlap import run_knn_overlap


def _plot_importance(importance_table, out_path: Path, top_k: int = 20) -> None:
    top = importance_table[:top_k][::-1]
    names = [row["feature"] for row in top]
    vals = [row["auc_drop"] for row in top]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(top) + 1))
    ax.barh(range(len(top)), vals, color="#c0392b")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("AUC drop when feature is shuffled  (larger = drifted more)")
    ax.set_title(f"C2ST permutation importance: top {len(top)} drifted features")
    ax.axvline(0.0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_knn(knn: dict, out_path: Path) -> None:
    train_nn = knn["_train_nn"]
    test_nn = knn["_test_nn"]
    hi = float(np.quantile(np.concatenate([train_nn, test_nn]), 0.995))
    bins = np.linspace(0, hi, 60)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(train_nn, bins=bins, density=True, alpha=0.55, color="#2e7d32",
            label=f"train->train (median {knn['train_nn_median']:.2f})")
    ax.hist(test_nn, bins=bins, density=True, alpha=0.55, color="#c0392b",
            label=f"test->train (median {knn['test_nn_median']:.2f})")
    ax.axvline(knn["train_nn_quantiles"]["0.95"], color="k", ls="--", lw=1.0,
               label="train p95")
    ax.set_xlabel("Euclidean nearest-neighbor distance (standardized features)")
    ax.set_ylabel("density")
    ax.set_title(
        f"kNN overlap: {knn['frac_test_beyond_train_p95']*100:.1f}% of test beyond train p95"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Train-vs-test distribution-shift tests (C2ST + kNN).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="paper")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--decorrelate-stride", type=int, default=24,
                   help="Keep every k-th time-ordered row before testing (fights hourly "
                        "autocorrelation; ~150 independent episodes). 1 disables.")
    p.add_argument("--n-repeats", type=int, default=5, help="C2ST train/val splits to average.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--reference-size", type=int, default=4000, help="train subset for kNN.")
    p.add_argument("--out-dir", type=str, default="reports/distribution_shift_testing")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Run parameters:", flush=True)
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}", flush=True)

    # ---- load + summarize + decorrelate ----------------------------------------
    dataset, train_idx, test_idx = load_split(split=args.split, seed=args.seed)
    print(f"\nfull split: train={len(train_idx)} test={len(test_idx)} rows", flush=True)

    train_idx = decorrelate(train_idx, args.decorrelate_stride)
    test_idx = decorrelate(test_idx, args.decorrelate_stride)
    print(f"after stride {args.decorrelate_stride}: train={len(train_idx)} "
          f"test={len(test_idx)} (approx. independent episodes)", flush=True)

    names = summary_feature_names(dataset.feature_names)
    feats_train = summarize_trajectories(dataset.x[train_idx])
    feats_test = summarize_trajectories(dataset.x[test_idx])
    feats_train_z, feats_test_z = standardize_by_train(feats_train, feats_test)
    print(f"summary features: {feats_train_z.shape[1]} "
          f"({len(dataset.feature_names)} channels x 6 stats)", flush=True)

    # ---- C2ST ------------------------------------------------------------------
    print("\n=== C2ST (classifier two-sample test) ===", flush=True)
    c2st = run_c2st(
        feats_train_z, feats_test_z, names,
        device=args.device, n_repeats=args.n_repeats, epochs=args.epochs, seed=args.seed,
    )
    print(f"MLP    AUC = {c2st['mlp_auc_mean']:.3f} +/- {c2st['mlp_auc_std']:.3f}", flush=True)
    print(f"linear AUC = {c2st['linear_auc_mean']:.3f} +/- {c2st['linear_auc_std']:.3f}", flush=True)
    print("top drifted features (AUC drop when shuffled):", flush=True)
    for row in c2st["importance_table"][:12]:
        print(f"  {row['feature']:28s} {row['auc_drop']:+.4f}", flush=True)

    # ---- kNN overlap -----------------------------------------------------------
    print("\n=== kNN overlap (test->train vs train->train) ===", flush=True)
    knn = run_knn_overlap(
        feats_train_z, feats_test_z, reference_size=args.reference_size, seed=args.seed
    )
    print(f"median nn distance: train->train {knn['train_nn_median']:.3f}  "
          f"test->train {knn['test_nn_median']:.3f}  (ratio {knn['median_ratio']:.2f})", flush=True)
    print(f"test beyond train p95: {knn['frac_test_beyond_train_p95']*100:.1f}%   "
          f"beyond p99: {knn['frac_test_beyond_train_p99']*100:.1f}%", flush=True)

    # ---- charts + JSON ---------------------------------------------------------
    _plot_importance(c2st["importance_table"], out_dir / "c2st_feature_importance.png")
    _plot_knn(knn, out_dir / "knn_overlap.png")

    knn_json = {k: v for k, v in knn.items() if not k.startswith("_")}
    payload = {
        "params": vars(args),
        "n_full_train": int(len(dataset.x[train_idx])),  # post-decorrelation
        "c2st": c2st,
        "knn_overlap": knn_json,
        "interpretation": {
            "c2st": "AUC ~0.5 no shift; >0.6 clear shift. MLP>>linear => nonlinear shift.",
            "knn": "median_ratio ~1 and small beyond-p95 => test stays in train support.",
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_dir}/results.json, c2st_feature_importance.png, knn_overlap.png",
          flush=True)


if __name__ == "__main__":
    main()
