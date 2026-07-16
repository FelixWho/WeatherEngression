"""Evaluate embedding-space OOD calibration from a saved checkpoint (no retraining).

Reloads a fitted LSTM-engression model from a checkpoint, rebuilds the same
data split, and reports kNN and Mahalanobis distance-from-training diagnostics
(coverage stratified by distance) plus the chart -- reusing the exact helpers
from ``real_data_diagnostic`` so the numbers match a fresh run.

Defaults reproduce the full-data log10(CCN) / paper run.

Example
-------
```bash
python experiments/evaluate_oos_from_checkpoint.py --prediction-samples 400
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    load_ena_supervised_dataset,
    parse_count_or_all,
    select_real_split,
)
from engression_modifications.lstm import load_lstm_engressor_checkpoint
from experiments.real_data_diagnostic import (
    QUANTILE_LEVELS,
    embedding_oos_report,
    predict_quantiles_and_samples,
)
from experiments.real_metrics import (
    empirical_quantile_metrics,
    pit_calibration_metrics,
    pit_values,
)
from experiments.real_plotting import (
    write_coverage_calibration_curve_chart,
    write_coverage_vs_distance_chart,
    write_pit_by_distance_chart,
    write_pit_histogram_chart,
    write_real_posterior_band_chart,
)

DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather" / "log10_ccn" / "paper" / "lstm"


def build_parser() -> argparse.ArgumentParser:
    """CLI; defaults reproduce the full-data log10(CCN) / paper run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN_DIR / "checkpoint_best.pt")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to the checkpoint's folder.")
    parser.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    parser.add_argument("--target", type=str, default="ccn")
    parser.add_argument("--log-ccn", dest="log_ccn", action="store_true", default=True)
    parser.add_argument("--no-log-ccn", dest="log_ccn", action="store_false")
    parser.add_argument("--split", type=str, default="paper")
    parser.add_argument("--event-flag", type=str, default="dust")
    parser.add_argument("--ccn-tail-quantile", type=float, default=0.80)
    parser.add_argument("--max-samples", type=parse_count_or_all, default=None)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--train-size", type=parse_count_or_all, default=None)
    parser.add_argument("--test-size", type=parse_count_or_all, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prediction-samples", type=int, default=400)
    parser.add_argument("--oos-reference-size", type=int, default=2000)
    parser.add_argument("--oos-bins", type=int, default=10)
    parser.add_argument("--oos-batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device; defaults to cuda when available, else cpu.",
    )
    return parser


def main() -> None:
    """Reload the checkpoint and compute embedding-space OOD calibration."""

    args = build_parser().parse_args()
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Compute device: {args.device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    engressor = load_lstm_engressor_checkpoint(args.checkpoint, device=args.device)
    cfg = engressor.config
    # Same label mapping as real_data_diagnostic, covering every noise variant so
    # the sweep configs are named correctly on their charts.
    if getattr(cfg, "pre_additive", False):
        model_label = "LSTM + pre-additive"
    elif getattr(cfg, "stonet_head", False):
        model_label = "LSTM + StoNet"
    elif getattr(cfg, "appending_noise", False):
        model_label = "LSTM + appending noise"
    elif getattr(cfg, "additive_noise", False):
        model_label = "LSTM + additive noise"
    elif getattr(cfg, "per_timestep_noise", False):
        model_label = "LSTM + per-timestep noise"
    elif getattr(cfg, "global_latent_noise", False):
        model_label = "LSTM + global latent"
    elif getattr(cfg, "stochastic_init_noise", False):
        model_label = "LSTM + stochastic init"
    elif getattr(cfg, "recurrent_state_noise", False):
        model_label = "LSTM + recurrent noise"
    else:
        model_label = "LSTM (default head)"

    print("Loading data and rebuilding the split (no retraining)...")
    dataset = load_ena_supervised_dataset(
        mat_path=args.mat_path,
        target=args.target,
        max_samples=args.max_samples,
        seq_stride=args.seq_stride,
        seed=args.seed,
        log_ccn=args.log_ccn,
    )
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=args.split,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed + 17,
        event_flag=args.event_flag,
        ccn_tail_quantile=args.ccn_tail_quantile,
    )
    x_train = torch.from_numpy(dataset.x[train_idx])
    x_test = torch.from_numpy(dataset.x[test_idx])
    y_test = dataset.y[test_idx].astype(np.float64)
    print(f"  train rows: {len(train_idx)}   test rows: {len(test_idx)}")

    torch.manual_seed(args.seed + 101)
    predicted_quantiles, samples = predict_quantiles_and_samples(
        engressor=engressor,
        x_test=x_test,
        sample_size=args.prediction_samples,
        levels=QUANTILE_LEVELS,
    )
    coverage = empirical_quantile_metrics(predicted_quantiles, y_test, levels=QUANTILE_LEVELS)
    covered = (y_test >= predicted_quantiles[:, 0]) & (y_test <= predicted_quantiles[:, 4])
    # Shape calibration across ALL levels (the PIT charts), not just 50%/90%.
    pit = pit_values(samples, y_test, seed=args.seed + 307)
    pit_metrics = pit_calibration_metrics(pit)

    report = embedding_oos_report(
        engressor=engressor,
        x_train=x_train,
        x_test=x_test,
        covered=covered,
        reference_size=args.oos_reference_size,
        batch_size=args.oos_batch_size,
        n_bins=args.oos_bins,
        seed=args.seed + 211,
    )
    knn_distances_per_point = report.pop("knn_distances_per_point")  # per-point, for stratified PIT

    write_real_posterior_band_chart(
        out=out_dir / "posterior_bands.png",
        target_name=dataset.target_name,
        split=args.split,
        y_test=y_test,
        predicted_quantiles=predicted_quantiles,
        levels=QUANTILE_LEVELS,
        model_label=model_label,
    )
    chart_path = out_dir / "coverage_vs_distance.png"
    write_coverage_vs_distance_chart(
        out=chart_path,
        target_name=dataset.target_name,
        knn_bins=report["knn"]["coverage_by_distance_bin"],
        maha_bins=report["mahalanobis"]["coverage_by_distance_bin"],
        model_label=model_label,
    )
    # Shape-calibration charts, alongside the bands and OOS chart.
    write_pit_histogram_chart(
        out=out_dir / "pit_histogram.png",
        target_name=dataset.target_name,
        pit=pit,
        model_label=model_label,
    )
    write_coverage_calibration_curve_chart(
        out=out_dir / "coverage_calibration_curve.png",
        target_name=dataset.target_name,
        pit=pit,
        model_label=model_label,
    )
    write_pit_by_distance_chart(
        out=out_dir / "pit_by_distance.png",
        target_name=dataset.target_name,
        pit=pit,
        distances=knn_distances_per_point,
        n_bins=min(5, args.oos_bins),
        model_label=model_label,
    )
    payload = {
        "source_checkpoint": str(args.checkpoint),
        "split": args.split,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "prediction_samples": args.prediction_samples,
        "overall_coverage_90": coverage["coverage_90"],
        "overall_coverage_50": coverage["coverage_50"],
        "pit_calibration": pit_metrics,
        "embedding_oos": report,
    }
    (out_dir / "oos_from_checkpoint.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    knn_bins = report["knn"]["coverage_by_distance_bin"]
    maha_bins = report["mahalanobis"]["coverage_by_distance_bin"]
    print(f"\noverall 90% coverage: {coverage['coverage_90']:.3f}")
    print(
        f"PIT shape: {pit_metrics['pit_shape']}  "
        f"(mean {pit_metrics['pit_mean']:.3f}/0.5, var {pit_metrics['pit_var']:.3f}/"
        f"{pit_metrics['pit_var_ideal']:.3f}, KS {pit_metrics['pit_ks']:.3f})"
    )
    print(f"embedding dim: {report['embedding_dim']}  kNN/Maha distance corr: {report['knn_mahalanobis_distance_corr']:.3f}")
    print("coverage by distance bin (closest -> farthest):")
    print("  kNN :", [round(b["coverage_90"], 3) for b in knn_bins])
    print("  Maha:", [round(b["coverage_90"], 3) for b in maha_bins])
    print(f"\nwrote: {chart_path}")
    print(f"wrote: {out_dir / 'oos_from_checkpoint.json'}")


if __name__ == "__main__":
    main()
