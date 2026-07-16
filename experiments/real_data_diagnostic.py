"""Fit LSTM engression on the real ENA weather data and score the forecast.

This is the real-data counterpart to ``experiments/engression_diagnostic.py``.
The synthetic diagnostic compares predicted quantiles against a known
conditional law; real data has no such truth, so this script instead scores the
predictive distribution against held-out realized targets (calibration of 50%
and 90% intervals, central error, and the energy score / CRPS).

It fits the sequence-native ``lstm`` engression variant on airmass-trajectory
inputs and writes a chart, metrics, and a run README under
``runs/real_data_diagnostics/`` by default.

Example
-------
```bash
python experiments/real_data_diagnostic.py --target ccn --split paper \
    --log-ccn --max-samples all --train-size all --test-size all --seq-stride 4 --epochs 40
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    format_count_or_all,
    load_ena_supervised_dataset,
    parse_count_or_all,
    select_real_split,
)
from engression_modifications import get_engression_model
from experiments.oos import (
    choose_reference_indices,
    mahalanobis_distances,
    nearest_neighbor_distances,
    standardize_by_train,
)
from experiments.pipeline import set_reproducible_seeds
from experiments.real_metrics import (
    coverage_by_distance_bins,
    empirical_quantile_metrics,
    energy_score_samples,
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

QUANTILE_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
SPLIT_MODES = ("random", "chronological", "paper", "event", "ccn_tail")
REAL_RUNS_ROOT = REPO_ROOT / "runs" / "real_data_diagnostics" / "ena_weather"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    parser.add_argument(
        "--engression-model",
        type=str,
        default="lstm",
        choices=("lstm", "vanilla", "regularized", "adamw"),
        help=(
            "Engression variant. 'lstm' is sequence-native; 'vanilla' is the public "
            "engression package's flat MLP (trajectory is flattened). 'regularized'/'adamw' "
            "add weight decay. Only 'lstm' supports checkpointing and embedding-OOD."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the run output folder (defaults to runs/.../<target>/<split>/lstm). Use for sweeps.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="ccn",
        help="'ccn' (scalar target) or a trajectory variable name/index (one step ahead).",
    )
    parser.add_argument("--split", choices=SPLIT_MODES, default="paper")
    ccn_transform = parser.add_mutually_exclusive_group()
    ccn_transform.add_argument(
        "--log-ccn",
        dest="log_ccn",
        action="store_true",
        default=True,
        help="Use log10(CCN) when target='ccn' (paper-style default).",
    )
    ccn_transform.add_argument(
        "--no-log-ccn",
        dest="log_ccn",
        action="store_false",
        help="Use raw CCN when target='ccn'.",
    )
    parser.add_argument(
        "--event-flag",
        type=str,
        default="dust",
        help="Event mask held out as the test set when --split event (e.g. dust, BB_criterion1).",
    )
    parser.add_argument(
        "--ccn-tail-quantile",
        type=float,
        default=0.80,
        help=(
            "When --split ccn_tail, train on CCN below this quantile and test on the "
            "held-out high-CCN tail above it (default 0.80 -> hold out the top 20%%)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for every usable sample.",
    )
    parser.add_argument("--seq-stride", type=int, default=4)
    parser.add_argument(
        "--train-size",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for the full train pool.",
    )
    parser.add_argument(
        "--test-size",
        type=parse_count_or_all,
        default=None,
        help="Positive integer cap, or 'all' for the full test pool.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--noise-dim", type=int, default=96)
    parser.add_argument("--num-layer", type=int, default=3)
    parser.add_argument(
        "--pre-additive",
        action="store_true",
        help="Use the engression-paper pre-ANM head: Y = g(phi(X) + eta) with monotone g.",
    )
    parser.add_argument("--index-dim", type=int, default=None, help="Latent index width for --pre-additive.")
    parser.add_argument(
        "--stonet-head",
        action="store_true",
        help="Loose mode: package StoNet head (noise at input + every layer, monotonicity not forced).",
    )
    parser.add_argument("--resblock", action="store_true", help="Use residual StoNet blocks with --stonet-head.")
    parser.add_argument(
        "--appending-noise",
        action="store_true",
        help="Noise-in-encoder head: append a fresh noise timestep to the sequence; deterministic MLP head.",
    )
    parser.add_argument(
        "--additive-noise",
        action="store_true",
        help="Noise-in-encoder head: add fresh noise onto the final timestep; deterministic MLP head.",
    )
    parser.add_argument(
        "--per-timestep-noise",
        action="store_true",
        help="Recurrence noise: concat fresh noise to every timestep -> LSTM reads (B, S, F+noise_dim).",
    )
    parser.add_argument(
        "--global-latent-noise",
        action="store_true",
        help="Recurrence noise: one latent z per sample, broadcast to every timestep (CVAE-style).",
    )
    parser.add_argument(
        "--stochastic-init-noise",
        action="store_true",
        help="Recurrence noise: seed the LSTM initial (h0, c0) from noise; input unchanged.",
    )
    parser.add_argument(
        "--recurrent-state-noise",
        action="store_true",
        help="Recurrence noise: per-step noise added to the hidden state (single-layer LSTMCell loop).",
    )
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--prediction-samples", type=int, default=800)
    parser.add_argument(
        "--no-oos",
        action="store_true",
        help="Skip the embedding-space kNN/Mahalanobis OOD distance diagnostics.",
    )
    parser.add_argument(
        "--oos-reference-size",
        type=int,
        default=2000,
        help="Train embeddings sampled as the reference set for OOD distances.",
    )
    parser.add_argument(
        "--oos-bins",
        type=int,
        default=10,
        help="Equal-count distance bins for coverage-vs-distance stratification.",
    )
    parser.add_argument("--oos-batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save checkpoint_latest.pt every N epochs. Use with default checkpointing.",
    )
    parser.add_argument(
        "--no-checkpoint",
        dest="checkpoint",
        action="store_false",
        default=True,
        help="Disable checkpoint_latest.pt and checkpoint_best.pt writes.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help=(
            "LSTM only: stop training if the mean epoch energy loss has not improved "
            "for this many epochs. Saves compute; monitors training loss (not calibration)."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch energy loss.")
    parser.add_argument(
        "--skip-assertions",
        action="store_true",
        help="Write artifacts without failing on loose sanity thresholds.",
    )
    parser.add_argument("--min-coverage-90", type=float, default=0.60)
    parser.add_argument("--max-coverage-90", type=float, default=0.99)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser


def default_run_dir(target_slug: str, split: str, model: str = "lstm") -> Path:
    """Return the default folder for one real-data engression diagnostic run."""

    return REAL_RUNS_ROOT / target_slug / split / model


def shell_command() -> str:
    """Return the current command in copy-pasteable form."""

    parts = ["python", "experiments/real_data_diagnostic.py", *sys.argv[1:]]
    return " ".join(shlex.quote(part) for part in parts)


def predict_quantiles_and_samples(
    engressor: object,
    x_test: torch.Tensor,
    sample_size: int,
    levels: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Draw conditional samples once, then derive quantiles and a sample matrix.

    Sampling once and reusing the draws keeps the quantiles and the energy score
    consistent and avoids generating the (expensive) sample set twice.
    """

    samples = engressor.sample(x_test, sample_size=sample_size, expand_dim=True)
    samples = samples.detach().cpu().numpy()[:, 0, :]  # (n, sample_size), out_dim == 1
    quantiles = np.quantile(samples, np.asarray(levels), axis=1).T  # (n, len(levels))
    return quantiles, samples


def embedding_oos_report(
    engressor: object,
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    covered: np.ndarray,
    reference_size: int,
    batch_size: int,
    n_bins: int,
    seed: int,
) -> dict[str, object]:
    """Score each test point's distance from training in the LSTM embedding space.

    Uses the model's own representation ``h(x)`` (not raw flattened trajectories):
    a sampled training reference is encoded once, then every test point gets a
    kNN distance (standardized Euclidean) and a Mahalanobis distance to that
    reference. Coverage is stratified by each distance so calibration can be read
    as a function of distance from the training feature space.
    """

    reference_idx = choose_reference_indices(int(x_train.shape[0]), reference_size, seed)
    h_reference = engressor.encode(x_train[torch.as_tensor(reference_idx)]).detach().cpu().numpy()
    h_test = engressor.encode(x_test).detach().cpu().numpy()

    reference_z, test_z = standardize_by_train(h_reference, h_test)
    knn_distances = nearest_neighbor_distances(test_z, reference_z, batch_size=batch_size)
    mahalanobis = mahalanobis_distances(h_reference, h_test)

    knn_bins = coverage_by_distance_bins(knn_distances, covered, n_bins)
    mahalanobis_bins = coverage_by_distance_bins(mahalanobis, covered, n_bins)
    corr = (
        float(np.corrcoef(knn_distances, mahalanobis)[0, 1]) if len(knn_distances) > 1 else float("nan")
    )
    return {
        "embedding_dim": int(h_test.shape[1]),
        "reference_size": int(len(reference_idx)),
        "n_bins": int(n_bins),
        "knn_mahalanobis_distance_corr": corr,
        # per-point kNN distances (test order) for stratified shape diagnostics;
        # popped before metrics.json is written so it does not bloat the file.
        "knn_distances_per_point": np.asarray(knn_distances, dtype=np.float64),
        "knn": {
            "mean_test_distance": float(np.mean(knn_distances)),
            "max_test_distance": float(np.max(knn_distances)),
            "coverage_by_distance_bin": knn_bins,
        },
        "mahalanobis": {
            "mean_test_distance": float(np.mean(mahalanobis)),
            "max_test_distance": float(np.max(mahalanobis)),
            "coverage_by_distance_bin": mahalanobis_bins,
        },
    }


def write_run_artifacts(
    run_dir: Path,
    plot_name: str,
    args: argparse.Namespace,
    metrics: dict[str, object],
    command: str,
) -> None:
    """Write README.md and metrics.json for one real-data run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_note = {
        "random": (
            "A shuffled in-distribution split: held-out rows are drawn from the "
            "same airmass population as training. This checks conditional "
            "distribution learning, not forecasting or shift."
        ),
        "chronological": (
            "Earliest trajectories train, later ones test. This is a genuine "
            "forecast-the-future split over the ENA record."
        ),
        "paper": (
            "The test pool follows the source paper: the held-out months are "
            "January, March, May, July, September, and November 2022. Training "
            "rows are drawn from all other months. When provided as positive "
            "integers, the CLI train/test size arguments cap how many rows are "
            "used from each pool; `all` uses the full pools."
        ),
        "event": (
            f"The test set is the held-out `{args.event_flag}` event period and "
            "training uses the remaining clean periods. This is a real "
            "distribution-shift stress test, the data analogue of the synthetic "
            "out-of-support splits."
        ),
        "ccn_tail": (
            f"Training uses CCN below the {args.ccn_tail_quantile:.0%} quantile; the "
            f"test set is the held-out high-CCN tail (top {1 - args.ccn_tail_quantile:.0%}). "
            "Because the target never reaches these values in training, this directly "
            "probes whether engression EXTRAPOLATES its conditional distribution "
            "beyond the observed target support -- the engression paper's central "
            "claim (which assumes pre-additive, monotone structure that real CCN "
            "need not satisfy). Coverage well below nominal here is the expected, "
            "informative failure mode, not a bug."
        ),
    }[args.split]
    output_lines = [
        f"- `{plot_name}`: predicted conditional bands with realized targets overlaid.",
        "- `pit_histogram.png`: PIT / rank histogram - shape calibration across ALL "
        "quantile levels (flat = calibrated; U = too narrow, dome = too wide, "
        "sloped = biased). Judges the whole distribution, not just the 50%/90% bands.",
        "- `coverage_calibration_curve.png`: empirical vs nominal coverage at every "
        "level (the 50%/90% coverage numbers are two points on this curve).",
        "- `metrics.json`: scalar metrics printed by the run.",
    ]
    if "embedding_oos" in metrics:
        output_lines.append(
            "- `coverage_vs_distance.png`: 90% coverage stratified by kNN and "
            "Mahalanobis distance from training in the LSTM embedding space."
        )
        output_lines.append(
            "- `pit_by_distance.png`: PIT histogram per distance-from-training bin - "
            "does the predictive SHAPE (not just interval width) degrade out of support?"
        )
    if args.checkpoint:
        output_lines.extend(
            [
                "- `checkpoint_latest.pt`: latest epoch model, optimizer state, and standardization stats.",
                "- `checkpoint_best.pt`: best training-energy-loss checkpoint.",
            ]
        )

    readme = run_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Real-Data Engression Run: ENA {metrics['target_name']} / {args.split} / {metrics['engression_model']}",
                "",
                "This folder contains one LSTM-engression run on the **real** ENA",
                "weather data (`weather_data.mat`). The true conditional law is",
                "unknown, so the predictive distribution is scored against held-out",
                "realized targets rather than against generator truth.",
                "",
                "## Command",
                "",
                "```bash",
                command,
                "```",
                "",
                "## Outputs",
                "",
                *output_lines,
                "",
                "## Main Metrics",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| energy score (CRPS, lower better) | `{metrics['energy_score']:.6f}` |",
                f"| median abs error | `{metrics['median_abs_error']:.6f}` |",
                f"| 90% interval coverage | `{metrics['coverage_90']:.6f}` |",
                f"| 50% interval coverage | `{metrics['coverage_50']:.6f}` |",
                f"| mean 90% interval width | `{metrics['mean_width_90']:.6f}` |",
                f"| mean 50% interval width | `{metrics['mean_width_50']:.6f}` |",
                "",
                "Nominal coverage is 0.90 and 0.50; closer is better-calibrated.",
                "",
                "## Run Parameters",
                "",
                "| Parameter | Value |",
                "|---|---:|",
                f"| data | `{metrics['mat_path']}` |",
                f"| target | `{metrics['target_name']}` |",
                f"| log_ccn | `{args.log_ccn}` |",
                f"| split | `{args.split}` |",
                f"| event_flag | `{args.event_flag if args.split == 'event' else 'n/a'}` |",
                f"| ccn_tail_quantile | `{args.ccn_tail_quantile if args.split == 'ccn_tail' else 'n/a'}` |",
                f"| max_samples | `{format_count_or_all(args.max_samples)}` |",
                f"| seq_stride | `{args.seq_stride}` |",
                f"| sequence shape (train) | `{tuple(metrics['x_train_shape'])}` |",
                f"| train_size arg | `{format_count_or_all(args.train_size)}` |",
                f"| test_size arg | `{format_count_or_all(args.test_size)}` |",
                f"| train rows used | `{metrics['train_rows']}` |",
                f"| test rows used | `{metrics['test_rows']}` |",
                f"| epochs | `{args.epochs}` |",
                f"| batch_size | `{args.batch_size}` |",
                f"| hidden_dim | `{args.hidden_dim}` |",
                f"| noise_dim | `{args.noise_dim}` |",
                f"| num_layer | `{args.num_layer}` |",
                f"| learning_rate | `{args.lr}` |",
                f"| weight_decay | `{args.weight_decay}` |",
                f"| prediction_samples | `{args.prediction_samples}` |",
                f"| checkpointing | `{args.checkpoint}` |",
                f"| checkpoint_every | `{args.checkpoint_every if args.checkpoint else 'n/a'}` |",
                f"| seed | `{args.seed}` |",
                "",
                "## Split Interpretation",
                "",
                split_note,
                "",
                f"The `{metrics['engression_model']}` engression model was fit; the training",
                f"tensor has shape `{tuple(metrics['x_train_shape'])}`. The `lstm` variant keeps",
                "lag windows unflattened `(n, seq_len, n_features)`; the flat variants "
                "(`vanilla`, `regularized`, `adamw`) take the flattened `(n, seq_len * n_features)`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run the real-data LSTM-engression diagnostic.

    Pipeline (see the numbered banners below): load -> split -> pick model ->
    build tensors -> train -> predict -> score -> OOS -> write artifacts -> check.
    """

    args = build_arg_parser().parse_args()
    if args.checkpoint and args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")
    set_reproducible_seeds(args.seed)

    print("Run parameters:")
    for key, value in sorted(vars(args).items()):
        print(f"  {key} = {value}")

    # 1) LOAD: read weather_data.mat into dataset.x (n, seq_len, n_features) and
    #    dataset.y (n,). See data_generation/ena_weather.py.
    dataset = load_ena_supervised_dataset(
        mat_path=args.mat_path,
        target=args.target,
        max_samples=args.max_samples,
        seq_stride=args.seq_stride,
        seed=args.seed,
        log_ccn=args.log_ccn,
    )
    # 2) SPLIT: choose which rows train vs. test (paper months, event mask, tail, ...).
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=args.split,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed + 17,
        event_flag=args.event_flag,
        ccn_tail_quantile=args.ccn_tail_quantile,
    )

    # 3) PICK MODEL: look up the fit function by name in the registry. The spec's
    #    input_kind decides whether X stays a sequence (lstm) or gets flattened.
    model_spec = get_engression_model(args.engression_model)
    is_lstm = model_spec.name == "lstm"
    # Flat (non-sequence) models take the flattened trajectory (n, seq_len * n_features).
    flatten = model_spec.input_kind == "flat"
    # Checkpointing and embedding-OOD are lstm-only capabilities.
    args.checkpoint = args.checkpoint and is_lstm

    def to_model_x(rows: np.ndarray) -> torch.Tensor:
        arr = dataset.x[rows]
        if flatten:
            arr = arr.reshape(len(rows), -1)
        return torch.from_numpy(arr)

    # 4) BUILD TENSORS: slice the split indices into train/test tensors.
    x_train = to_model_x(train_idx)
    y_train = torch.from_numpy(dataset.y[train_idx].reshape(-1, 1))
    x_test = to_model_x(test_idx)
    y_test = dataset.y[test_idx].astype(np.float64)

    run_dir = args.out_dir or default_run_dir(dataset.target_slug, args.split, model_spec.name)
    checkpoint_latest = run_dir / "checkpoint_latest.pt"
    checkpoint_best = run_dir / "checkpoint_best.pt"
    if args.checkpoint:
        run_dir.mkdir(parents=True, exist_ok=True)

    # 5) TRAIN: assemble the config kwargs, then model_spec.fit(...) runs the energy-
    #    loss training loop and returns a fitted engressor. lstm-only knobs (head
    #    choice, checkpointing) are added below only when the model is lstm.
    fit_kwargs: dict[str, object] = dict(
        num_layer=args.num_layer,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        add_bn=False,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        standardize=True,
        device=args.device,
        verbose=args.verbose,
    )
    if model_spec.name in {"regularized", "adamw", "lstm"}:
        fit_kwargs["weight_decay"] = args.weight_decay
    if is_lstm:
        fit_kwargs.update(
            pre_additive=args.pre_additive,
            index_dim=args.index_dim,
            stonet_head=args.stonet_head,
            appending_noise=args.appending_noise,
            additive_noise=args.additive_noise,
            per_timestep_noise=args.per_timestep_noise,
            global_latent_noise=args.global_latent_noise,
            stochastic_init_noise=args.stochastic_init_noise,
            recurrent_state_noise=args.recurrent_state_noise,
            resblock=args.resblock,
            early_stop_patience=args.early_stop_patience,
            checkpoint_path=str(checkpoint_latest) if args.checkpoint else None,
            checkpoint_best_path=str(checkpoint_best) if args.checkpoint else None,
            checkpoint_every_nepoch=args.checkpoint_every if args.checkpoint else None,
        )
    engressor = model_spec.fit(x_train, y_train, **fit_kwargs)

    # 6) PREDICT: draw many conditional samples per test point, then reduce them to
    #    the target quantiles (the predictive intervals) and keep the raw samples.
    torch.manual_seed(args.seed + 101)
    predicted_quantiles, samples = predict_quantiles_and_samples(
        engressor=engressor,
        x_test=x_test,
        sample_size=args.prediction_samples,
        levels=QUANTILE_LEVELS,
    )

    # 7) SCORE: record run provenance, then append the computed metrics (interval
    #    calibration/width + energy score / CRPS) comparing predictions to y_test.
    metrics: dict[str, object] = {
        "dataset": "ena_weather",
        "engression_model": model_spec.name,
        "mat_path": args.mat_path,
        "target_name": dataset.target_name,
        "target_slug": dataset.target_slug,
        "target_transform": dataset.target_transform,
        "log_ccn": args.log_ccn,
        "split": args.split,
        "event_flag": args.event_flag if args.split == "event" else None,
        "ccn_tail_quantile": args.ccn_tail_quantile if args.split == "ccn_tail" else None,
        "max_samples_arg": args.max_samples,
        "train_size_arg": args.train_size,
        "test_size_arg": args.test_size,
        "checkpoint": args.checkpoint,
        "checkpoint_every": args.checkpoint_every if args.checkpoint else None,
        "checkpoint_latest": str(checkpoint_latest) if args.checkpoint else None,
        "checkpoint_best": str(checkpoint_best) if args.checkpoint else None,
        "feature_names": list(dataset.feature_names),
        "sequence_length": dataset.sequence_length,
        "n_features": dataset.n_features,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
        "quantile_levels": list(QUANTILE_LEVELS),
    }
    metrics.update(empirical_quantile_metrics(predicted_quantiles, y_test, levels=QUANTILE_LEVELS))
    metrics.update(energy_score_samples(samples, y_test))
    # Shape calibration (whole distribution, not just the 50%/90% levels): the PIT
    # values feed the histogram / calibration-curve charts and their scalar summary.
    pit = pit_values(samples, y_test, seed=args.seed + 307)
    metrics.update(pit_calibration_metrics(pit))

    print(f"Real-data engression [{model_spec.name}]: ENA {dataset.target_name} / {args.split}")
    print(f"  usable samples loaded: {len(dataset.y)}  sequence shape: {tuple(x_train.shape[1:])}")
    print(f"  train rows: {len(train_idx)}   test rows: {len(test_idx)}")
    print(f"  energy score (CRPS): {metrics['energy_score']:.4f}")
    print(f"  median abs error:    {metrics['median_abs_error']:.4f}")
    print(f"  90% coverage: {metrics['coverage_90']:.3f}  (width {metrics['mean_width_90']:.3f})")
    print(f"  50% coverage: {metrics['coverage_50']:.3f}  (width {metrics['mean_width_50']:.3f})")
    print(
        f"  PIT shape:    {metrics['pit_shape']}  "
        f"(mean {metrics['pit_mean']:.3f}/0.5, var {metrics['pit_var']:.3f}/{metrics['pit_var_ideal']:.3f}, "
        f"KS {metrics['pit_ks']:.3f})"
    )

    # 8) OOS (lstm-only): how far is each test point from training in the model's
    #    own embedding space, and does coverage decay with that distance? Needs the
    #    engressor to expose encode(); flat models skip this.
    oos_report: dict[str, object] | None = None
    if not args.no_oos and hasattr(engressor, "encode"):
        # cols 0 and 4 are the 5th/95th percentiles -> the nominal 90% band.
        covered = (y_test >= predicted_quantiles[:, 0]) & (y_test <= predicted_quantiles[:, 4])
        oos_report = embedding_oos_report(
            engressor=engressor,
            x_train=x_train,
            x_test=x_test,
            covered=covered,
            reference_size=args.oos_reference_size,
            batch_size=args.oos_batch_size,
            n_bins=args.oos_bins,
            seed=args.seed + 211,
        )
        metrics["embedding_oos"] = oos_report
        knn = oos_report["knn"]
        maha = oos_report["mahalanobis"]
        print(
            f"  embedding OOD ({oos_report['embedding_dim']}-d): "
            f"kNN mean dist {knn['mean_test_distance']:.3f}, "
            f"Mahalanobis mean dist {maha['mean_test_distance']:.3f}"
        )
        print(
            f"    coverage closest vs farthest bin: "
            f"kNN {knn['coverage_by_distance_bin'][0]['coverage_90']:.3f} -> "
            f"{knn['coverage_by_distance_bin'][-1]['coverage_90']:.3f}"
        )

    # 9) WRITE ARTIFACTS: a human-readable label, then the charts, metrics.json and
    #    run README under run_dir.
    if not is_lstm:
        model_label = f"{model_spec.name} (flat)"
    elif args.pre_additive:
        model_label = "LSTM + pre-additive"
    elif args.stonet_head:
        model_label = "LSTM + StoNet"
    elif args.appending_noise:
        model_label = "LSTM + appending noise"
    elif args.additive_noise:
        model_label = "LSTM + additive noise"
    elif args.per_timestep_noise:
        model_label = "LSTM + per-timestep noise"
    elif args.global_latent_noise:
        model_label = "LSTM + global latent"
    elif args.stochastic_init_noise:
        model_label = "LSTM + stochastic init"
    elif args.recurrent_state_noise:
        model_label = "LSTM + recurrent noise"
    else:
        model_label = "LSTM (default head)"

    # per-point distances are only needed for the stratified shape chart; pull them
    # out of the report so they never land in metrics.json.
    knn_distances_per_point = (
        oos_report.pop("knn_distances_per_point") if oos_report is not None else None
    )

    plot_name = "posterior_bands.png"
    if not args.no_plot:
        write_real_posterior_band_chart(
            out=run_dir / plot_name,
            target_name=dataset.target_name,
            split=args.split,
            y_test=y_test,
            predicted_quantiles=predicted_quantiles,
            levels=QUANTILE_LEVELS,
            show=args.show,
            model_label=model_label,
        )
        # Shape-calibration charts (whole distribution), separate from the bands.
        write_pit_histogram_chart(
            out=run_dir / "pit_histogram.png",
            target_name=dataset.target_name,
            pit=pit,
            show=args.show,
            model_label=model_label,
        )
        write_coverage_calibration_curve_chart(
            out=run_dir / "coverage_calibration_curve.png",
            target_name=dataset.target_name,
            pit=pit,
            show=args.show,
            model_label=model_label,
        )
        if oos_report is not None:
            write_coverage_vs_distance_chart(
                out=run_dir / "coverage_vs_distance.png",
                target_name=dataset.target_name,
                knn_bins=oos_report["knn"]["coverage_by_distance_bin"],
                maha_bins=oos_report["mahalanobis"]["coverage_by_distance_bin"],
                show=args.show,
                model_label=model_label,
            )
            if knn_distances_per_point is not None:
                write_pit_by_distance_chart(
                    out=run_dir / "pit_by_distance.png",
                    target_name=dataset.target_name,
                    pit=pit,
                    distances=knn_distances_per_point,
                    n_bins=min(5, args.oos_bins),
                    show=args.show,
                    model_label=model_label,
                )
    if not args.no_readme:
        write_run_artifacts(run_dir, plot_name, args, metrics, shell_command())
    print(f"  wrote run to: {run_dir}")

    # 10) SANITY CHECKS: fail loudly on broken predictions (unless --skip-assertions).
    if args.skip_assertions:
        return
    if not np.isfinite(predicted_quantiles).all():
        raise AssertionError("engression produced non-finite quantile predictions")
    if not np.all(np.diff(predicted_quantiles, axis=1) >= -1e-4):
        raise AssertionError("predicted quantiles are not monotonically non-decreasing")
    # The ccn_tail split is an extrapolation test where under-coverage is the
    # informative result, so the in-distribution coverage band does not apply.
    if args.split != "ccn_tail" and not (
        args.min_coverage_90 <= metrics["coverage_90"] <= args.max_coverage_90
    ):
        raise AssertionError(
            f"90% interval coverage {metrics['coverage_90']:.3f} outside "
            f"[{args.min_coverage_90:.3f}, {args.max_coverage_90:.3f}]"
        )


if __name__ == "__main__":
    main()
