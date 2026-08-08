"""Partial-learner training: a clean arm and an all-data arm.

Same as ``experiments.t_learner.train``, except the second model sees all the
training data instead of the wildfire rows only:

  - ``clean`` arm : trained on the wildfire-free samples (identical to the
    T-learner's clean arm).
  - ``all``   arm : trained on clean + wildfire samples together.

Writes ``checkpoint_{all,clean}_{best,latest}.pt`` plus the pickled dataset /
index arrays into the save dir (on storage3), exactly like the T-learner, so
``partial_learner.test.load_saved`` can reload without retraining.

Shell:
    python -m experiments.partial_learner.train --recurrent-state-noise \
      --epochs 100 --early-stop-patience 12 --hidden-dim 192 --num-layer 5 \
      --device cuda --save-checkpoint-dir some/run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/partial_learner/train.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import DEFAULT_MAT_PATH, parse_count_or_all
from engression_modifications import get_engression_model
# Reuse the T-learner's generic infrastructure (path resolution, split/partition,
# head-flag list) so the only real difference lives in load_and_fit below.
from experiments.t_learner.train import (
    DEFAULT_CKPT_ROOT,  # noqa: F401 (re-exported for parity)
    _LSTM_HEAD_FLAGS,
    _load_dataset_and_arms,
    _resolve_save_dir,
)


def load_and_fit(
    *,
    mat_path: str = DEFAULT_MAT_PATH,
    target: str = "ccn",
    log_ccn: bool = True,
    split: str = "paper",
    seq_stride: int = 1,
    max_samples: int | None = None,
    train_size: int | None = None,
    test_size: int | None = None,
    engression_model: str = "lstm",
    epochs: int = 40,
    batch_size: int = 512,
    lr: float = 0.003,
    weight_decay: float = 0.0,
    hidden_dim: int = 192,
    num_layer: int = 3,
    noise_dim: int = 96,
    device: str = "cpu",
    seed: int = 2026,
    fit_overrides: dict | None = None,
    save_checkpoint_dir: str | Path | None = None,
    silent: bool = False,
    wildfire_flag: str = "BB_criterion1",
    early_stop_patience: int | None = 12,
    wildfire_oversample: int = 1,
):
    """Fit the clean arm and the all-data arm; return them + dataset + indices.

    Returns ``(eng_all, eng_clean, dataset, train_all_idx, train_clean_idx,
    test_wildfire_idx, test_no_wildfire_idx)``.
    """
    # Same split/partition as the T-learner. train_clean_idx is the wildfire-free
    # set; train_all_idx is clean + wildfire together (the tiny crit2-only buffer
    # is excluded, matching how the arms are defined).
    (
        dataset,
        train_wildfire_idx,
        train_no_wildfire_idx,
        test_wildfire_idx,
        test_no_wildfire_idx,
    ) = _load_dataset_and_arms(
        mat_path=mat_path,
        target=target,
        log_ccn=log_ccn,
        split=split,
        seq_stride=seq_stride,
        max_samples=max_samples,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        wildfire_flag=wildfire_flag,
    )
    if wildfire_oversample < 1:
        raise ValueError("wildfire_oversample must be >= 1")
    train_clean_idx = train_no_wildfire_idx
    # Oversample the wildfire rows N times to MIMIC upweighting them in the all-data
    # arm: a duplicated row contributes N x to the loss, i.e. weight N. N=1 -> the
    # natural ~11% wildfire mix; N ~= n_clean/n_wildfire (~8) -> roughly balanced.
    train_all_idx = np.concatenate([
        train_no_wildfire_idx,
        np.tile(train_wildfire_idx, wildfire_oversample),
    ])

    model_spec = get_engression_model(engression_model)
    is_lstm = model_spec.name == "lstm"
    flatten = model_spec.input_kind == "flat"

    def to_model_x(rows: np.ndarray) -> torch.Tensor:
        arr = dataset.x[rows]
        if flatten:
            arr = arr.reshape(len(rows), -1)
        return torch.from_numpy(arr)

    x_all_train = to_model_x(train_all_idx)
    y_all_train = torch.from_numpy(dataset.y[train_all_idx].reshape(-1, 1))
    x_clean_train = to_model_x(train_clean_idx)
    y_clean_train = torch.from_numpy(dataset.y[train_clean_idx].reshape(-1, 1))

    def generate_fit_kwargs(model_name: str) -> dict:
        fit_kwargs: dict[str, object] = dict(
            num_layer=num_layer,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            add_bn=False,
            lr=lr,
            num_epochs=epochs,
            batch_size=batch_size,
            standardize=True,
            device=device,
            verbose=False,
        )
        if model_spec.name in {"regularized", "adamw", "lstm"}:
            fit_kwargs["weight_decay"] = weight_decay
        if is_lstm and early_stop_patience is not None:
            fit_kwargs["early_stop_patience"] = early_stop_patience
        if is_lstm and save_checkpoint_dir is not None:
            ckpt_dir = _resolve_save_dir(save_checkpoint_dir)   # -> storage3
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            fit_kwargs.update(
                checkpoint_path=str(ckpt_dir / f"checkpoint_{model_name}_latest.pt"),
                checkpoint_best_path=str(ckpt_dir / f"checkpoint_{model_name}_best.pt"),
                checkpoint_every_nepoch=max(1, epochs // 4),
            )
        if fit_overrides:
            fit_kwargs.update(fit_overrides)
        return fit_kwargs

    sink = io.StringIO() if silent else sys.stdout
    with contextlib.redirect_stdout(sink):
        engressor_all = model_spec.fit(x_all_train, y_all_train, **generate_fit_kwargs("all"))
        engressor_clean = model_spec.fit(x_clean_train, y_clean_train, **generate_fit_kwargs("clean"))

    return (
        engressor_all,
        engressor_clean,
        dataset,
        train_all_idx,
        train_clean_idx,
        test_wildfire_idx,
        test_no_wildfire_idx,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit the clean arm and the all-data arm (partial-learner).")
    p.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    p.add_argument("--engression-model", type=str, default="lstm")
    p.add_argument("--target", type=str, default="ccn")
    p.add_argument("--log-ccn", dest="log_ccn", action="store_true", default=True)
    p.add_argument("--no-log-ccn", dest="log_ccn", action="store_false")
    p.add_argument("--split", type=str, default="paper")
    p.add_argument("--seq-stride", type=int, default=1)
    p.add_argument("--max-samples", type=parse_count_or_all, default=None)
    p.add_argument("--train-size", type=parse_count_or_all, default=None)
    p.add_argument("--test-size", type=parse_count_or_all, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--num-layer", type=int, default=3)
    p.add_argument("--noise-dim", type=int, default=96)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--wildfire-flag", type=str, default="BB_criterion1",
                   help="Flag defining wildfire samples (union goes into the all-data arm). "
                        "Clean arm is always NOT(crit1|crit2).")
    p.add_argument("--save-checkpoint-dir", type=str, default=None,
                   help="Save the two arm checkpoints here; a relative path lands under storage3.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-epoch training energy-loss prints. Verbose by default.")
    p.add_argument("--early-stop-patience", type=int, default=12,
                   help="Stop an arm after this many epochs with no training-loss improvement. "
                        "0 or negative disables early stopping.")
    p.add_argument("--wildfire-oversample", type=int, default=1,
                   help="Duplicate wildfire rows this many times in the all-data arm to mimic "
                        "upweighting them (1 = natural ~11 pct wildfire; ~8 ~= balanced 50/50).")
    for flag in _LSTM_HEAD_FLAGS:
        p.add_argument(f"--{flag.replace('_', '-')}", dest=flag, action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    fit_overrides = {flag: True for flag in _LSTM_HEAD_FLAGS if getattr(args, flag, False)}

    print("Run parameters:", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"  {key} = {value}", flush=True)
    print(f"  -> active head variant = {sorted(fit_overrides) or ['default (plain lstm head)']}", flush=True)
    if args.save_checkpoint_dir is not None:
        print(f"  -> resolved save dir  = {_resolve_save_dir(args.save_checkpoint_dir)}", flush=True)

    (
        engressor_all,
        engressor_clean,
        dataset,
        train_all_idx,
        train_clean_idx,
        test_wildfire_idx,
        test_no_wildfire_idx,
    ) = load_and_fit(
        mat_path=args.mat_path,
        target=args.target,
        log_ccn=args.log_ccn,
        split=args.split,
        seq_stride=args.seq_stride,
        max_samples=args.max_samples,
        train_size=args.train_size,
        test_size=args.test_size,
        engression_model=args.engression_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        noise_dim=args.noise_dim,
        device=args.device,
        seed=args.seed,
        wildfire_flag=args.wildfire_flag,
        fit_overrides=fit_overrides or None,
        save_checkpoint_dir=args.save_checkpoint_dir,
        silent=args.quiet,
        early_stop_patience=(args.early_stop_patience if args.early_stop_patience > 0 else None),
        wildfire_oversample=args.wildfire_oversample,
    )

    n_wf_eff = len(train_all_idx) - len(train_clean_idx)
    print(f"  all-data arm: {len(train_clean_idx)} clean + {n_wf_eff} wildfire "
          f"(oversample {args.wildfire_oversample}x) -> {100 * n_wf_eff / len(train_all_idx):.1f}% wildfire",
          flush=True)

    if args.save_checkpoint_dir is not None:
        save_dataset_dir = _resolve_save_dir(args.save_checkpoint_dir) / "dataset"
        save_dataset_dir.mkdir(parents=True, exist_ok=True)
        to_pickle = {
            "dataset_obj.pkl": dataset,
            "train_all_idx.pkl": train_all_idx,
            "train_clean_idx.pkl": train_clean_idx,
            "test_wildfire_idx.pkl": test_wildfire_idx,
            "test_no_wildfire_idx.pkl": test_no_wildfire_idx,
        }
        for name, obj in to_pickle.items():
            with open(save_dataset_dir / name, "wb") as file:
                pickle.dump(obj, file)


if __name__ == "__main__":
    main()
