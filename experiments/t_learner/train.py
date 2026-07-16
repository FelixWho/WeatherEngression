"""T-learner base: read the ENA dataset and fit the wildfire / wildfire-free arms.

Stripped-down counterpart to ``real_data_diagnostic.py`` -- no OOS diagnostics,
no metrics, no charts, no prints. Just ``load -> split -> partition by wildfire ->
fit two models``: one engressor on the wildfire (treatment) samples and one on the
wildfire-FREE (control) samples, returned alongside the dataset and split indices.

Programmatic use
----------------
```python
from experiments.t_learner.train import load_and_fit

(eng_wildfire, eng_no_wildfire, dataset,
 train_idx, test_idx) = load_and_fit(
    target="ccn", log_ccn=True, split="paper", epochs=40,
    wildfire_flag="BB_criterion1",                   # treatment definition
    fit_overrides={"recurrent_state_noise": True},   # pick a head variant
)
```

Shell use (prints per-epoch energy-loss; add --save-checkpoint-dir to persist)
-----------------------------------------------------------------------------
```bash
python -m experiments.t_learner.train --target ccn --log-ccn --recurrent-state-noise \
  --epochs 40 --lr 0.003 --hidden-dim 192 --num-layer 5 --device cuda \
  --save-checkpoint-dir runs/scratch/my_fit
```
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import sys
import pickle

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/t_learner/train.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    load_ena_supervised_dataset,
    parse_count_or_all,
    select_real_split,
)
from engression_modifications import get_engression_model
from experiments.pipeline import set_reproducible_seeds

# All checkpoints go to storage3 (the /home quota is tiny). A relative
# save_checkpoint_dir is resolved UNDER this root, mirroring the runs/ tree; an
# absolute path is used as-is (explicit override).
DEFAULT_CKPT_ROOT = Path("/storage3/fs1/myu/Active/felixhu/weather_checkpoints")


def _resolve_save_dir(save_checkpoint_dir: str | Path) -> Path:
    """Send saves to storage3: relative paths land under DEFAULT_CKPT_ROOT."""
    p = Path(save_checkpoint_dir)
    return p if p.is_absolute() else DEFAULT_CKPT_ROOT / p


# Head-noise flags accepted by the lstm fit(); expose them so every variant is reachable.
_LSTM_HEAD_FLAGS = (
    "pre_additive",
    "stonet_head",
    "appending_noise",
    "additive_noise",
    "per_timestep_noise",
    "global_latent_noise",
    "stochastic_init_noise",
    "recurrent_state_noise",
)


def _load_dataset_and_arms(
    *,
    mat_path: str,
    target: str,
    log_ccn: bool,
    split: str,
    seq_stride: int,
    max_samples: int | None,
    train_size: int | None,
    test_size: int | None,
    seed: int,
    wildfire_flag: str,
):
    """Load the dataset, build the split, and partition each split into the
    wildfire (treatment) and wildfire-free (control) arms.

    Shared by ``load_and_fit`` (to train) and ``load_saved`` (to reload) so the arm
    indices always match a given set of split params. See ``load_and_fit`` for the
    control-arm definition (excludes both BB criteria).
    """
    set_reproducible_seeds(seed)
    dataset = load_ena_supervised_dataset(
        mat_path=mat_path,
        target=target,
        max_samples=max_samples,
        seq_stride=seq_stride,
        seed=seed,
        log_ccn=log_ccn,
    )
    train_idx, test_idx = select_real_split(
        dataset=dataset,
        split=split,
        train_size=train_size,
        test_size=test_size,
        seed=seed + 17,
    )
    if wildfire_flag not in dataset.flags:
        raise KeyError(
            f"unknown wildfire flag {wildfire_flag!r}; available: {sorted(dataset.flags)}"
        )
    treated_idx = np.flatnonzero(dataset.flags[wildfire_flag])
    clean_idx = np.flatnonzero(
        ~(dataset.flags["BB_criterion1"] | dataset.flags["BB_criterion2"])
    )
    return (
        dataset,
        np.intersect1d(train_idx, treated_idx),
        np.intersect1d(train_idx, clean_idx),
        np.intersect1d(test_idx, treated_idx),
        np.intersect1d(test_idx, clean_idx),
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
):
    """Read the dataset, build the training split, and fit the model.

    Returns ``(engressor, dataset, train_idx, test_idx)``. ``fit_overrides`` is
    merged into the fit kwargs last -- use it to select a head variant (e.g.
    ``{"recurrent_state_noise": True}``) or override any training knob. Training is
    VERBOSE by default (per-epoch energy-loss / CRPS goes to stdout); pass
    ``silent=True`` to swallow it.
    """

    # Load + split + partition into the wildfire (treatment) and wildfire-FREE
    # (control) arms. The control excludes BOTH BB criteria (union), so a strong
    # event missed by the relaxed criterion (crit2>0 but crit1==0) -- or a NaN-flag
    # period that binarizes to False -- does NOT leak into the "clean" arm; points
    # flagged only by the other criterion fall in neither arm (a deliberate buffer).
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

    model_spec = get_engression_model(engression_model)
    is_lstm = model_spec.name == "lstm"
    flatten = model_spec.input_kind == "flat"

    def to_model_x(rows: np.ndarray) -> torch.Tensor:
        arr = dataset.x[rows]
        if flatten:
            arr = arr.reshape(len(rows), -1)
        return torch.from_numpy(arr)

    x_wildfire_train = to_model_x(train_wildfire_idx)
    y_wildfire_train = torch.from_numpy(dataset.y[train_wildfire_idx].reshape(-1, 1))
    x_no_wildfire_train = to_model_x(train_no_wildfire_idx)
    y_no_wildfire_train = torch.from_numpy(dataset.y[train_no_wildfire_idx].reshape(-1, 1))

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

    # Training is verbose by default: the fit loop prints per-epoch energy-loss
    # (CRPS) to stdout. silent=True swallows it into a throwaway buffer.
    sink = io.StringIO() if silent else sys.stdout
    with contextlib.redirect_stdout(sink):
        engressor_wildfire = model_spec.fit(x_wildfire_train, y_wildfire_train, **generate_fit_kwargs("wildfire"))
        engressor_no_wildfire = model_spec.fit(x_no_wildfire_train, y_no_wildfire_train, **generate_fit_kwargs("no_wildfire"))

    return engressor_wildfire, engressor_no_wildfire, dataset, train_wildfire_idx, train_no_wildfire_idx, test_wildfire_idx, test_no_wildfire_idx


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read the ENA dataset and fit an engression model.")
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
                   help="Flag defining the treatment (wildfire) arm: BB_criterion1 (total effect) "
                        "or BB_criterion2 (strong events). Control is always NOT(crit1|crit2).")
    p.add_argument("--save-checkpoint-dir", type=str, default=None,
                   help="Save the two arm checkpoints here; a relative path lands under storage3.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the per-epoch training energy-loss (CRPS) prints. Verbose by default.")
    # one boolean flag per head variant (default = plain lstm head)
    for flag in _LSTM_HEAD_FLAGS:
        p.add_argument(f"--{flag.replace('_', '-')}", dest=flag, action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    fit_overrides = {flag: True for flag in _LSTM_HEAD_FLAGS if getattr(args, flag, False)}

    # Log every parsed parameter (defaults included) so the run is fully reproducible
    # from the log alone. Printed here in main(), outside load_and_fit's silent block.
    print("Run parameters:", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"  {key} = {value}", flush=True)
    print(f"  -> active head variant = {sorted(fit_overrides) or ['default (plain lstm head)']}", flush=True)
    if args.save_checkpoint_dir is not None:
        print(f"  -> resolved save dir  = {_resolve_save_dir(args.save_checkpoint_dir)}", flush=True)
    engressor_wildfire, engressor_no_wildfire, dataset, train_wildfire_idx, train_no_wildfire_idx, test_wildfire_idx, test_no_wildfire_idx = load_and_fit(
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
    )

    if args.save_checkpoint_dir is not None:
        save_dataset_dir = _resolve_save_dir(args.save_checkpoint_dir) / "dataset"
        save_dataset_dir.mkdir(parents=True, exist_ok=True)
        to_pickle = {
            "dataset_obj.pkl": dataset,
            "train_wildfire_idx.pkl": train_wildfire_idx,
            "train_no_wildfire_idx.pkl": train_no_wildfire_idx,
            "test_wildfire_idx.pkl": test_wildfire_idx,
            "test_no_wildfire_idx.pkl": test_no_wildfire_idx,
        }
        for name, obj in to_pickle.items():
            with open(save_dataset_dir / name, "wb") as file:
                pickle.dump(obj, file)

if __name__ == "__main__":
    main()
