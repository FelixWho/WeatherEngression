"""Estimate the wildfire effect on CCN from the two trained T-learner arms.

SKELETON -- the plumbing (reload the arms, pick the covariates, sample both) is
wired up; the actual estimation is left as TODO stubs to fill in.

Idea (distributional ATT): at the wildfire-period covariates, sample the wildfire
arm ("with wildfire") and the clean arm ("counterfactual no-wildfire"), then
contrast the two CCN distributions. Evaluate at ``test_wildfire_idx`` -> ATT;
swap in all/clean covariates for ATE/ATC.

Run:
    python -m experiments.t_learner.test \
        --checkpoint-dir /storage3/.../t_learner/BB_criterion1_recurrent
(Split params default to train.py's, so a run trained with defaults reloads as-is.)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # experiments/t_learner/test.py -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import ENADataset  # noqa: F401 (needed to unpickle dataset_obj.pkl)
from engression_modifications.lstm import load_lstm_engressor_checkpoint, LSTMEngressor
from experiments.t_learner.train import _resolve_save_dir


# --------------------------------------------------------------------------- #
# Reload the two trained arms (no retraining). Lives here because loading a
# fitted model is an analysis concern; train.py only trains + saves.
# --------------------------------------------------------------------------- #
def load_saved(
    save_checkpoint_dir: str | Path,
    *,
    which: str = "best",
    device: str | None = None,
):
    """Reload the two arm models AND the pickled dataset / arm indices that
    ``train.load_and_fit(save_checkpoint_dir=...)`` wrote -- WITHOUT retraining.

    Returns the SAME 7-tuple as ``load_and_fit``. The dataset and the four arm
    index arrays are read straight from the ``dataset/`` pickles under the
    checkpoint dir, so they are the EXACT arrays training used -- no split
    reconstruction, no seed/param matching to get wrong.
    """
    ckpt_dir = _resolve_save_dir(save_checkpoint_dir)   # same storage3 resolution as saving
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    engressor_wildfire = load_lstm_engressor_checkpoint(
        ckpt_dir / f"checkpoint_wildfire_{which}.pt", device=device
    )
    engressor_no_wildfire = load_lstm_engressor_checkpoint(
        ckpt_dir / f"checkpoint_no_wildfire_{which}.pt", device=device
    )

    # Dataset + arm indices come from the pickles train.py saved (see its main()).
    ds_dir = ckpt_dir / "dataset"
    if not (ds_dir / "dataset_obj.pkl").exists():
        raise FileNotFoundError(
            f"no pickled dataset under {ds_dir}. This run predates the pickle dump "
            "in train.py -- retrain with the current train.py, or reconstruct the "
            "split from the seed instead."
        )

    def _unpickle(name: str):
        with open(ds_dir / name, "rb") as f:
            return pickle.load(f)

    dataset: ENADataset = _unpickle("dataset_obj.pkl")
    return (
        engressor_wildfire,
        engressor_no_wildfire,
        dataset,
        _unpickle("train_wildfire_idx.pkl"),
        _unpickle("train_no_wildfire_idx.pkl"),
        _unpickle("test_wildfire_idx.pkl"),
        _unpickle("test_no_wildfire_idx.pkl"),
    )


# --------------------------------------------------------------------------- #
# Sampling helpers (working -- reuse these in the estimation below).
# --------------------------------------------------------------------------- #
def sample_conditional(engressor: LSTMEngressor, x: torch.Tensor, n_samples: int = 400) -> np.ndarray:
    """Draw ``n_samples`` from an arm's conditional CCN distribution at each row of x.

    Returns ``(n, n_samples)`` (out_dim == 1). These are draws in the model's target
    space (log10 CCN if the run used --log-ccn).
    """
    samples = engressor.sample(x, sample_size=n_samples, expand_dim=True)
    return samples.detach().cpu().numpy()[:, 0, :]


def counterfactual_pair(eng_wildfire: LSTMEngressor, eng_no_wildfire: LSTMEngressor, x: torch.Tensor, n_samples: int = 400):
    """At covariates ``x``, sample both arms.

    Returns ``(with_wildfire, counterfactual_clean)``, each ``(n, n_samples)`` --
    the factual "with wildfire" draws and the counterfactual "no wildfire" draws.
    This is the raw material for every effect below.
    """
    with_wildfire = sample_conditional(eng_wildfire, x, n_samples)
    counterfactual_clean = sample_conditional(eng_no_wildfire, x, n_samples)
    return with_wildfire, counterfactual_clean

# --------------------------------------------------------------------------- #
# Loss function.  
# --------------------------------------------------------------------------- #

def energy_loss(engressor: LSTMEngressor, xs: np.ndarray, ys: np.ndarray, n_samples_per_x: int = 100) -> float:
    """
    Paper CRPS empirical loss
    """
    if n_samples_per_x <= 1:
        raise ValueError("n_samples_per_x must be > 1")
    assert len(xs) == len(ys)

    xs_torch = torch.as_tensor(xs, dtype=torch.float32)
    ys = ys.reshape(-1)

    samples = sample_conditional(engressor, xs_torch, n_samples_per_x) # (n, n_samples_per_x)
    lhs = np.mean(np.abs(ys[:, None] - samples))
    pairwise = np.abs(samples[:, :, None] - samples[:, None, :])
    rhs = np.mean(
        pairwise.sum(axis=(1, 2))
        / (2 * n_samples_per_x * (n_samples_per_x - 1))
    )

    return float(lhs - rhs)


# --------------------------------------------------------------------------- #
# TODO: estimation -- fill these in.
# --------------------------------------------------------------------------- #
def distributional_treatment_effect(with_wildfire, counterfactual_clean, quantiles=None):
    """TODO: the headline (distributional ATT).

    Pool the draws across units, then difference the two distributions
    quantile-by-quantile: ``QTE(tau) = Q_withwildfire(tau) - Q_clean(tau)``.
    Return the QTE curve plus the scalar mean effect (mean ATT).
    """
    raise NotImplementedError


def cate(with_wildfire, counterfactual_clean):
    """TODO: per-unit effect (CATE / heterogeneity view).

    For each covariate row, effect = summary(with_wildfire[i]) - summary(clean[i])
    (e.g. mean, or a per-quantile difference). Return one effect per unit so its
    spread describes where the effect is large vs small.
    """
    raise NotImplementedError


def episode_bootstrap(dataset, arm_idx, estimator, n_boot: int = 500, seed: int = 0):
    """TODO: uncertainty via BLOCK bootstrap over wildfire episodes (~150), not rows.

    Group ``arm_idx`` into contiguous episodes (consecutive time indices), resample
    whole episodes with replacement, re-run ``estimator`` on each resample, and
    collect the distribution of the effect -> CIs. Row-level bootstrap would be
    overconfident because hourly samples are near-duplicates.
    """
    raise NotImplementedError


def report(effect, ci=None, out_dir: Path | None = None):
    """TODO: write the deliverable -- the money plot (factual vs counterfactual
    distributions), the QTE curve, and a metrics JSON. Left for you to style.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate the wildfire effect from the saved T-learner arms.")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Dir holding checkpoint_{wildfire,no_wildfire}_best.pt (abs, or relative to storage3).")
    p.add_argument("--which", choices=("best", "latest"), default="best")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--n-samples", type=int, default=400, help="conditional draws per covariate")
    # No split params needed: the dataset + arm indices are read from the pickles
    # under <checkpoint-dir>/dataset/, so they always match the trained models.
    return p


def main() -> None:
    args = _build_parser().parse_args()
    print("Parsed arguments:", flush=True)
    for name, value in sorted(vars(args).items()):
        print(f"  {name}: {value}", flush=True)

    # 1) Reload both arms + the dataset/arm indices (no retraining).
    (
        eng_wildfire,
        eng_no_wildfire,
        dataset,
        train_wildfire_idx,
        train_no_wildfire_idx,
        test_wildfire_idx,
        test_no_wildfire_idx,
    ) = load_saved(
        args.checkpoint_dir,
        which=args.which,
        device=args.device,
    )

    print("Loaded checkpointed arms and dataset")

    # 2) Headline = ATT: evaluate at the wildfire-period TEST covariates.
    #    (For ATE use all test covariates; for ATC use test_no_wildfire_idx.)
    eval_idx = test_wildfire_idx
    test_wildfire_x = dataset.x[test_wildfire_idx]
    test_wildfire_y = dataset.y[test_wildfire_idx]
    test_no_wildfire_x = dataset.x[test_no_wildfire_idx]
    test_no_wildfire_y = dataset.y[test_no_wildfire_idx]

    # with_wildfire, counterfactual_clean = counterfactual_pair(
    #     eng_wildfire, eng_no_wildfire, x_eval, n_samples=args.n_samples
    # )
    loss_score_wildfire = energy_loss(
        eng_wildfire,
        test_wildfire_x,
        test_wildfire_y,
        n_samples_per_x=args.n_samples,
    )
    loss_score_no_wildfire = energy_loss(
        eng_no_wildfire,
        test_no_wildfire_x,
        test_no_wildfire_y,
        n_samples_per_x=args.n_samples,
    )
    print("Wildfire crps:", loss_score_wildfire)
    print("No wildfire crps:", loss_score_no_wildfire)


if __name__ == "__main__":
    main()
