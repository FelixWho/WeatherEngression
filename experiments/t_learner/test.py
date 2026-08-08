"""Estimate the wildfire effect on CCN from the two trained T-learner arms.

Reloads the two trained arms, picks the covariates to evaluate at, samples both,
and contrasts the resulting CCN distributions.

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
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Headless plotting (compute nodes have no display / writable HOME cache).
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig_tlearner"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    ``train.load_and_fit(save_checkpoint_dir=...)`` wrote, WITHOUT retraining.

    Returns the SAME 7-tuple as ``load_and_fit``. The dataset and the four arm
    index arrays are read straight from the ``dataset/`` pickles under the
    checkpoint dir, so they are the EXACT arrays training used. No split
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
            "in train.py. Retrain with the current train.py, or reconstruct the "
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
# Sampling helpers, reused by the estimation below.
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

    Returns ``(with_wildfire, counterfactual_clean)``, each ``(n, n_samples)``: the
    factual "with wildfire" draws and the counterfactual "no wildfire" draws.
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
# Per-arm calibration: coverage + PIT, on the arm's OWN held-out test set.
# --------------------------------------------------------------------------- #
def arm_calibration(engressor: LSTMEngressor, x: np.ndarray, y: np.ndarray,
                    n_samples: int = 400, seed: int = 0) -> dict:
    """Coverage (50%/90%) + randomized PIT for one arm on its own test set.

    Draw samples at x, compare to realized y. PIT = rank of y among the samples
    (randomized so it's exactly Uniform(0,1) under correct predictive shape).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    samples = sample_conditional(engressor, torch.as_tensor(x, dtype=torch.float32), n_samples)  # (n, S)
    q05, q25, q50, q75, q95 = np.quantile(samples, [0.05, 0.25, 0.5, 0.75, 0.95], axis=1)
    below = np.sum(samples < y[:, None], axis=1)
    at_or_below = np.sum(samples <= y[:, None], axis=1)
    u = np.random.default_rng(seed).random(len(y))
    pit = (below + u * (at_or_below - below + 1)) / (n_samples + 1)
    return {
        "coverage_90": float(np.mean((y >= q05) & (y <= q95))),
        "coverage_50": float(np.mean((y >= q25) & (y <= q75))),
        "mean_width_90": float(np.mean(q95 - q05)),
        "median_abs_error": float(np.mean(np.abs(y - q50))),
        "pit": pit,
    }


# --------------------------------------------------------------------------- #
# Effect estimands: run BOTH arms at the SAME covariates, then difference.
#   average over wildfire x -> ATT ; clean x -> ATC ; all x -> ATE
# --------------------------------------------------------------------------- #
def estimand(eng_wildfire: LSTMEngressor, eng_no_wildfire: LSTMEngressor,
             x: np.ndarray, n_samples: int = 400) -> dict:
    """Treatment effect at covariates ``x``: effect(x) = wildfire(x) - clean(x).

    Both arms are sampled at the SAME x (that's what makes it causal, not a
    smoky-vs-clean comparison). Returns the mean effect in log10(CCN), the
    multiplicative ratio 10**mean, the raw-CCN mean difference, and the pooled
    factual / counterfactual sample sets (for the money plot).
    """
    xt = torch.as_tensor(x, dtype=torch.float32)
    with_wildfire = sample_conditional(eng_wildfire, xt, n_samples)        # (n, S) log10 CCN
    counterfactual = sample_conditional(eng_no_wildfire, xt, n_samples)    # (n, S) log10 CCN
    per_x_effect = with_wildfire.mean(axis=1) - counterfactual.mean(axis=1)  # log10, per period
    mean_log = float(per_x_effect.mean())
    factual_pool = with_wildfire.reshape(-1)
    counterfactual_pool = counterfactual.reshape(-1)
    mean_raw = float((10.0 ** factual_pool).mean() - (10.0 ** counterfactual_pool).mean())
    return {
        "n_periods": int(len(x)),
        "mean_log_effect": mean_log,                 # Δ in log10(CCN)
        "ratio": float(10.0 ** mean_log),            # wildfire CCN / clean CCN (×)
        "mean_raw_ccn_effect": mean_raw,             # Δ in CCN (cm^-3)
        "factual": factual_pool,
        "counterfactual": counterfactual_pool,
    }


# --------------------------------------------------------------------------- #
# Charts.
# --------------------------------------------------------------------------- #
def plot_pit(pit: np.ndarray, title: str, out_path: Path, n_bins: int = 20) -> None:
    """PIT histogram with the uniform line + 95% consistency band."""
    n = len(pit)
    counts, edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    expected = n / n_bins
    half = 1.96 * np.sqrt(n * (1.0 / n_bins) * (1.0 - 1.0 / n_bins))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
           color="#4c72b0", alpha=0.75, edgecolor="white")
    ax.axhline(expected, ls="--", color="#2a2a2a", alpha=0.8, label="calibrated (uniform)")
    ax.fill_between([0.0, 1.0], expected - half, expected + half, color="#dd8452", alpha=0.2,
                    label="95% band")
    ax.set_xlim(0, 1); ax.set_ylim(0, max(counts.max(), expected + half) * 1.15)
    ax.set_xlabel("PIT value"); ax.set_ylabel("count"); ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def plot_money(factual: np.ndarray, counterfactual: np.ndarray, title: str, out_path: Path) -> None:
    """The money plot: pooled factual vs counterfactual CCN distributions overlaid."""
    lo = float(min(factual.min(), counterfactual.min()))
    hi = float(max(factual.max(), counterfactual.max()))
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(counterfactual, bins=bins, density=True, alpha=0.55, color="#55a868",
            label="counterfactual: no wildfire")
    ax.hist(factual, bins=bins, density=True, alpha=0.55, color="#c44e52",
            label="factual: with wildfire")
    ax.axvline(counterfactual.mean(), color="#55a868", ls="--", lw=1.5)
    ax.axvline(factual.mean(), color="#c44e52", ls="--", lw=1.5)
    ax.set_xlabel("log10(CCN)"); ax.set_ylabel("density"); ax.set_title(title)
    ax.legend(fontsize="small")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate the wildfire effect from the saved T-learner arms.")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Dir holding checkpoint_{wildfire,no_wildfire}_best.pt (abs, or relative to storage3).")
    p.add_argument("--which", choices=("best", "latest"), default="best")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--n-samples", type=int, default=400, help="conditional draws per covariate")
    p.add_argument("--out-dir", type=str, default="reports/t_learner_effect",
                   help="Where to write charts + metrics.json (repo-relative).")
    p.add_argument("--seed", type=int, default=0, help="seed for the randomized PIT tie-break")
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

    print("Loaded checkpointed arms and dataset", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Covariates + realized targets for each arm's own held-out test set.
    xw, yw = dataset.x[test_wildfire_idx], dataset.y[test_wildfire_idx]
    xc, yc = dataset.x[test_no_wildfire_idx], dataset.y[test_no_wildfire_idx]

    # --- 1) Per-arm quality: CRPS + coverage + PIT, each on its OWN test set. ---
    crps_wildfire = energy_loss(eng_wildfire, xw, yw, n_samples_per_x=args.n_samples)
    crps_clean = energy_loss(eng_no_wildfire, xc, yc, n_samples_per_x=args.n_samples)
    cal_wildfire = arm_calibration(eng_wildfire, xw, yw, args.n_samples, seed=args.seed)
    cal_clean = arm_calibration(eng_no_wildfire, xc, yc, args.n_samples, seed=args.seed)
    plot_pit(cal_wildfire["pit"], "PIT - wildfire arm (own test set)", out_dir / "pit_wildfire.png")
    plot_pit(cal_clean["pit"], "PIT - clean arm (own test set)", out_dir / "pit_clean.png")

    # --- 2) Effects: ATT (wildfire x), ATC (clean x), ATE (all x). ---
    att = estimand(eng_wildfire, eng_no_wildfire, xw, args.n_samples)
    atc = estimand(eng_wildfire, eng_no_wildfire, xc, args.n_samples)
    ate = estimand(eng_wildfire, eng_no_wildfire, np.concatenate([xw, xc], axis=0), args.n_samples)

    # --- 3) Money plot from ATT (the headline). ---
    plot_money(att["factual"], att["counterfactual"],
               "With wildfire vs counterfactual no-wildfire (ATT)", out_dir / "money_plot.png")

    # --- 4) Write metrics.json (drop the big sample arrays). ---
    def _effect(d):
        return {k: d[k] for k in ("n_periods", "mean_log_effect", "ratio", "mean_raw_ccn_effect")}
    metrics = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_samples": args.n_samples,
        "arms": {
            "wildfire": {"n_test": int(len(yw)), "crps": crps_wildfire,
                         "coverage_90": cal_wildfire["coverage_90"], "coverage_50": cal_wildfire["coverage_50"],
                         "mean_width_90": cal_wildfire["mean_width_90"]},
            "clean": {"n_test": int(len(yc)), "crps": crps_clean,
                      "coverage_90": cal_clean["coverage_90"], "coverage_50": cal_clean["coverage_50"],
                      "mean_width_90": cal_clean["mean_width_90"]},
        },
        "effects": {"ATT": _effect(att), "ATC": _effect(atc), "ATE": _effect(ate)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # --- 5) Console summary. ---
    print("\n=== per-arm quality (each on its own test set) ===", flush=True)
    print(f"  wildfire arm: CRPS {crps_wildfire:.4f} | cov90 {cal_wildfire['coverage_90']:.3f} "
          f"| cov50 {cal_wildfire['coverage_50']:.3f} | width90 {cal_wildfire['mean_width_90']:.3f}")
    print(f"  clean    arm: CRPS {crps_clean:.4f} | cov90 {cal_clean['coverage_90']:.3f} "
          f"| cov50 {cal_clean['coverage_50']:.3f} | width90 {cal_clean['mean_width_90']:.3f}")
    print("\n=== wildfire effect on CCN (log10 Δ | ×ratio | raw Δ cm^-3) ===", flush=True)
    for name, d in (("ATT", att), ("ATC", atc), ("ATE", ate)):
        print(f"  {name} (n={d['n_periods']:5d}): {d['mean_log_effect']:+.3f} log10 "
              f"| {d['ratio']:.2f}x | {d['mean_raw_ccn_effect']:+.1f} cm^-3")
    print(f"\nwrote charts + metrics.json to: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
