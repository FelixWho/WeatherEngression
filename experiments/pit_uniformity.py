"""Uniformity p-values for the PIT histograms across the stochastic-sweep grid.

The PIT charts show the *shape* of miscalibration; this script attaches a number
to the question "is this histogram flat?" for every cell of the 48-config LSTM
noise-injection sweep, so configs can be ranked and filtered rather than eyeballed.

For each checkpoint we rebuild the identical ``paper`` split, resample, and form
the randomized PIT (reusing ``pit_values`` so the values match the plotted
histograms), then run three goodness-of-fit tests against Uniform(0, 1):

- Anderson-Darling (A^2 computed directly against F(x)=x, since SciPy's
  ``anderson`` has no uniform null). Tail-weighted, which is what we want because
  CCN miscalibration tends to live in the extreme tail. This is the headline
  effect-size statistic.
- chi-squared on the 20 histogram bins (the bins you already look at).
- Kolmogorov-Smirnov (central-deviation sensitive, weakest in the tails).

The catch this split carries: test rows are distinct back-trajectories, but the
paper split keeps *every* trajectory arriving in six 2022 months, and trajectories
with nearby arrival times share overlapping air-mass histories. So the ~3220 PIT
values are not guaranteed independent, and all three tests assume they are ->
their naive p-values are anti-conservative under positive autocorrelation.

Rather than assume, we measure it. Ordering PIT by arrival time
(``dataset.time``), we estimate the integrated autocorrelation time tau and the
effective sample size n_eff = n / tau, and report a dependence-robust p-value
alongside the naive one:

- chi-squared re-scaled to n_eff (the statistic scales linearly with n at fixed
  bin proportions), and
- Anderson-Darling on a time-thinned subsample (one point per ~tau, median
  p-value over the tau phase offsets).

If tau ~ 1 the rows are effectively independent and robust == naive; if tau is
large the gap tells you how much the dependence mattered. Finally, since the grid
is 48 simultaneous tests, we add Benjamini-Hochberg adjusted p-values across cells
so the best-looking cell is not just multiple-comparisons noise.

The AD null uses a Monte-Carlo simulation (fixed seed), so p-values are
reproducible. Recomputing PIT for 48 checkpoints needs a GPU; run under sbatch
(see slurm/pit_uniformity/run.slurm), not in the foreground.

Example
-------
```bash
python experiments/pit_uniformity.py --prediction-samples 400
```
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import stats

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
from experiments.real_data_diagnostic import QUANTILE_LEVELS, predict_quantiles_and_samples
from experiments.real_metrics import pit_values

# Same grid and ordering as slurm/pit_charts_stochastic_sweep_oos/run.slurm.
VARIANTS = ("additive", "appending", "per_timestep", "global_latent", "init_state", "recurrent")
LRS = ("0.003", "0.01")
LAYERS = (3, 5)
HIDDENS = (128, 192)

DEFAULT_SWEEP_SUBDIR = Path("runs/real_data_diagnostics/ena_weather/log10_ccn/paper/lstm_stochastic_sweep_oos")
DEFAULT_CKPT_ROOT = Path("/storage3/fs1/myu/Active/felixhu/weather_checkpoints")

_EPS = 1e-12


def grid_tags() -> list[str]:
    """The 48 config tags, one per sweep cell (``variant_lr..._l..._h...``)."""

    tags = []
    for variant, lr, layers, hidden in itertools.product(VARIANTS, LRS, LAYERS, HIDDENS):
        tags.append(f"{variant}_lr{lr}_l{layers}_h{hidden}")
    return tags


def ad_statistic_uniform(pit: np.ndarray) -> float:
    """Anderson-Darling A^2 of ``pit`` against a fully specified Uniform(0, 1).

    A^2 = -n - (1/n) * sum_i (2i-1) * [ ln u_(i) + ln(1 - u_(n+1-i)) ],

    with ``u`` the sorted PIT clipped off the 0/1 boundary so the logs stay finite.
    """

    u = np.sort(np.clip(np.asarray(pit, dtype=np.float64), _EPS, 1.0 - _EPS))
    n = u.size
    idx = np.arange(1, n + 1)
    total = np.sum((2 * idx - 1) * (np.log(u) + np.log(1.0 - u[::-1])))
    return float(-n - total / n)


def _ad_statistic_rows(sorted_rows: np.ndarray) -> np.ndarray:
    """Vectorized A^2 for many already-sorted uniform samples (rows of equal n)."""

    n = sorted_rows.shape[1]
    idx = np.arange(1, n + 1)
    log_u = np.log(np.clip(sorted_rows, _EPS, 1.0 - _EPS))
    log_1m = np.log(np.clip(1.0 - sorted_rows, _EPS, 1.0 - _EPS))
    total = np.sum((2 * idx - 1) * (log_u + log_1m[:, ::-1]), axis=1)
    return -n - total / n


def ad_mc_pvalue(a2_obs: float, n: int, n_sim: int, seed: int) -> float:
    """Monte-Carlo p-value for A^2 under the Uniform(0, 1) null at sample size n."""

    rng = np.random.default_rng(seed)
    sims = rng.random((n_sim, n))
    sims.sort(axis=1)
    a2 = _ad_statistic_rows(sims)
    return float((1 + np.sum(a2 >= a2_obs)) / (n_sim + 1))


def integrated_autocorr_time(z: np.ndarray, max_lag: int | None = None) -> float:
    """Integrated autocorrelation time via the initial-positive-sequence rule.

    tau = 1 + 2 * sum_k rho_k, truncated at the first non-positive lag (Geyer),
    where ``z`` is the PIT series in arrival-time order. Returns >= 1.
    """

    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean()
    n = z.size
    var = float(np.dot(z, z) / n)
    if var <= 0.0 or n < 3:
        return 1.0
    if max_lag is None:
        max_lag = min(n - 1, max(50, n // 20))
    tau = 1.0
    for k in range(1, max_lag + 1):
        rho = float(np.dot(z[:-k], z[k:]) / (n * var))
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    return max(tau, 1.0)


def ad_thinned_pvalue(pit_time_ordered: np.ndarray, spacing: int, n_sim: int, seed: int) -> tuple[float, int]:
    """AD p-value on a ~independent time-thinned subsample.

    Takes one point every ``spacing`` steps in arrival-time order; repeats over all
    ``spacing`` phase offsets and returns the median p-value plus the subsample size.
    """

    spacing = max(1, int(spacing))
    pvals: list[float] = []
    n_sub = 0
    for offset in range(spacing):
        sub = pit_time_ordered[offset::spacing]
        if sub.size < 10:
            continue
        n_sub = sub.size
        a2 = ad_statistic_uniform(sub)
        pvals.append(ad_mc_pvalue(a2, sub.size, n_sim=n_sim, seed=seed + offset))
    if not pvals:
        return float("nan"), 0
    return float(np.median(pvals)), n_sub


def block_thinned_pvalue(
    pit_ordered: np.ndarray,
    times_ordered_hours: np.ndarray,
    block_hours: float,
    n_sim: int,
    seed: int,
    n_starts: int = 10,
) -> tuple[float, int]:
    """AD p-value on a time-blocked subsample: points >= ``block_hours`` apart.

    Greedily walks the arrival-time-ordered points and keeps one whenever it is at
    least ``block_hours`` past the last kept point, so the retained points do not
    share a trajectory window (241 h) and are approximately independent. This is
    robust to the QC gaps in the six test months because it thins on real elapsed
    time, not on row index. Repeats from several start points spread across the
    first block and returns the median p-value plus the subsample size.
    """

    n = pit_ordered.size
    span = times_ordered_hours - times_ordered_hours[0]
    first_block = int(np.searchsorted(span, block_hours))
    first_block = max(1, min(first_block, n))
    starts = np.unique(np.linspace(0, first_block - 1, num=min(n_starts, first_block)).astype(int))

    pvals: list[float] = []
    n_sub = 0
    for start in starts:
        keep = [int(start)]
        last_t = times_ordered_hours[start]
        for j in range(start + 1, n):
            if times_ordered_hours[j] - last_t >= block_hours:
                keep.append(j)
                last_t = times_ordered_hours[j]
        sub = pit_ordered[keep]
        if sub.size < 10:  # MC-AD stays valid at small n, just low-power; below ~10 it is meaningless
            continue
        n_sub = sub.size
        a2 = ad_statistic_uniform(sub)
        pvals.append(ad_mc_pvalue(a2, sub.size, n_sim=n_sim, seed=seed + int(start)))
    if not pvals:
        return float("nan"), n_sub
    return float(np.median(pvals)), n_sub


def uniformity_row(
    pit: np.ndarray,
    times_hours: np.ndarray,
    n_bins: int,
    n_sim: int,
    seed: int,
    block_hours: float = 0.0,
) -> dict:
    """All uniformity statistics for one cell's PIT sample.

    ``times_hours`` are the per-point arrival times (aligned to ``pit``); they set
    the ordering for the autocorrelation estimate and, when ``block_hours`` > 0,
    the time-based thinning that floors the effective sample size at the number of
    non-overlapping blocks. With ``block_hours`` == 0 the thinning falls back to the
    data-driven spacing ceil(tau).
    """

    pit = np.asarray(pit, dtype=np.float64).reshape(-1)
    n = pit.size

    a2 = ad_statistic_uniform(pit)
    ad_p = ad_mc_pvalue(a2, n, n_sim=n_sim, seed=seed)

    counts, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    chi2 = stats.chisquare(counts)
    chi2_stat = float(chi2.statistic)
    chi2_p = float(chi2.pvalue)

    ks = stats.kstest(pit, "uniform")
    ks_stat = float(ks.statistic)
    ks_p = float(ks.pvalue)

    # Dependence diagnostic + robust p-values, from arrival-time ordering.
    order = np.argsort(times_hours, kind="stable")
    pit_ordered = pit[order]
    times_ordered = np.asarray(times_hours, dtype=np.float64)[order]
    tau = integrated_autocorr_time(pit_ordered - 0.5)
    n_eff = n / tau

    if block_hours > 0.0:
        # Time-based blocking: thin to >= block_hours apart, and floor n_eff at the
        # number of independent blocks (can't have more independent points than that).
        ad_p_thin, n_thin = block_thinned_pvalue(
            pit_ordered, times_ordered, block_hours=block_hours, n_sim=n_sim, seed=seed + 1000
        )
        if n_thin:
            n_eff = min(n_eff, float(n_thin))
    else:
        ad_p_thin, n_thin = ad_thinned_pvalue(
            pit_ordered, spacing=int(np.ceil(tau)), n_sim=n_sim, seed=seed + 1000
        )

    df = n_bins - 1
    chi2_stat_eff = chi2_stat * (n_eff / n)
    chi2_p_eff = float(stats.chi2.sf(chi2_stat_eff, df))

    return {
        "n": int(n),
        "ad_stat": a2,
        "ad_p_iid": ad_p,
        "chi2_stat": chi2_stat,
        "chi2_p_iid": chi2_p,
        "ks_stat": ks_stat,
        "ks_p_iid": ks_p,
        "tau": float(tau),
        "n_eff": float(n_eff),
        "chi2_p_eff": chi2_p_eff,
        "ad_p_thin": ad_p_thin,
        "n_thin": int(n_thin),
        "block_hours": float(block_hours),
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI; data defaults mirror the sweep so PIT matches the plotted histograms."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / DEFAULT_SWEEP_SUBDIR,
                        help="Local run dir holding the per-cell TAG folders (for --out default).")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CKPT_ROOT,
                        help="Storage root; checkpoints live at <root>/<sweep-subdir>/<TAG>/checkpoint_best.pt.")
    parser.add_argument("--out", type=Path, default=None, help="CSV path (default: <sweep-dir>/pit_uniformity.csv).")
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
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--n-sim", type=int, default=2000, help="Monte-Carlo draws for the AD null.")
    parser.add_argument(
        "--block-hours",
        type=float,
        default=0.0,
        help="Min elapsed hours between retained points for the robust test. 0 = data-driven "
             "(spacing ceil(tau)). Set to the trajectory length (241) to block by one full "
             "airmass window so the thinned points can't share trajectory history.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    """Loop the 48 checkpoints, test each PIT for uniformity, write a ranked table."""

    args = build_parser().parse_args()
    out_csv = args.out or (args.sweep_dir / "pit_uniformity.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sweep_subdir = args.sweep_dir.relative_to(REPO_ROOT) if args.sweep_dir.is_absolute() else args.sweep_dir

    # The split depends only on the data args + seed, so build it ONCE and reuse it
    # for every checkpoint (matching evaluate_oos_from_checkpoint's seed offsets so
    # the PIT reproduces the plotted histograms).
    print(f"Compute device: {args.device}")
    print("Loading data and rebuilding the paper split (shared across all cells)...")
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
    x_test = torch.from_numpy(dataset.x[test_idx])
    y_test = dataset.y[test_idx].astype(np.float64)
    times_hours = dataset.time[test_idx].astype(np.float64) * 24.0  # datenum (days) -> hours
    block_note = f"block-hours={args.block_hours:g}" if args.block_hours > 0 else "data-driven thinning (tau)"
    print(f"  test rows: {len(test_idx)}   robust null: {block_note}")

    rows: list[dict] = []
    for tag in grid_tags():
        ckpt = args.checkpoint_root / sweep_subdir / tag / "checkpoint_best.pt"
        if not ckpt.is_file():
            print(f"[skip] {tag}: missing checkpoint")
            rows.append({"tag": tag, "status": "missing_checkpoint"})
            continue
        print(f"[test] {tag}")
        try:
            engressor = load_lstm_engressor_checkpoint(ckpt, device=args.device)
            torch.manual_seed(args.seed + 101)  # matches evaluate_oos_from_checkpoint
            _, samples = predict_quantiles_and_samples(
                engressor=engressor,
                x_test=x_test,
                sample_size=args.prediction_samples,
                levels=QUANTILE_LEVELS,
            )
            pit = pit_values(samples, y_test, seed=args.seed + 307)
            stats_row = uniformity_row(
                pit, times_hours, n_bins=args.n_bins, n_sim=args.n_sim,
                seed=args.seed, block_hours=args.block_hours,
            )
        except Exception as exc:  # one bad checkpoint should not sink the other 47
            print(f"       ERROR: {exc}")
            rows.append({"tag": tag, "status": f"error: {exc}"})
            continue
        finally:
            if "engressor" in dir():
                del engressor
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        rows.append({"tag": tag, "status": "ok", **stats_row})
        print(
            f"       A2={stats_row['ad_stat']:.2f}  p_iid={stats_row['ad_p_iid']:.4g}  "
            f"tau={stats_row['tau']:.2f}  n_eff={stats_row['n_eff']:.0f}  "
            f"chi2_p_eff={stats_row['chi2_p_eff']:.4g}"
        )

    # Benjamini-Hochberg across the grid, on both the naive AD p and the robust
    # chi-squared p, so the multiple-comparisons correction is available for either.
    tested = [r for r in rows if r.get("status") == "ok"]
    for key, adj_key in (("ad_p_iid", "ad_p_iid_bh"), ("chi2_p_eff", "chi2_p_eff_bh")):
        pvals = np.array([r[key] for r in tested], dtype=np.float64)
        if pvals.size:
            adjusted = stats.false_discovery_control(pvals, method="bh")
            for r, q in zip(tested, adjusted):
                r[adj_key] = float(q)

    fields = [
        "tag", "status", "n", "ad_stat", "ad_p_iid", "ad_p_iid_bh",
        "chi2_stat", "chi2_p_iid", "ks_stat", "ks_p_iid",
        "tau", "n_eff", "chi2_p_eff", "chi2_p_eff_bh", "ad_p_thin", "n_thin", "block_hours",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    out_json = out_csv.with_suffix(".json")
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    # Leaderboard: most uniform first, ranked by the tail-weighted AD statistic
    # (smaller = flatter), with the robust BH-adjusted p as the significance flag.
    ranked = sorted(tested, key=lambda r: r["ad_stat"])
    print("\nMost uniform PIT (low A^2 = flat) -> least uniform:")
    print(f"  {'tag':<34} {'A^2':>8} {'ad_p_bh':>9} {'tau':>5} {'chi2_p_eff_bh':>13}")
    for r in ranked:
        flag = "" if r.get("chi2_p_eff_bh", 1.0) >= 0.05 else "  * non-uniform (q<.05)"
        print(f"  {r['tag']:<34} {r['ad_stat']:>8.2f} {r.get('ad_p_iid_bh', float('nan')):>9.3g}"
              f" {r['tau']:>5.2f} {r.get('chi2_p_eff_bh', float('nan')):>13.3g}{flag}")
    print(f"\nwrote: {out_csv}")
    print(f"wrote: {out_json}")


if __name__ == "__main__":
    main()
