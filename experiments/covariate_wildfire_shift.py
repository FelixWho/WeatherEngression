"""Per-covariate comparison of wildfire vs non-wildfire trajectory weather.

For each of the 22 input covariates, this script compares its distribution inside
wildfire periods (``BB_criterion1``, the T-learner treatment definition) against
outside them, using the per-trajectory mean of the covariate as the sample unit
(the same convention as the correlation heatmap in the cumulative summary).

Season is a confounder here, and a severe one: about 84% of wildfire trajectories
arrive in June-September, while January and December contain none at all. Any
covariate with a seasonal cycle (surface temperature, solar radiation, wind speed)
therefore differs between the two groups whether or not smoke affects it. To
separate the two, every statistic is computed twice:

- ``raw``: wildfire vs all non-wildfire trajectories;
- ``matched``: wildfire vs a non-wildfire subsample drawn to have the same
  calendar-month distribution as the wildfire group, which removes the seasonal
  cycle as an explanation.

Reported per covariate, for both comparisons:

- quantiles (5%, 25%, 50%, 75%, 95%) inside and outside wildfire periods;
- the two-sample Anderson-Darling statistic (scale-free, tail-sensitive; its
  p-value is floored/capped by scipy at [0.001, 0.25]);
- the Wasserstein-1 distance and the energy distance, both unit-bearing, so the
  CSV also carries versions standardized by the pooled standard deviation
  (``W1/sd`` and ``energy^2/sd``) that are comparable across covariates.

Note what this can and cannot establish. Even month-matched, a distributional
difference does not prove a covariate responds to smoke: smoke arrives in air
masses that differ for reasons of transport as well (continental origin, for
instance). Identifying genuinely wildfire-sensitive variables requires the
counterfactual model, not a two-sample comparison.

Pure CPU analysis over the loaded dataset; no model or GPU involved.

Example
-------
```bash
python experiments/covariate_wildfire_shift.py
```
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import warnings

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_generation.ena_weather import (
    DEFAULT_MAT_PATH,
    _matlab_datenum_to_datetime,
    load_ena_supervised_dataset,
    parse_count_or_all,
)

QUANTS = (0.05, 0.25, 0.50, 0.75, 0.95)


def two_sample_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """AD statistic, Wasserstein-1, and energy distance for one covariate.

    Both distances are also returned standardized by the pooled standard deviation.
    scipy's ``energy_distance`` returns the square root of the energy statistic, so
    it scales as sqrt(units); squaring first makes ``energy^2/sd`` dimensionless.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy warns when the AD p hits its floor/cap
        ad = stats.anderson_ksamp([a, b])
    w1 = float(stats.wasserstein_distance(a, b))
    en = float(stats.energy_distance(a, b))
    pooled_sd = float(np.std(np.concatenate([a, b])))
    return {
        "ad_stat": float(ad.statistic),
        "ad_p": float(ad.significance_level),  # capped to [0.001, 0.25] by scipy
        "wasserstein": w1,
        "energy_distance": en,
        "pooled_sd": pooled_sd,
        "wasserstein_std": w1 / pooled_sd if pooled_sd else float("nan"),
        "energy_distance_std": en ** 2 / pooled_sd if pooled_sd else float("nan"),
    }


def month_matched_clean(
    fire: np.ndarray, months: np.ndarray, seed: int
) -> np.ndarray:
    """Indices of non-wildfire trajectories matching the wildfire month distribution.

    Takes ``k * n_fire[m]`` clean trajectories from each month ``m``, where ``k`` is
    the largest factor every month can supply. This preserves the wildfire group's
    monthly proportions exactly while retaining as much clean data as possible.
    """

    ratios = []
    for m in range(1, 13):
        nf = int((fire & (months == m)).sum())
        if nf:
            ratios.append(int((~fire & (months == m)).sum()) / nf)
    k = min(ratios)
    rng = np.random.default_rng(seed)
    picks = []
    for m in range(1, 13):
        nf = int((fire & (months == m)).sum())
        if not nf:
            continue
        pool = np.flatnonzero((~fire) & (months == m))
        picks.append(rng.choice(pool, size=int(round(k * nf)), replace=False))
    return np.concatenate(picks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mat-path", type=str, default=DEFAULT_MAT_PATH)
    parser.add_argument("--wildfire-flag", type=str, default="BB_criterion1",
                        help="Event mask defining wildfire periods (T-learner treatment).")
    parser.add_argument("--max-samples", type=parse_count_or_all, default=None)
    parser.add_argument("--seq-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "runs" / "covariate_wildfire_shift" / "covariate_wildfire_shift.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset (target ccn keeps all 22 covariates in X)...")
    dataset = load_ena_supervised_dataset(
        mat_path=args.mat_path,
        target="ccn",
        max_samples=args.max_samples,
        seq_stride=args.seq_stride,
        seed=args.seed,
        log_ccn=True,
    )
    if args.wildfire_flag not in dataset.flags:
        raise SystemExit(f"unknown flag {args.wildfire_flag!r}; available: {sorted(dataset.flags)}")
    fire = dataset.flags[args.wildfire_flag]
    # One value per trajectory per covariate: the mean over the 241 timesteps.
    means = dataset.x.mean(axis=1).astype(np.float64)  # (n, d)
    names = dataset.feature_names
    n_fire, n_clean = int(fire.sum()), int((~fire).sum())
    months = np.array([_matlab_datenum_to_datetime(t).month for t in dataset.time])
    matched = month_matched_clean(fire, months, seed=args.seed)
    print(f"trajectories: {means.shape[0]}  wildfire: {n_fire}  clean: {n_clean}  covariates: {len(names)}")
    print(f"month-matched clean subsample: {matched.size}")
    fire_share_jjas = float((fire & np.isin(months, (6, 7, 8, 9))).sum()) / max(n_fire, 1)
    print(f"share of wildfire trajectories in Jun-Sep: {100 * fire_share_jjas:.1f}%")

    rows: list[dict] = []
    for j, name in enumerate(names):
        a = means[fire, j]           # inside wildfire periods
        b = means[~fire, j]          # outside, all of it
        b_matched = means[matched, j]  # outside, month-matched to the wildfire group

        row: dict = {"covariate": name, "n_wildfire": n_fire, "n_clean": n_clean,
                     "n_clean_matched": int(matched.size)}
        row.update(two_sample_stats(a, b))
        row.update({f"{k}_matched": v for k, v in two_sample_stats(a, b_matched).items()})

        for q, va, vb, vm in zip(QUANTS, np.quantile(a, QUANTS), np.quantile(b, QUANTS),
                                 np.quantile(b_matched, QUANTS)):
            pct = int(q * 100)
            row[f"q{pct:02d}_wildfire"] = float(va)
            row[f"q{pct:02d}_clean"] = float(vb)
            row[f"q{pct:02d}_clean_matched"] = float(vm)
        rows.append(row)

    # Rank by the month-matched statistic: the raw one is dominated by seasonality.
    rows.sort(key=lambda r: r["ad_stat_matched"], reverse=True)
    fields = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    raw_rank = {r["covariate"]: i + 1 for i, r in
                enumerate(sorted(rows, key=lambda r: -r["ad_stat"]))}
    print(f"\n{'covariate':<12} {'AD raw':>9} {'AD matched':>11} {'W1/sd m':>8} {'E2/sd m':>8} "
          f"{'rank raw':>8} {'rank m':>6}")
    for i, r in enumerate(rows):
        print(f"{r['covariate']:<12} {r['ad_stat']:>9.1f} {r['ad_stat_matched']:>11.1f} "
              f"{r['wasserstein_std_matched']:>8.3f} {r['energy_distance_std_matched']:>8.3f} "
              f"{raw_rank[r['covariate']]:>8} {i + 1:>6}")
    print(f"\nwrote: {args.out}")


if __name__ == "__main__":
    main()
