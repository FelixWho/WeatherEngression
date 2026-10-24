"""Separate simulation audit: select time-spaced PIT observations, then IID AD.

No training, recalibration fitting, or changes to the sensitivity gate. Reuses
known-null and miscalibrated cases from validate_ad_bootstrap. Those simulations
place all observations of a synthetic weather block at one timestamp, 30 days
apart. This tests skipping blocks, not the actual ENA sampling calendar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from experiments.calibration import ad_statistic_uniform, _ad_statistic_rows
from experiments.validate_ad_bootstrap import simulate_pit

SCENARIOS = ("iid_uniform", "moderate_dependent_uniform", "dependent_uniform",
             "across_block_uniform", "miscalibrated")


def time_spaced_indices(time_days, gap_days):
    """Keep the earliest row, then the earliest at least gap_days after each pick.

    Selection depends only on timestamps. Equal-time ties retain original row
    order. Zero gap retains every row; positive gaps exclude duplicate times.
    """
    times = np.asarray(time_days, dtype=float)
    if times.ndim != 1 or not times.size or not np.isfinite(times).all():
        raise ValueError("require a nonempty vector of finite timestamps")
    if not np.isfinite(gap_days) or gap_days < 0:
        raise ValueError("gap_days must be finite and nonnegative")
    order = np.argsort(times, kind="stable")
    selected = [order[0]]
    for row in order[1:]:
        if times[row] - times[selected[-1]] >= gap_days:
            selected.append(row)
    return np.asarray(selected, dtype=int)


def iid_ad_reference(n, n_sim, seed):
    """Sorted AD scores from IID uniforms; batch generation bounds memory."""
    if n < 2 or n_sim < 1:
        raise ValueError("require at least two observations and one null simulation")
    rng = np.random.default_rng(seed)
    scores = np.empty(n_sim)
    for start in range(0, n_sim, 128):
        stop = min(start + 128, n_sim)
        draws = np.sort(rng.random((stop - start, n)), axis=1)
        scores[start:stop] = _ad_statistic_rows(draws)
    return np.sort(scores)


def iid_ad_pvalue(pit, sorted_reference):
    """Add-one Monte Carlo p-value, using a reference for the same sample size."""
    score = ad_statistic_uniform(pit)
    exceed = len(sorted_reference) - np.searchsorted(sorted_reference, score, side="left")
    return float((1 + exceed) / (len(sorted_reference) + 1))


def run_audit(block_sizes, gaps, repetitions, n_sim, seed):
    sizes = np.asarray(block_sizes)
    if sizes.ndim != 1 or len(sizes) < 2 or sizes.dtype.kind not in "iu" or np.any(sizes < 1):
        raise ValueError("require at least two positive integer block sizes")
    if repetitions < 1 or n_sim < 1 or seed < 0:
        raise ValueError("require positive simulation counts and a nonnegative seed")
    times = np.repeat(np.arange(len(sizes)) * 30. + .5, sizes)
    selections = {gap: time_spaced_indices(times, gap) for gap in gaps}
    if not selections or any(len(rows) < 2 for rows in selections.values()):
        raise ValueError("each gap must retain at least two observations")
    # Reference distribution depends only on retained sample size. Reuse it
    # across trials/scenarios to avoid millions of redundant null draws.
    references = {n: iid_ad_reference(n, n_sim, np.random.SeedSequence([seed, 1, n]))
                  for n in sorted({len(rows) for rows in selections.values()})}
    results = []
    for scenario_id, scenario in enumerate(SCENARIOS):
        p_values = {gap: [] for gap in selections}
        for repetition in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, 0, scenario_id, repetition]))
            pit, generated_times = simulate_pit(scenario, len(sizes), rng, block_sizes=sizes)
            if not np.array_equal(times, generated_times):
                raise ValueError("simulation timestamps changed; recompute time-based selections")
            for gap, rows in selections.items():
                p_values[gap].append(iid_ad_pvalue(pit[rows], references[len(rows)]))
        for gap, rows in selections.items():
            checks = {}
            for alpha in (.01, .05, .10):
                rejected = sum(p < alpha for p in p_values[gap])
                ci = binomtest(rejected, repetitions).proportion_ci()
                checks[str(alpha)] = dict(rejected=rejected, rate=rejected / repetitions,
                                         ci95=[ci.low, ci.high])
            result = dict(scenario=scenario, known_uniform_null=scenario != "miscalibrated",
                          gap_days=gap, n_selected=len(rows), selected_indices=rows.tolist(),
                          threshold_checks=checks, p_values=p_values[gap])
            results.append(result)
            print(json.dumps({k: v for k, v in result.items()
                              if k not in ("p_values", "selected_indices")}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-sizes-from", type=Path,
                        help="Read only block sizes from an existing candidate calibration JSON.")
    parser.add_argument("--gap-days", type=float, nargs="+", default=[0., 30., 60., 90.])
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--null-simulations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2431)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = ([20] * 33 if args.block_sizes_from is None else
             json.loads(args.block_sizes_from.read_text())["metrics"]["ad"]["block_sizes"])
    results = run_audit(sizes, args.gap_days, args.repetitions, args.null_simulations, args.seed)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(block_sizes=sizes, n_values=sum(sizes), n_blocks=len(sizes),
                    selection="earliest timestamp, then greedy minimum gap; stable ties",
                    data_seed_rule="SeedSequence([seed, 0, scenario_index, repetition])",
                    reference_seed_rule="SeedSequence([seed, 1, retained_sample_size])")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(settings=settings, results=results,
        interpretation="Known-uniform scenarios assess false rejections; miscalibrated assesses power. "
        "Spacing does not guarantee independence. Each row reports one sample and one p-value, "
        "never a median or minimum across subsets. Confidence intervals describe trial variation "
        "conditional on the reused Monte Carlo reference. Synthetic blocks are 30 days apart; "
        "these are not real ENA timestamps. No spacing is selected for production by this audit."),
        indent=2) + "\n")


if __name__ == "__main__":
    main()
