"""Compare saved control PIT rank correlations with the AD audit's scenarios.

Simulate on the same dates and measure the same lag pairs. Reference ranges
describe simulation variation at each lag, not confidence intervals for real
dependence or an automatic choice of a true scenario. No models are fitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from data_generation.ena_weather import slugify
from experiments.validate_ad_control_spacing import SCENARIOS, simulate_pit
from experiments.wildfire_sensitivity.pit_dependence import correlation, lag_pairs, write_table


def rank_correlations(pit, pairs):
    """Spearman correlation for each fixed earlier/later observation pair set."""
    pit = np.asarray(pit, dtype=float)
    if pit.ndim != 1 or not np.isfinite(pit).all() or np.any((pit < 0) | (pit > 1)):
        raise ValueError('require finite scalar PIT values in [0, 1]')
    return np.asarray([correlation(rankdata(pit[left]), rankdata(pit[right]))
                       for left, right in pairs], dtype=float)


def synthetic_correlations(times, pairs, repetitions, seed):
    """One reference per scenario, reused across models sharing the calendar."""
    if repetitions < 1 or seed < 0:
        raise ValueError('require positive repetitions and nonnegative seed')
    references = {}
    for scenario_id, (name, parameters) in enumerate(SCENARIOS.items()):
        if parameters['miscalibrated']:
            continue
        draws = np.empty((repetitions, len(pairs)))
        for trial in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id, trial]))
            draws[trial] = rank_correlations(simulate_pit(times, rng, **parameters), pairs)
        references[name] = draws
    return references


def comparison_rows(pit, gaps, pairs, references):
    observed = rank_correlations(pit, pairs)
    result = []
    for column, (gap, (left, right)) in enumerate(zip(gaps, pairs)):
        for scenario, samples in references.items():
            values = samples[:, column]
            values = values[np.isfinite(values)]
            limits = np.quantile(values, [.025, .5, .975]).tolist() if len(values) else [None] * 3
            result.append(dict(
                gap_days=gap, n_pairs=len(left), n_unique_rows=len(np.unique(np.r_[left, right])),
                observed_spearman=float(observed[column]) if np.isfinite(observed[column]) else None,
                scenario=scenario, simulated_valid=len(values),
                simulated_lower95=limits[0], simulated_median=limits[1], simulated_upper95=limits[2]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2435)
    args = parser.parse_args()
    if args.repetitions < 1 or args.seed < 0:
        parser.error('require positive repetitions and nonnegative seed')
    manifest = json.loads((args.pit_dir / 'manifest.json').read_text())
    if manifest['role'] != 'control' or not manifest['models']:
        raise ValueError('require a completed control PIT diagnostic')
    gaps, half_width = manifest['gaps_days'], manifest['half_width_days']
    calendar, references, rows = None, None, []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in manifest['models']:
        run_id = Path(model['run_dir']).parent.name
        path = args.pit_dir / f"seed_{model['order_seed']}_{run_id}" / f"{slugify(model['candidate'])}_pit.npz"
        with np.load(path) as saved:
            if saved['evaluation_role'].item() != 'control':
                raise ValueError(f'expected control PIT: {path}')
            current = tuple(saved[key] for key in ('time_days', 'row_index', 'source_index'))
            pit = saved['corrected_pit']
        if calendar is None:
            calendar = current
            pairs = [lag_pairs(calendar[0], gap, half_width) for gap in gaps]
            references = synthetic_correlations(calendar[0], pairs, args.repetitions, args.seed)
            np.savez_compressed(args.output_dir / 'synthetic_correlations.npz',
                                time_days=calendar[0], row_index=calendar[1], source_index=calendar[2],
                                gap_days=gaps, **references)
        elif any(not np.array_equal(a, b) for a, b in zip(calendar, current)):
            raise ValueError('all models must use identical control rows and dates')
        if pit.shape != calendar[0].shape:
            raise ValueError('require one corrected PIT per control date')
        rows.extend(dict(candidate=model['candidate'], order_seed=model['order_seed'],
                         screening_round=model['screening_round'],
                         n_control=len(pit), n_distinct_pit=len(np.unique(pit)), **row)
                    for row in comparison_rows(pit, gaps, pairs, references))
    write_table(args.output_dir / 'comparison.csv', rows)
    settings = dict(pit_dir=str(args.pit_dir), repetitions=args.repetitions, seed=args.seed,
                    seed_rule='SeedSequence([seed, scenario_index, trial])',
                    gaps_days=gaps, half_width_days=half_width, n_control=len(calendar[0]),
                    n_models=len(manifest['models']), pit_kind='corrected',
                    scenarios={name: SCENARIOS[name] for name in references},
                    interpretation=[
                        'Pointwise simulated ranges; overlapping pairs are reproduced in every simulation.',
                        'Models share the same controls and are not independent data replicates.',
                        'Scenarios are assumptions, not fitted decay times or verified AD p-value models.',
                        'Real PIT uses fresh predictive draws and saved mappings; synthetic PIT is exact.',
                        'Rank correlation describes one aspect of dependence; ties and sparse pairs limit comparisons.'])
    (args.output_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print(f"Compared {settings['n_models']} models on {settings['n_control']} controls; saved {args.output_dir}", flush=True)


if __name__ == '__main__':
    main()
