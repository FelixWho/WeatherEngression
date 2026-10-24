"""Descriptive PIT dependence on saved clean recalibration rows, via SLURM.

Reloads the latest saved arrival model for each candidate in an until-stable run.
The mapping was fitted on these same recalibration rows: this is development
analysis of dependence, NOT independent validation of calibration. No control
outcomes are evaluated and no acceptance decisions or spacings are selected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch

from data_generation.ena_weather import DEFAULT_MAT_PATH, load_ena_supervised_dataset, slugify
from engression_modifications.lstm import load_lstm_engressor_checkpoint
from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.calibration import validate_evaluation_targets, validate_evaluation_outputs
from experiments.pipeline import set_reproducible_seeds
from experiments.validate_ad_thinning import time_spaced_indices

GAPS = (1., 3., 7., 10., 14., 21., 30., 60., 90.)


def latest_candidates(run_dir):
    """Recover each candidate's last checkpoint AND its original feature order."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'screening_order.json').read_text())
    if metadata['screening_mode'] != 'until_stable' or metadata['evaluation_scope'] != 'arrival':
        raise ValueError('requires an arrival-only until-stable run')
    latest = {}
    for path in run_dir.glob('round_*/*_calibration.json'):
        record = json.loads(path.read_text())
        if record['evaluation_scope'] != 'arrival':
            raise ValueError(f'non-arrival candidate: {path}')
        name = record['candidate']
        if name not in latest or record['screening_round'] > latest[name][1]['screening_round']:
            latest[name] = (path, record)
    if not latest:
        raise ValueError(f'no candidate records under {run_dir}')
    result = []
    for name, (path, record) in sorted(latest.items()):
        if name in record['conditioning_covariates']:
            raise ValueError('candidate must not be a conditioning feature')
        checkpoint = path.parent / f'{slugify(name)}_best.pt'
        mapping = path.parent / f'{slugify(name)}_quantile_levels.json'
        if not checkpoint.is_file() or not mapping.is_file():
            raise FileNotFoundError(f'missing model or mapping beside {path}')
        result.append((record, checkpoint, mapping))
    return metadata, result


def recalibration_rows(run_dir, dataset):
    with np.load(Path(run_dir) / 'screening_split.npz') as split:
        if split['evaluation_scope'].item() != 'arrival' or split['training_scope'].item() != 'arrival':
            raise ValueError('saved split must be arrival-only')
        rows = split['recalibration'].copy()
        if (rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows)
                or rows.min() < 0 or rows.max() >= len(dataset.x) or len(np.unique(rows)) != len(rows)):
            raise ValueError('invalid saved recalibration indices')
        for role in ('train', 'validation', 'control', 'wildfire'):
            if np.intersect1d(rows, split[role]).size:
                raise ValueError(f'recalibration overlaps {role}')
    if np.any(dataset.flags['BB_criterion1'][rows] | dataset.flags['BB_criterion2'][rows]):
        raise ValueError('recalibration rows must be clean under both flags')
    return rows[np.argsort(dataset.time[rows], kind='stable')]


def lag_pairs(time_days, lag_days, half_width_days=.5):
    """All earlier/later pairs in an actual elapsed-time bin, inclusive edges."""
    times = np.asarray(time_days, dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) < 0)):
        raise ValueError('require finite timestamps sorted in time')
    if not 0 < half_width_days < lag_days:
        raise ValueError('lag must exceed positive bin half-width')
    left, right = [], []
    for i, time in enumerate(times):
        start = np.searchsorted(times, time + lag_days - half_width_days, side='left')
        stop = np.searchsorted(times, time + lag_days + half_width_days, side='right')
        if stop > start:
            left.extend([i] * (stop - start))
            right.extend(range(start, stop))
    return np.asarray(left, dtype=int), np.asarray(right, dtype=int)


def correlation(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def pair_metrics(pit, left, right):
    a, b = pit[left], pit[right]
    low_a, low_b = a <= .05, b <= .05
    high_a, high_b = a >= .95, b >= .95
    return dict(n_pairs=len(left), n_unique_rows=len(np.unique(np.r_[left, right])),
                pit_pearson=correlation(a, b),
                pit_spearman=correlation(rankdata(a), rankdata(b)),
                lower_tail_correlation=correlation(low_a, low_b),
                upper_tail_correlation=correlation(high_a, high_b),
                lower_tail_left=int(low_a.sum()), lower_tail_right=int(low_b.sum()),
                upper_tail_left=int(high_a.sum()), upper_tail_right=int(high_b.sum()))


def dependence_rows(pit, times, gaps=GAPS, half_width_days=.5):
    """Lag-bin correlations plus adjacent correlations after greedy thinning.

    Overlapping pairs are not independent. Counts are descriptive; no IID
    confidence intervals, independence claims, or automatic cutoffs are added.
    """
    pit, times = np.asarray(pit, dtype=float), np.asarray(times, dtype=float)
    if (pit.ndim != 1 or pit.shape != times.shape or not np.isfinite(pit).all()
            or np.any((pit < 0) | (pit > 1))):
        raise ValueError('require one finite PIT in [0,1] per timestamp')
    result = []
    for gap in gaps:
        left, right = lag_pairs(times, gap, half_width_days)
        selected = time_spaced_indices(times, gap)
        deltas = np.diff(times[selected])
        result.append(dict(view='lag_bin', gap_days=gap, half_width_days=half_width_days,
                           n_selected=None, actual_gap_min=None, actual_gap_median=None,
                           actual_gap_max=None, **pair_metrics(pit, left, right)))
        result.append(dict(view='thinned_adjacent', gap_days=gap, half_width_days=None,
                           n_selected=len(selected),
                           actual_gap_min=float(deltas.min()) if len(deltas) else None,
                           actual_gap_median=float(np.median(deltas)) if len(deltas) else None,
                           actual_gap_max=float(deltas.max()) if len(deltas) else None,
                           **pair_metrics(pit, selected[:-1], selected[1:])))
    return result


def sample_pit(model, x, y, batch_size, draws, pit_seed):
    validate_evaluation_targets(x, y, 'arrival')
    raw = RecalibratedEngressor(model.engressor)
    raw_batches, corrected_batches = [], []
    for start in range(0, len(x), batch_size):
        observed = y[start:start + batch_size]
        samples = model.engressor.sample(torch.from_numpy(x[start:start + batch_size]),
                                         sample_size=draws).detach().cpu()
        validate_evaluation_outputs(samples, observed)
        raw_batches.append(raw.pit_from_samples(samples, observed, seed=pit_seed + start).ravel())
        corrected_batches.append(model.pit_from_samples(samples, observed, seed=pit_seed + start).ravel())
    return np.concatenate(raw_batches), np.concatenate(corrected_batches)


def write_table(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [r for r in rows if r['pit_kind'] == 'corrected' and r['view'] == 'lag_bin']
    labels = list(dict.fromkeys(f"s{r['order_seed']} {r['candidate']}" for r in selected))
    gaps = sorted({r['gap_days'] for r in selected})
    fields = ['pit_spearman', 'lower_tail_correlation', 'upper_tail_correlation']
    titles = ['PIT rank correlation', 'Lower 5% events', 'Upper 5% events']
    fig, axes = plt.subplots(1, 3, figsize=(12, max(6, len(labels) * .22)), sharey=True)
    for ax, field, title in zip(axes, fields, titles):
        values = np.full((len(labels), len(gaps)), np.nan)
        for r in selected:
            value = r[field]
            if value is not None:
                values[labels.index(f"s{r['order_seed']} {r['candidate']}"), gaps.index(r['gap_days'])] = value
        colors = plt.get_cmap('coolwarm').copy()
        colors.set_bad('lightgray')
        im = ax.imshow(np.ma.masked_invalid(values), vmin=-1, vmax=1, cmap=colors, aspect='auto')
        ax.set_xticks(range(len(gaps)), [f'{g:g}' for g in gaps])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(title)
        ax.set_xlabel('Elapsed days (±0.5-day bins)')
    fig.suptitle('Recalibration-set dependence: descriptive, overlapping pairs\nGray = unavailable; pair/event counts in dependence.csv', fontsize=11)
    fig.colorbar(im, ax=axes, shrink=.55, label='Correlation')
    fig.subplots_adjust(left=.19, right=.86, top=.9, bottom=.08, wspace=.08)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mat-path', default=DEFAULT_MAT_PATH)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prediction-samples', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2433)
    args = parser.parse_args()
    if args.batch_size < 1 or args.prediction_samples < 2 or args.seed < 0:
        parser.error('require positive batch size, >=2 samples, and nonnegative seed')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    runs = [(p, *latest_candidates(p)) for p in args.run_dirs]
    total_models = sum(len(candidates) for _, _, candidates in runs)
    dataset_seeds = {m['seeds']['dataset'] for _, m, _ in runs}
    if len(dataset_seeds) != 1:
        raise ValueError('runs must share the same dataset seed/order')
    dataset_seed = dataset_seeds.pop()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(settings={k: str(v) if isinstance(v, Path) else
                             [str(p) for p in v] if k == 'run_dirs' else v for k, v in vars(args).items()},
                    dataset_seed=dataset_seed, gaps_days=GAPS, half_width_days=.5,
                    role='recalibration', model_selection='latest saved round per candidate',
                    interpretation='Development diagnostic on the same rows used to fit the mapping. '
                    'No control outcomes evaluated. Pairs overlap; no independence significance test. '
                    'Thinning changes time/season weighting. No production spacing is selected.', models=[])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Loading data once with original dataset seed {dataset_seed}', flush=True)
    dataset = load_ena_supervised_dataset(mat_path=args.mat_path, target='ccn', log_ccn=True,
                                         seq_stride=1, max_samples=None, seed=dataset_seed)
    all_rows = []
    features = {name: index for index, name in enumerate(dataset.feature_names)}
    for run_dir, metadata, candidates in runs:
        rows = recalibration_rows(run_dir, dataset)
        times = dataset.time[rows]
        # Copy only the small recalibration subset, preserving original row IDs.
        data = dataset.x[rows]
        run_id = run_dir.parent.name
        order_seed = metadata['candidate_order_seed']
        for record, checkpoint, mapping in candidates:
            model_index = len(manifest['models'])
            sampling_seed, pit_seed = args.seed + model_index, args.seed + 1000 + model_index
            candidate = record['candidate']
            conditioning = record['conditioning_covariates']
            x = data[:, :, [features[name] for name in conditioning]]
            y = data[:, -1:, features[candidate]]
            set_reproducible_seeds(sampling_seed)
            base = load_lstm_engressor_checkpoint(str(checkpoint), device=args.device)
            wrapped = RecalibratedEngressor.load(base, mapping)
            wrapped.require_full_mapping()
            if wrapped.calibration_pit is None or len(wrapped.calibration_pit) != len(rows):
                raise ValueError('require an arrival recalibration map fitted on the saved recalibration rows')
            print(f'[{model_index + 1}/{total_models}] seed {order_seed}: {candidate}, round '
                  f'{record["screening_round"]}, {len(rows)} recalibration rows', flush=True)
            raw, corrected = sample_pit(wrapped, x, y, args.batch_size, args.prediction_samples, pit_seed)
            destination = output / f'seed_{order_seed}_{run_id}'
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / f'{slugify(candidate)}_pit.npz',
                                row_index=rows, source_index=dataset.source_index[rows],
                                time_days=times, raw_pit=raw, corrected_pit=corrected)
            model_info = dict(candidate=candidate, order_seed=order_seed, run_dir=str(run_dir),
                              screening_round=record['screening_round'], checkpoint=str(checkpoint),
                              mapping=str(mapping), conditioning_covariates=conditioning,
                              n_recalibration=len(rows), sampling_seed=sampling_seed, pit_batch_base=pit_seed,
                              pit_seed_rule='pit_batch_base + sorted batch start row')
            model_rows = []
            for kind, pit in [('raw', raw), ('corrected', corrected)]:
                for diagnostic in dependence_rows(pit, times):
                    model_rows.append(dict(candidate=candidate, order_seed=order_seed,
                                           screening_round=record['screening_round'], pit_kind=kind, **diagnostic))
            write_table(destination / f'{slugify(candidate)}_dependence.csv', model_rows)
            all_rows.extend(model_rows)
            manifest['models'].append(model_info)
            write_table(output / 'dependence.csv', all_rows)
            (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            del wrapped, base
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            print('  saved PIT values and dependence diagnostics', flush=True)
    plot_summary(output / 'dependence.png', all_rows)
    print(f'Complete: {len(manifest["models"])} models; outputs in {output}', flush=True)


if __name__ == '__main__':
    main()
