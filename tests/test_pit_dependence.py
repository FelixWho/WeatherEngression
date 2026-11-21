"""Small synthetic checks; no model fitting or real dataset loads."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.wildfire_sensitivity import pit_dependence as diagnostic
from experiments.wildfire_sensitivity.pit_dependence import (
    clean_evaluation_rows, dependence_rows, lag_pairs, latest_candidates, pair_metrics,
    recalibration_rows, sample_pit,
)


class DependenceTests(unittest.TestCase):
    def test_actual_elapsed_time_pairs_match_brute_force(self):
        times = np.array([0., .5, 1., 2., 7., 10., 30., 60., 90.])
        for lag in [1., 3., 7., 10., 14., 15., 20., 21., 30., 60., 90.]:
            left, right = lag_pairs(times, lag)
            expected = [(i, j) for i in range(len(times)) for j in range(i + 1, len(times))
                        if lag - .5 <= times[j] - times[i] <= lag + .5]
            self.assertEqual(list(zip(left, right)), expected)
        with self.assertRaises(ValueError):
            lag_pairs(times[::-1], 1.)

    def test_counts_constant_tails_and_missing_lags(self):
        times = np.array([0., 1., 2., 3., 4.])
        pit = np.array([.2, .3, .4, .5, .6])
        rows = dependence_rows(pit, times, gaps=[1., 30.])
        near = rows[0]
        self.assertEqual(near['n_pairs'], 4)
        self.assertEqual(near['n_unique_rows'], 5)
        self.assertAlmostEqual(near['pit_pearson'], 1.)
        self.assertAlmostEqual(near['pit_spearman'], 1.)
        self.assertIsNone(near['lower_tail_correlation'])
        self.assertEqual(rows[2]['n_pairs'], 0)
        self.assertIsNone(rows[2]['pit_pearson'])
        self.assertEqual(rows[3]['n_selected'], 1)
        self.assertIsNone(rows[3]['actual_gap_median'])

    def test_thinning_reports_actual_gaps_and_tail_events(self):
        times = np.array([0., 1., 3., 7., 20., 30.])
        pit = np.array([.01, .03, .2, .98, .5, .99])
        selected = dependence_rows(pit, times, gaps=[3.])[1]
        self.assertEqual(selected['n_selected'], 5)
        self.assertEqual(selected['actual_gap_min'], 3.)
        self.assertEqual(selected['actual_gap_max'], 13.)
        result = pair_metrics(pit, np.array([0, 1, 2]), np.array([3, 4, 5]))
        self.assertEqual(result['lower_tail_left'], 2)
        self.assertEqual(result['upper_tail_right'], 2)
        with self.assertRaises(ValueError):
            dependence_rows(np.array([np.nan] * 6), times)

    def test_only_recalibration_rows_returned_in_time_order(self):
        dataset = SimpleNamespace(x=np.zeros((8, 2, 1)), time=np.array([5, 4, 3, 2, 1, 0, 8, 9.]),
                                  flags={name: np.zeros(8, dtype=bool) for name in ['BB_criterion1', 'BB_criterion2']})
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'screening_split.npz'
            np.savez(path, recalibration=[2, 3], train=[0], validation=[1], control=[4, 5],
                     wildfire=[6, 7], evaluation_scope='arrival', training_scope='arrival')
            np.testing.assert_array_equal(recalibration_rows(directory, dataset), [3, 2])
            dataset.flags['BB_criterion2'][2] = True
            with self.assertRaisesRegex(ValueError, 'clean'):
                recalibration_rows(directory, dataset)
            dataset.flags['BB_criterion2'][2] = False
            np.savez(path, recalibration=[2, 3], train=[0], validation=[1], control=[3, 4, 5],
                     wildfire=[6, 7], evaluation_scope='arrival', training_scope='arrival')
            with self.assertRaisesRegex(ValueError, 'overlaps control'):
                recalibration_rows(directory, dataset)

    def test_latest_candidate_preserves_saved_feature_order(self):
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'screening_order.json').write_text(json.dumps(dict(
                screening_mode='until_stable', evaluation_scope='arrival')))
            for round_number, names in [(1, ['A', 'B']), (2, ['B', 'A'])]:
                folder = run / f'round_{round_number}'
                folder.mkdir()
                (folder / 't_calibration.json').write_text(json.dumps(dict(candidate='T',
                    screening_round=round_number, conditioning_covariates=names, evaluation_scope='arrival')))
                (folder / 't_best.pt').touch()
                (folder / 't_quantile_levels.json').touch()
            _, selected = latest_candidates(run)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0][0]['conditioning_covariates'], ['B', 'A'])
            self.assertEqual(selected[0][1].parent.name, 'round_2')

    def test_control_selection_rejects_overlap_with_every_other_role(self):
        dataset = SimpleNamespace(x=np.zeros((10, 3, 2)), time=np.arange(10)[::-1],
                                  flags={name: np.zeros(10, dtype=bool)
                                         for name in ['BB_criterion1', 'BB_criterion2']})
        roles = dict(train=[0], validation=[1], recalibration=[2, 3], control=[4, 5, 6], wildfire=[7, 8, 9])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'screening_split.npz'
            np.savez(path, **roles, evaluation_scope='arrival', training_scope='arrival')
            np.testing.assert_array_equal(clean_evaluation_rows(directory, dataset, 'control'), [6, 5, 4])
            for other in ['train', 'validation', 'recalibration', 'wildfire']:
                bad = dict(roles, **{other: roles[other] + [4]})
                np.savez(path, **bad, evaluation_scope='arrival', training_scope='arrival')
                with self.subTest(other=other), self.assertRaisesRegex(ValueError, f'control overlaps {other}'):
                    clean_evaluation_rows(directory, dataset, 'control')
            np.savez(path, **roles, evaluation_scope='arrival', training_scope='arrival')
            dataset.flags['BB_criterion1'][4] = True
            with self.assertRaisesRegex(ValueError, 'control rows must be clean'):
                clean_evaluation_rows(directory, dataset, 'control')
            with self.assertRaisesRegex(ValueError, 'evaluation role'):
                clean_evaluation_rows(directory, dataset, 'train')

    def test_control_cli_evaluates_control_but_checks_mapping_against_recalibration_count(self):
        # Deliberately different sizes catch accidental reuse of evaluation N
        # as the expected number of observations that fitted the mapping.
        x = np.zeros((10, 3, 2), dtype=np.float32)
        x[:, :, 1] = np.arange(10)[:, None] / 10
        dataset = SimpleNamespace(x=x, time=np.arange(10)[::-1].astype(float),
                                  source_index=np.arange(10) + 100, feature_names=['A', 'B'],
                                  flags={name: np.zeros(10, dtype=bool)
                                         for name in ['BB_criterion1', 'BB_criterion2']})
        base = SimpleNamespace(sample=lambda x, sample_size:
                               torch.linspace(0., 1., sample_size).expand(len(x), 1, sample_size))
        wrapped = RecalibratedEngressor(base).fit_from_pit([.2, .8], [.05, .95])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'job' / 'arrival'
            run.mkdir(parents=True)
            output = root / 'out'
            np.savez(run / 'screening_split.npz', train=[0], validation=[1], recalibration=[2, 3],
                     control=[4, 5, 6], wildfire=[7, 8, 9], evaluation_scope='arrival', training_scope='arrival')
            record = dict(candidate='B', conditioning_covariates=['A'], screening_round=1)
            metadata = dict(seeds={'dataset': 2026}, candidate_order_seed=2027)
            args = ['pit_dependence', '--run-dirs', str(run), '--output-dir', str(output),
                    '--role', 'control', '--gap-days', '10', '15', '20', '--device', 'cpu',
                    '--prediction-samples', '11', '--batch-size', '2']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(diagnostic, 'latest_candidates', return_value=(metadata, [(record, run/'b.pt', run/'b.json')])), \
                 patch.object(diagnostic, 'load_ena_supervised_dataset', return_value=dataset), \
                 patch.object(diagnostic, 'load_lstm_engressor_checkpoint', return_value=base), \
                 patch.object(diagnostic, 'set_reproducible_seeds'), \
                 patch.object(RecalibratedEngressor, 'load', return_value=wrapped), \
                 patch.object(diagnostic, 'plot_summary') as plot:
                diagnostic.main()
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['role'], 'control')
            self.assertEqual(manifest['gaps_days'], [10., 15., 20.])
            self.assertEqual(manifest['models'][0]['n_evaluation'], 3)
            self.assertEqual(manifest['models'][0]['n_recalibration'], 2)
            self.assertEqual(plot.call_args.kwargs['role'], 'control')
            with np.load(output / 'seed_2027_job' / 'b_pit.npz') as saved:
                np.testing.assert_array_equal(saved['row_index'], [6, 5, 4])
                np.testing.assert_array_equal(saved['source_index'], [106, 105, 104])
                self.assertEqual(saved['corrected_pit'].shape, (3,))
                self.assertEqual(saved['evaluation_role'].item(), 'control')

    def test_raw_and_corrected_pit_share_draws_and_target_shape(self):
        calls = []
        def draw(x, sample_size):
            calls.append(len(x))
            return torch.linspace(0., 1., sample_size).expand(len(x), 1, sample_size)
        wrapped = RecalibratedEngressor(SimpleNamespace(sample=draw)).fit_from_pit([.2, .8], [.05, .95])
        x = np.zeros((5, 3, 2), np.float32)
        y = np.array([.3, .4, .5, .6, .7])[:, None]
        raw, corrected = sample_pit(wrapped, x, y, 2, 100, 5)
        self.assertEqual(calls, [2, 2, 1])
        self.assertEqual(raw.shape, (5,))
        np.testing.assert_allclose(corrected, (raw - .2) / .6, atol=1e-12)
        with self.assertRaises(ValueError):
            sample_pit(wrapped, x, np.repeat(y, 3, axis=1), 2, 100, 5)


if __name__ == '__main__':
    unittest.main()
