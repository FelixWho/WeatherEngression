"""Tiny saved-PIT comparisons; no real dataset loads or model evaluation."""
import contextlib
import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.wildfire_sensitivity import compare_pit_dependence as comparison
from experiments.wildfire_sensitivity.pit_dependence import lag_pairs, pair_metrics


class ComparisonTests(unittest.TestCase):
    def test_rank_measure_matches_existing_diagnostic_including_ties_and_empty_pairs(self):
        times = np.arange(12) * 15.
        pairs = [lag_pairs(times, gap) for gap in (15., 30., 300.)]
        pit = np.array([.1, .1, .2, .4, .3, .8, .6, .9, 1., 1., .7, .3])
        actual = comparison.rank_correlations(pit, pairs)
        expected = np.asarray([pair_metrics(pit, a, b)['pit_spearman'] for a, b in pairs], dtype=float)
        np.testing.assert_allclose(actual, expected, equal_nan=True)
        self.assertTrue(np.isnan(actual[-1]))
        with self.assertRaises(ValueError):
            comparison.rank_correlations([np.nan] * 12, pairs)

    def test_scenarios_reproduce_without_changing_global_rng(self):
        times = np.arange(20) * 15.
        pairs = [lag_pairs(times, gap) for gap in (15., 30., 90.)]
        before = np.random.get_state()
        first = comparison.synthetic_correlations(times, pairs, 3, 2435)
        second = comparison.synthetic_correlations(times, pairs, 3, 2435)
        self.assertEqual(set(first), {'iid_uniform', 'moderate_3day', 'strong_3day',
                                     'strong_10day', 'strong_30day'})
        for scenario in first:
            self.assertEqual(first[scenario].shape, (3, 3))
            np.testing.assert_array_equal(first[scenario], second[scenario])
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        self.assertEqual(before[2:], np.random.get_state()[2:])

    def test_unavailable_correlations_stay_unavailable(self):
        pairs = [(np.array([0, 1]), np.array([1, 2]))]
        rows = comparison.comparison_rows(np.array([.4, .4, .4]), [15.], pairs,
                                         {'iid_uniform': np.full((3, 1), np.nan)})
        self.assertEqual(rows[0]['n_pairs'], 2)
        self.assertEqual(rows[0]['n_unique_rows'], 3)
        self.assertIsNone(rows[0]['observed_spearman'])
        self.assertIsNone(rows[0]['simulated_lower95'])
        self.assertEqual(rows[0]['simulated_valid'], 0)

    def test_cli_compares_models_separately_and_rejects_calendar_mismatch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'comparison'
            models = [dict(candidate=name, order_seed=2027, screening_round=2,
                           run_dir='/saved/100/arrival') for name in ['A', 'B']]
            (root / 'manifest.json').write_text(json.dumps(dict(
                role='control', gaps_days=[15., 30.], half_width_days=.5, models=models)))
            destination = root / 'seed_2027_100'
            destination.mkdir()
            times = np.arange(20) * 15.
            rows = np.arange(20)
            for name in ['a', 'b']:
                np.savez(destination / f'{name}_pit.npz', time_days=times, row_index=rows,
                         source_index=rows + 100, evaluation_role='control',
                         corrected_pit=np.linspace(.01, .99, 20))
            args = ['compare_pit_dependence', '--pit-dir', str(root), '--output-dir', str(output),
                    '--repetitions', '3']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            settings = json.loads((output / 'settings.json').read_text())
            self.assertEqual(settings['n_models'], 2)
            self.assertEqual(settings['n_control'], 20)
            with (output / 'comparison.csv').open() as handle:
                results = list(csv.DictReader(handle))
            self.assertEqual(len(results), 2 * 2 * 5)
            self.assertEqual({row['candidate'] for row in results}, {'A', 'B'})
            self.assertTrue(all(float(row['observed_spearman']) > .99 for row in results))
            np.savez(destination / 'b_pit.npz', time_days=times + 1, row_index=rows,
                     source_index=rows + 100, evaluation_role='control', corrected_pit=np.ones(20))
            with patch('sys.argv', args), self.assertRaisesRegex(ValueError, 'identical control rows and dates'):
                comparison.main()


if __name__ == '__main__':
    unittest.main()
