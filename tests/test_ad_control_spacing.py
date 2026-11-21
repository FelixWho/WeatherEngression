"""Small checks for the real-calendar synthetic audit; no data/model fitting."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.stats import norm

from experiments.calibration import ad_block_bootstrap
from experiments.validate_ad_control_spacing import (
    load_comparison, load_saved_control, load_selections, matched_random_rows,
    rejection_summary, run_audit, simulate_pit,
)


class ControlSpacingTests(unittest.TestCase):
    def test_gaussian_covariance_has_unit_variance_and_elapsed_time_decay(self):
        # Recover the linear Gaussian transform using basis innovations. Its
        # covariance must equal the analytic law, including repeated dates.
        times = np.array([0., 0., 2., 15.])
        rho, tau = .8, 10.
        columns = []
        for basis in np.eye(2 * len(times)):
            rng = Mock()
            rng.normal.return_value = basis.reshape(2, -1).copy()
            columns.append(norm.ppf(simulate_pit(times, rng, rho=rho, tau_days=tau)))
        transform = np.array(columns).T
        expected = rho * np.exp(-abs(times[:, None] - times[None, :]) / tau)
        np.fill_diagonal(expected, 1.)
        np.testing.assert_allclose(transform @ transform.T, expected, atol=2e-15)
        # Each latent Z is N(0,1), so Phi(Z) has a uniform marginal.

    def test_alternative_and_invalid_simulation_inputs(self):
        times = np.array([0., 3., 9., 10.])
        uniform = simulate_pit(times, np.random.default_rng(9), rho=.3, tau_days=3.)
        wide = simulate_pit(times, np.random.default_rng(9), rho=.3, tau_days=3., miscalibrated=True)
        np.testing.assert_array_equal(wide, .1 + .8 * uniform)
        for bad_times, rho, tau in [([], .3, 3), ([np.nan], .3, 3), ([1., 0.], .3, 3),
                                    ([0.], 1.1, 3), ([0.], .3, 0)]:
            with self.subTest(times=bad_times, rho=rho, tau=tau), self.assertRaises(ValueError):
                simulate_pit(bad_times, np.random.default_rng(0), rho=rho, tau_days=tau)

    def test_random_comparator_preserves_count_months_and_rng(self):
        clean = np.arange(120)
        months = np.tile(np.arange(1, 13), 10)
        selected = np.array([0, 12, 24, 3, 7, 19])
        before = np.random.get_state()
        rows = matched_random_rows(clean, months, selected, 8)
        np.testing.assert_array_equal(rows, matched_random_rows(clean, months, selected, 8))
        self.assertEqual(len(rows), len(np.unique(rows)))
        np.testing.assert_array_equal(np.bincount(months[rows], minlength=13),
                                      np.bincount(months[selected], minlength=13))
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])

    def test_loads_control_rows_and_keeps_pair_seasons(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            times = 737791. + np.arange(1500)
            np.savez(root / "population.npz", time_days=times, clean=np.arange(1500),
                     source_index=np.arange(1500) + 100)
            metadata = {"settings": {"spaced_role": "control", "gap_days": 10}}
            (root / "results.json").write_text(json.dumps(metadata))
            for filename, rows in [("current", np.arange(200)),
                                    ("n75_seed2228", np.arange(0, 750, 10)),
                                    ("earliest_first_maximum", np.arange(0, 1500, 10))]:
                np.savez(root / f"{filename}.npz", control=rows, recalibration=np.array([1499]))
            actual_times, source, selections, _ = load_selections(root, 12)
            np.testing.assert_array_equal(actual_times, times)
            self.assertEqual(len(selections), 5)
            self.assertEqual(len(selections['spaced_75']), 75)
            self.assertEqual(len(selections['random_maximum']), 150)
            self.assertTrue(all(1499 not in selections[name] for name in ['current', 'spaced_75']))
            self.assertEqual(source[0], 100)
            metadata['settings']['spaced_role'] = 'recalibration'
            (root / "results.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'control, not recalibration'):
                load_selections(root, 12)

    def test_paired_trials_use_production_bootstrap_and_reproduce(self):
        times = np.arange(24) * 30.
        selections = dict(all=np.arange(24), subset=np.arange(0, 24, 2))
        kwargs = dict(repetitions=2, bootstrap_replicates=19, null_simulations=29, seed=5)
        before = np.random.get_state()
        with patch('experiments.validate_ad_control_spacing.ad_block_bootstrap',
                   wraps=ad_block_bootstrap) as check:
            first = run_audit(times, selections, 'strong_10day', **kwargs)
        calls = check.call_args_list
        self.assertEqual(len(calls), 4)
        for trial in range(2):
            np.testing.assert_array_equal(calls[2 * trial].args[0][::2], calls[2 * trial + 1].args[0])
            self.assertEqual(calls[2 * trial].kwargs['n_bootstrap'], 19)
        self.assertEqual(first, run_audit(times, selections, 'strong_10day', **kwargs))
        self.assertEqual(first['all']['n_blocks'], 24)
        self.assertEqual(first['subset']['n_blocks'], 12)
        self.assertEqual(first['all']['invalid_reasons'], {})
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])
        self.assertEqual(before[2:], np.random.get_state()[2:])

    def test_saved_control_preserves_rows_and_checks_calendar_and_scope(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            times = 737791. + np.arange(120)
            source = np.arange(120) + 200
            wildfire = np.array([110, 111])
            np.savez(root / 'population.npz', time_days=times, source_index=source,
                     clean=np.arange(100), wildfire=wildfire)
            (root / 'results.json').write_text(json.dumps({'settings': {'dataset_seed': 2026}}))
            split = root / 'screening_split.npz'
            metadata = dict(settings={'control_gap_days': 15}, counts={'control': 4},
                            seeds={'dataset': 2026})
            split.with_suffix('.json').write_text(json.dumps(metadata))
            rows = np.array([42, 1, 17, 81])
            np.savez(split, control=rows, wildfire=wildfire, evaluation_scope='arrival',
                     recalibration=np.array([99]))
            actual_times, actual_source, selections, actual_metadata = load_saved_control(root, split)
            self.assertEqual(set(selections), {'saved_control'})
            np.testing.assert_array_equal(selections['saved_control'], [1, 17, 42, 81])
            np.testing.assert_array_equal(actual_times, times)
            np.testing.assert_array_equal(actual_source, source)
            self.assertEqual(actual_metadata, metadata)
            for bad_rows in ([1, 1, 17, 81], [1, 17, 42, 110], [1, 17, 42],
                             [1, 2, 17, 81], [1., 17., 42., 81.]):
                with self.subTest(rows=bad_rows):
                    np.savez(split, control=bad_rows, wildfire=wildfire, evaluation_scope='arrival')
                    with self.assertRaises(ValueError):
                        load_saved_control(root, split)
            np.savez(split, control=rows, wildfire=wildfire, evaluation_scope='trajectory')
            with self.assertRaisesRegex(ValueError, 'arrival split'):
                load_saved_control(root, split)
            np.savez(split, control=rows, wildfire=[112, 113], evaluation_scope='arrival')
            with self.assertRaisesRegex(ValueError, 'different wildfire rows'):
                load_saved_control(root, split)
            (root / 'results.json').write_text(json.dumps({'settings': {'dataset_seed': 123}}))
            with self.assertRaisesRegex(ValueError, 'different dataset seeds'):
                load_saved_control(root, split)

    def test_multiple_gaps_share_baseline_and_enforce_equal_seasonal_counts(self):
        with TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ['gap10', 'gap20']]
            times = 737791. + np.arange(1500)
            for root, gap in zip(roots, [10, 20]):
                root.mkdir()
                (root / 'results.json').write_text(json.dumps(
                    {'settings': {'spaced_role': 'control', 'gap_days': gap}}))
                np.savez(root / 'population.npz', time_days=times, clean=np.arange(1500),
                         source_index=np.arange(1500))
                np.savez(root / 'current.npz', control=np.arange(200))
                np.savez(root / 'n40_seed2228.npz', control=np.arange(40) * gap)
                np.savez(root / 'earliest_first_maximum.npz', control=np.arange(0, 1500, gap))
            _, _, selections, metadata = load_comparison(roots, 12, seasonal_count=40)
            self.assertEqual(len(selections), 9)
            self.assertEqual(set(metadata['settings']['inventories']), {'gap10', 'gap20'})
            for gap in [10, 20]:
                self.assertEqual(len(selections[f'gap{gap}_spaced_40']), 40)
                self.assertEqual(len(selections[f'gap{gap}_random_40']), 40)
                self.assertGreaterEqual(np.diff(times[selections[f'gap{gap}_spaced_40']]).min(), gap)
            with self.assertRaisesRegex(ValueError, 'distinct minimum gaps'):
                load_comparison([roots[0], roots[0]], 12, seasonal_count=40)
            np.savez(roots[1] / 'current.npz', control=np.arange(201))
            with self.assertRaisesRegex(ValueError, 'same current-control baseline'):
                load_comparison(roots, 12, seasonal_count=40)
            np.savez(roots[1] / 'n40_seed2228.npz', control=np.arange(39) * 20)
            with self.assertRaisesRegex(ValueError, '40-point selection is incomplete'):
                load_selections(roots[1], 12, seasonal_count=40)

    def test_invalid_checks_are_not_counted_as_acceptances(self):
        results = run_audit(np.arange(4), {'small': np.arange(4)}, 'iid_uniform',
                            repetitions=2, bootstrap_replicates=19, null_simulations=29, seed=5)
        row = results['small']
        self.assertEqual(row['block_bootstrap'], [None, None])
        self.assertEqual(row['invalid_reasons'], {'insufficient blocks: 1 < 10': 2})
        self.assertEqual(rejection_summary(row['block_bootstrap'], .05),
                         dict(valid=0, invalid=2, rejected=0, rejection_rate=None, ci95=None))
        summary = rejection_summary([None, .01, .05, .9], .05)
        self.assertEqual(summary['valid'], 3)
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['rejection_rate'], 1 / 3)
        self.assertLess(summary['ci95'][0], 1 / 3)
        self.assertGreater(summary['ci95'][1], 1 / 3)


if __name__ == '__main__':
    unittest.main()
