"""Seconds-long split checks on synthetic timestamps; no real-data loading."""
from itertools import combinations
import unittest

import numpy as np

from experiments.wildfire_sensitivity.control import _months
from experiments.wildfire_sensitivity.split_feasibility import (
    earliest_spaced, month_quotas, purge_near_selected, select_spaced, split_inventory,
)


class SplitFeasibilityTests(unittest.TestCase):
    def test_earliest_maximum_matches_exhaustive_search(self):
        times = np.array([15., 0., 8., 10., 20., 20., 29., 40.])
        rows = np.arange(len(times))
        chosen = earliest_spaced(rows, times, 10)
        brute_max = max(len(c) for n in range(1, len(rows) + 1)
                        for c in combinations(rows, n)
                        if np.all(np.diff(np.sort(times[list(c)])) >= 10))
        self.assertEqual(len(chosen), brute_max)
        np.testing.assert_array_equal(times[chosen], [0, 10, 20, 40])

    def test_random_selection_uses_full_pool_and_preserves_spacing_and_quotas(self):
        times = np.arange(400, dtype=float)
        months = np.tile(np.repeat([6, 7], 100), 2)
        quotas = month_quotas([6, 6, 7, 7], 12)
        rows = np.arange(400)
        selected = select_spaced(rows, times, months, quotas, 10, 2228)
        np.testing.assert_array_equal(selected, select_spaced(rows, times, months, quotas, 10, 2228))
        self.assertFalse(np.array_equal(selected, select_spaced(rows, times, months, quotas, 10, 2229)))
        self.assertEqual(len(selected), 12)
        np.testing.assert_array_equal(np.bincount(months[selected], minlength=13), quotas)
        self.assertGreaterEqual(np.diff(np.sort(times[selected])).min(), 10)
        # No prior recalibration indices restrict selection; both distant periods contribute.
        self.assertTrue(np.any(selected < 200) and np.any(selected >= 200))

    def test_quotas_and_shortfall_are_explicit(self):
        q = month_quotas([1, 1, 1, 2], 7)
        self.assertEqual(q.sum(), 7)
        np.testing.assert_array_equal(q[1:3], [5, 2])
        selected = select_spaced(np.arange(4), np.array([0., 1., 2., 20.]),
                                 np.ones(4, dtype=int), month_quotas([1], 4), 10, 5)
        self.assertEqual(len(selected), 2)
        for gap in [-1, 0, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                earliest_spaced(np.arange(4), np.arange(4), gap)

    def test_purge_both_sides_and_exact_boundary(self):
        times = np.array([-11., -10., -9., 0., 0., 9., 10., 11.])
        remaining, excluded = purge_near_selected(np.arange(8), np.array([3]), times, 10)
        np.testing.assert_array_equal(remaining, [0, 1, 6, 7])
        np.testing.assert_array_equal(excluded, [2, 4, 5])
        remaining, excluded = purge_near_selected(np.arange(8), np.array([3]), times, 0)
        np.testing.assert_array_equal(remaining, [0, 1, 2, 4, 5, 6, 7])
        self.assertEqual(len(excluded), 0)
        for gap in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                purge_near_selected(np.arange(8), np.array([3]), times, gap)

    def test_no_recalibration_guard_keeps_nearby_rows_and_baseline_unchanged(self):
        times = 737791. + np.arange(1965)
        clean, wildfire = np.arange(1600), np.arange(1600, 1965)
        settings = dict(block_days=30, control_fraction=.25, recalibration_fraction=.2,
                        validation_fraction=.15, control_seed=2026, recalibration_seed=2228,
                        validation_seed=2127)
        selected = earliest_spaced(clean, times, 10)
        _, baseline = split_inventory(clean, wildfire, times, 241, settings)
        settings['recalibration_guard_days'] = 0
        _, unchanged = split_inventory(clean, wildfire, times, 241, settings)
        for role in baseline:
            np.testing.assert_array_equal(baseline[role], unchanged[role])
        counts, subsets = split_inventory(clean, wildfire, times, 241, settings, selected)
        self.assertEqual(counts['status'], 'ok', counts['reason'])
        self.assertEqual(counts['recalibration_guard'], 0)
        self.assertEqual(counts['after_recalibration_guard'], len(clean) - len(selected))
        np.testing.assert_array_equal(np.sort(np.concatenate(list(subsets.values()))), clean)
        for role in ['train', 'validation', 'control']:
            distances = abs(times[subsets[role]][:, None] - times[selected][None, :])
            self.assertLess(distances.min(), 10)
            self.assertEqual(len(np.intersect1d(subsets[role], selected)), 0)

    def test_full_inventory_accounts_for_all_rows_and_separates_roles(self):
        times = 737791. + np.arange(1965)  # daily observations starting in 2020
        clean, wildfire = np.arange(1600), np.arange(1600, 1965)
        months = _months(times)
        settings = dict(block_days=30, control_fraction=.25, recalibration_fraction=.2,
                        validation_fraction=.15, control_seed=2026, recalibration_seed=2228,
                        validation_seed=2127)
        selected = select_spaced(clean, times, months, month_quotas(months[wildfire], 20), 10, 2228)
        for recal in [None, selected]:
            counts, subsets = split_inventory(clean, wildfire, times, 241, settings, recal)
            self.assertEqual(counts['status'], 'ok', counts['reason'])
            all_rows = np.concatenate(list(subsets.values()))
            np.testing.assert_array_equal(np.sort(all_rows), clean)
            self.assertEqual(sum(counts[k] for k in ['train', 'validation', 'control',
                                                    'recalibration', 'excluded_total']), len(clean))
            if recal is not None:
                np.testing.assert_array_equal(subsets['recalibration'], selected)
                for role in ['train', 'validation', 'control']:
                    distances = abs(times[subsets[role]][:, None] - times[selected][None, :])
                    self.assertGreaterEqual(distances.min(), 241 / 24)

    def test_depleted_pool_reports_invalid_without_losing_accounting(self):
        times = 737791. + np.arange(25)
        clean, wildfire = np.arange(20), np.arange(20, 25)
        settings = dict(block_days=30, control_fraction=.25, recalibration_fraction=.2,
                        validation_fraction=.15, control_seed=2026, recalibration_seed=2228,
                        validation_seed=2127)
        counts, subsets = split_inventory(clean, wildfire, times, 241, settings, np.array([10]))
        self.assertEqual(counts['status'], 'invalid')
        self.assertTrue(counts['reason'])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(subsets.values()))), clean)


if __name__ == '__main__':
    unittest.main()
