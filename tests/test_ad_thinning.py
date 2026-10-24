"""Tiny checks for timestamp selection and IID AD; no research sweep."""
import contextlib
import io
import unittest

import numpy as np

from experiments.calibration import ad_mc_pvalue, ad_statistic_uniform
from experiments.validate_ad_thinning import (
    iid_ad_pvalue, iid_ad_reference, run_audit, time_spaced_indices,
)


class ThinnedADTests(unittest.TestCase):
    def test_unsorted_times_duplicates_and_exact_gap(self):
        times = np.array([60., 0., 29., 30., 30., 90., 61.])
        selected = time_spaced_indices(times, 30.)
        np.testing.assert_array_equal(selected, [1, 3, 0, 5])
        self.assertTrue(np.all(np.diff(times[selected]) >= 30))
        self.assertEqual(len(time_spaced_indices(times, 0)), len(times))
        # No PIT values enter selection at all.
        np.testing.assert_array_equal(selected, time_spaced_indices(times, 30.))

    def test_selection_rejects_invalid_inputs(self):
        for times, gap in [([], 1), ([np.nan], 1), ([[0., 1.]], 1), ([0., 1.], -1), ([0., 1.], np.inf)]:
            with self.subTest(times=times, gap=gap), self.assertRaises(ValueError):
                time_spaced_indices(times, gap)

    def test_cached_reference_matches_existing_iid_monte_carlo(self):
        pit = np.array([.02, .18, .4, .4, .6, .8, .97])
        reference = iid_ad_reference(len(pit), 301, 7)
        self.assertEqual(iid_ad_pvalue(pit, reference),
                         ad_mc_pvalue(ad_statistic_uniform(pit), len(pit), 301, 7))
        np.testing.assert_array_equal(reference, iid_ad_reference(len(pit), 301, 7))

    def test_tiny_audit_preserves_sample_sizes_and_global_rng(self):
        before = np.random.get_state()
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_audit([2, 1, 3, 2, 2], [0., 30., 60.], 3, 19, 4)
            repeated = run_audit([2, 1, 3, 2, 2], [0., 30., 60.], 3, 19, 4)
        self.assertEqual(result, repeated)
        self.assertEqual(len(result), 15)
        for row in result:
            self.assertEqual(row['n_selected'], {0: 10, 30: 5, 60: 3}[row['gap_days']])
            self.assertEqual(len(row['p_values']), 3)
            self.assertEqual(row['known_uniform_null'], row['scenario'] != 'miscalibrated')
            self.assertEqual(set(row['threshold_checks']), {'0.01', '0.05', '0.1'})
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


if __name__ == '__main__':
    unittest.main()
