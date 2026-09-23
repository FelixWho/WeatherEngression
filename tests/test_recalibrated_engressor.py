"""Quantile mapping checks with fixed synthetic samples; no network training."""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.wildfire_sensitivity import main as screening
from experiments.wildfire_sensitivity.control import split_clean_holdout


class RecalibratedEngressorTests(unittest.TestCase):
    def setUp(self):
        # The base forecast is Uniform(-1, 1) around each input's location.
        self.base = SimpleNamespace(
            device=torch.device("cpu"),
            sample=lambda x, sample_size: (
                x[:, :1, :1] + torch.linspace(-1, 1, sample_size).reshape(1, 1, -1)
            ),
        )

    def test_pooled_mapping_corrects_overcoverage_on_different_observations(self):
        # True outcomes span (-.5, .5), so the base forecast is twice as wide.
        model = RecalibratedEngressor(self.base).fit_from_pit(
            np.linspace(.25, .75, 1001), screening.QUANTILE_LEVELS
        )
        self.assertAlmostEqual(model.quantile_levels[.05], .275)
        self.assertAlmostEqual(model.quantile_levels[.95], .725)
        x = np.zeros((200, 1, 1), dtype=np.float32)
        y = ((np.arange(200) + .5) / 200 - .5).reshape(-1, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            raw = screening.interval_coverage(self.base, x, y, "raw", .90)
            corrected = screening.interval_coverage(model, x, y, "corrected", .90)
            metrics = screening.print_calibration(model, x, y, "corrected", np.arange(200))
            corrected95 = screening.interval_coverage(model, x, y, "corrected", .95)
        self.assertEqual(raw, 1.)
        self.assertAlmostEqual(corrected, .90)
        self.assertAlmostEqual(corrected95, .95)
        self.assertAlmostEqual(metrics["central"][.90], corrected)
        self.assertAlmostEqual(metrics["central"][.95], corrected95)
        self.assertAlmostEqual(metrics["below"][.90], .90)
        self.assertFalse(metrics["pit_is_recalibrated"])

    def test_asymmetric_mapping_and_input_dependent_boundaries(self):
        model = RecalibratedEngressor(self.base).fit_from_pit(
            np.linspace(.4, .8, 1001), [.05, .5, .95]
        )
        x = torch.tensor([[[0.]], [[10.]]])
        lower, upper = model.predict(x, target=[.05, .95], sample_size=101)
        torch.testing.assert_close(lower[:, 0], torch.tensor([-.16, 9.84]))
        torch.testing.assert_close(upper[:, 0], torch.tensor([.56, 10.56]))
        torch.testing.assert_close(model.predict(x, .5)[:, 0], torch.tensor([.2, 10.2]))

    def test_fitting_helper_uses_base_samples_and_saves_correction(self):
        x = np.zeros((200, 1, 1), dtype=np.float32)
        y_fit = ((np.arange(200) + .25) / 200 - .5).reshape(-1, 1)
        y_check = ((np.arange(200) + .75) / 200 - .5).reshape(-1, 1)
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "quantiles.json"
            with patch.object(screening, "EVALUATION_BATCH_SIZE", 50):
                wrapped = screening.recalibrate_counterfactual(self.base, x, y_fit, path)
            loaded = RecalibratedEngressor.load(self.base, path)
            coverage = screening.interval_coverage(loaded, x, y_check, "check", .90)
        self.assertIs(wrapped.engressor, self.base)
        self.assertAlmostEqual(coverage, .90, delta=.01)

    def test_mapping_round_trip_and_float_level_lookup(self):
        model = RecalibratedEngressor(self.base).fit_from_pit(
            [.1, .3, .8, .9], [.025, .975]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "quantiles.json"
            model.save(path)
            loaded = RecalibratedEngressor.load(self.base, path)
            # (1-.95)/2 is not exactly .025 in floating-point arithmetic.
            samples = self.base.sample(torch.zeros(2, 1, 1), 101)
            torch.testing.assert_close(
                model.quantiles_from_samples(samples, [.025, .975]),
                loaded.quantiles_from_samples(samples, [(1-.95)/2, 1-(1-.95)/2]),
            )

    def test_invalid_mapping_and_missing_level_are_rejected(self):
        for pit in ([], [np.nan], [-.1], [1.1]):
            with self.subTest(pit=pit), self.assertRaises(ValueError):
                RecalibratedEngressor(self.base).fit_from_pit(pit, [.5])
        with self.assertRaises(ValueError):
            RecalibratedEngressor(self.base, {.05: .8, .95: .2})
        with self.assertRaises(KeyError):
            RecalibratedEngressor(self.base, {.5: .5}).predict(torch.zeros(1, 1, 1), .95)


class TimeSeparationTests(unittest.TestCase):
    def test_both_sides_of_holdout_have_a_history_gap(self):
        times = 730000. + np.arange(240 * 24) / 24
        rows = np.arange(len(times))
        # Isolate boundary purging from calendar stratification.
        with patch("experiments.wildfire_sensitivity.control._months",
                   return_value=np.ones(len(times), dtype=int)):
            fit, holdout = split_clean_holdout(
                rows, times, window_hours=241, block_days=30,
                holdout_fraction=.25, seed=2026,
            )
        self.assertTrue(len(fit) and len(holdout))
        self.assertFalse(np.intersect1d(fit, holdout).size)
        # Each history spans the previous 240 hours. Both neighboring fit rows
        # must be more than 240 hours away from every retained holdout arrival.
        fit_times = np.sort(times[fit])
        positions = np.searchsorted(fit_times, times[holdout])
        before = positions > 0
        after = positions < len(fit_times)
        self.assertTrue(np.all((times[holdout][before] - fit_times[positions[before]-1]) * 24 > 240))
        self.assertTrue(np.all((fit_times[positions[after]] - times[holdout][after]) * 24 > 240))

    def test_sequential_splits_keep_all_four_roles_separate(self):
        times = 730000. + np.arange(360 * 24) / 24
        fit = np.arange(len(times))
        held_out = []
        with patch("experiments.wildfire_sensitivity.control._months",
                   return_value=np.ones(len(times), dtype=int)):
            for seed, fraction in [(2026, .25), (2228, .20), (2127, .15)]:
                fit, holdout = split_clean_holdout(
                    fit, times, window_hours=241, block_days=30,
                    holdout_fraction=fraction, seed=seed,
                )
                held_out.append(holdout)
        roles = [fit] + held_out
        for i, left in enumerate(roles):
            for right in roles[i+1:]:
                self.assertFalse(np.intersect1d(left, right).size)
                # Synthetic rows are hourly, so row distances are hour distances.
                positions = np.searchsorted(left, right)
                before, after = positions > 0, positions < len(left)
                self.assertTrue(np.all(right[before] - left[positions[before]-1] > 240))
                self.assertTrue(np.all(left[positions[after]] - right[after] > 240))


if __name__ == "__main__":
    unittest.main()
