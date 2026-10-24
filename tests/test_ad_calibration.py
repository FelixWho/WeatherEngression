"""Small numerical and integration checks; no fitting or real data loading."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.integrate import quad
import torch

from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.calibration import (
    _block_residual_gram, _bootstrap_statistics, ad_block_bootstrap,
    ad_statistic_uniform, ad_mc_pvalue, evaluate_ad,
)
from experiments.real_metrics import pit_values
from experiments.wildfire_sensitivity import main as screening
from experiments.wildfire_sensitivity import evaluation


def uniform_model(outputs=1):
    return SimpleNamespace(device=torch.device("cpu"), sample=Mock(side_effect=lambda x, sample_size:
        torch.linspace(0, 1, sample_size).expand(len(x), outputs, sample_size)))


class ADNumericalTests(unittest.TestCase):
    def test_statistic_against_independent_cdf_integral(self):
        u = np.array([.04, .12, .12, .6, .75, .95])
        edges = np.r_[0., np.unique(u), 1.]
        expected = 0.
        for lo, hi in zip(edges[:-1], edges[1:]):
            empirical = np.mean(u <= (lo + hi) / 2)
            expected += len(u) * quad(lambda t: (empirical - t)**2 / (t * (1 - t)), lo, hi)[0]
        self.assertAlmostEqual(ad_statistic_uniform(u), expected, places=12)
        self.assertTrue(np.isfinite(ad_statistic_uniform([0., 1.])))
        for bad in [[], [np.nan], [np.inf], [-.1], [1.1]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ad_statistic_uniform(bad)
        self.assertEqual(ad_mc_pvalue(1., 10, 31, 7), ad_mc_pvalue(1., 10, 31, 7))

    def test_bootstrap_matches_direct_resampled_cdfs_with_unequal_blocks(self):
        pit = np.array([.01, .2, .4, .4, .55, .9, .97])
        blocks = np.array([0, 0, 0, 1, 2, 2, 2])
        sizes = np.bincount(blocks)
        b, reps, seed = 3, 23, 52
        actual = _bootstrap_statistics(pit, blocks, sizes, reps, seed)
        multiplicities = np.random.default_rng(seed).multinomial(b, [1 / b] * b, reps)
        edges = np.unique(pit)
        mids = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(np.log(edges) - np.log1p(-edges))
        original = np.array([(pit <= mid).mean() for mid in mids])
        expected = []
        for counts in multiplicities:
            resample = np.repeat(pit, counts[blocks])
            empirical = np.array([(resample <= mid).mean() for mid in mids])
            expected.append(len(resample) * np.sum((empirical - original)**2 * widths))
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        np.testing.assert_allclose(_block_residual_gram(pit, blocks, sizes, chunk_size=2),
                                   _block_residual_gram(pit, blocks, sizes), atol=1e-13)

    def test_centering_and_whole_trajectory_resampling(self):
        # Identical blocks have no bootstrap variation, even when not uniform.
        bad = ad_block_bootstrap(np.full(20, .2), np.arange(20) * 30., n_bootstrap=99)
        self.assertEqual(bad["p_value"], .01)
        self.assertFalse(bad["accepted"])
        u = np.random.default_rng(19).uniform(size=30)
        times = np.arange(30) * 30.
        scalar = ad_block_bootstrap(u, times, n_bootstrap=99)
        trajectory = ad_block_bootstrap(np.repeat(u[:, None], 7, axis=1), times, n_bootstrap=99)
        self.assertAlmostEqual(trajectory["statistic"], 7 * scalar["statistic"])
        self.assertEqual(trajectory["p_value"], scalar["p_value"])
        self.assertEqual(trajectory["block_sizes"], [7] * 30)

    def test_invalid_checks_and_settings(self):
        for pit, times in [([], []), ([.5], [np.nan]), ([.5, .2], [0]), ([np.nan], [0])]:
            result = ad_block_bootstrap(pit, times)
            self.assertFalse(result["valid"])
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["p_value"])
            self.assertTrue(result["reason"])
        insufficient = ad_block_bootstrap([.2, .8], [0., 30.])
        self.assertEqual(insufficient["n_blocks"], 2)
        self.assertIn("insufficient blocks", insufficient["reason"])
        for kwargs in [dict(alpha=0), dict(alpha=1), dict(n_bootstrap=0), dict(seed=-1), dict(block_days=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ad_block_bootstrap([.2, .8], [0., 30.], **kwargs)

    def test_bootstrap_reproducibility_and_no_global_rng_changes(self):
        np.random.seed(321)
        before = np.random.get_state()
        pit = np.random.default_rng(4).uniform(size=(20, 3))
        times = np.arange(20) * 30.
        first = ad_block_bootstrap(pit, times, n_bootstrap=43, seed=51)
        second = ad_block_bootstrap(pit, times, n_bootstrap=43, seed=51)
        self.assertEqual(first, second)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


class CorrectedPITTests(unittest.TestCase):
    def test_identity_matches_existing_randomized_rank_including_ties(self):
        samples = np.array([[[0., 0., 1., 2.], [1., 2., 2., 3.]],
                            [[0., 0., 0., 0.], [2., 3., 4., 5.]]])
        y = np.array([[0., 2.], [0., 6.]])
        wrapped = RecalibratedEngressor(uniform_model(2))
        np.testing.assert_array_equal(wrapped.pit_from_samples(samples, y, seed=4).ravel(),
                                      pit_values(samples.reshape(-1, 4), y.ravel(), seed=4))

    def test_nonlinear_correction_transforms_bounds_before_randomizing_ties(self):
        wrapped = RecalibratedEngressor(uniform_model()).fit_from_pit([0., .25, 1.], [.5])
        samples = np.array([[[0., 0., 0., 1.]]])
        y = np.zeros((1, 1))
        random = np.random.default_rng(3).random()
        result = wrapped.pit_from_samples(samples, y, seed=3).item()
        expected = random * (.5 + (.8 - .25) / .75 * .5)
        self.assertAlmostEqual(result, expected)
        wrong = wrapped.transform_cdf(random * .8).item()
        self.assertNotAlmostEqual(result, wrong)
        self.assertEqual(wrapped.cdf_from_samples(samples, y, side="left").item(), 0.)
        self.assertAlmostEqual(wrapped.cdf_from_samples(samples, y).item(), 5 / 6)

    def test_inverse_map_handles_repeated_knots_and_exterior_probabilities(self):
        wrapped = RecalibratedEngressor(uniform_model()).fit_from_pit([.2, .4, .4, .8], [.1, .9])
        np.testing.assert_allclose(wrapped.transform_cdf([0, .2, .4, .6, .8, 1]),
                                   [0, 0, 2 / 3, 5 / 6, 1, 1])
        self.assertAlmostEqual(wrapped.transform_cdf(.4, side="left").item(), 1 / 3)
        samples = torch.linspace(0, 1, 101).reshape(1, 1, -1)
        for q in [.1, .3, .9]:
            self.assertAlmostEqual(wrapped.quantiles_from_samples(samples, [q]).item(),
                                   np.quantile([.2, .4, .4, .8], q), places=6)

    def test_mapping_serialization_and_legacy_quantiles(self):
        base = uniform_model()
        fitted = RecalibratedEngressor(base).fit_from_pit([.1, .21, .43, .6, .8], [.05, .95])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            fitted.save(path)
            loaded = RecalibratedEngressor.load(base, path)
            np.testing.assert_array_equal(fitted.calibration_pit, loaded.calibration_pit)
            self.assertEqual(fitted.quantile_levels, loaded.quantile_levels)
            np.testing.assert_array_equal(fitted.transform_cdf([.2, .5]), loaded.transform_cdf([.2, .5]))
            path.write_text(json.dumps({.05: .2, .95: .8}))
            legacy = RecalibratedEngressor.load(base, path)
            self.assertEqual(len(legacy.predict(torch.zeros(1, 2, 1), [.05, .95])), 2)
            with self.assertRaisesRegex(ValueError, "refit recalibration"):
                legacy.pit_from_samples(np.zeros((1, 1, 5)), np.zeros((1, 1)))

    def test_empirical_cdf_limits_with_atoms_in_the_correction(self):
        wrapped = RecalibratedEngressor(uniform_model()).fit_from_pit([0, 0, .5, .5, 1, 1], [.5])
        samples = np.tile([0., 1., 2., 3.], (4, 1))
        y = np.array([-1., 0., 1.5, 4.])
        left = wrapped.cdf_from_samples(samples, y, side="left")
        right = wrapped.cdf_from_samples(samples, y)
        self.assertEqual(left[0], 0.)
        self.assertEqual(right[0], 0.)
        self.assertEqual(left[1], 0.)
        self.assertEqual(left[2], right[2])  # No atom at an unobserved y.
        self.assertEqual(left[-1], 1.)
        self.assertEqual(right[-1], 1.)


class ADIntegrationTests(unittest.TestCase):
    def test_model_evaluator_in_both_scopes_and_shape_errors(self):
        for scope, outputs in [("arrival", 1), ("trajectory", 3)]:
            x = np.zeros((60, 3, 1), dtype=np.float32)
            y = .25 + .5 * np.random.default_rng(11).uniform(size=(60, outputs))
            times = np.arange(60) * 30.
            base = uniform_model(outputs)
            wrapped = RecalibratedEngressor(base).fit_from_pit(np.linspace(.25, .75, 1001), [.05, .95])
            corrected = evaluate_ad(wrapped, x, y, times, evaluation_scope=scope, batch_size=17, n_bootstrap=99)
            raw = evaluate_ad(base, x, y, times, evaluation_scope=scope, batch_size=17, n_bootstrap=99)
            self.assertTrue(corrected["pit_is_recalibrated"])
            self.assertTrue(corrected["valid"])
            self.assertLess(corrected["statistic"], raw["statistic"])
            self.assertEqual(corrected["n_values"], 60 * outputs)
            with self.assertRaises(ValueError):
                evaluate_ad(uniform_model(outputs + 1), x, y, times, evaluation_scope=scope)
            with self.assertRaises(ValueError):
                evaluate_ad(base, x, np.zeros((60, outputs + 1)), times, evaluation_scope=scope)

    def test_sampling_seed_is_isolated(self):
        model = SimpleNamespace(device=torch.device("cpu"),
                                sample=lambda x, sample_size: torch.rand(len(x), 1, sample_size))
        x, y = np.zeros((20, 3, 1), np.float32), np.full((20, 1), .5)
        before = torch.get_rng_state().clone()
        a = evaluate_ad(model, x, y, np.arange(20) * 30., n_bootstrap=19)
        b = evaluate_ad(model, x, y, np.arange(20) * 30., n_bootstrap=19)
        self.assertEqual(a, b)
        torch.testing.assert_close(before, torch.get_rng_state())

    def test_ad_gate_is_independent_of_coverage_and_reuses_draws(self):
        for scope, outputs in [("arrival", 1), ("trajectory", 2)]:
            for acceptance, y_row in [(True, [.5] * 20), (False, [.5] * 18 + [.03, .01])]:
                y = np.repeat(np.array(y_row)[:, None], outputs, axis=1)
                x = np.zeros((20, 2, 1), dtype=np.float32)
                base = uniform_model(outputs)
                wrapped = RecalibratedEngressor(base).fit_from_pit([0., 1.], screening.QUANTILE_LEVELS)
                ad = dict(accepted=acceptance, valid=True, statistic=1.,
                          p_value=.5 if acceptance else .01, reason="test")
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(evaluation, "ad_block_bootstrap", return_value=ad) as check:
                    result = screening.print_calibration(wrapped, x, y, "control", np.arange(20) * 30.,
                                                         evaluation_scope=scope, calibration_metric="ad")
                self.assertEqual(result["calibrated"], acceptance)
                self.assertEqual(result["coverage_calibrated"], not acceptance)
                self.assertEqual(base.sample.call_count, 1)
                np.testing.assert_array_equal(check.call_args.args[0],
                    wrapped.pit_from_samples(torch.linspace(0, 1, 400).expand(20, outputs, 400), y, seed=2026))

    def test_selecting_ad_does_not_change_predictive_draws_or_coverage(self):
        base = SimpleNamespace(device=torch.device("cpu"),
                               sample=lambda x, sample_size: torch.rand(len(x), 1, sample_size))
        wrapped = RecalibratedEngressor(base).fit_from_pit([.1, .9], screening.QUANTILE_LEVELS)
        x, y = np.zeros((20, 2, 1), np.float32), np.full((20, 1), .5)
        results, states = [], []
        for metric in ["coverage", "ad"]:
            torch.manual_seed(9)
            with contextlib.redirect_stdout(io.StringIO()):
                results.append(screening.print_calibration(
                    wrapped, x, y, "control", np.arange(20) * 30., evaluation_scope="arrival",
                    calibration_metric=metric, ad_bootstrap_replicates=19))
            states.append(torch.get_rng_state().clone())
        self.assertEqual(results[0]["central"], results[1]["central"])
        np.testing.assert_array_equal(results[0]["pit_counts"], results[1]["pit_counts"])
        torch.testing.assert_close(*states)

    def test_legacy_ad_check_is_invalid_without_raw_fallback(self):
        base = uniform_model()
        legacy = RecalibratedEngressor(base, {q: q for q in screening.QUANTILE_LEVELS})
        with contextlib.redirect_stdout(io.StringIO()):
            result = screening.print_calibration(legacy, np.zeros((20, 2, 1), np.float32),
                np.full((20, 1), .5), "control", np.arange(20) * 30.,
                evaluation_scope="arrival", calibration_metric="ad", ad_bootstrap_replicates=19)
        self.assertFalse(result["calibrated"])
        self.assertFalse(result["ad"]["valid"])
        self.assertIn("refit recalibration", result["acceptance_reason"])

    def test_cli_defaults_and_json_invalid_values(self):
        args = screening.build_parser().parse_args([])
        self.assertEqual((args.calibration_metric, args.ad_alpha, args.ad_bootstrap_replicates,
                          args.ad_bootstrap_seed), ("coverage", .05, 2000, 2430))
        args = screening.build_parser().parse_args(["--calibration-metric", "ad", "--ad-alpha", ".1",
                                                   "--ad-bootstrap-replicates", "31", "--ad-bootstrap-seed", "42"])
        self.assertEqual((args.calibration_metric, args.ad_alpha, args.ad_bootstrap_replicates,
                          args.ad_bootstrap_seed), ("ad", .1, 31, 42))
        self.assertEqual(json.loads(json.dumps(screening._json_ready({"p": np.nan, "counts": np.array([1])}),
                                              allow_nan=False)), {"p": None, "counts": [1]})

    def test_slurm_passes_metric_settings_without_running_python_or_submitting(self):
        repo = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            capture = Path(directory) / "args"
            data = Path(directory) / "empty.mat"
            data.touch()
            env = dict(os.environ, REPO_ROOT=str(repo), MAT_PATH=str(data), CKPT_DIR=directory,
                       CAPTURE_ARGS=str(capture), SCREENING_MODE="until_stable",
                       EVALUATION_SCOPE="arrival", CALIBRATION_METRIC="ad", AD_ALPHA="0.1",
                       AD_BOOTSTRAP_REPLICATES="31", AD_BOOTSTRAP_SEED="42")
            # A shell function intercepts BOTH Python invocations, including the
            # CUDA preflight. This tests argument wiring without running a job.
            shell = '''
python() {
  if [[ "$2" == "-" ]]; then
    cat >/dev/null
  else
    printf '%s\\n' "$@" > "$CAPTURE_ARGS"
  fi
}
export -f python
bash slurm/wildfire_sensitivity/run.slurm
'''
            subprocess.run(["bash", "-c", shell], cwd=repo, env=env,
                           check=True, capture_output=True, text=True)
            args = capture.read_text().splitlines()
            for flag, value in [("--calibration-metric", "ad"), ("--ad-alpha", "0.1"),
                                ("--ad-bootstrap-replicates", "31"), ("--ad-bootstrap-seed", "42"),
                                ("--screening-mode", "until_stable"), ("--evaluation-scope", "arrival")]:
                self.assertEqual(args[args.index(flag) + 1], value)

    def test_audit_scenarios_produce_valid_pit_and_keep_requested_blocks(self):
        from experiments.validate_ad_bootstrap import SCENARIOS, simulate_pit
        for scenario in SCENARIOS:
            pit, times = simulate_pit(scenario, 10, np.random.default_rng(7), prediction_samples=20)
            self.assertEqual(pit.shape, times.shape)
            self.assertTrue(np.all((pit >= 0) & (pit <= 1)))
            self.assertEqual(len(np.unique(times)), 10)


if __name__ == "__main__":
    unittest.main()
