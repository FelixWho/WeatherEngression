"""Independent numerical oracles for both screening modes; no fitting or real data."""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from engression_modifications.lstm import LSTMEngressionConfig, LSTMEngressor
from engression_modifications.lstm.checkpoint import _checkpoint_payload, _write_checkpoint
from engression_modifications.lstm.generators import build_lstm_model
from engression_modifications.lstm.preprocessing import standardization_stats
from engression_modifications.recalibrated_engressor import RecalibratedEngressor
from experiments.wildfire_sensitivity import main as screening
from experiments.wildfire_sensitivity.control import month_matched_subsample


def uniform_model(outputs):
    # The conditioning input sets a different location for each row and output.
    return SimpleNamespace(
        device=torch.device("cpu"),
        sample=lambda x, sample_size: (
            x[:, -1:, 0] if outputs == 1 else x[:, :, 0]
        ).unsqueeze(-1) + torch.linspace(0, 1, sample_size),
    )


class NumericalAuditTests(unittest.TestCase):
    def test_month_matching_keeps_reference_mix_and_rejects_missing_months(self):
        # Pool has 4 January + 3 February; reference has 4 January + 2 February.
        months = np.array([1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2])
        times = np.arange(len(months), dtype=float)
        with patch("experiments.wildfire_sensitivity.control._months", return_value=months):
            selected = month_matched_subsample(np.arange(7), np.arange(7, 13), times)
            self.assertEqual(list(np.bincount(months[selected])[1:]), [4, 2])
            self.assertEqual(len(np.unique(selected)), len(selected))
            with self.assertRaisesRegex(ValueError, "no clean holdout rows"):
                month_matched_subsample(np.arange(4), np.arange(7, 13), times)

    def test_adjusted_endpoints_counts_and_weighted_errors_in_both_modes(self):
        x = np.arange(15, dtype=np.float32).reshape(5, 3, 1)
        offsets = np.array([
            [0., .3, .5], [.9, .8, .1], [.4, .5, .99],
            [.5, .6, .7], [.1, .9, .5],
        ], dtype=np.float32)
        times = np.array([0., 0., 30., 60., 60.])
        # Adjusted central 95% endpoints are location + .215 and + .785;
        # 90% endpoints are +.23 and +.77, with the same inside counts here.
        mapping = {q: .2 + .6 * q for q in screening.QUANTILE_LEVELS}
        for scope, outputs, covered, variance, below in [
            ("arrival", 1, 3 / 5, .0624, 4 / 5),
            ("trajectory", 3, 8 / 15, 3.36 / 225, 11 / 15),
        ]:
            y = x[:, :, 0] + offsets
            if scope == "arrival":
                y = y[:, -1:]
            model = RecalibratedEngressor(uniform_model(outputs), mapping)
            # Last batch has one row: averaging batch percentages would be wrong.
            for batch_size in [2, 5]:
                with self.subTest(scope=scope, batch_size=batch_size), \
                     patch.object(screening, "EVALUATION_BATCH_SIZE", batch_size), \
                     contextlib.redirect_stdout(io.StringIO()):
                    metrics = screening.print_calibration(
                        model, x, y, "control", times, evaluation_scope=scope
                    )
                    for level in [.9, .95]:
                        result = screening.interval_coverage(
                            model, x, y, "wildfire", level, evaluation_scope=scope
                        )
                        self.assertAlmostEqual(result, covered)
                        self.assertAlmostEqual(metrics["central"][level], covered)
                        self.assertAlmostEqual(metrics["standard_error"][level], np.sqrt(variance))
                        self.assertAlmostEqual(metrics["tolerance"][level], 3 * np.sqrt(variance))
                        self.assertAlmostEqual(metrics["below"][level], below)
                    self.assertEqual(sum(metrics["pit_counts"]), len(x) * outputs)

    def test_fitted_recalibration_and_independent_control_in_both_modes(self):
        n = 200
        for scope, outputs in [("arrival", 1), ("trajectory", 3)]:
            x = np.tile(np.array([0., 10., 20.], dtype=np.float32)[None, :, None], (n, 1, 1))
            centers = x[:, -1:, 0] if outputs == 1 else x[:, :, 0]
            # Narrower true distribution (.25,.75) than the base forecast (0,1).
            fit_y = centers + (.25 + .5 * (np.arange(n) + .25) / n)[:, None]
            check_y = centers + (.25 + .5 * (np.arange(n) + .75) / n)[:, None]
            base = uniform_model(outputs)
            with self.subTest(scope=scope), TemporaryDirectory() as directory, \
                 contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(screening, "EVALUATION_BATCH_SIZE", 37):
                path = Path(directory) / "mapping.json"
                fitted = screening.recalibrate_counterfactual(base, x, fit_y, path, scope)
                loaded = RecalibratedEngressor.load(base, path)
                self.assertAlmostEqual(fitted.quantile_levels[.05], .275, delta=.005)
                self.assertAlmostEqual(fitted.quantile_levels[.95], .725, delta=.005)
                metrics = screening.print_calibration(
                    loaded, x, check_y, "control", np.arange(n), evaluation_scope=scope
                )
                for level in [.9, .95]:
                    coverage = screening.interval_coverage(
                        loaded, x, check_y, "wildfire", level, scope
                    )
                    self.assertAlmostEqual(coverage, level, delta=.015)
                    self.assertEqual(metrics["central"][level], coverage)

    def test_invalid_data_fails_in_all_three_evaluation_stages(self):
        x = np.zeros((4, 3, 1), dtype=np.float32)
        times = np.arange(4.)
        for scope in ["arrival", "trajectory"]:
            width = 1 if scope == "arrival" else 3
            y = np.zeros((4, width), dtype=np.float32)
            # Wrong output width, missing target axis, and nonfinite predictions.
            bad_samples = [torch.zeros(4, 2, 10), torch.full((4, width, 10), float("nan"))]
            with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
                for samples in bad_samples:
                    base = SimpleNamespace(device=torch.device("cpu"), sample=Mock(return_value=samples))
                    actions = [
                        lambda: screening.recalibrate_counterfactual(base, x, y, Path(directory) / "map.json", scope),
                        lambda: screening.print_calibration(base, x, y, "test", times, evaluation_scope=scope),
                        lambda: screening.interval_coverage(base, x, y, "test", evaluation_scope=scope),
                    ]
                    for action in actions:
                        with self.subTest(scope=scope, shape=samples.shape), self.assertRaises(ValueError):
                            action()


class TrainingAndReloadAuditTests(unittest.TestCase):
    def test_actual_sampler_and_checkpoint_preserve_output_shape_and_units(self):
        for scope, width in [("arrival", 1), ("trajectory", 241)]:
            data = np.arange(4 * 241 * 3, dtype=np.float32).reshape(4, 241, 3)
            x, y = screening.filter_dataset_columns(data, [0, 2], 1, scope)
            xt, yt = torch.from_numpy(x), torch.from_numpy(y)
            config = LSTMEngressionConfig(
                hidden_dim=4, num_layer=1, lstm_num_layers=1, noise_dim=2,
                global_latent_noise=True, device="cpu",
            )
            model = build_lstm_model(config, input_dim=2, out_dim=y.shape[1])
            # Zero output in standardized units must become the training y mean.
            for parameter in model.parameters():
                parameter.data.zero_()
            xm, xs, ym, ys = standardization_stats(xt, yt, True)
            engressor = LSTMEngressor(model, config, xm, xs, ym, ys)
            with self.subTest(scope=scope), TemporaryDirectory() as directory:
                path = str(Path(directory) / "checkpoint.pt")
                payload = _checkpoint_payload(
                    model, torch.optim.Adam(model.parameters()), config,
                    2, .5, 2, width, xm, xs, ym, ys, val_loss=.4,
                )
                _write_checkpoint(path, payload)
                loaded = screening.load_lstm_engressor_checkpoint(path, device="cpu")
                for sampler in [engressor, loaded]:
                    for draws in [1, 3]:
                        samples = sampler.sample(xt, sample_size=draws)
                        self.assertEqual(tuple(samples.shape), (4, width, draws))
                        torch.testing.assert_close(samples, ym.unsqueeze(-1).expand(4, width, draws))
                        screening.validate_evaluation_outputs(samples if draws > 1 else samples.expand(-1, -1, 2), y)

    def test_counterfactual_returns_best_checkpoint_not_final_epoch(self):
        for scope, width in [("arrival", 1), ("trajectory", 241)]:
            x = np.zeros((4, 241, 2), dtype=np.float32)
            y = np.zeros((4, width), dtype=np.float32)
            final, best = object(), object()
            with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(screening, "fit_lstm_engression", return_value=final) as fit, \
                 patch.object(screening, "load_lstm_engressor_checkpoint", return_value=best) as reload:
                result = screening.train_counterfactual(x, y, "T", Path(directory), x.copy(), y.copy())
                self.assertIs(result, best)
                self.assertEqual(tuple(fit.call_args.args[1].shape), (4, width))
                self.assertEqual(tuple(fit.call_args.kwargs["y_val"].shape), (4, width))
                self.assertEqual(reload.call_args.args[0], fit.call_args.args[2].checkpoint_best_path)


if __name__ == "__main__":
    unittest.main()
