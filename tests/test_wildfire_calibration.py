"""Small calibration and screening checks; no training or real data loading."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from experiments.wildfire_sensitivity import main as screening


class CalibrationAcceptanceTests(unittest.TestCase):
    def diagnostic(self, pit_row, n_blocks=4, scope="trajectory"):
        # A uniform base forecast makes these outcomes known quantile positions.
        y = np.tile(pit_row, (n_blocks, 1))
        x = np.zeros((n_blocks, len(pit_row), 1), dtype=np.float32)
        times = np.arange(n_blocks) * 30.0
        if scope == "arrival":
            y = y.reshape(-1, 1)
            x = np.zeros((len(y), 241, 1), dtype=np.float32)
            times = np.repeat(times, len(pit_row))
        model = SimpleNamespace(
            device=torch.device("cpu"),
            sample=lambda x, sample_size: torch.linspace(0, 1, sample_size).expand(
                len(x), y.shape[1], sample_size
            ),
        )
        with patch.object(screening, "pit_values", side_effect=lambda samples, y, seed: y):
            with contextlib.redirect_stdout(io.StringIO()):
                return screening.print_calibration(
                    model, x, y, "test", times, evaluation_scope=scope
                )

    def test_only_required_levels_determine_acceptance(self):
        for scope in ["trajectory", "arrival"]:
            with self.subTest(scope=scope):
                result = self.diagnostic([0.5] * 18 + [0.03, 0.01], scope=scope)
                self.assertFalse(result["passes"][0.50])
                self.assertFalse(result["passes"][0.75])
                self.assertTrue(result["passes"][0.90])
                self.assertTrue(result["passes"][0.95])
                self.assertTrue(result["calibrated"])

    def test_failure_at_either_required_level_rejects(self):
        for pit_row, failed_level in [
            ([0.5] * 17 + [0.03, 0.03, 0.01], 0.90),
            ([0.5] * 18 + [0.01, 0.01], 0.95),
        ]:
            for scope in ["trajectory", "arrival"]:
                with self.subTest(level=failed_level, scope=scope):
                    result = self.diagnostic(pit_row, scope=scope)
                    other_level = 0.95 if failed_level == 0.90 else 0.90
                    self.assertTrue(result["passes"][other_level])
                    self.assertFalse(result["passes"][failed_level])
                    self.assertFalse(result["calibrated"])

    def test_insufficient_blocks_do_not_pass(self):
        for scope in ["trajectory", "arrival"]:
            with self.subTest(scope=scope):
                result = self.diagnostic([0.5] * 18 + [0.03, 0.01], n_blocks=1, scope=scope)
                self.assertTrue(np.isnan(result["standard_error"][0.90]))
                self.assertFalse(result["calibrated"])


class ScreeningGateTests(unittest.TestCase):
    def run_screening(self, acceptance, mode, scope="trajectory", control_coverage=.95,
                      wildfire_coverages=None, candidate_order_seed=None):
        names = (
            "P", "LAND", "SLP", "WS", "TS", "DUST^(1/5)", "EPV",
            "CHL^(1/2)", "DMS^(1/2)", "SO2EM^(1/5)", "CO^(1/5)", "LWP^(1/5)",
        )
        dataset = SimpleNamespace(
            feature_names=names,
            x=np.broadcast_to(np.arange(12)[:, None, None], (12, 2, len(names))).copy(),
            time=np.arange(12, dtype=float),
        )
        arms = SimpleNamespace(
            train_clean=np.arange(4), test_clean=np.arange(4, 8),
            train_wildfire=np.array([8, 9]), test_wildfire=np.array([10, 11]),
        )
        def train(x, y, candidate, path, val_x, val_y):
            np.testing.assert_array_equal(x[:, 0, 0], [0, 1])
            np.testing.assert_array_equal(val_x[:, 0, 0], [2, 3])
            self.assertEqual(x.shape[1], 2)  # Inputs keep all timesteps.
            self.assertEqual(val_x.shape[1], 2)
            expected_steps = 1 if scope == "arrival" else 2
            self.assertEqual(y.shape, (2, expected_steps))
            self.assertEqual(val_y.shape, (2, expected_steps))
            return candidate

        def recalibrate(model, x, y, path, evaluation_scope):
            self.assertEqual(evaluation_scope, scope)
            self.assertEqual(y.shape[1], 1 if scope == "arrival" else 2)
            np.testing.assert_array_equal(x[:, 0, 0], [4, 5])
            return model

        def assess(model, x, y, *args, **kwargs):
            self.assertEqual(kwargs["evaluation_scope"], scope)
            self.assertEqual(y.shape[1], 1 if scope == "arrival" else 2)
            np.testing.assert_array_equal(x[:, 0, 0], [6, 7])
            return {"calibrated": acceptance(model), "central": {0.95: control_coverage}}

        fit = Mock(side_effect=train)
        calibration = Mock(side_effect=assess)

        def coverage(model, x, y, label, evaluation_scope):
            self.assertEqual(evaluation_scope, scope)
            self.assertEqual(y.shape[1], 1 if scope == "arrival" else 2)
            self.assertEqual(label, "wildfire")
            np.testing.assert_array_equal(x[:, 0, 0], [8, 9, 10, 11])
            if wildfire_coverages is not None:
                return wildfire_coverages[model]
            return 0.90 if model == "CO^(1/5)" else 0.94

        intervals = Mock(side_effect=coverage)
        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            directory = stack.enter_context(TemporaryDirectory())
            for name, replacement in {
                "ALL_COVARIATE_NAMES": names,
                "load_wildfire_arms": Mock(return_value=(dataset, arms)),
                "split_clean_holdout": Mock(side_effect=[
                    (np.arange(6), np.array([6, 7])),
                    (np.arange(4), np.array([4, 5])),
                    (np.arange(2), np.array([2, 3])),
                ]),
                "month_matched_subsample": Mock(side_effect=lambda pool, *args, **kwargs: pool),
                "train_counterfactual": fit,
                "recalibrate_counterfactual": Mock(side_effect=recalibrate),
                "print_calibration": calibration,
                "interval_coverage": intervals,
            }.items():
                stack.enter_context(patch.object(screening, name, replacement))
            stack.enter_context(contextlib.redirect_stdout(output))
            screening.main(mode, "unused.mat", Path(directory), evaluation_scope=scope,
                           candidate_order_seed=candidate_order_seed)
            self.assertEqual(screening.load_wildfire_arms.call_args.kwargs["seed"], 2026)
            self.assertEqual([call.kwargs["seed"] for call in screening.split_clean_holdout.call_args_list],
                             [2026, 2228, 2127])
            artifacts = Path(directory) / "arrival" if scope == "arrival" else Path(directory)
            order = json.loads((artifacts / "screening_order.json").read_text())
            self.assertEqual(order["candidate_order_seed"], candidate_order_seed)
            self.assertEqual(order["sweep_seed"], 2026)
            expected_seeds = {
                "sweep": 2026, "dataset": 2026, "paper_split": 2043,
                "training_each_candidate": 2026, "control_split": 2026,
                "control_month_matching": 2026, "recalibration_split": 2228,
                "recalibration_month_matching": 2228, "validation_split": 2127,
                "pit_batch_base": 2026, "candidate_order": candidate_order_seed,
            }
            self.assertEqual(order["seeds"], expected_seeds)
            for name, seed in expected_seeds.items():
                self.assertIn(f"  {name}: {seed}\n", output.getvalue())
            self.assertEqual(order["evaluation_batch_size"], screening.EVALUATION_BATCH_SIZE)
            self.assertEqual(order["prediction_samples"], screening.PREDICTION_SAMPLES)
            self.assertIn("zero-based batch start row", order["seed_usage"]["pit_batches"])
            self.assertIn("no separate sampling seed", order["seed_usage"]["prediction_draws"])
            self.assertEqual(order["initial_candidate_order"], [call.args[2] for call in fit.call_args_list[:2]])
            with np.load(artifacts / "screening_split.npz") as splits:
                self.assertEqual(splits["evaluation_scope"].item(), scope)
                self.assertEqual(splits["training_scope"].item(), scope)
                for role, expected in {
                    "train": [0, 1], "validation": [2, 3],
                    "recalibration": [4, 5], "control": [6, 7],
                }.items():
                    np.testing.assert_array_equal(splits[role], expected)
        return fit, intervals, output.getvalue()

    def test_order_seed_changes_order_reproducibly_without_changing_global_rng(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        for seed, expected in [
            (None, ["CO^(1/5)", "LWP^(1/5)", "CO^(1/5)"]),
            (2027, ["LWP^(1/5)", "CO^(1/5)", "CO^(1/5)"]),
            (2027, ["LWP^(1/5)", "CO^(1/5)", "CO^(1/5)"]),
            (1, ["CO^(1/5)", "LWP^(1/5)", "CO^(1/5)"]),
        ]:
            with self.subTest(seed=seed):
                fit, _, _ = self.run_screening(lambda model: True, "until_stable", "arrival",
                                               candidate_order_seed=seed)
                self.assertEqual([call.args[2] for call in fit.call_args_list], expected)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        torch.testing.assert_close(torch_state, torch.random.get_rng_state())

    def test_arrival_scope_reaches_all_evaluation_stages(self):
        fit, intervals, output = self.run_screening(lambda model: True, "until_stable", "arrival")
        self.assertGreater(intervals.call_count, 0)
        self.assertIn("evaluation scope: arrival", output)

    def test_arrival_rejects_bad_calibration_and_retries_after_inputs_grow(self):
        _, intervals, output = self.run_screening(lambda model: False, "until_stable", "arrival")
        intervals.assert_not_called()
        self.assertIn("undetermined: ['CO^(1/5)', 'LWP^(1/5)']", output)
        attempts = {}

        def acceptance(model):
            attempts[model] = attempts.get(model, 0) + 1
            return model != "CO^(1/5)" or attempts[model] > 1

        fit, intervals, output = self.run_screening(acceptance, "until_stable", "arrival")
        self.assertEqual([call.args[2] for call in fit.call_args_list],
                         ["CO^(1/5)", "LWP^(1/5)", "CO^(1/5)"])
        self.assertEqual([call.args[0].shape[2] for call in fit.call_args_list], [10, 10, 11])
        self.assertEqual(intervals.call_count, 2)
        self.assertIn("sensitive: ['CO^(1/5)']", output)
        self.assertIn("undetermined: []", output)

    def test_two_point_boundary_and_larger_drop_in_both_modes(self):
        for scope in ["arrival", "trajectory"]:
            with self.subTest(scope=scope):
                _, _, output = self.run_screening(
                    lambda model: True, "until_stable", scope, control_coverage=.97,
                    wildfire_coverages={"CO^(1/5)": .949, "LWP^(1/5)": .95},
                )
                self.assertIn("sensitive: ['CO^(1/5)']", output)
                self.assertIn("undetermined: []", output)

    def test_all_undetermined_stops_without_wildfire_evaluation(self):
        fit, intervals, output = self.run_screening(lambda model: False, "until_stable")
        self.assertEqual(fit.call_count, 2)
        intervals.assert_not_called()
        self.assertIn("undetermined: ['CO^(1/5)', 'LWP^(1/5)']", output)
        self.assertIn("sensitive: []", output)

    def test_undetermined_retried_only_after_input_set_grows(self):
        attempts = {}

        def acceptance(model):
            attempts[model] = attempts.get(model, 0) + 1
            return model != "CO^(1/5)" or attempts[model] > 1

        fit, intervals, output = self.run_screening(acceptance, "until_stable")
        self.assertEqual([call.args[2] for call in fit.call_args_list],
                         ["CO^(1/5)", "LWP^(1/5)", "CO^(1/5)"])
        self.assertEqual([call.args[0].shape[2] for call in fit.call_args_list], [10, 10, 11])
        self.assertEqual(intervals.call_count, 2)
        self.assertIn("sensitive: ['CO^(1/5)']", output)
        self.assertIn("undetermined: []", output)

    def test_single_pass_does_not_retry_undetermined(self):
        fit, intervals, output = self.run_screening(
            lambda model: model != "CO^(1/5)", "single_pass"
        )
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(intervals.call_count, 1)
        self.assertIn("undetermined: ['CO^(1/5)']", output)


class EvaluationScopeTests(unittest.TestCase):
    def test_arrival_training_target_is_final_value_and_model_has_one_output(self):
        from engression_modifications.lstm.generators import build_lstm_model

        data = np.arange(2 * 241 * 3, dtype=np.float32).reshape(2, 241, 3)
        x, y = screening.filter_dataset_columns(data, [0, 2], 1, "arrival")
        self.assertEqual(x.shape, (2, 241, 2))
        self.assertEqual(y.shape, (2, 1))
        np.testing.assert_array_equal(y[:, 0], data[:, -1, 1])
        np.testing.assert_array_equal(x, data[:, :, [0, 2]])
        config = screening.LSTMEngressionConfig(
            hidden_dim=4, num_layer=1, lstm_num_layers=1,
            noise_dim=2, global_latent_noise=True, device="cpu",
        )
        model = build_lstm_model(config, input_dim=x.shape[2], out_dim=y.shape[1])
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.from_numpy(x)).shape), (2, 1))

    def setUp(self):
        self.x = np.zeros((4, 2, 1), dtype=np.float32)
        # Earlier timesteps all miss; three of the four arrivals are covered.
        self.y = np.array([[2, .5], [2, .01], [2, .5], [2, .5]], dtype=np.float32)
        self.model = SimpleNamespace(
            device=torch.device("cpu"),
            sample=lambda x, sample_size: torch.linspace(0, 1, sample_size).expand(
                len(x), 2, sample_size
            ),
        )
        self.arrival_model = SimpleNamespace(
            device=torch.device("cpu"),
            sample=lambda x, sample_size: torch.linspace(0, 1, sample_size).expand(
                len(x), 1, sample_size
            ),
        )

    def test_arrival_coverage_and_block_error_ignore_earlier_timesteps(self):
        with contextlib.redirect_stdout(io.StringIO()):
            full = screening.interval_coverage(self.model, self.x, self.y, "test")
            arrival = screening.interval_coverage(
                self.arrival_model, self.x, self.y[:, -1:], "test", evaluation_scope="arrival"
            )
            metrics = screening.print_calibration(
                self.arrival_model, self.x, self.y[:, -1:], "test", np.array([0., 0., 30., 60.]),
                evaluation_scope="arrival",
            )
        self.assertEqual(full, 3 / 8)
        self.assertEqual(arrival, 3 / 4)
        self.assertEqual(metrics["central"][.95], arrival)
        # Three blocks: sizes (2,1,1), covered (1,1,1), pooled coverage .75.
        self.assertAlmostEqual(metrics["standard_error"][.95], .1875)
        self.assertEqual(sum(metrics["pit_counts"]), 4)

    def test_arrival_recalibration_uses_only_final_values(self):
        changed_history = self.y.copy()
        changed_history[:, 0] = -2
        changed_arrival = self.y.copy()
        changed_arrival[:, -1] = .9
        with TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            mappings = [screening.recalibrate_counterfactual(
                self.arrival_model, self.x, y[:, -1:], Path(directory) / f"{i}.json",
                evaluation_scope="arrival",
            ).quantile_levels for i, y in enumerate((self.y, changed_history, changed_arrival))]
        self.assertEqual(mappings[0], mappings[1])
        self.assertNotEqual(mappings[0], mappings[2])

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            screening.validate_evaluation_targets(self.x, self.y, "typo")

    def test_wrong_target_or_model_dimensions_cannot_be_silently_sliced(self):
        for scope, y, model in [
            ("arrival", self.y, self.model),
            ("arrival", self.y[:, -1:], self.model),
            ("trajectory", self.y[:, -1:], self.arrival_model),
            ("trajectory", self.y, self.arrival_model),
        ]:
            with self.subTest(scope=scope, shape=y.shape), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    screening.interval_coverage(model, self.x, y, "test", evaluation_scope=scope)

    def test_nonfinite_forecasts_and_target_leakage_are_rejected(self):
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                screening.validate_evaluation_outputs(torch.full((4, 2, 3), value), self.y)
        with self.assertRaises(ValueError):
            screening.filter_dataset_columns(self.x, [0], 0)


if __name__ == "__main__":
    unittest.main()
