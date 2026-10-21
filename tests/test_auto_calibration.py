import copy
import unittest
import numpy as np
import torch

from online_ts.auto_calibration import (
    ForecastSamples, GatedLinearCalibrator, Recipe, context_from_inputs,
    evaluate, state_digest, train_candidate, verify_time_order,
)


def samples(n=32, horizon=3, channels=2, first_row=0, offset=.4):
    prediction = np.zeros((n, horizon, channels), dtype=np.float32)
    context = np.zeros((n, channels, 3), dtype=np.float32)
    starts = first_row + np.arange(n, dtype=np.int64)
    return ForecastSamples(prediction, context, prediction + offset, starts, starts+8)


class AutoCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_identity_initialization_and_rng_preservation(self):
        torch.manual_seed(6)
        before = torch.get_rng_state().clone()
        model = GatedLinearCalibrator(3, 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        p, c = torch.randn(9, 3, 2), torch.randn(9, 2, 3)
        self.assertTrue(torch.equal(model(p, c), p))

    def test_forward_matches_channel_by_channel_reference(self):
        model = GatedLinearCalibrator(3, 2).double()
        with torch.no_grad():
            model.gate.copy_(torch.tensor([.4, -.2]))
            model.bias.copy_(torch.arange(6).reshape(2, 3)*.01)
        p, c = torch.randn(5, 3, 2, dtype=torch.float64), torch.randn(5, 2, 3, dtype=torch.float64)
        expected = torch.stack([
            p[:, :, k] + model.bias[k] + torch.tanh(model.gate[k]) *
            (torch.cat((p[:, :, k], c[:, k]), dim=1) @ model.weight[k]) for k in range(2)
        ], dim=-1)
        torch.testing.assert_close(model(p, c), expected, rtol=1e-12, atol=1e-12)

    def test_parameter_gradients_match_finite_differences(self):
        model = GatedLinearCalibrator(2, 1).double()
        with torch.no_grad():
            model.gate.fill_(.3)
        p = torch.tensor([[[.2], [.7]], [[-.3], [.4]]], dtype=torch.float64)
        c = torch.tensor([[[1., .1, .5]], [[.5, -.2, .3]]], dtype=torch.float64)
        y = p + .2
        ((model(p, c)-y)**2).mean().backward()
        for tensor, index in ((model.gate, (0,)), (model.bias, (0, 1)), (model.weight, (0, 2, 1))):
            analytic = tensor.grad[index].item()
            old = tensor[index].item()
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    tensor[index] = old + sign * 1e-6
                values.append(((model(p, c)-y)**2).mean().item())
            with torch.no_grad():
                tensor[index] = old
            self.assertAlmostEqual(analytic, (values[0]-values[1])/2e-6, places=8)

    def test_zero_input_offset_is_learnable(self):
        fitting, validation = samples(), samples(first_row=100)
        model, record = train_candidate(fitting, Recipe(learning_rate=.005, steps=160))
        result, _ = evaluate(model, validation)
        self.assertLess(result['mse'], .001)
        self.assertEqual(record['steps'], 160)

    def test_training_is_reproducible_and_does_not_modify_inputs(self):
        fitting = samples()
        before = [a.copy() for a in (fitting.prediction, fitting.context, fitting.truth)]
        recipe = Recipe(steps=40, learning_rate=.005, loss='huber')
        first, _ = train_candidate(fitting, recipe)
        second, _ = train_candidate(fitting, recipe)
        self.assertEqual(state_digest(first), state_digest(second))
        for actual, expected in zip((fitting.prediction, fitting.context, fitting.truth), before):
            np.testing.assert_array_equal(actual, expected)

    def test_incumbent_is_copied_and_evaluation_never_updates(self):
        incumbent = GatedLinearCalibrator(3, 2)
        before = state_digest(incumbent)
        child, _ = train_candidate(samples(), Recipe(steps=40, initialization='incumbent'), incumbent=incumbent)
        self.assertEqual(state_digest(incumbent), before)
        child_before = state_digest(child)
        result, _ = evaluate(child, samples(first_row=100, offset=100.))
        self.assertEqual(state_digest(child), child_before)
        self.assertGreater(result['mse'], 9000.)

    def test_chronology_and_strict_dimensions(self):
        verify_time_order(samples(), samples(first_row=100))
        with self.assertRaises(ValueError):
            verify_time_order(samples(), samples(first_row=39))
        with self.assertRaises(ValueError):
            verify_time_order(samples(), samples(first_row=100, channels=1))
        s = samples()
        with self.assertRaises(ValueError):
            ForecastSamples(s.prediction, s.context[:1], s.truth, s.input_start, s.label_end)
        with self.assertRaises(ValueError):
            ForecastSamples(s.prediction, s.context, s.truth[:, :, :1], s.input_start, s.label_end)

    def test_context_uses_only_given_observations(self):
        x = np.arange(24, dtype=float).reshape(2, 4, 3)
        context = context_from_inputs(x)
        np.testing.assert_allclose(context[:, :, 0], x.mean(axis=1))
        np.testing.assert_array_equal(context[:, :, 1], x[:, -1])
        np.testing.assert_allclose(context[:, :, 2], x.std(axis=1))

    def test_window_choice_is_chronological_and_nonempty(self):
        s = samples(n=5)
        np.testing.assert_array_equal(s.indices_for('all'), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(s.indices_for('recent_half'), [2, 3, 4])
        np.testing.assert_array_equal(s.indices_for('recent_quarter'), [3, 4])

    def test_recipe_and_bad_data_are_rejected(self):
        for fields in ({'steps': True}, {'learning_rate': float('nan')}, {'loss': 'unknown'}, {'steps': 40.}):
            with self.assertRaises(ValueError):
                Recipe(**fields)
        with self.assertRaises(ValueError):
            Recipe.from_dict({'steps': 40})
        s = samples()
        bad = s.truth.copy(); bad[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            ForecastSamples(s.prediction, s.context, bad, s.input_start, s.label_end)
        with self.assertRaises(ValueError):
            train_candidate(s, Recipe(initialization='incumbent'))


if __name__ == '__main__':
    unittest.main()
