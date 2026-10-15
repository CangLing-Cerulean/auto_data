"""Synthetic algebra checks; no dataset, checkpoints, or training are accessed."""

import unittest

import numpy as np

from fourgroup_policy import (
    ACTIONS, A_INTERVALS, METHODS, SEEDS, SHORT_STEPS, TASKS, WINDOW_COUNT,
    BudgetCalibrationError, action_weights, actual_weights, assign_groups,
    calibrate_short, recommend, recommend_from_scores, recommendation_stop,
)


def gain_rows(value: float = 0.0) -> dict:
    return {seed: {action: value for action in ACTIONS} for seed in SEEDS}


class FourGroupPolicyChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.cycles = np.repeat(np.arange(308), 21)
        self.groups = assign_groups("G1", self.cycles, (0, 308))

    def test_group_boundaries_and_arbitrary_window_order(self) -> None:
        for task, interval in A_INTERVALS.items():
            cycles = np.repeat(np.arange(*interval), 21)
            groups = assign_groups(task, cycles, interval)
            np.testing.assert_array_equal(groups, np.repeat(np.arange(4), 1617))
            np.testing.assert_array_equal(assign_groups(task, cycles[::-1], interval), groups[::-1])
        self.assertEqual(self.groups[1616], 0)
        self.assertEqual(self.groups[1617], 1)

    def test_invalid_cycle_coverage_and_types(self) -> None:
        changed = self.cycles.copy()
        changed[0] = 1
        outside = self.cycles.copy()
        outside[-1] = 308
        for cycles in (changed, outside, self.cycles[:-1], self.cycles.astype(float)):
            with self.assertRaises(ValueError):
                assign_groups("G1", cycles, (0, 308))
        for task, interval in (("G4", (0, 308)), ("G1", (1, 309)), ("G1", (0.0, 308.0))):
            with self.assertRaises(ValueError):
                assign_groups(task, self.cycles, interval)

    def test_fixed_menu_actual_float32_mass_and_ess(self) -> None:
        self.assertEqual(ACTIONS, tuple(f"G{g}_{d}" for g in range(4) for d in ("UP", "DOWN")))
        for index, action in enumerate(ACTIONS):
            plan = action_weights(self.groups, action)
            sign = 1 if index % 2 == 0 else -1
            expected = np.full(4, 1 - sign * .2 / 3, dtype=np.float32)
            expected[index // 2] = 1 + sign * .2
            np.testing.assert_array_equal(plan.group_weights, expected)
            self.assertEqual(plan.window_weights.dtype, np.dtype("float32"))
            self.assertEqual(plan.moment_mass.dtype, np.dtype("float64"))
            np.testing.assert_array_equal(plan.window_weights, expected[self.groups])
            np.testing.assert_array_equal(plan.sample_probability, np.full(WINDOW_COUNT, 1 / WINDOW_COUNT))
            np.testing.assert_array_equal(plan.moment_mass, plan.sample_probability * plan.window_weights.astype(float))
            self.assertNotEqual(plan.mass_deviation, 0.0)
            self.assertEqual(plan.mean_weight, float(expected.astype(float).mean()))
            self.assertLessEqual(abs(plan.mass_deviation), 1e-7)
            self.assertLessEqual(plan.effective_radius, .2 + 1e-7)
            independent_ess = plan.window_weights.astype(float).sum()**2 / np.square(plan.window_weights.astype(float)).sum()
            self.assertAlmostEqual(plan.ess, independent_ess, places=9)
        self.assertEqual(action_weights(self.groups, "PARENT").ess_fraction, 1)

    def test_weight_validation_never_repairs(self) -> None:
        accepted = actual_weights(self.groups, [1.20000005, .79999995, 1, 1])
        self.assertGreater(accepted.effective_radius, .2)
        for weights in ([1.2000002, .7999998, 1, 1], [1.000001] * 4,
                        [np.nan, 1, 1, 1], [np.inf, 1, 1, 1], [1, 1, 1]):
            with self.assertRaises(ValueError):
                actual_weights(self.groups, weights)
        with self.assertRaises(ValueError):
            action_weights(np.zeros(WINDOW_COUNT, dtype=int), "G0_UP")

    def test_mean_relative_gain_not_ratio_of_means(self) -> None:
        parents = dict(zip(SEEDS, (1.0, 10.0, 100.0)))
        scores = {s: {a: parents[s] for a in ACTIONS} for s in SEEDS}
        for seed, gain in zip(SEEDS, (30, 0, -15)):
            scores[seed][ACTIONS[0]] = parents[seed] * (1 - gain / 100)
        result = recommend_from_scores(parents, scores)
        self.assertEqual(result.action, ACTIONS[0])
        self.assertAlmostEqual(result.best_mean_gain, 5.0)
        ratio_of_means = 100 * (1 - np.mean([scores[s][ACTIONS[0]] for s in SEEDS]) / np.mean(list(parents.values())))
        self.assertLess(ratio_of_means, 0)

    def test_tie_threshold_and_common_action(self) -> None:
        rows = gain_rows(1.0)
        self.assertEqual(recommend(rows).action, ACTIONS[0])
        self.assertEqual(recommend(gain_rows(np.nextafter(1.0, 0.0))).action, "PARENT")
        rows = gain_rows(-10)
        for seed, action in zip(SEEDS, ACTIONS[:3]):
            rows[seed][action] = 10
        self.assertEqual(recommend(rows).action, "PARENT")

    def test_missing_nonfinite_and_wrong_seed_action_order(self) -> None:
        rows = gain_rows()
        for invalid in ({s: rows[s] for s in reversed(SEEDS)}, {SEEDS[0]: rows[SEEDS[0]]},
                        {s: dict(reversed(list(rows[s].items()))) for s in SEEDS}):
            with self.assertRaises(ValueError):
                recommend(invalid)
        for invalid_value in (np.nan, np.inf, True):
            rows = gain_rows()
            rows[SEEDS[0]][ACTIONS[0]] = invalid_value
            with self.assertRaises(ValueError):
                recommend(rows)
        for invalid_parent in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                recommend_from_scores(dict.fromkeys(SEEDS, invalid_parent), gain_rows())

    def test_budget_maxima_minima_equality_and_no_fallback(self) -> None:
        proxy = dict(zip(SEEDS, (4, 8, 6)))
        identity = dict(zip(SEEDS, (3, 1, 2)))
        prefixes = {s: dict(zip(SHORT_STEPS, values)) for s, values in zip(
            SEEDS, ((16, 17, 18, 19, 20), (14, 15, 16, 17, 18), (20, 21, 22, 23, 24)))}
        result = calibrate_short(proxy, identity, prefixes)
        self.assertEqual(result.candidate_seconds, 12)
        self.assertEqual(result.required_prefix_seconds, 15)
        self.assertEqual(result.short_steps, 1000)
        prefixes[SEEDS[1]] = dict.fromkeys(SHORT_STEPS, 14.99)
        with self.assertRaises(BudgetCalibrationError):
            calibrate_short(proxy, identity, prefixes)
        prefixes[SEEDS[1]][2500] = np.nan
        with self.assertRaises(ValueError):
            calibrate_short(proxy, identity, prefixes)
        with self.assertRaises(ValueError):
            calibrate_short(dict(reversed(list(proxy.items()))), identity, prefixes)

    def test_stop_requires_one_baseline_across_all_tasks(self) -> None:
        recommendations = {m: dict.fromkeys(TASKS, "PARENT") for m in METHODS}
        recommendations["M1"] = dict(zip(TASKS, ACTIONS[:3]))
        for method, task in zip(("RAW", "SOURCE", "SHORT"), TASKS):
            recommendations[method][task] = recommendations["M1"][task]
        self.assertFalse(recommendation_stop(recommendations).stop)
        recommendations["SOURCE"] = recommendations["M1"].copy()
        result = recommendation_stop(recommendations)
        self.assertTrue(result.stop)
        self.assertEqual(result.matching_baselines, ("SOURCE",))
        recommendations["M1"] = dict.fromkeys(TASKS, "PARENT")
        self.assertIn("PARENT", recommendation_stop(recommendations).matching_baselines)

    def test_stop_invalid_task_method_action_order(self) -> None:
        base = {m: dict.fromkeys(TASKS, "PARENT") for m in METHODS}
        for invalid in (dict(reversed(list(base.items()))), {m: base[m] for m in METHODS[:-1]}):
            with self.assertRaises(ValueError):
                recommendation_stop(invalid)
        for task_map in ({"G1": "PARENT", "G2": "PARENT", "G4": "PARENT"},
                         dict.fromkeys(reversed(TASKS), "PARENT"), dict.fromkeys(TASKS, "UNKNOWN")):
            base["RAW"] = task_map
            with self.assertRaises(ValueError):
                recommendation_stop(base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
