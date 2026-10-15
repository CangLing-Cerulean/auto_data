"""Synthetic-only N2 contract/math checks; never loads research assets."""

import json
import unittest

import numpy as np

from response_weight_proposal import (
    CHECKPOINTS, Q, SEEDS, W, Z, ProposalError, Response, SeedResponse,
    _audit, _metrics, _parents, dependency_versions, propose,
)


def synthetic(*, tie=False, zero_channel=False):
    seeds = []
    for seed in SEEDS:
        b, c = {}, {}
        for i, checkpoint in enumerate(CHECKPOINTS):
            r = np.ones(8)
            if zero_channel:
                r[0] = 0
            k = np.zeros((8, 4))
            k[:, 0] = -0.5
            k[:, 3] = 0.25  # Independently detect dropped rounding direction.
            if zero_channel:
                k[0] = 0
            for target, scale in ((b, 1.0 if tie else 1 + i / 10), (c, 1.0)):
                target[checkpoint] = Response(
                    r=r * scale, K_all=k.copy(), e=r * 3 * scale,
                    J_all=k * 2, mae_weights=np.full(8, 1 / 24),
                    channel_index=np.arange(8), channel_nmse_multiplier=np.full(8, 8.0))
        seeds.append(SeedResponse(seed, b, c))
    return seeds


class AlgebraChecks(unittest.TestCase):
    def test_basis(self):
        np.testing.assert_allclose(W.T @ W, np.eye(4), atol=1e-15)
        np.testing.assert_allclose(Q @ Z, np.zeros(3), atol=1e-15)

    def test_different_nmse_nmae_scales(self):
        x = synthetic()[0].C[500]
        metrics = _metrics(x, np.zeros(4))
        self.assertEqual(metrics["nmse"], 8)
        self.assertEqual(metrics["nmae"], 1)
        self.assertEqual(metrics["channel_nmse"], [8.0] * 8)

    def test_all_four_columns_and_actual_ess(self):
        seeds = synthetic()
        parents = _parents(seeds)
        w = np.array([1.1, 0.9, 1.05, 0.95], dtype=np.float32)
        u = w.astype(np.float64) - 1
        audit = _audit(seeds, parents, u, actual=True)
        self.assertNotEqual(audit["coordinates"][3], 0)
        a = W.T @ u
        x = seeds[0].C[500]
        expected = np.sum((x.r + x.K_all @ a)**2)
        self.assertEqual(audit["predictions"][0]["curves"]["C"][500]["nmse"], expected)
        self.assertEqual(audit["ess_fraction"], (Q @ w.astype(float))**2
                         / (Q @ w.astype(float)**2))
        dropped = np.sum((x.r + x.K_all[:, :3] @ a[:3])**2)
        self.assertNotEqual(expected, dropped)

    def test_ess_rejects_radius_valid_extreme(self):
        seeds = synthetic()
        audit = _audit(seeds, _parents(seeds), np.array([.2, .2, -.2, -.2]), actual=True)
        self.assertFalse(audit["passed"])
        self.assertGreater(audit["normalized_residuals"]["ess"], 0)

    def test_branch_curvature_is_sufficient(self):
        rng = np.random.default_rng(17)
        for _ in range(30):
            rk, rj = rng.normal(size=(2, 8))
            kk, kj = rng.normal(size=(2, 8, 4))
            a = rng.normal(size=4)
            gap = rj @ rj - rk @ rk
            bound = 2 * (kk.T @ rk - kj.T @ rj) @ a + np.sum((kk @ a)**2)
            difference = np.sum((rk + kk @ a)**2) - np.sum((rj + kj @ a)**2)
            self.assertAlmostEqual(difference, -gap + bound - np.sum((kj @ a)**2), places=12)

    def test_tiny_gap_returns_parent_before_solver(self):
        report = propose(synthetic(tie=True), "FULL", n1_cost_seconds=2.0)
        self.assertEqual(report["status"], "branch_unsupported")
        self.assertEqual(report["recommended"], "PARENT")
        self.assertEqual(len(report["unsupported_branches"]), 15)
        self.assertIsNone(report["solver"])

    def test_missing_checkpoint_retains_failure(self):
        seeds = synthetic()
        del seeds[0].B[3000]
        with self.assertRaises(ProposalError) as caught:
            propose(seeds, "FULL", n1_cost_seconds=2.0)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(caught.exception.report["status"], "engineering_failure")
        self.assertTrue(caught.exception.report["errors"])

    def test_zero_channel_is_absolute_zero(self):
        seeds = synthetic(zero_channel=True)
        audit = _audit(seeds, _parents(seeds), np.zeros(4), actual=True)
        for seed in SEEDS:
            self.assertEqual(audit["normalized_residuals"][f"{seed}/C/channel/0"], 0)

    def test_missing_dependency_is_explicit_failure(self):
        if dependency_versions()["cvxpy"] is not None:
            self.skipTest("CVXPY exists; covered by solver checks")
        with self.assertRaises(ProposalError) as caught:
            propose(synthetic(), "FULL", n1_cost_seconds=2.0)
        self.assertIsInstance(caught.exception.__cause__, ModuleNotFoundError)
        self.assertIsNone(caught.exception.report["recommended"])
        self.assertEqual(caught.exception.report["errors"][0]["type"], "ModuleNotFoundError")


@unittest.skipUnless(all(dependency_versions()[p] is not None for p in ("cvxpy", "clarabel")),
                     "CVXPY/Clarabel unavailable: solving NOT verified")
class SolverChecks(unittest.TestCase):
    def test_interior_full_solution_matches_closed_form(self):
        seeds = synthetic()
        r = np.array([-.01, .006, -.004, 1., 1., 1., 1., 1.])
        k = np.zeros((8, 4))
        k[:3, :3] = np.eye(3)
        for seed in seeds:
            for checkpoint in CHECKPOINTS:
                seed.B[checkpoint].K_all[:] = 0.
                seed.B[checkpoint].J_all[:] = 0.
                seed.C[checkpoint] = Response(r.copy(), k.copy(), r.copy(), k.copy(),
                    np.full(8, 1/8), np.arange(8), np.full(8, 8.))
        report = propose(seeds, 'FULL', n1_cost_seconds=2.0)
        actual = np.array(report['raw']['v'])
        normal = (k[:, :3].T @ k[:, :3]) / (r @ r) + .01 * (Z.T @ np.diag(Q) @ Z)
        linear = k[:, :3].T @ r / (r @ r)
        expected = np.linalg.solve(normal, -linear)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-5)
        self.assertLess(np.linalg.norm(2 * (normal @ actual + linear)), 3e-6)
        self.assertEqual(report['solver']['status'], 'optimal')
        self.assertTrue(report['actual_audit']['passed'])

    def test_full_and_linear(self):
        reports = [propose(synthetic(), method, n1_cost_seconds=2.0)
                   for method in ("FULL", "LINEAR")]
        for report in reports:
            self.assertEqual(report["solver"]["status"], "optimal")
            self.assertTrue(report["actual_audit"]["passed"])
            self.assertGreaterEqual(report["actual_audit"]["mean_gain_percent"], 1.0)
            self.assertEqual(report["recommended"], report["method"])
            # Independent direct-array constraint checks after float32 conversion.
            weights = np.array(report["proposal"]["weights_float32"], dtype=np.float32)
            u = weights.astype(float) - 1
            a = W.T @ u
            self.assertLessEqual(abs(Q @ u), 1e-7)
            self.assertLessEqual(np.max(np.abs(u)), .2 + 1e-7)
            self.assertGreaterEqual((Q @ weights)**2 / (Q @ (weights.astype(float)**2)), .98 - 1e-7)
            for seed in synthetic():
                b = {k: np.sum((x.r + x.K_all @ a)**2) for k, x in seed.B.items()}
                self.assertEqual(min(CHECKPOINTS, key=b.get), 500)
                x = seed.C[500]
                self.assertLessEqual(np.dot(x.mae_weights, np.abs(x.e + x.J_all @ a)),
                                     1.01 * np.dot(x.mae_weights, np.abs(x.e)) + 1e-7)
                for c in range(8):
                    rows = x.channel_index == c
                    self.assertLessEqual(np.sum((x.r[rows] + x.K_all[rows] @ a)**2),
                                         1.05 * np.sum(x.r[rows]**2) + 1e-7)


if __name__ == "__main__":
    print(json.dumps({"dependency_versions": dependency_versions(),
                      "scope": "synthetic_only_no_scientific_training"}, sort_keys=True))
    unittest.main(verbosity=2)
