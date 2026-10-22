import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from m1_prefix_shrinkage import blend_parameters, fit_coefficients


class PrefixShrinkageTests(unittest.TestCase):
    def test_known_channel_scales_and_scalar_solution(self):
        u = np.arange(1, 25, dtype=np.float64).reshape(3, 4, 2)
        v = u * np.array([.25, .75])
        channel, _ = fit_coefficients(u, v, True)
        np.testing.assert_allclose(channel, [.25, .75], rtol=0, atol=1e-15)
        scalar, _ = fit_coefficients(u, v, False)
        np.testing.assert_allclose(scalar, np.sum(u * v) / np.sum(u * u))
        # Convex quadratic first-order condition independently checks the fit.
        self.assertAlmostEqual(float(np.sum(u * (v - scalar[None, None] * u))), 0., places=10)

    def test_constraints_and_no_direction(self):
        u = np.ones((2, 3, 3))
        u[:, :, 2] = 0
        v = np.ones_like(u) * np.array([-2., 4., 9.])
        result, _ = fit_coefficients(u, v, True)
        np.testing.assert_array_equal(result, [0., 1., 0.])

    def test_exact_endpoints_and_intermediate(self):
        raw = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
        paired = raw * -3 + 7
        mixed = blend_parameters(raw, paired, np.array([0., 1., .5]))
        np.testing.assert_array_equal(mixed[0], raw[0])
        np.testing.assert_array_equal(mixed[1], paired[1])
        np.testing.assert_array_equal(mixed[2], (raw[2] + paired[2]) / 2)

    def test_identity_discrepancy(self):
        u = np.arange(1, 25, dtype=np.float64).reshape(3, 4, 2)
        for per_channel in (False, True):
            alpha, _ = fit_coefficients(u, u, per_channel)
            np.testing.assert_array_equal(alpha, [1., 1.])

    def test_reject_invalid_inputs(self):
        u = np.ones((2, 3, 4))
        for invalid in (np.full_like(u, np.nan), np.ones((2, 3, 5))):
            with self.assertRaises(ValueError):
                fit_coefficients(u, invalid, True)
        with self.assertRaises(ValueError):
            blend_parameters(u, u, np.array([1., 1.01]))


if __name__ == '__main__':
    unittest.main()
