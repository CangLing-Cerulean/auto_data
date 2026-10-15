"""Asset-free N1 algebra checks; this is not real-path engineering acceptance.

Run: python scripts/check_response_tangent.py
Synthetic finite differences use predetermined h=1e-4 (ordinary trajectories).
The joint-zero scalar map has an O(|h|) central-difference error, because its
second derivative is not continuous at zero; it is tested separately.
"""
import importlib.util
import json
import unittest
import numpy as np
import response_tangent as rt


RTOL = 1e-9
ATOL = 1e-11


def adam(theta, m, v, step, g):
    """Independent ordinary Adam reference, also accepts complex numbers."""
    m1 = .9 * m + .1 * g
    v1 = .999 * v + .001 * g ** 2
    correction_m = 1. - .9 ** (step + 1)
    correction_v = 1. - .999 ** (step + 1)
    theta1 = theta - .001 * (m1 / correction_m) / (np.sqrt(v1 / correction_v) + 1e-8)
    return theta1, m1, v1


def synthetic_batch():
    rng = np.random.default_rng(1701)
    raw = rng.normal(size=(8, 2, 32))
    padded = np.pad(raw, ((0, 0), (0, 0), (12, 12)), mode='edge')
    trend = np.stack([padded[:, :, i:i + 25].mean(-1) for i in range(32)], -1)
    phi = np.concatenate((raw - trend, np.ones((8, 2, 1)), trend, np.ones((8, 2, 1))), -1)
    target = rng.normal(size=(8, 2, 2))
    theta = rng.normal(scale=.1, size=(2, 66, 2))
    s = rng.normal(scale=.01, size=(theta.size, 4))
    return phi, target, theta, np.arange(8) % 4, s


class TangentChecks(unittest.TestCase):
    def assertClose(self, actual, expected, *, rtol=RTOL, atol=ATOL):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def test_basis_actual_four_direction_projection(self):
        w = rt.direction_basis()
        self.assertClose(w.T @ w, np.eye(4))
        self.assertClose(np.ones(4) @ w[:, :3], np.zeros(3))
        # Saved float32 menu weights have nonzero common mass after rounding.
        weights = np.array([1.2, 1 - .2 / 3, 1 - .2 / 3, 1 - .2 / 3], dtype=np.float32)
        u = weights.astype(np.float64) - 1.
        a = rt.actual_coordinates(u)
        self.assertNotEqual(a[3], 0.)
        self.assertClose(w @ a, u)
        self.assertGreater(np.max(np.abs(w[:, :3] @ a[:3] - u)), 1e-9)
        rng = np.random.default_rng(2)
        s = rng.normal(size=(9, 4))
        self.assertClose(s @ a, sum(a[j] * s[:, j] for j in range(4)))

    def test_batch_terms_direct_blocks_and_redundant_bias(self):
        x, y, theta, groups, s = synthetic_batch()
        g, hs, forcing = rt.batch_terms(x, y, theta, groups, s)
        expected_g = np.zeros_like(theta)
        expected_hs = np.zeros((*theta.shape, 4))
        expected_c = np.zeros_like(expected_hs)
        w = rt.direction_basis()
        # Direct sample/channel/output loops are independent of einsum blocks.
        for b in range(len(x)):
            for c in range(2):
                hessian = 2 * np.outer(x[b, c], x[b, c]) / (len(x) * 2 * 2)
                for h in range(2):
                    error = x[b, c] @ theta[c, :, h] - y[b, h, c]
                    grad = 2 * x[b, c] * error / (len(x) * 2 * 2)
                    expected_g[c, :, h] += grad
                    expected_c[c, :, h] += grad[:, None] * w[groups[b]]
                    expected_hs[c, :, h] += hessian @ s.reshape(2, 66, 2, 4)[c, :, h]
        self.assertClose(g, expected_g.ravel())
        self.assertClose(hs, expected_hs.reshape(-1, 4))
        self.assertClose(forcing, expected_c.reshape(-1, 4))
        self.assertClose(g.reshape(theta.shape)[:, 32], g.reshape(theta.shape)[:, 65])
        self.assertClose(forcing.reshape(*theta.shape, 4)[:, 32],
                         forcing.reshape(*theta.shape, 4)[:, 65])
        null = np.zeros((*theta.shape, 4))
        null[:, 32] = 1.
        null[:, 65] = -1.
        _, hnull, _ = rt.batch_terms(x, y, theta, groups, null.reshape(-1, 4))
        self.assertClose(hnull, np.zeros_like(hnull))
        # Uniform forcing is one half of the parent gradient, not removed.
        self.assertClose(forcing[:, 3], g / 2.)

    def test_one_step_complex_differential(self):
        rng = np.random.default_rng(3)
        p = 7
        theta = rng.normal(size=p)
        m, v = rng.normal(size=p) * .03, rng.uniform(.1, .5, size=p)
        s, mt, vt = (rng.normal(size=(p, 4)) * .02 for _ in range(3))
        hessian = rng.normal(size=(p, p))
        hessian = hessian.T @ hessian / p
        c = rng.normal(size=(p, 4)) * .1
        g = hessian @ theta + .3
        result = rt.tangent_step(theta, m, v, 19, g, hessian @ s, c,
                                 rt.TangentState(s, mt, vt), history_all_zero=np.zeros(p, bool))
        for j in range(4):
            z = 1e-30j
            refs = adam(theta + z * s[:, j], m + z * mt[:, j], v + z * vt[:, j],
                        19, g + z * (hessian @ s[:, j] + c[:, j]))
            for actual, ref in zip((result.S, result.M, result.V), refs):
                self.assertClose(actual[:, j], ref.imag / z.imag)
        _, mnew, vnew = adam(theta, m, v, 19, g)
        bound = rt.tangent_step(theta, m, v, 19, g, hessian @ s, c,
                                rt.TangentState(s, mt, vt), history_all_zero=np.zeros(p, bool),
                                updated_parent_m=mnew, updated_parent_v=vnew)
        self.assertClose(bound.S, result.S)

    def test_one_and_multi_step_central_differences(self):
        rng = np.random.default_rng(4)
        p = 6
        design = rng.normal(scale=.15, size=(16, p))
        target = rng.normal(size=16)
        groups = np.arange(16) % 4
        w = rt.direction_basis()[groups]
        initial = rng.normal(scale=.1, size=p)
        hessian = design.T @ design / 16

        def run(a, steps):
            theta, m, v = initial.copy(), np.zeros(p), np.zeros(p)
            for step in range(steps):
                g = design.T @ ((design @ theta - target) * (1. + w @ a)) / 16
                theta, m, v = adam(theta, m, v, step, g)
            return theta, m, v

        theta, m, v = initial.copy(), np.zeros(p), np.zeros(p)
        tangent = rt.TangentState.zeros(p)
        history = np.ones(p, bool)
        for step in range(12):
            residual = design @ theta - target
            g = design.T @ residual / 16
            c = design.T @ (residual[:, None] * w) / 16
            tangent = rt.tangent_step(theta, m, v, step, g, hessian @ tangent.S, c,
                                      tangent, history_all_zero=history)
            history &= g == 0
            theta, m, v = adam(theta, m, v, step, g)
            if step in (0, 11):
                for j in range(4):
                    a = np.zeros(4)
                    a[j] = 1e-4
                    plus, minus = run(a, step + 1), run(-a, step + 1)
                    for actual, hi, lo in zip((tangent.S, tangent.M, tangent.V), plus, minus):
                        # Finite differences have truncation and cancellation;
                        # strict 1e-9/1e-11 is reserved for algebra/JVP above.
                        self.assertClose(actual[:, j], (hi - lo) / 2e-4, rtol=2e-6, atol=2e-10)

    def test_joint_zero_scalar_and_continuing_zero_history(self):
        forcing = np.array([[1e-10, -2e-10, 3e-10, 5e-11]])
        tangent = rt.TangentState.zeros(1)
        for step in range(3):
            tangent = rt.tangent_step(np.zeros(1), np.zeros(1), np.zeros(1), step,
                                      np.zeros(1), np.zeros((1, 4)), forcing, tangent,
                                      history_all_zero=np.ones(1, bool))
            self.assertClose(tangent.S, -(step + 1) * .001 * forcing / 1e-8)
            self.assertClose(tangent.V, np.zeros((1, 4)))
        # Fresh one-step joint mapping: a*c/(|a*c|+eps). At zero its
        # derivative exists, while the leading FD error is O(|h|), not O(h^2).
        c = forcing[0, 0]
        exact = -.001 * c / 1e-8
        errors = []
        for h in (1e-4, 5e-5):
            plus = adam(np.zeros(1), np.zeros(1), np.zeros(1), 0, np.array([h * c]))[0]
            minus = adam(np.zeros(1), np.zeros(1), np.zeros(1), 0, np.array([-h * c]))[0]
            fd = ((plus - minus) / (2 * h))[0]
            self.assertClose(fd, exact, rtol=2e-6, atol=1e-12)
            errors.append(abs(fd - exact))
        self.assertLess(errors[1], .501 * errors[0])

    def test_invalid_states(self):
        p = 2
        base = dict(parent_theta=np.zeros(p), parent_m=np.zeros(p), parent_v=np.zeros(p),
                    parent_step=0, gradient=np.ones(p), hvp_s=np.zeros((p, 4)),
                    forcing=np.ones((p, 4)), tangent=rt.TangentState.zeros(p),
                    history_all_zero=np.ones(p, bool))
        cases = [dict(parent_step=-1), dict(parent_step=True), dict(parent_m=np.ones(p)),
                 dict(parent_v=-np.ones(p)), dict(gradient=np.array([np.nan, 1.])),
                 dict(forcing=np.ones((p, 3))), dict(history_all_zero=np.ones(p)),
                 dict(parent_step=3, history_all_zero=np.zeros(p, bool), gradient=np.zeros(p)),
                 dict(updated_parent_m=np.ones(p)),
                 dict(updated_parent_m=np.zeros(p), updated_parent_v=np.zeros(p)),
                 dict(tangent=rt.TangentState(np.ones((p, 4)), np.zeros((p, 4)), np.zeros((p, 4))))]
        for change in cases:
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    rt.tangent_step(**(base | change))
        x, y, theta, groups, s = synthetic_batch()
        with self.assertRaises(ValueError):
            rt.batch_terms(x, y, theta, groups + 4, s)
        x[:, :, 65] = 0.
        with self.assertRaises(ValueError):
            rt.batch_terms(x, y, theta, groups, s)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'torch unavailable; JVP remains unverified')
    def test_torch_autograd_batch_and_adam_jvp(self):
        import torch
        x, y, theta, groups, s = synthetic_batch()
        g, hs, forcing = rt.batch_terms(x, y, theta, groups, s)
        tx, ty, tt = (torch.tensor(a, dtype=torch.float64) for a in (x, y, theta))
        tw = torch.tensor(rt.direction_basis()[groups], dtype=torch.float64)

        def gradient(t, a):
            residual = torch.einsum('bcd,cdh->bch', tx, t) - ty.transpose(1, 2)
            loss = (residual.square() * (1. + tw @ a)[:, None, None]).mean()
            return torch.autograd.grad(loss, t, create_graph=True)[0]

        tt.requires_grad_(True)
        a = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        self.assertClose(gradient(tt, a).detach().numpy().ravel(), g)
        for j in range(4):
            da = torch.eye(4, dtype=torch.float64)[j]
            _, product = torch.autograd.functional.jvp(gradient, (tt, a),
                           (torch.tensor(s[:, j].reshape(theta.shape)), da))
            self.assertClose(product.detach().numpy().ravel(), hs[:, j] + forcing[:, j])
        # Nonzero moments keep this reference in the smooth Adam domain.
        p = theta.size
        m, v = np.full(p, .03), np.full(p, .2)
        tangent = rt.TangentState(s, .1 * s, .01 * s)
        result = rt.tangent_step(theta.ravel(), m, v, 7, g, hs, forcing, tangent,
                                 history_all_zero=np.zeros(p, bool))

        def mapping(t, tm, tv, a):
            grad = gradient(t.reshape(theta.shape), a).reshape(-1)
            mm = .9 * tm + .1 * grad
            vv = .999 * tv + .001 * grad.square()
            out = t - .001 * (mm / (1. - .9 ** 8)) / (torch.sqrt(vv / (1. - .999 ** 8)) + 1e-8)
            return out, mm, vv

        for j in range(4):
            inputs = tuple(torch.tensor(z, dtype=torch.float64, requires_grad=True)
                           for z in (theta.ravel(), m, v, np.zeros(4)))
            tangents = tuple(torch.tensor(z, dtype=torch.float64)
                             for z in (s[:, j], .1 * s[:, j], .01 * s[:, j], np.eye(4)[j]))
            _, derivative = torch.autograd.functional.jvp(mapping, inputs, tangents)
            for actual, ref in zip((result.S, result.M, result.V), derivative):
                self.assertClose(actual[:, j], ref.detach().numpy())

        def trajectory(a):
            t = torch.tensor(theta.ravel(), dtype=torch.float64, requires_grad=True)
            tm, tv = torch.zeros_like(t), torch.zeros_like(t)
            for step in range(4):
                grad = gradient(t.reshape(theta.shape), a).reshape(-1)
                tm = .9 * tm + .1 * grad
                tv = .999 * tv + .001 * grad.square()
                t = t - .001 * (tm / (1. - .9 ** (step + 1))) / (torch.sqrt(tv / (1. - .999 ** (step + 1))) + 1e-8)
            return t, tm, tv

        t, mm, vv = theta.ravel().copy(), np.zeros(p), np.zeros(p)
        state = rt.TangentState.zeros(p)
        history = np.ones(p, bool)
        for step in range(4):
            gg, hss, cc = rt.batch_terms(x, y, t.reshape(theta.shape), groups, state.S)
            state = rt.tangent_step(t, mm, vv, step, gg, hss, cc, state,
                                    history_all_zero=history)
            history &= gg == 0
            t, mm, vv = adam(t, mm, vv, step, gg)
        for j in range(4):
            _, derivative = torch.autograd.functional.jvp(trajectory, torch.zeros(4, dtype=torch.float64),
                                                         torch.eye(4, dtype=torch.float64)[j])
            for actual, ref in zip((state.S, state.M, state.V), derivative):
                self.assertClose(actual[:, j], ref.detach().numpy())

        # Independent JVP of the joint first-step zero mapping. Rewriting the
        # scalar expression as g/(abs(g)+eps) avoids differentiating sqrt(g^2)
        # as two separate operations at their undefined intermediate derivative.
        coefficient = torch.tensor(1e-10, dtype=torch.float64)
        _, zero_jvp = torch.autograd.functional.jvp(
            lambda a: -.001 * coefficient * a / (torch.abs(coefficient * a) + 1e-8),
            torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
        self.assertClose(zero_jvp.detach().numpy(), -.001 * 1e-10 / 1e-8)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TangentChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'synthetic_checks_passed': result.wasSuccessful(),
                      'tests_run': result.testsRun, 'skipped': len(result.skipped),
                      'algebra_rtol': RTOL, 'algebra_atol': ATOL,
                      'torch_available': importlib.util.find_spec('torch') is not None,
                      'real_parent_replay_verified': False,
                      'real_batch_jvp_verified': False,
                      'float32_full_training_fd_verified': False,
                      'N1_engineering_accepted': False}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
