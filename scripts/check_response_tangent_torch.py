"""Asset-free checks against the frozen NumPy N1 and independent Torch AD."""
import json
import time
import unittest
import numpy as np
import torch
import response_tangent as reference
import response_tangent_torch as candidate

RTOL = 1e-9
ATOL = 1e-11


def reference_adam(theta, m, v, step, gradient):
    next_m = .9 * m + .1 * gradient
    next_v = .999 * v + .001 * gradient.square()
    # This joint-zero extension has the same first derivative as Adam at a
    # certified all-zero gradient history; avoids differentiating sqrt alone.
    zero = next_v.detach() == 0
    safe_v = torch.where(zero, torch.ones_like(next_v), next_v)
    normal = (next_m / (1 - .9 ** (step + 1))) / (torch.sqrt(safe_v / (1 - .999 ** (step + 1))) + 1e-8)
    change = torch.where(zero, next_m / ((1 - .9 ** (step + 1)) * 1e-8), normal)
    return theta - .001 * change, next_m, next_v


def fixture(device):
    rng = np.random.default_rng(1701)
    phi = rng.normal(size=(12, 2, 66))
    phi[:, :, [32, 65]] = 1
    values = [phi, rng.normal(size=(12, 3, 2)), rng.normal(scale=.1, size=(2, 66, 3))]
    return (*[torch.tensor(x, dtype=torch.float64, device=device) for x in values],
            torch.arange(12, dtype=torch.int64, device=device) % 4)


def gradient(phi, target, theta, a, groups):
    prediction = torch.einsum('bcd,cdh->bhc', phi, theta)
    weights = 1 + (candidate.basis(theta.device) @ a)[groups]
    loss = ((prediction - target).square() * weights[:, None, None]).mean()
    return torch.autograd.grad(loss, theta, create_graph=True)[0]


class KernelChecks(unittest.TestCase):
    def close(self, actual, expected):
        actual = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else actual
        expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else expected
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_basis_and_batch_reference(self):
        for device in self.devices:
            phi, y, theta, groups = fixture(device)
            s = torch.linspace(-.02, .03, theta.numel() * 4, device=device, dtype=torch.float64).reshape(-1, 4)
            self.close(candidate.basis(device), reference.direction_basis())
            actual = candidate.batch_terms(phi, y, theta, groups, s)
            expected = reference.batch_terms(*[x.cpu().numpy() for x in (phi, y, theta, groups, s)])
            for left, right in zip(actual, expected):
                self.close(left, right)

    def test_one_step_and_actual_float32_moments(self):
        rng = np.random.default_rng(3)
        for device in self.devices:
            p = 31
            arrays = [rng.normal(size=p), rng.normal(scale=.03, size=p), rng.uniform(.1, .5, size=p),
                      rng.normal(size=p), *[rng.normal(scale=.01, size=(p, 4)) for _ in range(5)]]
            theta, m, v, g, hs, forcing, s, mt, vt = [torch.tensor(x, dtype=torch.float64, device=device) for x in arrays]
            new_m = (.9 * m + .1 * g).float().double()
            new_v = (.999 * v + .001 * g.square()).float().double()
            state = candidate.TangentState(s, mt, vt)
            actual = candidate.tangent_step(theta, m, v, 17, g, hs, forcing, state,
                history_all_zero=torch.zeros(p, dtype=torch.bool, device=device),
                updated_parent_m=new_m, updated_parent_v=new_v)
            expected = reference.tangent_step(*arrays[:3], 17, arrays[3], arrays[4], arrays[5],
                reference.TangentState(*arrays[6:]), history_all_zero=np.zeros(p, dtype=bool),
                updated_parent_m=new_m.cpu().numpy(), updated_parent_v=new_v.cpu().numpy())
            for field in ('S', 'M', 'V'):
                self.close(getattr(actual, field), getattr(expected, field))

    def test_multistep_autodiff(self):
        for device in self.devices:
            phi, y, initial, groups = fixture(device)
            theta = initial.clone()
            m = torch.zeros_like(theta)
            v = torch.zeros_like(theta)
            state = candidate.TangentState.zeros(theta.numel(), device)
            history = torch.ones(theta.numel(), dtype=torch.bool, device=device)
            for step in range(12):
                g, hs, forcing = candidate.batch_terms(phi, y, theta, groups, state.S)
                state = candidate.tangent_step(theta.flatten(), m.flatten(), v.flatten(), step,
                    g, hs, forcing, state, history_all_zero=history)
                history &= g == 0
                theta, m, v = reference_adam(theta, m, v, step, g.reshape(theta.shape))
            def trajectory(a):
                t = initial.clone().requires_grad_(True)
                mm, vv = torch.zeros_like(t), torch.zeros_like(t)
                for step in range(12):
                    gg = gradient(phi, y, t, a, groups)
                    t, mm, vv = reference_adam(t, mm, vv, step, gg)
                return t, mm, vv
            zero = torch.zeros(4, dtype=torch.float64, device=device)
            for direction in range(4):
                vector = torch.zeros_like(zero)
                vector[direction] = 1
                _, derivative = torch.autograd.functional.jvp(trajectory, zero, vector, strict=True)
                for field, expected in zip(('S', 'M', 'V'), derivative):
                    self.close(getattr(state, field)[:, direction], expected.flatten())

    def test_joint_zero(self):
        for device in self.devices:
            theta = torch.zeros(2, dtype=torch.float64, device=device)
            state = candidate.TangentState.zeros(2, device)
            forcing = torch.arange(8, dtype=torch.float64, device=device).reshape(2, 4) / 16
            actual = candidate.tangent_step(theta, theta, theta, 0, theta, state.S, forcing, state,
                history_all_zero=torch.ones(2, dtype=torch.bool, device=device))
            def mapping(a):
                return reference_adam(theta, theta, theta, 0, forcing @ a)
            for j in range(4):
                vector = torch.eye(4, dtype=torch.float64, device=device)[j]
                _, expected = torch.autograd.functional.jvp(mapping, torch.zeros_like(vector), vector, strict=True)
                for field, value in zip(('S', 'M', 'V'), expected):
                    self.close(getattr(actual, field)[:, j], value)

    def test_invalid_state_is_rejected(self):
        for device in self.devices:
            zero = torch.zeros(2, dtype=torch.float64, device=device)
            state = candidate.TangentState.zeros(2, device)
            with self.assertRaises(ValueError):
                candidate.tangent_step(zero, zero, -torch.ones_like(zero), 1, zero, state.S, state.S, state,
                    history_all_zero=torch.zeros(2, dtype=torch.bool, device=device))
            with self.assertRaises(ValueError):
                candidate.tangent_step(zero, zero, zero, 0, zero, state.S, state.S, state,
                    history_all_zero=torch.ones(2, dtype=torch.bool, device=device), updated_parent_m=zero)
            with self.assertRaises(ValueError):
                candidate.tangent_step(zero, zero, zero, 0, zero, state.S, state.S, state,
                    history_all_zero=torch.ones(2, dtype=torch.bool, device=device),
                    updated_parent_m=torch.ones_like(zero), updated_parent_v=torch.ones_like(zero))

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('This engineering check requires the registered CUDA environment')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        cls.devices = [torch.device('cpu'), torch.device('cuda')]


if __name__ == '__main__':
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(KernelChecks))
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL', tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        seconds=time.perf_counter()-started, rtol=RTOL, atol=ATOL,
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda)))
    raise SystemExit(not result.wasSuccessful())
