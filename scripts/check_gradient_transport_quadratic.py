"""Fixed synthetic algebra checks, with all descriptive proxy outcomes retained."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from gradient_transport_quadratic import AdamState, adam_step, group_statistics, transported_gradient

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261015_gradient_transport_design_r1'


def direct_gradient(x, y, group, weights, theta):
    # Sample-wise reference; no use of group sufficient statistics.
    return np.mean((weights[group] * (x @ theta - y))[:, None] * x, axis=0)


def scalar_adam(state, grad, args):
    # Independent scalar-coordinate expression for the optimizer algebra check.
    k = state.step + 1
    first, second, theta = [], [], []
    for old_theta, old_first, old_second, value in zip(state.theta, state.first, state.second, grad):
        m = args['beta1'] * old_first + (1 - args['beta1']) * value
        v = args['beta2'] * old_second + (1 - args['beta2']) * value * value
        update = args['learning_rate'] * (m / (1 - args['beta1'] ** k)) / ((v / (1 - args['beta2'] ** k)) ** .5 + args['epsilon'])
        first.append(m)
        second.append(v)
        theta.append(old_theta - update)
    return AdamState(np.array(theta), np.array(first), np.array(second), k)


def initial(theta):
    return AdamState(theta.copy(), np.zeros_like(theta), np.zeros_like(theta))


def main():
    start = time.perf_counter()
    plan_path = STAGE / 'check_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    lock = json.loads((STAGE / 'source_lock.json').read_text(encoding='utf-8-sig'))
    for item in lock['files']:
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
    args = {key: plan[key] for key in ('learning_rate', 'beta1', 'beta2', 'epsilon')}
    tolerance = {key: plan['identity_' + key] for key in ('rtol', 'atol')}
    counts = {key: 0 for key in plan['required_checks']}
    counts['scalar_adam_agreement'] = 0
    counts['synthetic_optimizer_updates'] = 0
    maxima = {'gradient_identity': 0., 'defect_identity': 0., 'parent_path': 0., 'exact_batch_path': 0., 'full_batch_path': 0., 'optimizer': 0.}
    rows = []

    def advance(state, grad):
        actual = adam_step(state, grad, **args)
        expected = scalar_adam(state, grad, args)
        for name in ('theta', 'first', 'second'):
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), **tolerance)
            maxima['optimizer'] = max(maxima['optimizer'], float(np.max(np.abs(getattr(actual, name) - getattr(expected, name)))))
        counts['scalar_adam_agreement'] += 1
        counts['synthetic_optimizer_updates'] += 1
        return actual

    for seed in plan['seeds']:
        for fixture in plan['fixtures']:
            rng = np.random.default_rng(seed)
            n, d, groups = plan['samples'], plan['features'], plan['groups']
            assert d == 3 and groups == 3 and n == 48
            group = np.arange(n) % groups
            x = np.ones((n, d))  # Two duplicate bias coordinates are deliberately retained.
            x[:, 0] = rng.normal(size=n) * (group + 1) + group - 1
            if fixture == 'fixed_feature_curvature':
                x[:, 0] = .5
            y = (.4 + .3 * group) * x[:, 0] + .2 * group + rng.normal(scale=.1, size=n)
            theta0 = rng.normal(scale=.2, size=d)
            batches = [np.arange(n) if fixture == 'full_batch' else rng.integers(0, n, size=plan['batch_size']) for _ in range(plan['steps'])]
            _, full_h = group_statistics(x, y, group, groups, theta0)
            parent = initial(theta0)
            parent_before, parent_after, group_q, batch_h = [], [], [], []
            for idx in batches:
                parent_before.append(parent.theta.copy())
                q, h = group_statistics(x[idx], y[idx], group[idx], groups, parent.theta)
                group_q.append(q)
                batch_h.append(h)
                parent = advance(parent, direct_gradient(x[idx], y[idx], group[idx], np.ones(groups), parent.theta))
                parent_after.append(parent.theta.copy())

            for weights_list in plan['weights']:
                w = np.asarray(weights_list, dtype=np.float64)
                truth = initial(theta0)
                proxy = initial(theta0)
                exact = initial(theta0)
                mean_child = initial(theta0)
                mean_parent = initial(theta0)
                max_proxy_error, max_m1_error = 0., 0.
                for t, idx in enumerate(batches):
                    z = proxy.theta
                    actual = direct_gradient(x[idx], y[idx], group[idx], w, z)
                    batch_transport = transported_gradient(group_q[t], batch_h[t], w, z, parent_before[t])
                    np.testing.assert_allclose(batch_transport, actual, **tolerance)
                    counts['direct_gradient_vs_batch_transport'] += 1
                    maxima['gradient_identity'] = max(maxima['gradient_identity'], float(np.max(np.abs(batch_transport - actual))))
                    approx = transported_gradient(group_q[t], full_h, w, z, parent_before[t])
                    defect = np.einsum('g,gij,j->i', w, full_h - batch_h[t], z - parent_before[t])
                    np.testing.assert_allclose(approx - actual, defect, **tolerance)
                    counts['exact_gradient_defect_identity'] += 1
                    maxima['defect_identity'] = max(maxima['defect_identity'], float(np.max(np.abs(approx - actual - defect))))
                    zero_grad = transported_gradient(group_q[t], full_h, np.zeros(groups), z, parent_before[t])
                    np.testing.assert_array_equal(zero_grad, np.zeros(d))
                    counts['zero_weights_zero_gradient'] += 1
                    truth = advance(truth, direct_gradient(x[idx], y[idx], group[idx], w, truth.theta))
                    proxy = advance(proxy, approx)
                    exact = advance(exact, transported_gradient(group_q[t], batch_h[t], w, exact.theta, parent_before[t]))
                    mean_child = advance(mean_child, direct_gradient(x, y, group, w, mean_child.theta))
                    mean_parent = advance(mean_parent, direct_gradient(x, y, group, np.ones(groups), mean_parent.theta))
                    m1 = parent_after[t] + mean_child.theta - mean_parent.theta
                    np.testing.assert_allclose(exact.theta, truth.theta, **tolerance)
                    maxima['exact_batch_path'] = max(maxima['exact_batch_path'], float(np.max(np.abs(exact.theta - truth.theta))))
                    max_proxy_error = max(max_proxy_error, float(np.linalg.norm(proxy.theta - truth.theta)))
                    max_m1_error = max(max_m1_error, float(np.linalg.norm(m1 - truth.theta)))
                    if np.array_equal(w, np.ones(groups)):
                        np.testing.assert_allclose(proxy.theta, parent_after[t], **tolerance)
                        counts['parent_reproduction'] += 1
                        maxima['parent_path'] = max(maxima['parent_path'], float(np.max(np.abs(proxy.theta - parent_after[t]))))
                    if fixture == 'full_batch':
                        np.testing.assert_allclose(proxy.theta, truth.theta, **tolerance)
                        counts['full_batch_rollout_exactness'] += 1
                        maxima['full_batch_path'] = max(maxima['full_batch_path'], float(np.max(np.abs(proxy.theta - truth.theta))))
                rows.append({'seed': seed, 'fixture': fixture, 'weights': weights_list,
                    'transport_final_parameter_error_l2': float(np.linalg.norm(proxy.theta - truth.theta)),
                    'M1_form_final_parameter_error_l2': float(np.linalg.norm(m1 - truth.theta)),
                    'transport_max_path_parameter_error_l2': max_proxy_error,
                    'M1_form_max_path_parameter_error_l2': max_m1_error,
                    'true_final_theta': truth.theta.tolist(), 'transport_final_theta': proxy.theta.tolist(),
                    'M1_form_final_theta': m1.tolist()})

    invalid_calls = [
        lambda: group_statistics(np.ones((2, 3)), np.ones(3), np.zeros(2, dtype=int), 1, np.ones(3)),
        lambda: group_statistics(np.ones((2, 3)), np.ones(2), np.array([0, 2]), 2, np.ones(3)),
        lambda: transported_gradient(np.ones((2, 3)), np.ones((2, 3, 3)), np.array([1., -1.]), np.ones(3), np.ones(3)),
        lambda: transported_gradient(np.ones((2, 3)), np.ones((2, 3, 3)), np.array([1., np.nan]), np.ones(3), np.ones(3)),
        lambda: adam_step(initial(np.zeros(3)), np.zeros(3), learning_rate=-1),
    ]
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            counts['invalid_inputs_rejected'] += 1
        else:
            raise AssertionError('Invalid input was accepted')
    # Zero weighted gradients do not freeze an Adam state with existing momentum.
    moving = advance(initial(np.zeros(3)), np.ones(3))
    coasting = advance(moving, np.zeros(3))
    assert np.linalg.norm(coasting.theta - moving.theta) > 0
    for required in plan['required_checks']:
        assert counts[required] > 0, required
    result = {'status': 'PASS', 'scope': plan['scope'], 'counts': counts, 'max_abs_identity_residuals': maxima,
        'zero_gradient_with_existing_momentum_moves': True, 'rows': rows,
        'all_fixture_outcomes_retained': True, 'real_data_access': False, 'real_experiments_enabled': False,
        'new_full_trainings': 0, 'cumulative_full_trainings': 434,
        'engineering_seconds': time.perf_counter() - start,
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'scientific_accuracy_or_decision_claim': False}
    (STAGE / 'algebra_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'counts', 'max_abs_identity_residuals', 'engineering_seconds')}))


if __name__ == '__main__':
    main()
