"""An exact rational, nonzero-defect example independent of random fixtures."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from gradient_transport_quadratic import group_statistics, transported_gradient

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261015_gradient_transport_design_r1'


def main():
    x = [F(1), F(2), F(3), F(4)]
    y = [F(1), F(0), F(2), F(-1)]
    group = [0, 0, 1, 1]
    indices = [0, 2]
    parent, candidate = F(1, 2), F(3, 4)
    weights = [F(1, 2), F(3, 2)]
    q = [sum((x[i] * (x[i] * parent - y[i]) for i in indices if group[i] == g), F(0)) / len(indices) for g in range(2)]
    full_h = [sum((x[i] ** 2 for i in range(4) if group[i] == g), F(0)) / 4 for g in range(2)]
    batch_h = [sum((x[i] ** 2 for i in indices if group[i] == g), F(0)) / len(indices) for g in range(2)]
    direct = sum(weights[group[i]] * x[i] * (x[i] * candidate - y[i]) for i in indices) / len(indices)
    predicted = sum(weights[g] * (q[g] + full_h[g] * (candidate - parent)) for g in range(2))
    defect = sum(weights[g] * (full_h[g] - batch_h[g]) * (candidate - parent) for g in range(2))
    assert (direct, predicted, defect) == (F(1, 2), F(5, 4), F(3, 4))
    assert predicted - direct == defect
    X = np.array([[float(value)] for value in x])
    Y, groups = np.array([float(value) for value in y]), np.array(group)
    Q, Hbatch = group_statistics(X[indices], Y[indices], groups[indices], 2, np.array([.5]))
    _, Hfull = group_statistics(X, Y, groups, 2, np.array([.5]))
    actual_proxy = transported_gradient(Q, Hfull, np.array([.5, 1.5]), np.array([.75]), np.array([.5]))
    exact_proxy = transported_gradient(Q, Hbatch, np.array([.5, 1.5]), np.array([.75]), np.array([.5]))
    np.testing.assert_array_equal(actual_proxy, np.array([float(predicted)]))
    np.testing.assert_array_equal(exact_proxy, np.array([float(direct)]))
    lock = json.loads((STAGE / 'source_lock.json').read_text(encoding='utf-8-sig'))
    for entry in lock['files']:
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    check = json.loads((STAGE / 'algebra_check.json').read_text(encoding='utf-8'))
    assert check['status'] == 'PASS' and len(check['rows']) == 45
    summary = []
    for fixture in ('full_batch', 'fixed_feature_curvature', 'heterogeneous_minibatch'):
        rows = [r for r in check['rows'] if r['fixture'] == fixture and r['weights'] != [1, 1, 1]]
        assert len(rows) == 12
        summary.append({'fixture': fixture, 'nonidentity_cases': len(rows),
            'transport_mean_final_error_l2': float(np.mean([r['transport_final_parameter_error_l2'] for r in rows])),
            'M1_form_mean_final_error_l2': float(np.mean([r['M1_form_final_parameter_error_l2'] for r in rows])),
            'transport_lower_error_cases': None if fixture == 'full_batch' else sum(r['transport_final_parameter_error_l2'] < r['M1_form_final_parameter_error_l2'] for r in rows),
            'scope': 'constructed parameter-space examples, not forecasting or decision performance'})
    result = {'status': 'PASS', 'independent_reference': 'Python fractions exact rational arithmetic',
        'true_gradient': str(direct), 'transport_gradient': str(predicted), 'nonzero_defect': str(defect),
        'locked_sources_unchanged': True, 'numpy_version': np.__version__,
        'random_fixture_rows_verified_present': len(check['rows']), 'summaries': summary,
        'cannot_infer_general_accuracy_or_decision_gain': True,
        'new_full_trainings': 0, 'real_data_access': False}
    (STAGE / 'hand_check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
