"""Narrow r8 interface gates; never execute prepare/run or a real policy path."""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import conditional_interactions as impl

CHECKS = []

def require(value, message):
    if not value:
        raise AssertionError(message)

def close(a, b, message='numeric disagreement'):
    require(np.allclose(a, b, rtol=1e-11, atol=1e-12), message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(name, function):
    try:
        CHECKS.append(dict(check=name, status='PASS', details=function()))
    except Exception as exc:
        CHECKS.append(dict(check=name, status='FAIL', error=repr(exc), traceback=traceback.format_exc()))

def record(identity, task, signal, implementation='S'):
    features = {name: .05 * (i % 4 - 1) for i, name in enumerate(impl.CORE)}
    features['synthetic_q'] = signal
    return dict(id=identity, family='synthetic', task=task, action='synthetic',
                implementation=implementation, features=features,
                response=dict(Y=.02 + .01 * signal, Z=.03 * signal, NMAE_gain=.01),
                seed_Y=[.01, .02, .03], original_full_training_seconds=12.)

SOURCE = [record('a', 'g1', -1.), record('b', 'g1', 1., 'W'),
          record('c', 'g2', 2.), record('d', 'g2', 3., 'W')]
NAMES = ('synthetic_q',)
NORM = impl.normalizers(SOURCE, NAMES)
RHO = {'synthetic_q': {'S': 0., 'W': 0.}}

def locked_inputs_and_matrix():
    lock = json.loads((STAGE / 'execution_lock.json').read_text(encoding='utf-8'))
    require(sha(STAGE / 'PROTOCOL.md') == lock['protocol_sha256'], 'protocol hash')
    for name, expected in lock['scripts'].items():
        require(sha(ROOT / 'scripts' / name) == expected, 'script changed: ' + name)
    for path, expected in lock['inputs'].items():
        require(sha(ROOT / path) == expected, 'input changed: ' + path)
    require({k: list(v) for k, v in impl.METHODS.items()} == lock['methods'], 'method lock')
    tasks = []
    scales = []
    for source in ('ETT', 'Hydraulic'):
        pool = json.loads((impl.OLD / 'replay' / f'{source}_augmented_records.json').read_text(encoding='utf-8'))['records']
        training = [r for r in pool if r['family'] == source]
        groups = sorted({(r['family'], r['task']) for r in training})
        for name, saved in lock['normalizers'][source].items():
            # Stable independently weighted Euclidean norm; no production normalizer call.
            terms = []
            for group in groups:
                rows = [r for r in training if (r['family'], r['task']) == group]
                terms.extend(float(r['features'][name]) / math.sqrt(len(groups) * len(rows)) for r in rows)
            close(saved, math.hypot(*terms), 'source RMS mismatch: ' + source + ':' + name)
            scales.append(saved)
        for family, task in sorted({(r['family'], r['task']) for r in pool if r['family'] != source}):
            held = [r for r in pool if (r['family'], r['task']) == (family, task)]
            require(len(held) in (4, 8), 'unexpected target inventory')
            require(all(len(r['seed_Y']) == 3 for r in held), 'three-seed cost provenance')
            require(all(r['original_full_training_seconds'] is None or r['original_full_training_seconds'] >= 0 for r in held), 'invalid historical cost')
            tasks.append(dict(family=family, task=task, candidates=len(held),
                              unknown_costs=sum(r['original_full_training_seconds'] is None for r in held)))
    counts = dict(methods=len(impl.METHODS), paths=len(tasks)*len(impl.METHODS),
                  queries=sum(t['candidates'] for t in tasks)*len(impl.METHODS),
                  nodes=sum(sum(b <= t['candidates'] for b in (2, 4, 8)) for t in tasks)*len(impl.METHODS))
    require(counts == dict(methods=7, paths=35, queries=224, nodes=91), 'matrix counts')
    require(len(impl.CORE) == 26 and len(impl.CORE_NAMES) == 377, 'core dimensions')
    return dict(**counts, tasks=tasks, independently_verified_source_RMS=len(scales),
                inspected_only='locked metadata and preexisting augmented feature/label summaries; no raw arrays')

def source_weighting_zero_rule():
    records = [record('x', 'one', 2.), record('y', 'two', 4.), record('z', 'two', 4.)]
    close(impl.normalizers(records, NAMES)['synthetic_q'], math.sqrt(10.))
    for r in records:
        r['features']['synthetic_q'] = 0.
    norm = impl.normalizers(records, NAMES)
    candidate = record('target', 'target', 1e200)
    x, columns = impl.basis(candidate, NAMES, norm, RHO)
    require(np.all(x[len(impl.CORE_NAMES):] == 0), 'zero source feature revived by target')
    require(norm == {'synthetic_q': 0.}, 'zero source RMS')
    return dict(equal_task_RMS=True, zero_source_target_suppression=True)

def persistent_signal_and_gate():
    candidate = record('target', 'target', 2.)
    baseline, columns = impl.basis(candidate, NAMES, NORM, RHO)
    index = columns.index('signal:synthetic_q')
    for implementation in ('S', 'W'):
        candidate['implementation'] = implementation
        for extreme in (-1e6, 0., 1e6):
            rho = {'synthetic_q': {'S': extreme, 'W': -extreme}}
            x, actual_columns = impl.basis(candidate, NAMES, NORM, rho)
            require(columns == actual_columns, 'feature layout changed')
            close(x[:len(impl.CORE_NAMES)], baseline[:len(impl.CORE_NAMES)])
            close(x[index], baseline[index])
            require(np.isfinite(x).all() and np.max(np.abs(x[index:])) <= 1., 'unbounded gate features')
            inactive = 'W' if implementation == 'S' else 'S'
            require(x[columns.index('interaction:synthetic_q:' + inactive)] == 0., 'inactive implementation feature')
    close(impl.gain_gate(0.), 1 / math.sqrt(2))
    x = np.array([impl.basis(r, NAMES, NORM, {'synthetic_q': {'S': -5., 'W': 5.}})[0] for r in SOURCE])
    _, scales, _ = impl.DualUnitGaussian._fit_mean(x, np.array([r['response']['Y'] for r in SOURCE]), np.array([r['task'] for r in SOURCE]), 1.)
    require(np.all(scales[index:] == 1.), 'unit floor cancels bounded feature gate')
    return dict(ordinary_column_retained=True, bounded=True, gate_extremes=[-1e6, 1e6], unit_floor_exact=True)

def masked_policy_and_scorers():
    target = [record('target_a', 'held', .5), record('target_b', 'held', -.5, 'W')]
    changed = copy.deepcopy(target)
    for r in changed:
        r['response'] = dict(Y=999., Z=-999., NMAE_gain=-999.)
        r['seed_Y'] = [-999.] * 3
        r['original_full_training_seconds'] = 999999.
    candidates = [impl.view(r) for r in target]
    require(candidates == [impl.view(r) for r in changed], 'hidden data leaked into candidate schema')
    require(all(set(r) == {'id','family','task','action','implementation','features'} for r in candidates), 'candidate schema')
    visible = [impl.visible_response(r) for r in SOURCE]
    require(all(set(r['response']) == {'Y','Z','NMAE_gain'} for r in visible), 'full feedback schema')
    a = impl.policy(visible, candidates, NAMES, NORM, RHO)
    b = impl.policy(visible, [impl.view(r) for r in changed], NAMES, NORM, RHO)
    require(a[0]['id'] == b[0]['id'] and a[1] == b[1], 'hidden-label intervention changed predictions')
    c = impl.policy(visible, candidates, NAMES, NORM, {'synthetic_q': {'S': -20., 'W': 20.}})
    require(a[3].snapshot() == c[3].snapshot(), 'rho changes reference scorer')
    require({k:v.snapshot() for k,v in a[4].items()} == {k:v.snapshot() for k,v in c[4].items()}, 'rho changes diagnostic scorer')
    require(a[2].snapshot() != c[2].snapshot(), 'rho does not affect gain learner')
    require(a[2].snapshot() == b[2].snapshot(), 'nonreproducible model snapshots')
    return dict(hidden_label_intervention=True, repeat_exact=True, rho_independent_scoring_models=True,
                synthetic_policy_calls=3, real_policy_calls=0)

def buckets_and_adoption():
    scores = {'synthetic_q': dict(marginal=1., conditional=-2., joint=-1.)}
    saved = copy.deepcopy(RHO)
    for rule in ('none', 'conditional', 'joint'):
        for implementation in ('S', 'W'):
            actual = impl.update_support(RHO, scores, implementation, rule)
            expected = copy.deepcopy(RHO)
            if rule != 'none':
                expected['synthetic_q'][implementation] = scores['synthetic_q'][rule]
            require(actual == expected and RHO == saved, 'wrong update scope or input mutation')
    weak = record('weak', 't', 0.)
    weak['response'] = dict(Y=.0099, Z=0., NMAE_gain=0.)
    a, z = record('a', 't', 0.), record('z', 't', 0.)
    a['response'] = z['response'] = dict(Y=.01, Z=0., NMAE_gain=-.01)
    require(impl.choose([weak]) is None and impl.choose([a,z])['id'] == 'z', 'adoption thresholds/tie')
    bad = copy.deepcopy(z); bad['response']['NMAE_gain'] = -.01001
    require(impl.choose([weak,bad]) is None, 'adoption guardrail')
    return dict(update_cases=6, no_input_mutation=True, observed_gain_guardrail_and_tie=True)

def registration_order_static():
    tree = ast.parse(inspect.getsource(impl.run_task))
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    text = ast.unparse(loop)
    ordered = ["selected, predictions, gain, ref, experts = policy(", 'write(reg_path, registration)',
               "write(folder / 'initial_forecasts.json'", "revealed = oracle[selected['id']]",
               'reference_scores = ref.log_scores(', 'after = update_support(',
               "write(folder / f'{step:02d}_event.json'", 'visible.append(visible_response(revealed))']
    positions = [text.index(token) for token in ordered]
    require(positions == sorted(positions), 'registration/reveal/score/update/next-fit ordering')
    policy_tree = ast.parse(inspect.getsource(impl.policy))
    params = [arg.arg for arg in policy_tree.body[0].args.args]
    require(params == ['visible','candidates','names','norm','rho'], 'policy oracle input')
    require('oracle' not in inspect.getsource(impl.policy), 'policy accesses oracle')
    require('logical_candidate_full_trainings=3 * step' in text and 'shared_parent_full_trainings=3' in text, 'three-seed logical costs')
    require('None if any((c is None for c in costs)) else sum(costs)' in text, 'unknown costs replaced with zero')
    return dict(order=ordered, scope='AST/source structure; saved real event order pending result audit')

def degenerate_fit():
    constant = copy.deepcopy(SOURCE)
    for r in constant:
        r['features'] = {name: 0. for name in r['features']}
        r['response'] = dict(Y=.02, Z=0., NMAE_gain=.01)
    model = impl.fit(constant, NAMES, {'synthetic_q': 0.}, RHO)
    x, _ = impl.basis(constant[0], NAMES, {'synthetic_q': 0.}, RHO)
    require(model.predict_gain(x).mean == .02 and model.marginal.std == .01, 'constant predictor/floor')
    require(all(math.isfinite(v) for v in model.log_scores(x,.02,0.).values()), 'nonfinite constant score')
    return dict(constant_collinear_design=True, predictive_std_floor=.01)

def extreme_source_RMS():
    # Two finite magnitudes whose mathematical RMS is representable in float64.
    observations = []
    for value in (1e200, 1e-200):
        rows = [record('x', 'one', value), record('y', 'two', value)]
        try:
            actual = impl.normalizers(rows, NAMES)['synthetic_q']
            observations.append(dict(input=value, actual=actual, expected=abs(value),
                                     passed=math.isfinite(actual) and actual > 0 and math.isclose(actual, abs(value), rel_tol=1e-12)))
        except Exception as exc:
            observations.append(dict(input=value, expected=abs(value), passed=False, error=repr(exc)))
    require(all(r['passed'] for r in observations), 'finite source RMS overflow/underflow: ' + repr(observations))
    return observations

if __name__ == '__main__':
    for name, function in [('locked_inputs_matrix_and_source_RMS', locked_inputs_and_matrix),
                           ('task_equal_source_normalization_zero_support', source_weighting_zero_rule),
                           ('persistent_signal_bounded_interaction_unitfloor', persistent_signal_and_gate),
                           ('hidden_labels_reproducibility_and_scorer_independence', masked_policy_and_scorers),
                           ('none_conditional_joint_buckets_and_adoption', buckets_and_adoption),
                           ('registration_oracle_score_update_nextfit', registration_order_static),
                           ('constant_collinear_fit', degenerate_fit),
                           ('finite_extreme_source_RMS', extreme_source_RMS)]:
        check(name, function)
    receipt = dict(status='PASS' if all(r['status'] == 'PASS' for r in CHECKS) else 'FAIL', checks=CHECKS,
                   production_script_sha256=sha(ROOT/'scripts/conditional_interactions.py'),
                   protocol_sha256=sha(STAGE/'PROTOCOL.md'), execution_lock_sha256=sha(STAGE/'execution_lock.json'),
                   checker_sha256=sha(__file__), new_full_trainings=0, new_LLM_calls=0,
                   real_policy_trajectories_executed=0,
                   not_covered=['saved real snapshots/events/results: awaiting results-ready',
                                'raw arrays, D/test, servers, and external calls: excluded'])
    path = STAGE / 'gate_audit.json'
    if path.exists():
        index = 2
        while (STAGE / f'gate_audit_attempt_{index}.json').exists(): index += 1
        path = STAGE / f'gate_audit_attempt_{index}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], passed=sum(c['status']=='PASS' for c in CHECKS),
                          checks=len(CHECKS), receipt=str(path)), ensure_ascii=False))
    sys.exit(0 if receipt['status'] == 'PASS' else 1)
