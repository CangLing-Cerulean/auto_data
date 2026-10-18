"""Read-only r4 interface checks; synthetic probes are not strategy evaluations."""
from pathlib import Path
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import window_measurement_dsl as dsl
import replay_window_signals as replay
from conditional_predictive_support import UnitScaleGaussian
from conditional_diagnosis import PrequentialLedger


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_dsl():
    rng = np.random.default_rng(1810)
    # Equal lengths ensure rejection is due to axes, not NumPy shape errors.
    table = {"A_inputs": rng.normal(size=(7, 8)),
             "B_inputs": rng.normal(size=(7, 8)),
             "A_first": np.array([True] * 3 + [False] * 4)}
    for seed in (1, 2, 3):
        table[f"B_errors_s{seed}"] = rng.normal(size=(7, 5))
    features = dict(exposure_shift_first=.3, removed_fraction=.2,
                    weighted_implementation=1., child_response=123456.)
    invalid = [
        '__import__("os")', 'open("x")', 'eval("1")', 'globals()',
        'mean(B_level).__class__', 'B_level[0]', '[x for x in B_level]',
        '(lambda: 1)()', 'mean(B_level ** 2)', 'mean(B_level / 2)',
        'mean(B_level, axis=0)', 'mean(*B_level)', 'child_response',
        'B_level', 'True', '3', 'mean(A_level * B_level)',
        'mean(A_level + E_loss)', 'mean(safe_ratio(E_contrast, A_variance))',
        'mean(center(A_level) * unit(B_level))', 'first_mean(B_level)',
        'mean(A_level * (B_level - mean(B_level)))',
        '+'.join(['1'] * 40),
    ]
    rejected = []
    for expression in invalid:
        try:
            dsl.evaluate(expression, table, features)
        except (ValueError, SyntaxError):
            rejected.append(expression)
        else:
            raise AssertionError(f"Forbidden expression accepted: {expression}")
    expected = {
        'mean(A_level) * mean(B_level)': table['A_inputs'][:, 0].mean() * table['B_inputs'][:, 0].mean(),
        'first_mean(A_level)-second_mean(A_level)': table['A_inputs'][:3, 0].mean() - table['A_inputs'][3:, 0].mean(),
        'mean(unit(B_level))': 0.,
        'mean(E_contrast)': np.mean([table[f'B_errors_s{s}'] for s in (1, 2, 3)], axis=0)[:, 1:3].mean(axis=0) @ np.array([1., -1.]),
    }
    for expression, value in expected.items():
        np.testing.assert_allclose(dsl.evaluate(expression, table, features), value, atol=1e-12, rtol=1e-12)
    return dict(status="PASS", rejected=len(rejected), valid_checked=len(expected),
                equal_length_axis_probe=True, arbitrary_feature_inaccessible=True)


def check_packets():
    base = HERE.parent
    preparation = read(HERE / 'corrected_input_preparation.json')
    records = read(base / '20261018_conditional_diagnosis_r1/development_pool.json')['records']
    results = []
    for entry in preparation['packets']:
        family = entry['family']
        packet_path = HERE / 'corrected_inputs' / f'{family}_facts.json'
        packet = read(packet_path)
        assert sha(packet_path) == entry['sha256']
        original_path = base / '20261018_predictive_support_r2' / f'{family}_conditional_source_packet.json'
        assert sha(original_path) == entry['source_packet_sha256']
        expected_facts = [{k: v for k, v in row.items() if k != 'frozen_mean_functions'}
                          for row in read(original_path)['facts']]
        assert packet['response_facts'] == expected_facts
        source = [r for r in records if r['family'] == family]
        tasks = sorted({r['task'] for r in source})
        anonymous = {t: f'T{i+1}' for i, t in enumerate(tasks)}
        for row in packet['response_facts']:
            source_record = source[int(row['record'][1:]) - 1]
            assert row['task'] == anonymous[source_record['task']]
            for field in ('features', 'response', 'seed_Y', 'seed_Z'):
                assert row[field] == source_record[field]
            assert all(source[int(name[1:]) - 1]['task'] != source_record['task'] for name in row['train_records'])
        arrays = []
        for task in tasks:
            path = base / '20261018_parent_measurements_r3/frequency_corrected' / f'{family}_{task}.npz'
            assert sha(path) == entry['parent_asset_sha256'][anonymous[task]]
            with np.load(path, allow_pickle=False) as archive:
                table = dict(archive)
            error_keys = {key for key in table if key.startswith('B_errors_s')}
            assert len(error_keys) == 3
            assert set(table) - error_keys == {'A_inputs', 'B_inputs', 'A_first', 'A_starts', 'B_starts'}
            assert dsl.summary(table) == packet['parent_measurements'][anonymous[task]]
            arrays.append(dict(task=task, keys=sorted(table)))
        assert packet['held_target_records_included'] is False
        results.append(dict(family=family, source_records=len(source), facts=len(expected_facts),
                            source_facts_match=True, source_summary_match=True, arrays=arrays))
    assert preparation['interpreter_sha256'] == sha(ROOT / 'scripts/window_measurement_dsl.py')
    assert preparation['preparer_sha256'] == sha(ROOT / 'scripts/prepare_window_signal_inputs.py')
    return dict(status='PASS', packets=results,
                limitation='Checks corrected table schema/provenance and summaries, not reconstruction from raw parent tensors.')


def check_dual_ridge():
    rng = np.random.default_rng(181018)
    checks = []
    for n, p, rank_deficient in ((9, 3, False), (7, 15, False), (10, 6, True)):
        x = rng.normal(size=(n, p)) * np.geomspace(.0001, 100., p)
        if rank_deficient:
            x[:, 0] = 4.
            x[:, -1] = x[:, -2]
        y = rng.normal(size=n)
        group = np.array(['a'] * 2 + ['b'] * 3 + ['c'] * (n - 5))
        ids = [str(i) for i in range(n)]
        names = [f'x{i}' for i in range(p)]
        for ridge in (.01, 1., 5.):
            primal = UnitScaleGaussian.fit(x, y, group, ids, names, ridge=ridge)
            dual = replay.DualUnitGaussian.fit(x, y, group, ids, names, ridge=ridge)
            differences = {}
            for field in ('center', 'scale', 'beta', 'std'):
                a, b = np.asarray(getattr(primal, field)), np.asarray(getattr(dual, field))
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-10)
                differences[field] = float(np.max(np.abs(a-b)))
            probes = rng.normal(size=(5, p))
            prediction_error = max(abs(primal.predict(q).mean - dual.predict(q).mean) for q in probes)
            assert prediction_error < 1e-10
            checks.append(dict(rows=n, columns=p, rank_deficient=rank_deficient,
                               ridge=ridge, max_field_error=differences, prediction_error=prediction_error))
    return dict(status='PASS', cases=checks,
                algebra='With weighted-centered X, unpenalized intercept is weighted mean(y); A^T(AA^T+lambda I)^-1 equals (A^TA+lambda I)^-1 A^T.')


def check_policy_interface():
    # Stub models isolate feature wiring/order. This does not execute a fitted new strategy.
    active = {}
    observations = set()
    fits = []

    class GuardRecord(dict):
        def __getitem__(self, key):
            if key in ('response', 'seed_Y', 'original_full_training_seconds'):
                assert dict.__getitem__(self, 'id') in active, 'Held label read before registration'
            return super().__getitem__(key)

    class Model:
        def __init__(self, names): self.names = tuple(names)
        def snapshot(self): return dict(names=list(self.names))
        def predict_gain(self, x):
            class Prediction: mean = .02
            return Prediction()
        def log_scores(self, x, y, z):
            value = len(self.names) * .1
            return dict(marginal=value, conditional=2*value, joint=3*value)

    def stub_fit(records, names, quadratic=False):
        assert all(r['id'].startswith('train') or r['id'] in observations for r in records)
        fits.append(dict(names=list(names), quadratic=quadratic))
        return Model(names)

    class GuardLedger(PrequentialLedger):
        def register(self, event_id, *args):
            result = super().register(event_id, *args)
            active[event_id] = True
            return result
        def observe(self, event_id, *args):
            assert event_id in active
            result = super().observe(event_id, *args)
            observations.add(event_id)
            return result

    names = tuple(replay.BASE_FEATURES) + tuple(replay.EXPERTS.values()) + ('old_extra',)
    feature = {name: .2 for name in names}
    feature.update(window_coverage_level=.1, window_correlation_level_loss=.2,
                   ordinary_measurement_0=.4, mismatch_measurement_0=.6)
    training = [dict(id=f'train{i}', features=feature) for i in range(2)]
    held = [GuardRecord(id=f'held{i}', family='f', task='t', action='a', implementation='s',
                        features=feature, response=dict(Y=.02, Z=.01, NMAE_gain=.01),
                        seed_Y=[.01, .02, .03], original_full_training_seconds=None) for i in range(2)]
    signal_names = {mode: [f'{mode}_measurement_0'] for mode in ('ordinary', 'mismatch')}
    results = []
    with patch.object(replay, 'fit', stub_fit), patch.object(replay, 'PrequentialLedger', GuardLedger):
        for method in replay.METHODS:
            active.clear(); observations.clear(); fits.clear()
            result = replay.run(training, held, method, names, signal_names)
            first = fits[0]
            if method.startswith('window_'):
                assert set(first['names']) == set(names) | {'window_coverage_level', 'window_correlation_level_loss'}
            elif method == 'original_quadratic': assert set(first['names']) == set(names)
            elif method.endswith('_combined'):
                mode = method.split('_')[0]
                assert set(first['names']) == set(replay.BASE_FEATURES) | set(signal_names[mode])
            else: assert set(first['names']) == set(replay.BASE_FEATURES)
            expected_quadratic = method in ('original_quadratic', 'window_quadratic')
            assert first['quadratic'] == expected_quadratic
            extra_count = 4 if method.startswith('manual_') else 1 if method.startswith(('ordinary_', 'mismatch_')) and not method.endswith('_combined') else 0
            assert len(result['trace'][0]['model_snapshots']['diagnostics']) == extra_count
            mode = method.rsplit('_', 1)[-1]
            expected_mode = mode if mode in ('joint', 'conditional') else 'uniform'
            for event in [t['event'] for t in result['trace']]:
                for key, value in event['after'].items():
                    delta = value - event['before'][key]
                    np.testing.assert_allclose(delta, 0. if expected_mode == 'uniform' else event['scores'][key][expected_mode])
            assert observations == {'held0', 'held1'}
            assert result['nodes'][0]['query_seconds'] is None
            results.append(dict(method=method, mode=expected_mode, diagnostic_models=extra_count,
                                reference_features=first['names'], quadratic=first['quadratic'],
                                registered_before_label_access=True))
    return dict(status='PASS', methods=results,
                limitation='Two synthetic candidates and stub prediction models; no full real strategy replay or scientific effect test.')


def main():
    audited = ['window_measurement_dsl.py', 'prepare_window_signal_inputs.py', 'replay_window_signals.py']
    before = {name: sha(ROOT / 'scripts' / name) for name in audited}
    result = dict(status='PASS', scope='r4 interface, provenance, bounded AST, dual/primal algebra, synthetic policy wiring only',
                  scripts_sha256=before, dsl=check_dsl(), packets=check_packets(),
                  dual_ridge=check_dual_ridge(), policy=check_policy_interface(),
                  findings=[dict(
                      classification='suspected_freeze_provenance_gap',
                      location='scripts/replay_window_signals.py:90-99; scripts/replay_window_signals.py:47-48',
                      trigger='Parent NPZ or generation response plus its receipt change after evaluation freeze.',
                      actual='Replay main validates script and pool lock hashes and response-to-receipt consistency; parent arrays are loaded without a hash check, and response/receipt are not compared to evaluation_lock.',
                      expected='Frozen evaluation provenance should bind the exact parent assets and generation outputs, either in this runner or an explicit verified launch gate.',
                      impact='A changed measurement input or replaced receipt-consistent response could produce different replay features without rejection by this runner.',
                      limitation='Static control-flow finding; evaluation lock/launch gate did not yet exist, and no tampering or full replay was performed.'
                  )], harness_history=[
                      'Initial harness assumed Hydraulic seed IDs for ETT; schema assertion failed. Inspection showed ETT seeds 2026101901..03, Hydraulic 2026101701..03; corrected harness to require exactly three B-error seed arrays and A/B-only keys. This was an audit-harness failure, not a reviewed-code defect.',
                  ], limitations=[
                      'Generator and evaluation locks are not reviewed here; they were not present at audit start.',
                      'Replay main checks locked script and pool hashes, but does not itself check parent table hashes or bind generation response/receipt hashes to evaluation_lock.',
                      'Existing generation receipts are checked for response content hash only; generator input isolation and budget need their own lock/audit.',
                      'No LLM calls, new training, full new strategy replay, D/test reads, raw-data reconstruction, or scientific effectiveness claim.',
                  ])
    after = {name: sha(ROOT / 'scripts' / name) for name in audited}
    assert before == after, 'Audited scripts changed during checks'
    result['audited_scripts_stable'] = True
    result['audit_script_sha256'] = sha(__file__)
    (HERE / 'interface_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', rejected_expressions=result['dsl']['rejected'],
                         source_packets=len(result['packets']['packets']),
                         ridge_cases=len(result['dual_ridge']['cases']),
                         synthetic_method_wiring=len(result['policy']['methods']))))


if __name__ == '__main__': main()
