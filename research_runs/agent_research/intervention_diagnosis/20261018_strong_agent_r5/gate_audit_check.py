"""Read-only r5 gate checks. No prepare, Codex subprocess, or new strategy execution."""
import copy
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts'))
import strong_agent_baseline as s

REPORT_PATH = Path(__file__).with_name('gate_audit.json')
CHECKS = []


def note(name, **details):
    CHECKS.append(dict(check=name, status='PASS', **details))


def check_inputs():
    for family, task in s.TASKS:
        records = s.load_pool(family)
        target = sorted([r for r in records if (r['family'], r['task']) == (family, task)], key=lambda r: r['id'])
        source_ids = {r['id'] for r in records if r['family'] == s.source_family(family)}
        for prefix in (0, 2):
            history = [dict(id=r['id'], record=r, agent_decision=dict(candidate_id=f'Q{i+1}', expected_Y=0., expected_Z=0., decision_reason='audit prefix')) for i, r in enumerate(target[:prefix])]
            packet, models, ids = s.build_input(records, family, task, history)
            mutated = copy.deepcopy(records)
            for r in mutated:
                if r['family'] == family:
                    r['response'] = {k: 900000. + i for i, k in enumerate(r['response'])}
                    r['seed_Y'], r['seed_Z'], r['original_full_training_seconds'] = [999.] * 3, [-999.] * 3, -100.
            packet2, models2, ids2 = s.build_input(mutated, family, task, history)
            assert (packet, models, ids) == (packet2, models2, ids2)
            expected_ids = source_ids | {h['id'] for h in history}
            for model in models.values():
                for component in model.values():
                    assert set(component['train_record_ids']) == expected_ids
            assert len(s.original_features(records)) == 26
            for row in packet['unqueried_candidates']:
                assert set(row) == {'id', 'task', 'features', 'fitted_suggestions'}
                assert len(row['features']) == 28
                assert not any(name.startswith(('window_', 'ordinary_')) for name in row['features'])
                assert set(row['fitted_suggestions']) == {'reference', 'fixed_quadratic', 'measurement_combined'}
            assert {r['id'] for r in packet['unqueried_candidates']} == {ids[r['id']] for r in target[prefix:]}
            assert set(packet['measurement_definitions']) == {'mismatch_measurement_0', 'mismatch_measurement_1'}
            assert packet['query_number'] == prefix + 1
            if prefix:
                altered_history = copy.deepcopy(history)
                altered_history[0]['record']['response']['Y'] += .123
                packet3, models3, _ = s.build_input(records, family, task, altered_history)
                assert packet3 != packet and models3 != models
            note('real_pool_input_isolation', family=family, task=task, history=prefix, source_records=len(source_ids), pending=len(target)-prefix, model_components_checked=6)


def check_adoption_and_random():
    boundary = [
        dict(id='A', response=dict(Y=.01, NMAE_gain=-.01), seed_Y=[-.1, .1, .03], original_full_training_seconds=1.),
        dict(id='B', response=dict(Y=.1, NMAE_gain=-.0100000001), seed_Y=[.1]*3, original_full_training_seconds=2.),
        dict(id='C', response=dict(Y=.0099999999, NMAE_gain=.1), seed_Y=[.1]*3, original_full_training_seconds=None),
        dict(id='D', response=dict(Y=.02, NMAE_gain=0.), seed_Y=[.02]*3, original_full_training_seconds=4.),
    ]
    outcome = s.selected_recommendation([dict(record=r) for r in boundary[:3]])
    assert outcome['recommendation'] == 'A' and outcome['gain'] == .01 and outcome['query_seconds'] is None
    assert outcome['negative_seed_count'] == 1 and outcome['logical_candidate_full_trainings'] == 9
    assert s.selected_recommendation([dict(record=boundary[1])])['recommendation'] == 'PARENT'
    assert s.selected_recommendation([dict(record=r) for r in boundary])['recommendation'] == 'D'
    note('adoption_thresholds_and_unknown_cost')
    for family, task in s.TASKS:
        held = [r for r in s.load_pool(family) if (r['family'], r['task']) == (family, task)]
        for budget in (2, 4):
            subsets = list(itertools.combinations(held, budget))
            actual = sum(s.selected_recommendation([dict(record=r) for r in subset])['gain'] for subset in subsets) / len(subsets)
            expected = sum(max([0.] + [r['response']['Y'] for r in subset if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01]) for subset in subsets) / len(subsets)
            assert abs(actual-expected) < 1e-15
            note('uniform_subset_enumeration', family=family, task=task, queries=budget, subsets=len(subsets), expected_gain=expected)


class SealedTarget(dict):
    def __getitem__(self, key):
        if key in ('response', 'seed_Y', 'seed_Z', 'original_full_training_seconds'):
            assert s.read(s.OUT / 'calls' / 'ETT_ETTh1_1' / 'receipt.json')['status'] == 'PASS', 'target accessed before validated receipt'
        return super().__getitem__(key)


def check_call_gates(temp):
    original_out = s.OUT
    records = s.load_pool('ETT')
    for i, record in enumerate(records):
        if (record['family'], record['task']) == ('ETT', 'ETTh1'):
            records[i] = SealedTarget(record)
    try:
        for case in ('valid', 'invalid_candidate', 'nonfinite', 'tool', 'exit_failure', 'no_completed', 'transport_reconnect', 'malformed_event'):
            s.OUT = temp / case
            (s.OUT / 'calls').mkdir(parents=True)
            s.write(s.OUT / 'execution_lock.json', {'audit_stub': True})
            calls = []
            def fake_subprocess(command, input, stdout, stderr):
                calls.append(command)
                folder = s.OUT / 'calls' / 'ETT_ETTh1_1'
                assert all(s.sha(folder / name) == value for name, value in s.read(folder / 'input_lock.json')['files'].items())
                assert not (folder / 'receipt.json').exists() and not (folder / 'revealed.json').exists()
                packet = s.read(folder / 'input.json')
                decision = dict(candidate_id=packet['unqueried_candidates'][0]['id'], expected_Y=0., expected_Z=0., decision_reason='mock validation only')
                if case == 'invalid_candidate': decision['candidate_id'] = 'NOT_A_CANDIDATE'
                if case == 'nonfinite': decision['expected_Y'] = float('inf')
                (folder / 'response.json').write_text(json.dumps(decision), encoding='utf-8')
                events = []
                if case == 'tool': events.append(dict(type='item.completed', item=dict(type='command_execution')))
                if case == 'transport_reconnect': events.append(dict(type='item.completed', item=dict(type='error', message='Transport reconnect mock')))
                if case != 'no_completed': events.append(dict(type='turn.completed', usage=dict(input_tokens=1, output_tokens=1)))
                stdout.write(''.join(json.dumps(e) + '\n' for e in events).encode())
                if case == 'malformed_event': stdout.write(b'not-json\n')
                return SimpleNamespace(returncode=1 if case == 'exit_failure' else 0)
            with patch.object(s, 'load_pool', return_value=records), patch.object(s.subprocess, 'run', side_effect=fake_subprocess):
                try:
                    ok = s.call_task('ETT', 'ETTh1', 1)
                except json.JSONDecodeError:
                    assert case == 'malformed_event'
                    ok = False
                folder = s.OUT / 'calls' / 'ETT_ETTh1_1'
                assert len(calls) == 1
                assert (folder / 'revealed.json').exists() == (case in ('valid', 'transport_reconnect'))
                if case == 'transport_reconnect':
                    assert len(s.read(folder / 'receipt.json')['transport_errors']) == 1
                if case == 'malformed_event':
                    assert not (folder / 'receipt.json').exists()
                    note('malformed_event_fails_closed', limitation='JSON parsing occurs before receipt try/except; no FAILED receipt written')
                else:
                    assert ok == (case in ('valid', 'transport_reconnect'))
                    note('decision_validation_gate', case=case, subprocess_invocations=1, revealed=ok)
                try:
                    s.call_task('ETT', 'ETTh1', 1)
                    raise AssertionError('restart unexpectedly allowed')
                except FileExistsError:
                    assert len(calls) == 1
            note('step_restart_does_not_repeat', case=case)
    finally:
        s.OUT = original_out


def check_scheduler(temp):
    original_out = s.OUT
    try:
        for case in ('all_pass', 'first_round_failure', 'step_exception'):
            s.OUT = temp / case
            s.OUT.mkdir()
            s.write(s.OUT / 'execution_lock.json', dict(code_sha256={}, protocol_sha256='mock', pool_sha256={}, definitions_sha256={}))
            s.write(s.OUT / 'source_publication_receipt.json', dict(status='exact_commit_and_tree_verified'))
            invoked = []
            def fake_call(family, task, step):
                invoked.append((family, task, step))
                if case == 'step_exception' and task == 'G1':
                    raise json.JSONDecodeError('mock malformed event', '', 0)
                return case == 'all_pass' or task != 'G1'
            with patch.object(s, 'sha', return_value='mock'), patch.object(s, 'call_task', side_effect=fake_call):
                try:
                    s.run()
                    assert case == 'all_pass'
                except RuntimeError:
                    assert case == 'first_round_failure'
                except json.JSONDecodeError:
                    assert case == 'step_exception'
                    assert not (s.OUT / 'stopped.json').exists()
                invocation_count = len(invoked)
                if case == 'step_exception':
                    assert 3 <= invocation_count <= 5 and all(t[2] == 1 for t in invoked)
                else:
                    assert invocation_count == (20 if case == 'all_pass' else 5)
                    assert all(sum(t[:2] == task for t in invoked) == (4 if case == 'all_pass' else 1) for task in s.TASKS)
                try:
                    s.run()
                    raise AssertionError('restart unexpectedly allowed')
                except FileExistsError:
                    assert len(invoked) == invocation_count
            note('scheduler_gate', case=case, mock_step_calls=len(invoked), actual_LLM_calls=0, restart_repeats=0)
    finally:
        s.OUT = original_out


def main():
    audited_hash = s.sha(ROOT / 'scripts' / 'strong_agent_baseline.py')
    check_inputs()
    check_adoption_and_random()
    with tempfile.TemporaryDirectory(prefix='r5_gate_audit_') as directory:
        temp = Path(directory)
        check_call_gates(temp)
        check_scheduler(temp)
    assert s.sha(ROOT / 'scripts' / 'strong_agent_baseline.py') == audited_hash
    report = dict(status='PASS_WITH_LIMITATIONS', code_sha256=audited_hash, protocol_sha256=s.sha(Path(__file__).with_name('PROTOCOL.md')), checks=CHECKS, actual_LLM_calls=0, prepare_executed=False, new_full_trainings=0, findings=[dict(kind='reproduced', severity='medium', file='scripts/strong_agent_baseline.py', line=114, condition='A subprocess event line is malformed JSON, or call_task otherwise raises before returning False.', actual='The exception leaves call_task without receipt.json and leaves run without stopped.json; no target revelation or later round occurs.', expected='Preserve the failed call and a machine-readable failure/stop receipt, while stopping subsequent rounds.', impact='Failure/cost accounting and stage stop reason are incomplete; no observed label leak or repeated call.')], limitations=['Mock subprocess/scheduler checks do not validate live Codex transport behavior.', 'No full new policy/strategy, r4 mathematical audit, external calls or prepare was executed.', 'Restart is fail-closed and cannot resume a partial run.'])
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=len(CHECKS), actual_LLM_calls=0)))


if __name__ == '__main__':
    main()
