"""Narrow regression of three r5 exception paths; no prepare or real model call."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import strong_agent_baseline as s


class HiddenTarget(dict):
    def __getitem__(self, key):
        if key in ('response', 'seed_Y', 'seed_Z', 'original_full_training_seconds'):
            raise AssertionError(f'Unvalidated target read: {key}')
        return super().__getitem__(key)


def initialize(path):
    path.mkdir()
    (path / 'calls').mkdir()
    (path / 'PROTOCOL.md').write_text('Local exception regression stub\n', encoding='utf-8')
    s.write(path / 'execution_lock.json', dict(code_sha256={}, protocol_sha256=s.sha(path / 'PROTOCOL.md'), pool_sha256={}, definitions_sha256={}))
    s.write(path / 'source_publication_receipt.json', dict(status='exact_commit_and_tree_verified'))


def main():
    stable = {str(path): s.sha(path) for path in (ROOT / 'scripts/strong_agent_baseline.py', STAGE / 'gate_audit_check.py', STAGE / 'gate_audit.json')}
    records = s.load_pool('ETT')
    records = [HiddenTarget(r) if (r['family'], r['task']) == ('ETT', 'ETTh1') else r for r in records]
    checks = []
    with tempfile.TemporaryDirectory(prefix='r5_gate_resolution_') as directory:
        for case in ('malformed_event_json', 'subprocess_exception'):
            output = Path(directory) / case
            initialize(output)
            invocations = []
            def fake_subprocess(command, input, stdout, stderr):
                invocations.append(command)
                if case == 'subprocess_exception':
                    raise OSError('Synthetic subprocess launch failure')
                stdout.write(b'not-json\n')
                return SimpleNamespace(returncode=0)
            with patch.object(s, 'OUT', output), patch.object(s, 'TASKS', (('ETT', 'ETTh1'),)), patch.object(s, 'load_pool', return_value=records), patch.object(s.subprocess, 'run', side_effect=fake_subprocess):
                try:
                    s.run()
                    raise AssertionError('run unexpectedly passed')
                except RuntimeError as exc:
                    assert str(exc) == 'Decision validation failed'
            folder = output / 'calls/ETT_ETTh1_1'
            receipt = s.read(folder / 'receipt.json')
            stopped = s.read(output / 'stopped.json')
            assert receipt['status'] == 'FAILED'
            assert receipt['exit_code'] == (0 if case == 'malformed_event_json' else None)
            assert isinstance(receipt['seconds'], (int, float)) and receipt['seconds'] >= 0
            assert ('JSONDecodeError' if case == 'malformed_event_json' else 'OSError') in receipt['error']
            assert stopped['failed_round'] == 1
            assert not (folder / 'revealed.json').exists()
            assert not (output / 'completed.json').exists()
            assert len(invocations) == 1 and len(list((output / 'calls').iterdir())) == 1
            checks.append(dict(check=case, status='PASS', failed_receipt=receipt, stopped=stopped, later_rounds=0, target_revealed=False, mocked_subprocess_invocations=1))
        output = Path(directory) / 'run_task_exception'
        initialize(output)
        invocations = []
        def fake_task(family, task, step):
            invocations.append((family, task, step))
            raise ValueError('Synthetic task execution exception')
        with patch.object(s, 'OUT', output), patch.object(s, 'TASKS', (('ETT', 'ETTh1'),)), patch.object(s, 'call_task', side_effect=fake_task), patch.object(s.subprocess, 'run', side_effect=AssertionError('No subprocess permitted')):
            try:
                s.run()
                raise AssertionError('run unexpectedly passed')
            except ValueError as exc:
                assert str(exc) == 'Synthetic task execution exception'
        stopped = s.read(output / 'stopped.json')
        assert stopped['failed_round'] == 1 and 'ValueError' in stopped['error']
        assert invocations == [('ETT', 'ETTh1', 1)]
        assert not (output / 'completed.json').exists() and not list((output / 'calls').iterdir())
        checks.append(dict(check='run_task_exception', status='PASS', stopped=stopped, later_rounds=0, target_revealed=False))
    assert all(s.sha(path) == value for path, value in stable.items())
    report = dict(status='PASS', checks=checks, checks_count=3, code_sha256=stable[str(ROOT / 'scripts/strong_agent_baseline.py')], prior_audit_sha256=stable[str(STAGE / 'gate_audit.json')], prior_audit_unchanged=True, actual_LLM_calls=0, prepare_executed=False, new_full_trainings=0, scope='Only malformed event JSON, subprocess exception, and unexpected task exception paths. Prior 40-check audit was not rerun.', limitation='Mocked subprocess/task exceptions; no live Codex transport validation.')
    (STAGE / 'gate_resolution.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS', checks=3, actual_LLM_calls=0, prior_audit_unchanged=True)))


if __name__ == '__main__':
    main()
