"""One declared monitor repair retry; frozen scientific workers stay unchanged."""
import argparse
import importlib.util
import io
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_ts.agent_research import native
from online_ts.agent_research.runtime import ResearchRepository
from online_ts.agent_research.signal_gateway import SignalGateway, scope_summary
from online_ts.agent_research.signal_scoring import feedback_scores, outcome_gates
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.memory import MEMORY
from online_ts.agent_research.processing import DEFAULTS
from online_ts.config import GateError
from online_ts.selection.common import file_hash, read_json, write_json

FAILED_JOB = 'ad5444427bcea4c3dd470c902c611666714d8f041a40fc8624b9e288d9ff6c48'
RECOVERY_SEED = 2026101204


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    folder = Path(args.folder)
    if (folder/'confirmation_result.json').exists():
        raise GateError('CONFIRMATION_ALREADY_COMPLETED')
    engineering = folder/'engineering'
    fixed_path, tests_path = engineering/'native_fixed.py', engineering/'test_native_inventory.py'
    tests = load('inventory_regression_tests', tests_path)
    old_output = io.StringIO()
    old_test = unittest.TestSuite([tests.InventoryTests('test_atomic_temporary_file_disappears_after_enumeration')])
    old_result = unittest.TextTestRunner(stream=old_output).run(old_test)
    if len(old_result.errors) != 1 or 'NATIVE_OUTPUT_TYPE' not in old_result.errors[0][1]:
        raise GateError('EXPECTED_OLD_RACE_NOT_REPRODUCED')
    repaired = load('online_ts.agent_research.native_repair_snapshot', fixed_path)
    native.inventory = repaired.inventory
    fixed_output = io.StringIO()
    fixed_result = unittest.TextTestRunner(stream=fixed_output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests))
    write_json(engineering/'regression_result.json', dict(old_race_reproduced=True,
        old_output=old_output.getvalue(), repaired_output=fixed_output.getvalue(),
        tests_run=fixed_result.testsRun, repaired_passed=fixed_result.wasSuccessful(),
        interpretation='Deterministic reproduction of a monitor defect; original failure has no event-level trace proving this exact race.'))
    if not fixed_result.wasSuccessful():
        raise GateError('MONITOR_REPAIR_REGRESSION_FAILED')
    identity = read_json(folder/'preregistration.json')
    repo = ResearchRepository('solhigh_memory_r2', identity, folder=folder/'run')
    if not repo.state().incident_finished:
        raise GateError('SEARCH_NOT_FROZEN')
    gateway = SignalGateway(repo, TaskAssets(ROOT/'research_runs/agent_research/tasks/metropt3_dev'),
        read_json(MEMORY/'library.json'), folder)
    gateway.guards.update(identity['code'])
    confirm = ROOT/'scripts/confirm_autonomous_memory.py'
    gateway.guards[str(confirm)] = read_json(folder/'confirmation_freeze.json')['confirmation_runner_sha256']
    for row in read_json(folder/'memory_manifest.json')['sources']:
        gateway.guards[str(folder/row['file'])] = row['sha256']
    for path, expected in gateway.guards.items():
        if file_hash(path) != expected:
            raise GateError('FROZEN_ASSET_CHANGED')
    base = gateway.baseline()
    selected = read_json(folder/'run/final_policy.json')['selected_candidate']
    candidate = gateway.candidate(selected)
    signal = repo.get('signal:'+candidate['signal_id'])
    old = repo.get('training:'+FAILED_JOB)
    if old['status'] != 'failed' or old['error'] != 'NATIVE_OUTPUT_TYPE':
        raise GateError('EXPECTED_FAILURE_NOT_PRESENT')
    failure = read_json(folder/'run/training'/FAILED_JOB/'execution.json')
    control = read_json(folder/'confirmation_control.json')
    for path in (fixed_path, tests_path, Path(__file__)):
        gateway.guards[str(path)] = file_hash(path)
    amendment = dict(kind='engineering_monitor_retry', original_failed_job=FAILED_JOB,
        original_failure=failure, recovery_seed=RECOVERY_SEED,
        original_native_sha256=file_hash(native.__file__), repaired_native_sha256=file_hash(fixed_path),
        regression_test_sha256=file_hash(tests_path), recovery_runner_sha256=file_hash(__file__),
        implementation='Replace native.inventory in recovery controller memory only; original frozen server sources unchanged.',
        scientific_changes=[], candidate_feedback_after_freeze=False,
        retry_policy='Only missing third-seed control, once under a new protected-asset identity; retain and count failed execution.',
        causal_limit='Reproduced atomic-rename race fits the error; original event was not traced.')
    write_json(engineering/'amendment.json', amendment)
    if shutil.disk_usage(folder).free < 2*1024**3:
        raise GateError('DISK_RESERVE')
    replacement = gateway.adapter.evaluate('confirmation_control_recovery', control['asset'], DEFAULTS,
        seed=RECOVERY_SEED, transform=control['transform'], control_seed=20261012)
    scientific = lambda item: {k:v for k,v in item.items() if k != 'protected_assets'}
    if scientific(replacement['identity']) != scientific(old['identity']):
        raise GateError('RECOVERY_SCIENTIFIC_IDENTITY_CHANGED')
    amendment.update(recovery_job=replacement['job_id'], scientific_identity_equal=True)
    write_json(engineering/'amendment.json', amendment)
    records = list(repo.iter_records('research_training'))
    outcomes = []
    for seed in read_json(folder/'protocol.json')['confirmation_seeds']:
        values = {}
        for name in ('baseline', 'candidate', 'control'):
            tag = 'confirmation_control_recovery' if name == 'control' and seed == RECOVERY_SEED else 'confirmation_'+name
            matches = [r for r in records if r['candidate_id'] == tag and r['identity']['seed'] == seed and r['status'] == 'completed']
            if len(matches) != 1:
                raise GateError('CONFIRMATION_UNIQUE_COMPLETED_JOB_REQUIRED')
            row = matches[0]
            job = folder/'run/training'/row['job_id']
            if file_hash(job/'result.json') != row['result_sha256']:
                raise GateError('RESULT_CHANGED')
            result = read_json(job/'result.json')
            if any(file_hash(job/name) != expected for name, expected in result['assets'].items()):
                raise GateError('RESULT_ASSET_CHANGED')
            values[name] = (row['job_id'], result['metrics'], scope_summary(read_json(job/'scope_metrics.json'),
                signal['group'], signal['channels'], gateway.task.task['sensor_columns']))
        outcomes.append(dict(seed=seed, **{n:v[1] for n,v in values.items()},
            **{'local_'+n:v[2] for n,v in values.items()}, jobs={n:v[0] for n,v in values.items()}))
    gates = outcome_gates(outcomes, signal['local_gain_pct'])
    violations = repo.count_records('information_violation')
    valid = violations == 0 and gateway.journal.summary()['autonomy_eligible']
    result = dict(selected_candidate=selected, independent_task=False,
        feedback_checks=feedback_scores(gateway), information_violations=violations,
        registered_facts=[{'signal_id': s['signal_id'], **repo.get('signal-verification:'+s['signal_id'])} for s in gateway.signals()],
        exploration_NMSE_gain_pct=100*(1-candidate['result']['metrics']['NMSE']/base['metrics']['NMSE']),
        exploration_NMAE_deterioration_pct=100*(candidate['result']['metrics']['NMAE']/base['metrics']['NMAE']-1),
        status='completed_pending_semantic_review', results=outcomes, gates=gates,
        E=bool(gates['E_numeric'] and valid), information_conditions_valid=valid, control_kind=control['kind'],
        engineering_amendment='engineering/amendment.json', original_failed_job=FAILED_JOB,
        recovery_job=replacement['job_id'])
    write_json(folder/'confirmation_progress.json', dict(results=outcomes, completed_seeds=len(outcomes), engineering_amendment='engineering/amendment.json'))
    write_json(folder/'confirmation_result.json', result)
    repo.export()
    print('Repaired monitor verified; frozen confirmation completed.', flush=True)


if __name__ == '__main__':
    main()
