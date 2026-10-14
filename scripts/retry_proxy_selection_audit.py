"""Audit-only storage recovery; failed outputs and all training sources are immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, replace=False):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if replace:
        temp = path.with_suffix('.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def verify(root, hashes):
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert actual == set(hashes)
    for name, digest in hashes.items():
        assert sha(root / name) == digest, name


def run(source, out):
    started = time.perf_counter()
    plan = read(out / 'audit_retry_plan.json')
    sync = read(out / 'audit_retry_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['audit_retry_plan_sha256'] == sha(out / 'audit_retry_plan.json')
    assert str(source) == plan['recovery_root'] and str(out) == plan['audit_retry_root']
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    assert read(source / 'recovery_status.json')['status'] == 'completed'
    assert not (source / 'aggregate' / 'final_audit.json').exists()
    for name, digest in plan['failure_logs_sha256'].items():
        assert sha(source / name) == digest
    assert shutil.disk_usage(out).free >= 2 * 1024 ** 3
    original = Path(plan['original_root'])
    inspection = read(source / 'inspection.json')
    verify(original, inspection['original_files_sha256'])
    aggregate = source / 'aggregate'
    input_paths = sorted(p for p in aggregate.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.is_symlink() for p in input_paths)
    inputs = {p.relative_to(aggregate).as_posix(): sha(p) for p in input_paths}
    write(out / 'input_snapshot.json', dict(source=str(aggregate), files_sha256=inputs,
        first_audit_status='failed_no_space', original_status_file_is_stale=True))
    view = out / 'view'
    view.mkdir()
    for name, digest in inputs.items():
        target_name = f'recovery/first_failed_audit_residuals/{name[6:]}' if name.startswith('audit/') else name
        target = view / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(aggregate / name, target)
        assert sha(target) == digest
    for name in ('audit_retry_plan.json', 'audit_retry_sync_receipt.json', 'AUDIT_RETRY_PROTOCOL.md',
                 'retry_proxy_selection_audit.py', 'input_snapshot.json'):
        shutil.copy2(out / name, view / 'recovery' / name)
    failure = view / 'recovery' / 'first_audit_failure'
    failure.mkdir()
    for name, target in (('original_auditor.log', 'original_auditor.log'),
                         ('audit_control/run.log', 'wrapper.log'), ('audit_status.json', 'stale_status.json')):
        shutil.copy2(source / name, failure / target)
    # Original provenance checks remain mandatory after relocation.
    changed = {'labels_status.json', f'runs/{inspection["partial_id"]}/status.json'}
    for name, digest in inspection['original_files_sha256'].items():
        if name not in changed:
            assert sha(view / name) == digest
    check = read(view / 'recovery' / 'checkpoint_equivalence.json')
    assert check['status'] == 'PASS' and check['model_optimizer_numpy_torch_cuda_bitexact']
    timing = read(view / 'runs' / inspection['partial_id'] / 'timing.json')
    continuation = read(view / 'recovery' / 'continuation.json')
    assert timing['seconds'] is None and timing['pre_interruption_seconds'] is None
    assert continuation['total_logical_run_seconds'] is None
    assert timing['measured_continuation_seconds'] == continuation['measured_continuation_seconds'] > 0
    labels = read(view / 'labels.json')
    assert labels['phase_seconds'] is None and labels['partial_original_timing_unknown']
    assert sum(r['seconds'] is None for r in labels['results']) == 1
    assert next(r for r in labels['results'] if r['id'] == inspection['partial_id'])['seconds'] is None
    temp = out / 'temp'
    temp.mkdir()
    write(out / 'status.json', dict(status='running_original_auditor'))
    try:
        with (out / 'original_auditor.log').open('xb') as stream:
            subprocess.run([sys.executable, '-B', str(view / 'audit_response_proxy_selection.py'), '--output', str(view)],
                cwd=view, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', TMPDIR=str(temp)), check=True)
        result = read(view / 'final_audit.json')
        assert result['status'] == 'PASS' and result['cumulative_full_trainings'] == 367
        verify(original, inspection['original_files_sha256'])
        verify(aggregate, inputs)
        known = sum(r['seconds'] for r in labels['results'] if r['seconds'] is not None)
        known += timing['measured_continuation_seconds']
        write(out / 'recovery_audit.json', dict(status='PASS', original_auditor_unchanged=True,
            original_final_audit_sha256=sha(view / 'final_audit.json'), archive_receipt_sha256=sha(view / 'archive_receipt.json'),
            original_and_failed_audit_inputs_unchanged=True, restored_state_test=check,
            cumulative_full_trainings=367, original_complete_reused=11, resumed_logical_trainings=1,
            newly_started_full_trainings=24, known_label_run_seconds=known,
            unknown_pre_interruption_run_seconds=None, full_label_run_seconds=None,
            measured_recovery_phase_seconds=labels['measured_recovery_phase_seconds'],
            audit_storage_recovery_seconds=time.perf_counter() - started, prior_audit_failures=1,
            prior_failure='No space left on device; original numerical/scientific audit incomplete',
            new_training_steps_for_audit_retry=0, D_access=False, test_access=False, advance=result['advance']))
        write(out / 'status.json', dict(status='PASS', scientific_advance=result['advance']), replace=True)
    except Exception:
        write(out / 'status.json', dict(status='failed', error=traceback.format_exc()), replace=True)
        raise


def launch(source, out):
    control = out / 'control'
    control.mkdir()
    with (control / 'run.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(out / 'retry_proxy_selection_audit.py'), 'run',
            '--source', str(source), '--output', str(out)], cwd=out, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    write(control / 'launch.json', dict(pid=process.pid, started_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='launched', pid=process.pid)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('launch', 'run'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.source, args.output)
