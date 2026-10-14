"""Resume a shutdown-interrupted audit without replacing its original controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path('/root/auto_data_experiments/20261014_file_transfer_r1')
CONTROL = ROOT / 'control' / 'audit_forecast_recovery_20261015'
INTEGRITY = 'recovery_integrity_20261015.json'
PROTOCOL = 'AUDIT_RECOVERY_20261015.md'
PLAN_SHA256 = '9ccfa057242708f440a898f8e22df294086a345df918e1196b4653f791acc861'
PREDICTIONS_SHA256 = 'de55da829296ec3c3f3e06a40f0195bf403cb9ca169344c54def9234bda3faff'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def validate():
    started = time.perf_counter()
    assert ROOT.is_dir() and ROOT.resolve() == ROOT
    assert Path(__file__).resolve() == ROOT / 'recover_file_transfer_audit.py'
    sync = read(ROOT / 'recovery_source_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert len(sync['source_commit']) == 40
    assert sync['plan_sha256'] == sha(ROOT / 'plan.json') == PLAN_SHA256
    assert sync['predictions_sha256'] == sha(ROOT / 'predictions.json') == PREDICTIONS_SHA256
    for name in (INTEGRITY, PROTOCOL, 'recover_file_transfer_audit.py'):
        assert sync['files_sha256'][name] == sha(ROOT / name), name
    receipt = read(ROOT / INTEGRITY)
    assert receipt['status'] == 'PASS' and receipt['research_processes_absent']
    for name, expected in receipt['checked_sha256'].items():
        path = ROOT / name
        assert ROOT in path.parents and path.resolve() == path and path.is_file(), name
        assert sha(path) == expected, name
    original = ROOT / 'control' / 'audit_forecast'
    for name, info in receipt['original_audit_controls'].items():
        path = original / name
        assert path.resolve() == path and sha(path) == info['sha256'], name
    assert not (original / 'exit.json').exists()
    for name in ('forecast_audit.json', 'forecast_audit_failure.json', 'labels.lock', 'labels.json'):
        assert not (ROOT / name).exists(), name
    predictions = read(ROOT / 'predictions.json')
    assert not predictions['eligible_to_label']
    assert predictions['identical_recommendation_baselines'] == ['RAW', 'SOURCE']
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            args = (directory / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        assert str(ROOT / 'audit_response_proxy_file_transfer.py').encode() not in args, directory.name
        assert str(ROOT / 'launch_file_transfer.py').encode() not in args, directory.name
    return dict(checked_files=len(receipt['checked_sha256']), seconds=time.perf_counter() - started,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                integrity_receipt_sha256=sha(ROOT / INTEGRITY),
                recovery_source_sync_sha256=sha(ROOT / 'recovery_source_sync_receipt.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    verification = validate()
    if args.supervise:
        assert CONTROL.is_dir() and CONTROL.resolve() == CONTROL
        assert not (CONTROL / 'child.json').exists() and not (CONTROL / 'exit.json').exists()
        command = [sys.executable, '-B', str(ROOT / 'audit_response_proxy_file_transfer.py'),
                   'forecast', '--output', str(ROOT)]
        started = time.time()
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL)
        write(CONTROL / 'child.json', dict(pid=child.pid, supervisor_pid=os.getpid(), command=command,
              started_unix=started, prelaunch_verification=verification))
        code = child.wait()
        write(CONTROL / 'exit.json', dict(exit_code=code, started_unix=started,
              finished_unix=time.time(), phase='audit_forecast_recovery_20261015',
              original_control_preserved=True, experimental_training_runs_added=0))
        raise SystemExit(code)
    assert CONTROL.parent.resolve() == CONTROL.parent
    CONTROL.mkdir()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--supervise']
    with (CONTROL / 'run.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(CONTROL / 'launch.json', dict(pid=process.pid, phase='audit_forecast_recovery_20261015',
          started_unix=time.time(), prelaunch_verification=verification))
    print(json.dumps(dict(pid=process.pid, status='recovery_audit_supervisor_launched')), flush=True)


if __name__ == '__main__':
    main()
