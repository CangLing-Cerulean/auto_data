"""Run only the frozen nine missing C2 confirmations with durable local stdout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def spawn_logged(command, logfile):
    # Both output descriptors point to a regular file, never the SSH transport.
    with Path(logfile).open('xb') as log:
        return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def validate_missing_jobs(root, jobs):
    expected = [dict(task='C2', action=action, seed=seed, id=f'C2_{action}_s{seed}')
                for seed in (2026101402, 2026101403, 2026101404)
                for action in ('PARENT', 'P1', 'CP1')]
    if jobs != expected:
        raise ValueError('Only the exact registered nine C2 jobs may run')
    if any((root/'candidate_runs'/job['id']).exists() for job in jobs):
        raise FileExistsError('Run ID already occupied; no repeat or resume')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    plan = read(root/'confirmation_recovery_protocol.json')
    assert sha(__file__) == plan['supervisor_sha256']
    assert sha(root/'candidate_protocol.json') == plan['candidate_protocol_sha256']
    assert sha(root/'run_stage1_candidate.py') == plan['runner_sha256']
    assert sha(root/'audit_stage1_confirmation_recovered.py') == plan['auditor_sha256']
    preflight = read(root/'confirmation_recovery_preflight.json')
    assert preflight['status'] == 'PASS'
    assert preflight['protocol_sha256'] == sha(root/'confirmation_recovery_protocol.json')
    assert read(root/'confirmation_recovery_source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
    for relative, digest in plan['preserved_files_sha256'].items():
        assert sha(root/relative) == digest, relative
    jobs = plan['new_jobs']
    validate_missing_jobs(root, jobs)
    control = root/'confirmation_recovery_control'
    if not args.execute:
        control.mkdir()  # An existing controller is never silently restarted.
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--root', str(root), '--execute']
        process = spawn_logged(command, control/'supervisor.log')
        write(control/'launch.json', dict(pid=process.pid, protocol_sha256=sha(root/'confirmation_recovery_protocol.json'),
                                        launched_unix=time.time(), output_transport='regular_file'))
        print(json.dumps(dict(pid=process.pid, status='launched', new_jobs=len(jobs))))
        return
    with (control/'execution.lock').open('x') as guard:
        guard.write(str(os.getpid()))
    state = dict(status='running', pid=os.getpid(), active=None, completed=[], failed=None,
                 protocol_sha256=sha(root/'confirmation_recovery_protocol.json'))
    write(control/'status.json', state)
    try:
        for job in jobs:
            name = job['id']
            if (root/'candidate_runs'/name).exists():
                raise FileExistsError('Run ID occupied immediately before launch')
            state['active'] = name
            write(control/'status.json', state)
            command = [sys.executable, '-B', str(root/'run_stage1_candidate.py'), '--root', str(root),
                       '--task', job['task'], '--action', job['action'], '--seed', str(job['seed'])]
            process = spawn_logged(command, control/(name+'.log'))
            state['child_pid'] = process.pid
            write(control/'status.json', state)
            code = process.wait()
            status = read(root/'candidate_runs'/name/'status.json')
            if code != 0 or status['status'] != 'completed' or status['steps_completed'] != 3000:
                raise RuntimeError(f'{name}: exit={code}, status={status}')
            state['completed'].append(name)
            state['active'] = None
            state.pop('child_pid', None)
            write(control/'status.json', state)
        state['status'] = 'completed'
    except Exception as error:
        state['status'] = 'failed_stopped'
        state['failed'] = state['active']
        state['error'] = str(error)
        raise
    finally:
        state['ended_unix'] = time.time()
        write(control/'status.json', state)


if __name__ == '__main__':
    main()
