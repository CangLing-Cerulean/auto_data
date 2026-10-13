"""Detached create-only sampling study launcher with prediction-lock checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('forecast', 'labels'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    plan = read(out / 'plan.json')
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    assert read(out / 'preparation.json')['status'] == 'PASS'
    assert not (out / f'{args.phase}.lock').exists()
    if args.phase == 'forecast':
        assert not (out / 'runs').exists()
    else:
        assert read(out / 'forecast_status.json')['status'] == 'completed'
        assert read(out / 'predictions.json')['budget_gate']
        receipt = read(out / 'prediction_sync_receipt.json')
        assert receipt['status'] == 'exact_commit_and_tree_verified'
        assert receipt['predictions_sha256'] == sha(out / 'predictions.json')
        assert receipt['prediction_lock_sha256'] == sha(out / 'prediction_lock.json')
    control = out / 'control' / args.phase
    control.mkdir(parents=True)
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (control / 'run.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(out / 'response_proxy_selection.py'), args.phase,
            '--output', str(out)], cwd=out, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    with (control / 'launch.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=process.pid, phase=args.phase, started_unix=time.time(),
                       plan_sha256=sha(out / 'plan.json'), output_transport='regular_file'), stream, indent=2)
    print(json.dumps(dict(pid=process.pid, phase=args.phase, status='launched')), flush=True)


if __name__ == '__main__':
    main()
