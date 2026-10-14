"""Single-use phase launcher with a detached regular-file supervisor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import response_proxy_file_transfer as study

PHASES = ('prepare', 'preflight', 'parents', 'audit_parents', 'forecast', 'audit_forecast', 'labels', 'final')


def validate(out, phase):
    plan = study.verify(out)
    if phase == 'prepare':
        assert not (out / 'prepared').exists() and not (out / 'preparation.json').exists()
    elif phase == 'preflight':
        assert study.read(out / 'preparation.json')['status'] == 'PASS'
        assert not (out / 'preflight_audit.json').exists()
    else:
        study.check_preflight(out)
        if phase == 'parents':
            assert not (out / 'runs').exists() and not (out / 'parents.lock').exists()
        elif phase == 'audit_parents':
            assert (out / 'calibration.json').is_file() and not (out / 'parents_audit.json').exists()
        elif phase == 'forecast':
            study.check_calibration(out)
            assert not (out / 'forecast.lock').exists()
        elif phase == 'audit_forecast':
            study.check_calibration(out)
            assert (out / 'predictions.json').is_file() and not (out / 'forecast_audit.json').exists()
        elif phase == 'labels':
            study.check_calibration(out)
            p = study.read(out / 'predictions.json'); a = study.read(out / 'forecast_audit.json')
            s = study.read(out / 'prediction_sync_receipt.json')
            assert p['eligible_to_label'] and a['status'] == 'PASS' and a['eligible_to_label']
            assert s['status'] == 'exact_commit_and_tree_verified'
            assert s['plan_sha256'] == study.sha(out / 'plan.json')
            assert s['predictions_sha256'] == a['predictions_sha256'] == study.sha(out / 'predictions.json')
            assert s['forecast_audit_sha256'] == study.sha(out / 'forecast_audit.json')
            assert not (out / 'labels.lock').exists()
        else:
            assert (out / 'labels.json').is_file() and not (out / 'final_audit.json').exists()
    return plan


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args(); out, phase = args.output, args.phase
    validate(out, phase)
    control = out / 'control' / phase
    if args.supervise:
        assert control.is_dir() and not (control / 'child.json').exists() and not (control / 'exit.json').exists()
        audit_phase = phase in ('preflight', 'audit_parents', 'audit_forecast', 'final')
        script = 'audit_response_proxy_file_transfer.py' if audit_phase else 'response_proxy_file_transfer.py'
        command = [sys.executable, '-B', str(out / script), phase.removeprefix('audit_'), '--output', str(out)]
        started = time.time()
        child = subprocess.Popen(command, cwd=out, stdin=subprocess.DEVNULL)
        study.write(control / 'child.json', dict(pid=child.pid, supervisor_pid=os.getpid(), command=command, started_unix=started))
        code = child.wait()
        study.write(control / 'exit.json', dict(exit_code=code, started_unix=started, finished_unix=time.time(), phase=phase))
        raise SystemExit(code)
    control.mkdir(parents=True)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), phase, '--output', str(out), '--supervise']
    with (control / 'run.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=out, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'),
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    study.write(control / 'launch.json', dict(pid=process.pid, phase=phase, started_unix=time.time(), plan_sha256=study.sha(out / 'plan.json')))
    print(json.dumps(dict(pid=process.pid, phase=phase, status='supervisor_launched')), flush=True)


if __name__ == '__main__':
    main()
