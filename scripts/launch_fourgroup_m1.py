"""Single-use detached phase supervisor with ordinary persistent log files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import fourgroup_runtime as run
import run_fourgroup_m1 as study

PHASES = ('prepare', 'preflight', 'calibrate', 'audit_calibration',
          'forecast', 'audit_forecast', 'labels', 'final')


def validate(out, phase):
    run.verify(out)
    if phase == 'prepare':
        assert not (out / 'prepared').exists()
    elif phase == 'preflight':
        assert run.read(out / 'preparation.json')['status'] == 'PASS'
        assert not (out / 'preflight_audit.json').exists()
    else:
        study.preflight_pass(out)
        if phase == 'calibrate':
            assert not (out / 'calibration.lock').exists()
        elif phase == 'audit_calibration':
            assert (out / 'calibration.json').is_file() and not (out / 'calibration_audit.json').exists()
        elif phase == 'forecast':
            study.calibration_pass(out)
            assert not (out / 'forecast.lock').exists()
        elif phase == 'audit_forecast':
            study.calibration_pass(out)
            assert (out / 'predictions.json').is_file() and not (out / 'forecast_audit.json').exists()
        elif phase == 'labels':
            study.labels_pass(out)
            assert not (out / 'labels.lock').exists()
        else:
            assert (out / 'labels.json').is_file() and not (out / 'final_audit.json').exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    out, phase = args.output, args.phase
    validate(out, phase)
    control = out / 'control' / phase
    if args.supervise:
        assert control.is_dir() and not (control / 'child.json').exists() and not (control / 'exit.json').exists()
        audit = phase in ('preflight', 'audit_calibration', 'audit_forecast', 'final')
        script = 'audit_fourgroup_m1.py' if audit else 'run_fourgroup_m1.py'
        command = [sys.executable, '-B', str(out / script), phase.removeprefix('audit_'), '--output', str(out)]
        started = time.time()
        child = subprocess.Popen(command, cwd=out, stdin=subprocess.DEVNULL)
        run.write(control / 'child.json', dict(pid=child.pid, supervisor_pid=os.getpid(), command=command, started_unix=started))
        code = child.wait()
        run.write(control / 'exit.json', dict(exit_code=code, started_unix=started,
            finished_unix=time.time(), phase=phase))
        raise SystemExit(code)
    control.mkdir(parents=True)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), phase, '--output', str(out), '--supervise']
    with (control / 'run.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=out, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'))
    run.write(control / 'launch.json', dict(pid=process.pid, phase=phase,
        started_unix=time.time(), plan_sha256=run.sha(out / 'plan.json')))
    print(dict(status='supervisor_launched', phase=phase, pid=process.pid), flush=True)


if __name__ == '__main__':
    main()
