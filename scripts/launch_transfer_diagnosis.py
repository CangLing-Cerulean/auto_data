"""One-shot phase supervision; failures and partial outputs are never overwritten."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main(stage, source, shrinkage):
    control = stage / 'control'
    control.mkdir(exist_ok=False)
    status = dict(status='running', phase=None, started_utc=datetime.now(timezone.utc).isoformat())
    def save():
        (control / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    phases = [('prepare_transfer_diagnosis.py', ['--source', str(source), '--shrinkage', str(shrinkage)]),
              ('run_transfer_diagnosis.py', []), ('audit_transfer_diagnosis.py', []),
              ('evaluate_transfer_diagnosis.py', [])]
    for name, extra in phases:
        status['phase'] = name
        save()
        with (control / (name + '.log')).open('x', encoding='utf-8') as log:
            completed = subprocess.run([sys.executable, str(stage / 'source_snapshot' / name),
                                        '--stage', str(stage), *extra], stdout=log, stderr=subprocess.STDOUT)
        status['returncode'] = completed.returncode
        if completed.returncode:
            status['status'] = 'failed'
            save()
            raise SystemExit(completed.returncode)
    status['status'] = 'completed'
    status['finished_utc'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--shrinkage', type=Path, required=True)
    a = p.parse_args()
    main(a.stage, a.source, a.shrinkage)
