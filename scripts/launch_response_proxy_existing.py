"""Detached launcher, with regular-file output and one-shot phase guards."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import response_proxy_existing as study

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('parents','forecast','labels','audit_parents','audit_final'));p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);plan=study.verify(out)
    preflight=study.read(out/'preflight_audit.json')
    assert preflight['status']=='PASS'
    assert preflight['plan_sha256']==study.sha(out/'plan.json')
    assert preflight['preparation_sha256']==study.sha(out/'preparation.json')
    phase=args.phase
    if phase=='parents':assert not (out/'runs').exists() and not (out/'parents.lock').exists()
    elif phase=='forecast':
        parent_audit=study.read(out/'parents_audit.json')
        assert parent_audit['status']=='PASS' and parent_audit['calibration_sha256']==study.sha(out/'calibration.json')
        study.check_calibration(out)
        assert not (out/'forecast.lock').exists()
    elif phase=='labels':
        assert study.read(out/'predictions.json')['budget_gate'] and not (out/'labels.lock').exists()
        assert study.read(out/'prediction_sync_receipt.json')['predictions_sha256']==study.sha(out/'predictions.json')
    elif phase=='audit_parents':assert (out/'calibration.json').exists() and not (out/'parents_audit.json').exists()
    else:assert (out/'labels.json').exists() and not (out/'final_audit.json').exists()
    control=out/'control'/phase;control.mkdir(parents=True)
    command=[sys.executable,'-B',str(out/('audit_response_proxy_existing.py' if phase.startswith('audit_') else 'response_proxy_existing.py')),
        phase.removeprefix('audit_'),'--output',str(out)]
    with (control/'run.log').open('xb') as log:
        process=subprocess.Popen(command,cwd=out,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    study.write(control/'launch.json',dict(pid=process.pid,phase=phase,started_unix=time.time(),plan_sha256=study.sha(out/'plan.json')))
    print(json.dumps(dict(pid=process.pid,phase=phase,status='launched')),flush=True)
