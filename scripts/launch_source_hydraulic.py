"""One-shot detached launch; phase logs never depend on SSH stdout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import run_source_hydraulic as study

parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('forecast','audit'));parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output;plan=study.verify(out)
assert study.read(out/'storage_preflight.json')['status']=='PASS'
assert not (out/f'{args.phase}.lock').exists() and not (out/f'{args.phase}_failure.json').exists()
if args.phase=='audit':
    receipt=study.read(out/'prediction_sync_receipt.json')
    assert receipt['status']=='exact_commit_and_tree_verified' and receipt['predictions_sha256']==study.sha(out/'predictions.json')
control=out/'control'/args.phase;control.mkdir(parents=True)
command=[sys.executable,'-B',str(out/('run_source_hydraulic.py' if args.phase=='forecast' else 'audit_source_hydraulic.py')),'--output',str(out)]
with (control/'run.log').open('xb') as log:
    proc=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
study.write(control/'launch.json',dict(pid=proc.pid,phase=args.phase,started_unix=time.time(),plan_sha256=study.sha(out/'plan.json')))
print(json.dumps(dict(pid=proc.pid,phase=args.phase,status='launched')),flush=True)
