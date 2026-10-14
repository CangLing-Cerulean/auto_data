"""Launch one frozen phase with ordinary persistent stdout/stderr files."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import run_weighted_response as run

p=argparse.ArgumentParser();p.add_argument('phase',choices=['labels','audit']);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
plan=run.verify(a.output);assert run.read(a.output/'preflight.json')['status']=='PASS'
assert not (a.output/f'{a.phase}.lock').exists()
if a.phase=='audit':assert run.read(a.output/'status.json')['status']=='completed'
else:assert not (a.output/'runs').exists()
control=a.output/'control'/a.phase;control.mkdir(parents=True)
with (control/'run.log').open('xb') as log:
    proc=subprocess.Popen([sys.executable,'-B',str(a.output/('run_weighted_response.py' if a.phase=='labels' else 'audit_weighted_response.py')),'--output',str(a.output)],
        cwd=a.output,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
run.write(control/'launch.json',dict(pid=proc.pid,phase=a.phase,started_unix=time.time(),plan_sha256=run.sha(a.output/'plan.json')))
print(dict(pid=proc.pid,phase=a.phase),flush=True)
