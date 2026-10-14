"""Detached one-shot execution for the two training-only diagnostic phases."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import run_sampling_information as study

parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('diagnostic','audit'));parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output;study.verify(out)
assert study.read(out/'storage_preflight.json')['status']=='PASS'
assert not (out/f'{args.phase}.lock').exists() and not (out/f'{args.phase}_failure.json').exists()
if args.phase=='audit':assert study.read(out/'diagnostic_result_lock.json')['diagnostic_sha256']==study.sha(out/'diagnostic.json')
control=out/'control'/args.phase;control.mkdir(parents=True)
command=[sys.executable,'-B',str(out/('run_sampling_information.py' if args.phase=='diagnostic' else 'audit_sampling_information.py')),'--output',str(out)]
with (control/'run.log').open('xb') as log:
    process=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
study.write(control/'launch.json',dict(pid=process.pid,phase=args.phase,started_unix=time.time(),plan_sha256=study.sha(out/'plan.json')))
print(dict(pid=process.pid,phase=args.phase,status='launched'),flush=True)
