"""Resume-safe stage-0 test/evidence entry point, dedicated server only.

Copies existing server code into a new project; never writes the frozen source.
Training and external model calls are deliberately absent from this entry point.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--old-project',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--manifest',required=True)
    args=parser.parse_args()
    if os.name!='posix':raise RuntimeError('Dedicated research server required')
    root=Path(__file__).resolve().parents[1]
    old=Path(args.old_project).resolve();output=Path(args.output).resolve()
    if old==root or old in root.parents or root in old.parents:
        raise RuntimeError('Frozen source and new project must be independent directories')
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
    for name,expected in manifest['files'].items():
        if sha(root/name)!=expected:raise RuntimeError('Uploaded file identity differs: '+name)
    if output.exists():raise RuntimeError('Existing attempt will not be overwritten or rerun')
    if shutil.disk_usage(root).free<5*1024**3:raise RuntimeError('Insufficient evidence archive reserve')
    output.mkdir(parents=True)
    started=time.monotonic()
    status=dict(status='preparing',training_count=0,external_API_calls=0,test_access=False,
                manifest_sha256=sha(args.manifest),source_project=str(old),attempts=[])
    write(output/'progress.json',status)
    try:
        # Dependencies already on this same server; frozen R2 package is read only.
        shutil.copytree(old/'online_ts',root/'online_ts',dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name,expected in manifest['files'].items():
            if sha(root/name)!=expected:raise RuntimeError('Dependency copy changed a stage file')
        commands=[('unit_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(root/'tests'),
                                    '-p','test_intervention_phase0.py','-v'],120),
                  ('paired_evaluation',[sys.executable,'-B',str(root/'scripts/export_intervention_pairs.py'),
                     '--old-project',str(old),'--output',str(output/'paired_evaluation')],1800)]
        for name,command,timeout in commands:
            status['status']=name;write(output/'progress.json',status)
            step=time.monotonic()
            with (output/(name+'.log')).open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=False)
            attempt=dict(name=name,exit_code=result.returncode,process_seconds=time.monotonic()-step,
                         log_sha256=sha(output/(name+'.log')))
            status['attempts'].append(attempt)
            if result.returncode:raise RuntimeError(name+' failed; inspect retained log')
        status['status']='completed'
    except Exception as error:
        status['status']='failed';status['error']=str(error)
        raise
    finally:
        status['process_seconds']=time.monotonic()-started
        write(output/'progress.json',status)


if __name__=='__main__':main()
