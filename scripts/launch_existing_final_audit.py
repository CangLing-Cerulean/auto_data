"""Explicit one-shot launcher for the publicly locked final audit revision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output
    assert out==Path('/root/auto_data_experiments/20261014_existing_sources_r1') and out.resolve()==out
    lock=read(out/'audit_revision_lock.json');sync=read(out/'prediction_sync_receipt.json')
    assert sync['status']=='exact_commit_and_tree_verified'
    assert sync['audit_revision_lock_sha256']==sha(out/'audit_revision_lock.json')
    assert lock['auditor_sha256']==sha(out/'audit_response_proxy_existing_r2.py')
    assert lock['launcher_sha256']==sha(Path(__file__))
    assert read(out/'labels.json')['status']=='completed' and not (out/'final_audit.json').exists()
    control=out/'control'/'audit_final_r2';control.mkdir()
    with (control/'run.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-B',str(out/'audit_response_proxy_existing_r2.py'),'final','--output',str(out)],
            cwd=out,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    result=dict(pid=process.pid,phase='audit_final_r2',started_unix=time.time(),audit_revision_lock_sha256=sha(out/'audit_revision_lock.json'))
    with (control/'launch.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
