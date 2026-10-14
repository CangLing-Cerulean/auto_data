"""Preserve all new SOURCE study assets and failures after verified termination."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
args=parser.parse_args();out=args.output
assert out==Path('/root/auto_data_experiments/20261014_source_hydraulic_r1') and out.resolve()==out
assert sha(out/'plan.json')==args.plan_sha256
for launch in sorted((out/'control').glob('*/launch.json')):
    pid=read(launch)['pid'];stat=Path('/proc')/str(pid)/'stat'
    if stat.exists():assert stat.read_text().rsplit(')',1)[1].split()[0]=='Z',f'Process {pid} is still active'
assert (out/'final_audit.json').exists() or (out/'audit_failure.json').exists() or (out/'forecast_failure.json').exists()
started=time.perf_counter();paths=sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.md','.py','.npz','.log','.lock'))
assert all(not p.is_symlink() for p in paths)
assert shutil.disk_usage(out).free>sum(p.stat().st_size for p in paths)+256*1024**2
hashes={p.relative_to(out).as_posix():sha(p) for p in paths}
archive=out/'source_hydraulic_20261014.tar.gz'
with tarfile.open(archive,'x:gz') as tar:
    for name in hashes:tar.add(out/name,arcname=name,recursive=False)
with tarfile.open(archive,'r:gz') as tar:
    assert tar.getnames()==list(hashes)
    for m in tar.getmembers():assert m.isfile() and hashlib.file_digest(tar.extractfile(m),'sha256').hexdigest()==hashes[m.name]
for name,digest in hashes.items():assert sha(out/name)==digest
receipt=dict(status='PASS',file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),members_sha256=hashes,
    archive_seconds=time.perf_counter()-started,new_full_trainings=0,retrospective=True)
write(out/'archive_receipt.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','file','bytes','sha256')}))
