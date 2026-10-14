"""Archive this registered stage only after its processes have terminated."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--outcome',choices=('completed','stopped'),required=True)
    args=p.parse_args();out=args.output
    assert out==Path('/root/auto_data_experiments/20261014_existing_sources_r1') and out.resolve()==out
    assert sha(out/'plan.json')==args.plan_sha256
    for launch in sorted((out/'control').glob('*/launch.json')):
        pid=read(launch)['pid'];proc=Path('/proc')/str(pid)
        if proc.exists():
            status=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
            assert status=='Z',f'Process {pid} still active; do not snapshot changing files'
    if args.outcome=='completed':assert read(out/'final_audit.json')['status']=='PASS'
    else:assert (out/'STOP_REPORT.md').is_file()
    started=time.perf_counter()
    write(out/'ARCHIVE_LOCK.json',dict(outcome=args.outcome,plan_sha256=args.plan_sha256,created_unix=time.time(),
        base_validation_access=False,test_access=False))
    allowed={'.py','.md','.json','.npz','.pt','.log','.lock'}
    paths=sorted(f for f in out.rglob('*') if f.is_file() and (f.suffix in allowed or f.name=='LICENSE' or f==out/'input'/'skab_train.csv'))
    assert all(not f.is_symlink() for f in paths)
    assert not any(f.name in ('test.csv','validation.csv') for f in paths)
    total=sum(f.stat().st_size for f in paths)
    assert shutil.disk_usage(out).free>total+256*1024**2,'Reserve enough space for complete archive'
    hashes={f.relative_to(out).as_posix():sha(f) for f in paths}
    archive=out/'response_proxy_existing_sources_20261014.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in hashes:tar.add(out/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        assert tar.getnames()==list(hashes)
        for member in tar.getmembers():
            assert member.isfile()
            assert hashlib.file_digest(tar.extractfile(member),'sha256').hexdigest()==hashes[member.name]
    for name,digest in hashes.items():assert sha(out/name)==digest
    write(out/'archive_receipt.json',dict(status='PASS',file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),
        members_sha256=hashes,archive_seconds=time.perf_counter()-started,outcome=args.outcome))
    print(json.dumps(dict(status='PASS',file=archive.name,bytes=archive.stat().st_size,members=len(hashes))),flush=True)
