"""Publish-lock guarded deployment and exact result collection for cold-start r10."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time
from coldstart_trial import ROOT,STAGE,REMOTE,read,write,sha,remote,transport


def scp():
    c=read(ROOT/'research_runs/ett_confirmation/private/connection.json')
    return ['scp','-i',c['identity_file'],'-o','BatchMode=yes','-o','ConnectTimeout=15','-P',str(c['port'])],c['host']


def deploy():
    pub=read(STAGE/'source_publication_receipt.json');assert pub['status']=='exact_commit_and_tree_verified'
    lock=read(STAGE/'execution_lock.json')
    files={name:ROOT/'scripts'/name for name in lock['runtime_code_sha256']}
    files.update({name:STAGE/name for name in ('execution_lock.json','PROTOCOL.md','source_index.json')})
    files.update({name:STAGE/'inputs'/name for name in ('ABC.npz','data_summary.json','input_receipt.json')})
    for name,path in files.items():
        if name in lock['runtime_code_sha256']: assert sha(path)==lock['runtime_code_sha256'][name]
    receipt=read(STAGE/'inputs/input_receipt.json');assert receipt['status']=='PASS' and not receipt['D_values_parsed']
    assert sha(files['ABC.npz'])==receipt['arrays_sha256']
    assert sha(files['data_summary.json'])==receipt['data_summary_sha256']
    private=STAGE/'private';private.mkdir(exist_ok=True)
    bundle=private/'runtime_inputs.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    manifest=dict(files={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in files.items()},
                  bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,source_commit=pub['source_commit'])
    write(STAGE/'deployment_manifest.json',manifest)
    remote(f'''import pathlib,shutil,json
p=pathlib.Path({REMOTE!r});assert p.resolve()==p and not p.exists()
assert shutil.disk_usage(p.parent).free>{lock['minimum_deploy_free_bytes']}
p.mkdir();print(json.dumps(dict(created=str(p),free_bytes=shutil.disk_usage(p).free)))
''')
    command,host=scp();subprocess.run(command+[str(bundle),host+':'+REMOTE+'/runtime_inputs.tar.gz'],check=True)
    result=remote(f'''import pathlib,tarfile,hashlib,json,os,shutil
p=pathlib.Path({REMOTE!r});m=json.loads({json.dumps(manifest)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'runtime_inputs.tar.gz')==m['bundle_sha256']
with tarfile.open(p/'runtime_inputs.tar.gz') as archive:
 assert {{v.name for v in archive.getmembers()}}==set(m['files'])
 for member in archive.getmembers():
  assert member.isfile() and pathlib.Path(member.name).name==member.name
  with archive.extractfile(member) as src,(p/member.name).open('xb') as dst:dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert sha(p/member.name)==m['files'][member.name]['sha256']
receipt=dict(status='PASS',files=len(m['files']),source_commit=m['source_commit'],free_bytes=shutil.disk_usage(p).free,D_included=False)
with (p/'deployment_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
''')
    write(STAGE/'deployment_receipt.json',result)
    print(json.dumps(result))


def status():
    print(json.dumps(remote(f'''import pathlib,json
p=pathlib.Path({REMOTE!r});out=dict(controls={{}},runs=[])
if (p/'control').exists():
 for q in sorted((p/'control').iterdir()):
  out['controls'][q.name]=json.loads((q/'exit.json').read_text()) if (q/'exit.json').exists() else dict(running=True)
for q in sorted((p/'runs').glob('*/*/result.json')):out['runs'].append(dict(run=q.parent.parent.name,seed=q.parent.name,status=json.loads(q.read_text())['status']))
print(json.dumps(out))
''')))


def collect():
    result=remote(f'''import pathlib,json,hashlib,tarfile
p=pathlib.Path({REMOTE!r});out=p/'results.tar.gz'
assert not out.exists()
with tarfile.open(out,'x:gz') as archive:
 for category in ('runs','queries','control'):
  for f in sorted((p/category).rglob('*')):
   if f.is_file():archive.add(f,arcname=f.relative_to(p).as_posix())
 for name in ('parent_facts.json','parent_receipt.json','deployment_receipt.json'):archive.add(p/name,arcname=name)
with out.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=out.name,bytes=out.stat().st_size,sha256=digest)))
''')
    archives=STAGE/'archives';archives.mkdir(exist_ok=True)
    command,host=scp();target=archives/result['path'];assert not target.exists()
    subprocess.run(command+[host+':'+REMOTE+'/'+result['path'],str(target)],check=True)
    assert sha(target)==result['sha256'] and target.stat().st_size==result['bytes']
    write(STAGE/'archive_receipt.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('deploy','status','collect'));args=parser.parse_args();globals()[args.action]()
