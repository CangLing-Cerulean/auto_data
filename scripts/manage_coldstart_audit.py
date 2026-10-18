"""Deploy and collect the frozen post-run audit, without restarting training."""
import argparse
import json
import subprocess
import tarfile
from pathlib import Path

from coldstart_trial import ROOT, STAGE, REMOTE, read, write, sha, remote
from manage_coldstart_trial import scp


def launch():
    assert read(STAGE/'trial_completed.json')['status']=='completed'
    assert read(STAGE/'final_recommendations.json')['status']=='FROZEN'
    publication=read(STAGE/'auditor_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    lock=read(STAGE/'auditor_lock.json')
    auditor=ROOT/'scripts/audit_coldstart_trial.py'
    assert sha(auditor)==lock['auditor_sha256']
    files={'audit_coldstart_trial.py':auditor}
    for name in ('trial_completed.json','final_recommendations.json','parent_facts.json','auditor_lock.json'):
        files['audit_decisions/'+name]=STAGE/name
    for category in ('calls','snapshots'):
        for path in sorted((STAGE/category).rglob('*')):
            if path.is_file():files['audit_decisions/'+path.relative_to(STAGE).as_posix()]=path
    assert len(list((STAGE/'snapshots').glob('*.json')))==16
    bundle=STAGE/'private/audit_inputs.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    manifest=dict(bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,
                  files={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in files.items()},
                  auditor_source_commit=publication['source_commit'])
    write(STAGE/'audit_deployment_manifest.json',manifest)
    remote(f'''import pathlib,json
p=pathlib.Path({REMOTE!r});assert p.resolve()==p
assert not (p/'audit_inputs.tar.gz').exists()
assert not (p/'audit_control').exists()
assert not (p/'development_audit.json').exists()
print(json.dumps(dict(ready=True)))
''')
    command,host=scp()
    subprocess.run(command+[str(bundle),host+':'+REMOTE+'/audit_inputs.tar.gz'],check=True)
    receipt=remote(f'''import pathlib,json,tarfile,hashlib,subprocess,sys,os
p=pathlib.Path({REMOTE!r});m=json.loads({json.dumps(manifest)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'audit_inputs.tar.gz')==m['bundle_sha256']
with tarfile.open(p/'audit_inputs.tar.gz') as archive:
 assert {{v.name for v in archive.getmembers()}}==set(m['files'])
 for member in archive.getmembers():
  target=p/member.name
  assert member.isfile() and target.resolve().is_relative_to(p) and not target.exists()
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as src,target.open('xb') as dst:dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert sha(target)==m['files'][member.name]['sha256'] and target.stat().st_size==m['files'][member.name]['bytes']
c=p/'audit_control';c.mkdir()
command=[sys.executable,'-B',str(p/'audit_coldstart_trial.py'),'--root',str(p),'--decisions',str(p/'audit_decisions'),'--output',str(p/'development_audit.json')]
wrapper="import subprocess,json,pathlib,time; p=pathlib.Path("+repr(str(c))+"); started=time.time(); result=subprocess.run("+repr(command)+"); (p/'exit.json').write_text(json.dumps(dict(returncode=result.returncode,started=started,finished=time.time())))"
with (c/'run.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,'-B','-c',wrapper],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='started',pid=child.pid,files=len(m['files']),bundle_sha256=m['bundle_sha256'])
(c/'started.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
''')
    write(STAGE/'audit_launch_receipt.json',receipt)
    print(json.dumps(receipt))


def status():
    value=remote(f'''import pathlib,json,os
p=pathlib.Path({REMOTE!r});c=p/'audit_control'
if (c/'exit.json').exists():
 print(json.dumps(dict(done=True,exit=json.loads((c/'exit.json').read_text()),result=json.loads((p/'development_audit.json').read_text()) if (p/'development_audit.json').exists() else None,log=(c/'run.log').read_text()[-4000:])))
else:
 pid=json.loads((c/'started.json').read_text())['pid']
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 print(json.dumps(dict(done=False,alive=alive)))
''')
    print(json.dumps(value))


def collect():
    value=remote(f'''import pathlib,json,hashlib
p=pathlib.Path({REMOTE!r});assert (p/'audit_control/exit.json').exists()
files={{}}
for name in ('development_audit.json','audit_control/started.json','audit_control/exit.json','audit_control/run.log'):
 f=p/name
 if f.exists():
  with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  files[name]=dict(sha256=digest,bytes=f.stat().st_size)
print(json.dumps(dict(files=files,exit=json.loads((p/'audit_control/exit.json').read_text()))))
''')
    command,host=scp()
    for name,metadata in value['files'].items():
        target=STAGE/name;target.parent.mkdir(parents=True,exist_ok=True)
        assert not target.exists()
        subprocess.run(command+[host+':'+REMOTE+'/'+name,str(target)],check=True)
        assert sha(target)==metadata['sha256'] and target.stat().st_size==metadata['bytes']
    write(STAGE/'audit_collection_receipt.json',value)
    print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','status','collect'))
    globals()[parser.parse_args().action]()
