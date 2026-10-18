"""Published-source deployment and identity preflight for open processing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from open_processing_trial import ROOT,STAGE,REMOTE,read,write,sha,remote
from manage_coldstart_trial import scp


def deploy():
    publication=read(STAGE/'source_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    lock=read(STAGE/'execution_lock.json')
    files={name:ROOT/'scripts'/name for name in lock['runtime_code_sha256']}
    files.update({name:STAGE/name for name in ('execution_lock.json','PROTOCOL.md','AGENT_INTERFACE.txt')})
    for name,digest in lock['runtime_code_sha256'].items():assert sha(files[name])==digest
    assert sha(files['PROTOCOL.md'])==lock['protocol_sha256']
    assert sha(files['AGENT_INTERFACE.txt'])==lock['agent_interface_sha256']
    (STAGE/'private').mkdir(exist_ok=True);bundle=STAGE/'private/deployment.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    manifest=dict(source_commit=publication['source_commit'],bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,
                  files={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in files.items()})
    write(STAGE/'deployment_manifest.json',manifest)
    preparation=remote(f'''import pathlib,shutil,json,os,hashlib
p=pathlib.Path({REMOTE!r});assert p.resolve()==p and not p.exists()
assert shutil.disk_usage(p.parent).free>{lock['minimum_deploy_free_bytes']}
p.mkdir();probe=p/'storage_probe.bin';block=bytes(range(256))*4096
expected=hashlib.sha256()
with probe.open('xb') as stream:
 for i in range(64):stream.write(block);expected.update(block)
 stream.flush();os.fsync(stream.fileno())
with probe.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==expected.hexdigest() and probe.stat().st_size==64*1024**2
assert probe.resolve().parent==p
probe.unlink()
print(json.dumps(dict(status='PASS',created=str(p),free_bytes=shutil.disk_usage(p).free,probe_bytes=64*1024**2,probe_sha256=actual)))
''')
    write(STAGE/'storage_preflight_receipt.json',preparation)
    command,host=scp();subprocess.run(command+[str(bundle),host+':'+REMOTE+'/deployment.tar.gz'],check=True)
    result=remote(f'''import pathlib,json,tarfile,hashlib,os
p=pathlib.Path({REMOTE!r});m=json.loads({json.dumps(manifest)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'deployment.tar.gz')==m['bundle_sha256']
with tarfile.open(p/'deployment.tar.gz') as archive:
 assert {{v.name for v in archive.getmembers()}}==set(m['files'])
 for member in archive.getmembers():
  assert member.isfile() and pathlib.Path(member.name).name==member.name
  target=p/member.name;assert not target.exists()
  with archive.extractfile(member) as source,target.open('xb') as destination:
   destination.write(source.read());destination.flush();os.fsync(destination.fileno())
  assert sha(target)==m['files'][member.name]['sha256'] and target.stat().st_size==m['files'][member.name]['bytes']
receipt=dict(status='PASS',source_commit=m['source_commit'],files=len(m['files']),arrays_copied=False,D_accessed=False)
with (p/'deployment_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
''')
    write(STAGE/'deployment_receipt.json',result);print(json.dumps(result))


def preflight():
    assert read(STAGE/'deployment_receipt.json')['status']=='PASS'
    result=remote(f'''import pathlib,json,subprocess,sys
p=pathlib.Path({REMOTE!r});c=p/'preflight_control';c.mkdir()
assert not (p/'preflight_receipt.json').exists()
command=[sys.executable,'-B',str(p/'preflight_open_processing.py'),'--root',str(p)]
wrapper="import subprocess,pathlib,json,time; p=pathlib.Path("+repr(str(c))+"); start=time.time(); r=subprocess.run("+repr(command)+"); (p/'exit.json').write_text(json.dumps(dict(returncode=r.returncode,started=start,finished=time.time())))"
with (c/'run.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,'-B','-c',wrapper],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(status='started',pid=child.pid)
(c/'started.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
''')
    write(STAGE/'preflight_launch_receipt.json',result);print(json.dumps(result))


def status():
    value=remote(f'''import pathlib,json,os
p=pathlib.Path({REMOTE!r});c=p/'preflight_control'
if (c/'exit.json').exists():
 print(json.dumps(dict(done=True,exit=json.loads((c/'exit.json').read_text()),log=(c/'run.log').read_text()[-8000:],status=json.loads((p/'preflight_receipt.json').read_text())['status'] if (p/'preflight_receipt.json').exists() else None)))
else:
 pid=json.loads((c/'started.json').read_text())['pid']
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 print(json.dumps(dict(done=False,alive=alive)))
''')
    print(json.dumps(value))


def collect_preflight():
    value=remote(f'''import pathlib,json,hashlib
p=pathlib.Path({REMOTE!r});exit=json.loads((p/'preflight_control/exit.json').read_text());files={{}}
for name in ('preflight_receipt.json','parent_reuse_receipt.json','prepared_parent_facts.json','preflight_control/started.json','preflight_control/exit.json','preflight_control/run.log'):
 f=p/name
 if f.exists():
  with f.open('rb') as stream:dg=hashlib.file_digest(stream,'sha256').hexdigest()
  files[name]=dict(sha256=dg,bytes=f.stat().st_size)
print(json.dumps(dict(exit=exit,files=files)))
''')
    command,host=scp()
    for name,metadata in value['files'].items():
        target=STAGE/name;target.parent.mkdir(exist_ok=True,parents=True);assert not target.exists()
        subprocess.run(command+[host+':'+REMOTE+'/'+name,str(target)],check=True)
        assert sha(target)==metadata['sha256'] and target.stat().st_size==metadata['bytes']
    write(STAGE/'preflight_collection_receipt.json',value);print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('deploy','preflight','status','collect_preflight'))
    globals()[parser.parse_args().action]()
