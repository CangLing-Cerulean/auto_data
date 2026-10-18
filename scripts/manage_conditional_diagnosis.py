"""Transport the small read-only ETT extractor to the established research host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from diagnosis_pool_features import read,sha,write
from manage_ett_confirmation import remote,transport

LOCAL=Path(__file__).resolve().parents[1]
STAGE=LOCAL/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
REMOTE='/root/auto_data_experiments/20261018_conditional_diagnosis_r1'
FILES=('conditional_diagnosis.py','diagnosis_pool_features.py','extract_ett_diagnosis.py')


def prepare():
    private=STAGE/'private';private.mkdir(exist_ok=True)
    files={name:LOCAL/'scripts'/name for name in FILES}
    files['PROTOCOL.md']=STAGE/'PROTOCOL.md'
    lock=dict(authorization='2026-10-18 user requested experiments for conditional diagnosis paper',
              scope='read-only original A/B/C and 30 saved ETT checkpoints; zero optimizer updates',
              remote_root=REMOTE,files={name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in files.items()},
              minimum_free_bytes=64*1024**2,maximum_new_full_trainings=0,D_test_access=False)
    write(STAGE/'extraction_lock.json',lock)
    files['extraction_lock.json']=STAGE/'extraction_lock.json'
    bundle=private/'extractor.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    write(STAGE/'deployment_lock.json',dict(sha256=sha(bundle),bytes=bundle.stat().st_size,
        files={name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in files.items()}))


def deploy():
    source=read(STAGE/'source_publication_receipt.json')
    assert source['status']=='exact_commit_and_tree_verified'
    lock=read(STAGE/'deployment_lock.json');bundle=STAGE/'private/extractor.tar.gz'
    assert sha(bundle)==lock['sha256']
    for path in [LOCAL/'scripts'/name for name in FILES]+[STAGE/'PROTOCOL.md',STAGE/'extraction_lock.json',STAGE/'deployment_lock.json']:
        blob=subprocess.run(['git','-c',f'safe.directory={LOCAL.as_posix()}','show',
            source['source_commit']+':'+path.relative_to(LOCAL).as_posix()],capture_output=True,check=True).stdout
        assert __import__('hashlib').sha256(blob).hexdigest()==sha(path)
    remote(f'''import pathlib,shutil,json,os
p=pathlib.Path({REMOTE!r})
assert p.resolve()==p and not p.exists() and shutil.disk_usage(p.parent).free>64*1024**2
p.mkdir()
probe=p/'storage_probe.bin'
payload=bytes(1024*1024)
with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==payload
probe.unlink()
print(json.dumps(dict(created=str(p),free_bytes=shutil.disk_usage(p).free,storage_probe_bytes=len(payload))))
''')
    _,scp,host=transport();subprocess.run(scp+[str(bundle),host+':'+REMOTE+'/extractor.tar.gz'],check=True)
    remote(f'''import pathlib,tarfile,hashlib,json,os
p=pathlib.Path({REMOTE!r});lock=json.loads({json.dumps(lock)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'extractor.tar.gz')==lock['sha256']
with tarfile.open(p/'extractor.tar.gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(lock['files']) and {{m.name for m in members}}==set(lock['files'])
 for m in members:
  assert m.isfile() and pathlib.Path(m.name).name==m.name
  with archive.extractfile(m) as src,(p/m.name).open('xb') as dst:dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert sha(p/m.name)==lock['files'][m.name]['sha256']
with (p/'deployment_receipt.json').open('x') as f:json.dump(dict(status='PASS',source_commit={source['source_commit']!r},files=len(members)),f,indent=2)
print('Frozen extractor deployment PASS; no training launched')
''')


def extract():
    # Ordinary-file logs survive an SSH stdout interruption; registration is create-only.
    remote(f'''import pathlib,json,subprocess,time
p=pathlib.Path({REMOTE!r})
assert json.loads((p/'deployment_receipt.json').read_text())['status']=='PASS'
with (p/'extraction_started.json').open('x') as f:json.dump(dict(started=time.time()),f)
started=time.perf_counter()
with (p/'extraction.log').open('x') as log:
 result=subprocess.run(['/opt/miniconda3/bin/python','-B',str(p/'extract_ett_diagnosis.py')],stdout=log,stderr=subprocess.STDOUT)
receipt=dict(exit_code=result.returncode,seconds=time.perf_counter()-started)
with (p/'extraction_exit.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
print((p/'extraction.log').read_text()[-3000:])
''')


def collect():
    _,scp,host=transport()
    for name in ('deployment_receipt.json','extraction_started.json','extraction_exit.json','extraction.log','ett_extraction.json'):
        target=STAGE/name
        assert not target.exists()
        subprocess.run(scp+[host+':'+REMOTE+'/'+name,str(target)],check=True)
    assert read(STAGE/'extraction_exit.json')['exit_code']==0
    assert read(STAGE/'ett_extraction.json')['status']=='PASS'
    print('Existing-checkpoint extraction collected; full training count unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','deploy','extract','collect'))
    try:globals()[parser.parse_args().action]()
    except subprocess.CalledProcessError as exc:
        print(f'Transport command failed with exit {exc.returncode}; private endpoint omitted',file=sys.stderr)
        sys.exit(exc.returncode)
