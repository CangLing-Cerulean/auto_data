"""Deploy and collect a hash-locked A/B-only checkpoint reader on the existing host."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
from diagnosis_pool_features import read,sha,write
from manage_ett_confirmation import remote,transport

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_parent_measurements_r3'
REMOTE='/root/auto_data_experiments/20261018_parent_measurements_r3'
FILES=('conditional_diagnosis.py','diagnosis_pool_features.py','parent_measurement_tables.py','extract_ett_parent_measurements.py')

def prepare():
    private=STAGE/'private';private.mkdir()
    files={name:ROOT/'scripts'/name for name in FILES};files['PROTOCOL.md']=STAGE/'PROTOCOL.md'
    lock=dict(files={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in files.items()},remote_root=REMOTE,new_full_trainings=0,read_scope='existing ETT parent checkpoints and A/B arrays only')
    write(STAGE/'execution_lock.json',lock);files['execution_lock.json']=STAGE/'execution_lock.json'
    bundle=private/'reader.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,p in files.items():archive.add(p,arcname=name)
    write(STAGE/'deployment_lock.json',dict(bytes=bundle.stat().st_size,sha256=sha(bundle),files={name:sha(p) for name,p in files.items()}))

def deploy():
    lock=read(STAGE/'deployment_lock.json');bundle=STAGE/'private/reader.tar.gz';assert sha(bundle)==lock['sha256']
    for name in FILES:assert sha(ROOT/'scripts'/name)==lock['files'][name]
    assert sha(STAGE/'PROTOCOL.md')==lock['files']['PROTOCOL.md']
    remote(f'''import pathlib,shutil,os,json
p=pathlib.Path({REMOTE!r});assert p.resolve()==p and not p.exists() and shutil.disk_usage(p.parent).free>64*1024**2
p.mkdir();probe=p/'probe.bin';payload=bytes(1024*1024)
with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
assert probe.read_bytes()==payload
probe.unlink()
print(json.dumps(dict(created=str(p),free_bytes=shutil.disk_usage(p).free)))
''')
    _,scp,host=transport();subprocess.run(scp+[str(bundle),host+':'+REMOTE+'/reader.tar.gz'],check=True)
    remote(f'''import pathlib,tarfile,hashlib,json,os
p=pathlib.Path({REMOTE!r});lock=json.loads({json.dumps(lock)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'reader.tar.gz')==lock['sha256']
with tarfile.open(p/'reader.tar.gz') as tar:
 members=tar.getmembers();assert len(members)==len(lock['files']) and {{m.name for m in members}}==set(lock['files'])
 for m in members:
  assert m.isfile() and pathlib.Path(m.name).name==m.name
  with tar.extractfile(m) as src,(p/m.name).open('xb') as dst:dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert sha(p/m.name)==lock['files'][m.name]
with (p/'deployment_receipt.json').open('x') as f:json.dump(dict(status='PASS',bundle_sha256=lock['sha256']),f)
print('Read-only parent measurement deployment PASS')
''')

def extract():
    remote(f'''import pathlib,json,time,subprocess
p=pathlib.Path({REMOTE!r});assert json.loads((p/'deployment_receipt.json').read_text())['status']=='PASS'
with (p/'started.json').open('x') as f:json.dump(dict(started=time.time()),f)
start=time.perf_counter()
with (p/'extraction.log').open('x') as log:result=subprocess.run(['/opt/miniconda3/bin/python','-B',str(p/'extract_ett_parent_measurements.py')],stdout=log,stderr=subprocess.STDOUT)
receipt=dict(exit_code=result.returncode,seconds=time.perf_counter()-start)
with (p/'exit.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt));print((p/'extraction.log').read_text()[-2000:])
''')

def collect():
    _,scp,host=transport()
    for name in ('deployment_receipt.json','started.json','exit.json','extraction.log','ett_measurements.json','tables/ETT_ETTh1.npz','tables/ETT_ETTh2.npz'):
        target=STAGE/name;assert not target.exists()
        subprocess.run(scp+[host+':'+REMOTE+'/'+name,str(target)],check=True)
    assert read(STAGE/'exit.json')['exit_code']==0
    result=read(STAGE/'ett_measurements.json');assert result['status']=='PASS'
    for name,info in result['assets'].items():assert sha(STAGE/'tables'/name)==info['sha256'] and (STAGE/'tables'/name).stat().st_size==info['bytes']
    print('ETT parent-only measurement collection PASS')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','deploy','extract','collect'))
    try:globals()[parser.parse_args().action]()
    except subprocess.CalledProcessError as exc:raise SystemExit(f'Transport failed with exit {exc.returncode}; endpoint omitted')
