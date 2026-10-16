"""Local create-only ETT deployment, status and result transfer over existing SSH."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

LOCAL=Path(__file__).resolve().parents[1]
STAGE=LOCAL/'research_runs/ett_confirmation'
REMOTE='/root/auto_data_experiments/20261016_ett_confirmation_r1'
HOST='root@instance-lcpdyopw.zju.smartml.cn'
KEY=Path.home()/'.ssh/smartml_codex'
SSH=['ssh','-i',str(KEY),'-p','20000','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]
SCP=['scp','-i',str(KEY),'-P','20000','-o','BatchMode=yes','-o','ConnectTimeout=15']
CODE=('ett_confirmation.py','launch_ett_confirmation.py','run_research.py')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def remote(script):
    ast.parse(script)
    result=subprocess.run(SSH+['/opt/miniconda3/bin/python','-B','-'],input=script.encode(),capture_output=True,check=True)
    print(result.stdout.decode(),end='')
    return result.stdout


def prepare():
    lock=dict(authorization='2026-10-16 user requested completion of ETT confirmation',
        remote_root=REMOTE,protocol_sha256=sha(STAGE/'PROTOCOL.md'),manifest_sha256=sha(LOCAL/'data/ett_v1/manifest.json'),
        data_lock_sha256=sha(STAGE/'data_lock.json'),code_sha256={n:sha(LOCAL/'scripts'/n) for n in CODE},
        optimizer=dict(name='torch.optim.Adam',foreach=False,fused=False,learning_rate=.001,betas=[.9,.999],epsilon=1e-8,weight_decay=0),
        cumulative_before=434,maximum_new_full_trainings=30,estimated_saved_assets_upper_bytes=160*1024**2,
        minimum_free_bytes=256*1024**2,source_protocol_unchanged=True)
    write(STAGE/'execution_lock.json',lock)
    private=STAGE/'private';private.mkdir(exist_ok=True)
    files={n:LOCAL/'scripts'/n for n in CODE}
    files.update({n:STAGE/n for n in ('PROTOCOL.md','data_lock.json','execution_lock.json','ETTh1_windows.csv','ETTh2_windows.csv')})
    files.update({'manifest.json':LOCAL/'data/ett_v1/manifest.json','ETTh1.csv':LOCAL/'data/ett_v1/raw/ETTh1.csv'})
    bundle=private/'ETTh1_inputs.tar.gz'
    with tarfile.open(bundle,'x:gz') as archive:
        for name,path in files.items():archive.add(path,arcname=name)
    write(STAGE/'deployment_lock.json',dict(bundle_sha256=sha(bundle),bundle_bytes=bundle.stat().st_size,
        files={n:dict(sha256=sha(p),bytes=p.stat().st_size) for n,p in files.items()},ETTh2_measurements_included=False))


def deploy():
    publication=read(STAGE/'runtime_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    commit=publication['source_commit']
    for path in [LOCAL/'scripts'/n for n in CODE]+[STAGE/'execution_lock.json',STAGE/'deployment_lock.json']:
        blob=subprocess.run(['git','-c',f'safe.directory={LOCAL.as_posix()}','show',commit+':'+path.relative_to(LOCAL).as_posix()],capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==sha(path)
    lock=read(STAGE/'deployment_lock.json');bundle=STAGE/'private/ETTh1_inputs.tar.gz'
    assert sha(bundle)==lock['bundle_sha256']
    remote(f"""import pathlib,shutil,json
p=pathlib.Path({REMOTE!r})
assert p.resolve()==p and not p.exists()
assert shutil.disk_usage(p.parent).free>256*1024**2
p.mkdir()
print(json.dumps(dict(created=str(p),free_bytes=shutil.disk_usage(p).free)))
""")
    subprocess.run(SCP+[str(bundle),HOST+':'+REMOTE+'/ETTh1_inputs.tar.gz'],check=True)
    remote(f"""import pathlib,json,hashlib,tarfile,os
p=pathlib.Path({REMOTE!r});lock=json.loads({json.dumps(lock)!r})
def sha(f):
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert sha(p/'ETTh1_inputs.tar.gz')==lock['bundle_sha256']
with tarfile.open(p/'ETTh1_inputs.tar.gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(lock['files']) and {{m.name for m in members}}==set(lock['files'])
 for m in members:
  assert m.isfile() and pathlib.Path(m.name).name==m.name
  with archive.extractfile(m) as src,(p/m.name).open('xb') as dst:dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert sha(p/m.name)==lock['files'][m.name]['sha256']
with (p/'deployment_receipt.json').open('x') as f:json.dump(dict(status='PASS',source_commit={commit!r},files=len(lock['files'])),f,indent=2)
print('Deployment PASS; ETTh2 values absent')
""")


def status(station):
    remote(f"""import pathlib,json
p=pathlib.Path({REMOTE!r});s=p/{station!r}
out=dict(station={station!r},controls={{}},runs=[])
for phase in ('preflight','train','audit'):
 q=p/'control'/({station!r}+'_'+phase)
 if q.exists():
  out['controls'][phase]={{f.name:json.loads(f.read_text()) for f in q.glob('*.json')}}
  out['controls'][phase]['log_tail']=(q/'run.log').read_text()[-1600:]
if (s/'runs').exists():
 for q in sorted((s/'runs').iterdir()):
  if (q/'status.json').exists():out['runs'].append(dict(run=q.name,**json.loads((q/'status.json').read_text())))
print(json.dumps(out))
""")


def launch(station,phase):
    command=f'/opt/miniconda3/bin/python -B {REMOTE}/launch_ett_confirmation.py {phase} {station}'
    subprocess.run(SSH+[command],check=True)


def collect(station):
    remote(f"""import pathlib,tarfile,json,hashlib
p=pathlib.Path({REMOTE!r});s=p/{station!r};out=p/({station!r}+'_results.tar.gz')
assert json.loads((s/'audit.json').read_text())['status']=='PASS'
with tarfile.open(out,'x:gz') as archive:
 for f in sorted(s.rglob('*')):
  if f.is_file() and f.name!='ABC.npz':archive.add(f,arcname=f.relative_to(p).as_posix())
 for f in sorted((p/'control').glob({station!r}+'_*/*')):
  if f.is_file():archive.add(f,arcname=f.relative_to(p).as_posix())
with out.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
receipt=dict(path=out.name,bytes=out.stat().st_size,sha256=digest,prepared_sensor_table_excluded=True)
with (p/({station!r}+'_archive_receipt.json')).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
""")
    archives=STAGE/'archives';archives.mkdir(exist_ok=True)
    for name in (station+'_results.tar.gz',station+'_archive_receipt.json'):
        destination=archives/name if name.endswith('.gz') else STAGE/name
        assert not destination.exists()
        subprocess.run(SCP+[HOST+':'+REMOTE+'/'+name,str(destination)],check=True)
    receipt=read(STAGE/(station+'_archive_receipt.json'));archive=archives/receipt['path']
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    with tarfile.open(archive) as package:
        for member in package.getmembers():
            relative=Path(member.name);target=STAGE/relative
            assert not relative.is_absolute() and '..' not in relative.parts and member.isfile()
            assert not target.exists()
            target.parent.mkdir(exist_ok=True,parents=True)
            with package.extractfile(member) as src,target.open('xb') as dst:dst.write(src.read())


def unlock():
    publication=read(STAGE/'ETTh1_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    commit=publication['source_commit']
    for name in ('summary.json','audit.json'):
        path=STAGE/'ETTh1'/name
        blob=subprocess.run(['git','-c',f'safe.directory={LOCAL.as_posix()}','show',commit+':'+path.relative_to(LOCAL).as_posix()],capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==sha(path)
    record=dict(status='ETTh1_results_and_runtime_published_verified',verified_commit=commit,
        summary_sha256=sha(STAGE/'ETTh1/summary.json'),audit_sha256=sha(STAGE/'ETTh1/audit.json'))
    write(STAGE/'ETTh2_unlock.json',record)
    for path in (LOCAL/'data/ett_v1/raw/ETTh2.csv',STAGE/'ETTh2_unlock.json'):
        remote(f"import pathlib; p=pathlib.Path({(REMOTE+'/'+path.name)!r}); assert not p.exists() and p.resolve()==p\n")
        subprocess.run(SCP+[str(path),HOST+':'+REMOTE+'/'+path.name],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','deploy','status','preflight','train','audit','collect','unlock'))
    parser.add_argument('--station',choices=('ETTh1','ETTh2'),default='ETTh1')
    args=parser.parse_args()
    if args.action in ('preflight','train','audit'):launch(args.station,args.action)
    elif args.action in ('status','collect'):globals()[args.action](args.station)
    else:globals()[args.action]()
