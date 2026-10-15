"""Emit two exact create-only remote deployment programs; no credentials.

Run prepare, transfer the locked inputs.tar.gz once, then finish. No training
is dispatched by this helper. Activation binds all required runtime receipts.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261015_fourgroup_m1_r1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    plan = read(STAGE / 'plan.json')
    publication = read(STAGE / 'source_publication_receipt.json')
    assert publication['status'] == 'exact_commit_and_tree_verified'
    commit = publication['source_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Invalid published source commit')
    def git_bytes(*arguments):
        return subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}', *arguments],
            cwd=ROOT, check=True, capture_output=True).stdout
    assert git_bytes('rev-parse', commit + '^{tree}').decode().strip() == publication['tree_sha']
    # Bind every actual source/config/input-lock byte to the verified remote tree.
    for name in list(plan['code_sha256']) + ['plan.json', 'input_lock.json', 'storage_estimate.json']:
        path = ROOT / 'scripts' / name if name.endswith('.py') else STAGE / name
        committed = git_bytes('show', commit + ':' + path.relative_to(ROOT).as_posix())
        assert hashlib.sha256(committed).hexdigest() == sha(path), name
    receipt = dict(publication, plan_sha256=sha(STAGE / 'plan.json'),
        input_lock_sha256=sha(STAGE / 'input_lock.json'))
    with (STAGE / 'source_sync_receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    files = []
    for name in list(plan['code_sha256']) + ['plan.json', 'input_lock.json', 'storage_estimate.json', 'source_sync_receipt.json']:
        path = ROOT / 'scripts' / name if name.endswith('.py') else STAGE / name
        content = path.read_bytes(); digest = hashlib.sha256(content).hexdigest()
        if name in plan['code_sha256']:
            assert digest == plan['code_sha256'][name]
        files.append(dict(name=name, sha256=digest, content=base64.b64encode(content).decode('ascii')))
    payload = base64.b64encode(json.dumps(dict(files=files, plan=plan)).encode()).decode()
    prepare = "import base64,hashlib,json,os,pathlib,platform,shutil,sys,time\n"
    prepare += "payload=json.loads(base64.b64decode('" + payload + "'))\n"
    prepare += '''
if not __debug__ or os.name!='posix': raise RuntimeError('Unoptimized Linux deployment required')
root=pathlib.Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')
plan=payload['plan']
assert root.resolve()==root and not root.exists() and not root.is_symlink()
assert plan['output_root']==str(root)
free=shutil.disk_usage(root.parent).free
assert free>=plan['preflight_min_free_bytes'],(free,plan['preflight_min_free_bytes'])
import numpy,torch
env=dict(python=platform.python_version(),numpy=numpy.__version__,torch=torch.__version__,
 cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),platform=platform.system())
assert env==plan['expected_environment'],env
root.mkdir()
for item in payload['files']:
 assert pathlib.Path(item['name']).name==item['name']
 content=base64.b64decode(item['content'])
 assert hashlib.sha256(content).hexdigest()==item['sha256']
 with (root/item['name']).open('xb') as stream: stream.write(content);stream.flush();os.fsync(stream.fileno())
 assert hashlib.sha256((root/item['name']).read_bytes()).hexdigest()==item['sha256']
def write(name,value):
 with (root/name).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\\n')
probe=root/'storage_probe.bin'
block=bytes(range(256))*4096
hasher=hashlib.sha256();start=time.perf_counter()
with probe.open('xb') as stream:
 for _ in range(64): stream.write(block);hasher.update(block)
 stream.flush();os.fsync(stream.fileno())
assert probe.stat().st_size==64*1024*1024
with probe.open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
assert actual==hasher.hexdigest()
assert probe.resolve()==root/'storage_probe.bin'
probe.unlink()
storage=dict(status='PASS',plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
 required_free_bytes=plan['preflight_min_free_bytes'],free_before_bytes=free,
 probe_bytes=64*1024*1024,write_sha256=hasher.hexdigest(),read_sha256=actual,
 probe_removed=not probe.exists(),seconds=time.perf_counter()-start,observed_unix=time.time())
write('storage_preflight.json',storage)
write('runtime_environment.json',dict(environment=env,observed_unix=time.time(),python_full=sys.version))
write('deployment_prepared.json',dict(status='awaiting_exact_input_bundle',files=len(payload['files']),
 source_commit=json.loads((root/'source_sync_receipt.json').read_text())['source_commit']))
print(json.dumps(dict(status='awaiting_exact_input_bundle',root=str(root),free_bytes=shutil.disk_usage(root).free)))
'''
    finish = '''import hashlib,json,os,pathlib,shutil,tarfile,time
if not __debug__ or os.name!='posix': raise RuntimeError('Unoptimized Linux deployment required')
root=pathlib.Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')
assert root.resolve()==root and not (root/'execution_activation.json').exists()
def read(name):return json.loads((root/name).read_text())
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
plan=read('plan.json');lock=read('input_lock.json');source=read('source_sync_receipt.json')
assert source['plan_sha256']==sha(root/'plan.json')
assert source['input_lock_sha256']==plan['input_lock_sha256']==sha(root/'input_lock.json')
assert read('deployment_prepared.json')['status']=='awaiting_exact_input_bundle'
bundle=root/'inputs.tar.gz'
assert bundle.resolve()==bundle and bundle.stat().st_size==plan['input_bundle']['bytes']
assert sha(bundle)==plan['input_bundle']['sha256']
expected={i['path']:i for i in lock['files']}
assert len(expected)==len(lock['files']) and all(n.startswith('inputs/') for n in expected)
with tarfile.open(bundle,'r:gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for member in members:
  item=expected[member.name];rel=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
  assert member.size==item['bytes']
  target=root/member.name
  assert target.resolve()==target and not target.exists()
  target.parent.mkdir(parents=True,exist_ok=True)
  with archive.extractfile(member) as src,target.open('xb') as dst:
   shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
  assert sha(target)==item['sha256'],member.name
activation=dict(status='inputs_environment_storage_verified',plan_sha256=sha(root/'plan.json'),
 input_lock_sha256=sha(root/'input_lock.json'),source_commit=source['source_commit'],
 storage_preflight_sha256=sha(root/'storage_preflight.json'),environment_sha256=sha(root/'runtime_environment.json'),
 input_files=len(expected),input_bytes=sum(i['bytes'] for i in expected.values()),observed_unix=time.time())
with (root/'execution_activation.json').open('x',encoding='utf-8') as stream:json.dump(activation,stream,indent=2);stream.write('\\n')
import sys
sys.path.insert(0,str(root))
import fourgroup_runtime
fourgroup_runtime.configure();fourgroup_runtime.verify(root)
print(json.dumps(activation))
'''
    private = STAGE / 'private'
    for name, content in (('deployment_prepare.py', prepare), ('deployment_finish.py', finish)):
        with (private / name).open('x', encoding='utf-8') as stream:
            stream.write(content)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('Unoptimized deployment builder required')
    build()
