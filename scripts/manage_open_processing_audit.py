"""Create-only supervision for the already deployed r11 development auditor."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from open_processing_trial import ROOT, STAGE, REMOTE, read, write, sha, remote
from manage_coldstart_trial import scp


DOCUMENTS = ('AGENT_INTERFACE.txt', 'PROTOCOL.md', 'trial_started.json',
             'trial_completed.json', 'final_recommendations.json',
             'prepared_parent_facts.json', 'parent_reuse_receipt.json',
             'preflight_receipt.json')
OUTPUTS = ('development_audit.json', 'audit_control/started.json',
           'audit_control/exit.json', 'audit_control/run.log')


def server_checks():
    """Every server action stays under the exact non-symlink r11 root."""
    assert REMOTE == '/root/auto_data_experiments/20261018_open_processing_r11'
    return f'''import pathlib,json,hashlib,os,stat
p=pathlib.Path({REMOTE!r})
assert p.is_dir() and p.resolve()==p and not p.is_symlink()
def regular(f):
 assert f.resolve().is_relative_to(p) and not f.is_symlink()
 assert stat.S_ISREG(f.lstat().st_mode)
def digest(f):
 regular(f)
 with f.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
'''


def locked_server_checks(manifest):
    return server_checks()+f'''
m=json.loads({json.dumps(manifest)!r})
assert digest(p/'execution_lock.json')==m['execution_lock_sha256']
lock=json.loads((p/'execution_lock.json').read_text())
assert lock['execution_enabled']
for name,expected in lock['runtime_code_sha256'].items():
 assert pathlib.PurePosixPath(name).name==name
 assert digest(p/name)==expected
assert lock['runtime_code_sha256']['audit_open_processing.py']==m['auditor_sha256']
'''


def launch():
    assert read(STAGE/'trial_completed.json')['status'] == 'completed'
    assert read(STAGE/'final_recommendations.json')['status'] == 'FROZEN'
    publication = read(STAGE/'source_publication_receipt.json')
    assert publication['status'] == 'exact_commit_and_tree_verified'
    lock = read(STAGE/'execution_lock.json')
    auditor_hash = sha(ROOT/'scripts/audit_open_processing.py')
    assert lock['execution_enabled']
    assert lock['runtime_code_sha256']['audit_open_processing.py'] == auditor_hash
    snapshots = sorted((STAGE/'snapshots').glob('*.json'))
    assert {f.name for f in snapshots} == {
        f'{arm}_{number:02d}.json' for arm in ('STRONG', 'SEPARATE') for number in range(1, 9)}
    files = {'audit_decisions/'+name: STAGE/name for name in DOCUMENTS}
    assert (STAGE/'calls').is_dir() and not (STAGE/'calls').is_symlink()
    for path in sorted((STAGE/'calls').rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            files['audit_decisions/'+path.relative_to(STAGE).as_posix()] = path
        else:
            assert path.is_dir()
    files.update({'audit_decisions/snapshots/'+path.name: path for path in snapshots})
    for path in files.values():
        assert path.is_file() and not path.is_symlink()
        assert path.resolve() == path.absolute() and path.resolve().is_relative_to(STAGE.resolve())
    private = STAGE/'private'; private.mkdir(exist_ok=True)
    assert private.resolve() == private.absolute() and not private.is_symlink()
    bundle = private/'audit_inputs.tar.gz'
    assert not (STAGE/'audit_deployment_manifest.json').exists()
    assert not (STAGE/'audit_launch_receipt.json').exists()
    with tarfile.open(bundle, 'x:gz') as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
    manifest = dict(bundle_sha256=sha(bundle), bundle_bytes=bundle.stat().st_size,
                    files={name: dict(sha256=sha(path), bytes=path.stat().st_size)
                           for name, path in files.items()},
                    source_commit=publication['source_commit'],
                    execution_lock_sha256=sha(STAGE/'execution_lock.json'),
                    auditor_sha256=auditor_hash)
    write(STAGE/'audit_deployment_manifest.json', manifest)
    remote(locked_server_checks(manifest)+'''
for name in ('audit_inputs.tar.gz','audit_decisions','audit_control','development_audit.json'):
 assert not os.path.lexists(p/name)
print(json.dumps(dict(ready=True)))
''')
    command, host = scp()
    subprocess.run(command+[str(bundle), host+':'+REMOTE+'/audit_inputs.tar.gz'], check=True)
    receipt = remote(locked_server_checks(manifest)+'''
import tarfile,subprocess,sys
assert digest(p/'audit_inputs.tar.gz')==m['bundle_sha256']
assert (p/'audit_inputs.tar.gz').stat().st_size==m['bundle_bytes']
for name in ('audit_decisions','audit_control','development_audit.json'):
 assert not os.path.lexists(p/name)
with tarfile.open(p/'audit_inputs.tar.gz') as archive:
 members=archive.getmembers()
 assert len(members)==len(m['files']) and {v.name for v in members}==set(m['files'])
 for member in members:
  name=pathlib.PurePosixPath(member.name)
  assert member.isfile() and not name.is_absolute() and '..' not in name.parts
  assert name.parts[0]=='audit_decisions' and str(name)==member.name
  assert member.size==m['files'][member.name]['bytes']
  target=p/member.name
  assert target.resolve().is_relative_to(p) and not os.path.lexists(target)
 for member in members:
  target=p/member.name;target.parent.mkdir(parents=True,exist_ok=True)
  assert target.parent.resolve()==target.parent
  with archive.extractfile(member) as src,target.open('xb') as dst:
   dst.write(src.read());dst.flush();os.fsync(dst.fileno())
  assert digest(target)==m['files'][member.name]['sha256']
  assert target.stat().st_size==m['files'][member.name]['bytes']
c=p/'audit_control';c.mkdir()
command=[sys.executable,'-B',str(p/'audit_open_processing.py'),'--root',str(p),'--decisions',str(p/'audit_decisions'),'--output',str(p/'development_audit.json')]
wrapper="import subprocess,json,pathlib,time; p=pathlib.Path("+repr(str(c))+"); started=time.time(); result=subprocess.run("+repr(command)+"); f=(p/'exit.json').open('x',encoding='utf-8'); json.dump(dict(returncode=result.returncode,started=started,finished=time.time()),f); f.close()"
with (c/'run.log').open('xb') as log:
 regular(c/'run.log')
 child=subprocess.Popen([sys.executable,'-B','-c',wrapper],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=str(p))
receipt=dict(status='started',pid=child.pid,files=len(m['files']),bundle_sha256=m['bundle_sha256'],auditor_sha256=m['auditor_sha256'])
with (c/'started.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream)
print(json.dumps(receipt))
''')
    write(STAGE/'audit_launch_receipt.json', receipt)
    print(json.dumps(receipt))


def status():
    value = remote(server_checks()+'''
c=p/'audit_control';assert c.is_dir() and c.resolve()==c
regular(c/'started.json');regular(c/'run.log')
done=(c/'exit.json').exists()
exit_value=None;result=None
if done:
 regular(c/'exit.json');exit_value=json.loads((c/'exit.json').read_text())
if os.path.lexists(p/'development_audit.json'):
 regular(p/'development_audit.json');result=json.loads((p/'development_audit.json').read_text())
pid=json.loads((c/'started.json').read_text())['pid']
try:os.kill(pid,0);alive=True
except ProcessLookupError:alive=False
with (c/'run.log').open('rb') as stream:
 stream.seek(max(0,(c/'run.log').stat().st_size-8000));tail=stream.read().decode('utf-8',errors='replace')
print(json.dumps(dict(done=done,alive=alive,exit=exit_value,result=result,log=tail)))
''')
    print(json.dumps(value))


def collect():
    value = remote(server_checks()+f'''
c=p/'audit_control';assert c.is_dir() and c.resolve()==c
regular(c/'exit.json');files={{}}
for name in {OUTPUTS!r}:
 f=p/name
 if os.path.lexists(f):files[name]=dict(sha256=digest(f),bytes=f.stat().st_size)
assert all(name in files for name in {OUTPUTS[1:]!r})
print(json.dumps(dict(files=files,exit=json.loads((c/'exit.json').read_text()))))
''')
    assert set(OUTPUTS[1:]) <= set(value['files']) <= set(OUTPUTS)
    assert not (STAGE/'audit_collection_receipt.json').exists()
    for name in value['files']:
        target = STAGE/name
        assert not target.exists() and not target.is_symlink()
        assert target.resolve() == target.absolute()
    command, host = scp()
    private = STAGE/'private'; private.mkdir(exist_ok=True)
    assert private.resolve() == private.absolute() and not private.is_symlink()
    scratch = Path(tempfile.mkdtemp(prefix='audit_collect_', dir=private))
    for name, metadata in value['files'].items():
        temporary = scratch/Path(name).name
        subprocess.run(command+[host+':'+REMOTE+'/'+name, str(temporary)], check=True)
        target = STAGE/name; target.parent.mkdir(parents=True, exist_ok=True)
        assert target.parent.resolve() == target.parent.absolute()
        with temporary.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        assert sha(target) == metadata['sha256'] and target.stat().st_size == metadata['bytes']
    write(STAGE/'audit_collection_receipt.json', value)
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'status', 'collect'))
    globals()[parser.parse_args().action]()
