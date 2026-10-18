"""Create-only verified r11 result streaming; default execution is offline only."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time

from collect_coldstart_stream import digest, write_new


REMOTE_ROOT = '/root/auto_data_experiments/20261018_open_processing_r11'
DIRECTORIES = ('runs', 'queries', 'measurements', 'control', 'preflight_control', 'audit_control')
OPTIONAL_DIRECTORIES = ('runs', 'queries')
METADATA = ('prepared_parent_facts.json', 'preflight_receipt.json',
            'parent_reuse_receipt.json', 'deployment_receipt.json',
            'development_audit.json', 'execution_lock.json')

# Sent only to the existing SSH interpreter in the explicit collection branch.
REMOTE_HELPER = '''import hashlib,json,os,pathlib,stat,sys,tarfile
def checked_file(root, name):
 parts=pathlib.PurePosixPath(name).parts
 if not parts or pathlib.PurePosixPath(name).is_absolute() or name != pathlib.PurePosixPath(name).as_posix() or any(v in ('','.','..') for v in parts) or '\\\\' in name or ':' in name:
  raise ValueError('Unsafe relative path: '+name)
 current=root
 for part in parts:
  current=current/part
  info=current.lstat()
  if stat.S_ISLNK(info.st_mode):raise ValueError('Symlink: '+name)
 if not stat.S_ISREG(info.st_mode):raise ValueError('Not a regular file: '+name)
 if not current.resolve().is_relative_to(root):raise ValueError('Path escape: '+name)
 return current
def inventory(root, expected_root, directories, optional, metadata):
 if not root.is_absolute() or root.resolve()!=root or not root.is_dir() or root!=pathlib.Path(expected_root):raise ValueError('Unsafe or wrong root')
 audit=json.loads(checked_file(root,'development_audit.json').read_text())
 audit_exit=json.loads(checked_file(root,'audit_control/exit.json').read_text())
 completed=json.loads(checked_file(root,'audit_decisions/trial_completed.json').read_text())
 if audit['status']!='PASS' or audit_exit['returncode']!=0 or completed['status']!='completed':raise ValueError('Completion gates failed')
 names=[]
 for category in directories:
  base=root/category
  if base.is_symlink():raise ValueError('Symlink category: '+category)
  if not base.exists() and category in optional:continue
  if not base.is_dir():raise ValueError('Missing or non-directory category: '+category)
  for directory, dirs, files in os.walk(base,followlinks=False):
   for entry in dirs:
    path=pathlib.Path(directory)/entry
    if not stat.S_ISDIR(path.lstat().st_mode):raise ValueError('Non-directory or symlink: '+str(path))
   for entry in files:names.append((pathlib.Path(directory)/entry).relative_to(root).as_posix())
 names.extend(metadata)
 if len(names)!=len(set(names)):raise ValueError('Duplicate inventory path')
 result={}
 for name in sorted(names):
  path=checked_file(root,name)
  with path.open('rb') as stream:
   before=os.fstat(stream.fileno())
   value=hashlib.file_digest(stream,'sha256').hexdigest()
   after=os.fstat(stream.fileno())
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed: '+name)
  result[name]=dict(bytes=after.st_size,sha256=value)
 return result
'''


def validate_inventory(files):
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty inventory')
    for name, record in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or path.as_posix() != name or '\\' in name
                or ':' in name or not path.parts
                or any(part in ('', '.', '..') for part in path.parts)):
            raise ValueError('Unsafe inventory name: ' + name)
        if not (name in METADATA or (len(path.parts) > 1 and path.parts[0] in DIRECTORIES)):
            raise ValueError('Asset outside fixed allowlist: ' + name)
        if type(record['bytes']) is not int or record['bytes'] < 0:
            raise ValueError('Invalid inventory size')
        value = record['sha256']
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid inventory digest')
    if not set(METADATA).issubset(files):
        raise ValueError('Required metadata is missing')


def verify_archive(path, files):
    """Verify every member as opaque bytes, then consume the gzip trailer."""
    validate_inventory(files)
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if name in seen or name not in files or not member.isfile():
                raise ValueError('Duplicate, unexpected or non-file member: ' + name)
            seen.add(name)
            if member.size != files[name]['bytes']:
                raise ValueError('Member size mismatch: ' + name)
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != files[name]['sha256']:
                raise ValueError('Member digest mismatch: ' + name)
    if seen != set(files):
        raise ValueError('Missing archive members: ' + repr(sorted(set(files) - seen)))
    with gzip.open(path, 'rb') as stream:
        while stream.read(1024 * 1024):
            pass
    return dict(status='PASS', members=len(seen), bytes=path.stat().st_size, sha256=digest(path))


def collect():
    from open_processing_trial import STAGE, REMOTE, read, transport
    if REMOTE != REMOTE_ROOT:
        raise ValueError('Wrong remote root')
    if read(STAGE/'trial_completed.json')['status'] != 'completed':
        raise ValueError('Trial is not completed')
    if read(STAGE/'development_audit.json')['status'] != 'PASS':
        raise ValueError('Development audit is not PASS')
    folder = STAGE/'archives'
    if folder.is_symlink():
        raise ValueError('Symlink archive folder')
    folder.mkdir(exist_ok=True)
    target = folder/'results.tar.gz'
    partial = folder/'results.tar.gz.partial'
    inventory_path = folder/'archive_inventory.json'
    receipt_path = folder/'stream_collection_receipt.json'
    stderr_path = folder/'stream_collection_stderr.log'
    for path in (target, partial, inventory_path, receipt_path, stderr_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Collection already attempted; no retry/overwrite: ' + str(path))
    receipt = dict(status='FAILED', remote_root=REMOTE, started=time.time(), archive=str(target))
    try:
        command = transport() + ['/opt/miniconda3/bin/python', '-B', '-']
        setup = REMOTE_HELPER + f'\nr=pathlib.Path({REMOTE!r})\n'
        call = f'inventory(r,{REMOTE_ROOT!r},{DIRECTORIES!r},{OPTIONAL_DIRECTORIES!r},{METADATA!r})'
        result = subprocess.run(command, input=(setup + f'print(json.dumps(dict(files={call})))\n').encode('utf-8'),
                                capture_output=True, check=False)
        with stderr_path.open('xb') as stderr:
            stderr.write(result.stderr)
        if result.returncode:
            raise RuntimeError('Remote inventory failed: SSH return code ' + str(result.returncode))
        files = json.loads(result.stdout.decode('utf-8'))['files']
        validate_inventory(files)
        write_new(inventory_path, dict(status='FROZEN', remote_root=REMOTE, files=files))
        script = setup + f'''
expected=json.loads({json.dumps(files)!r})
if {call}!=expected:raise ValueError('Assets changed after frozen inventory')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,record in expected.items():
  path=checked_file(r,name)
  with path.open('rb') as source:
   info=archive.gettarinfo(str(path),arcname=name)
   if not info.isfile() or info.size!=record['bytes']:raise ValueError('File changed: '+name)
   archive.addfile(info,source)
'''
        with partial.open('xb') as destination, stderr_path.open('ab') as stderr:
            process = subprocess.run(command, input=script.encode('utf-8'), stdout=destination, stderr=stderr, check=False)
            destination.flush()
            os.fsync(destination.fileno())
        receipt['stream_returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError('Remote archive stream failed: SSH return code ' + str(process.returncode))
        receipt.update(verify_archive(partial, files))
        os.link(partial, target)
        partial.unlink()
        receipt.update(inventory_sha256=digest(inventory_path), finished=time.time())
    except Exception as exc:
        receipt.update(status='FAILED', error=repr(exc), finished=time.time(), partial=str(partial),
                       partial_exists=partial.exists())
        if partial.exists():
            receipt.update(partial_bytes=partial.stat().st_size, partial_sha256=digest(partial))
        write_new(receipt_path, receipt)
        raise
    write_new(receipt_path, receipt)
    print(json.dumps(receipt))


def synthetic_check():
    """Exercise inventory and archive rejection rules using artificial bytes only."""
    helper = {}
    exec(compile(REMOTE_HELPER, 'remote_inventory_helper', 'exec'), helper)
    checks = []
    with tempfile.TemporaryDirectory(prefix='open_processing_stream_') as temporary:
        root = Path(temporary).resolve()
        for category in DIRECTORIES:
            if category not in OPTIONAL_DIRECTORIES:
                (root/category).mkdir()
        (root/'audit_decisions').mkdir()
        for name in METADATA:
            write_new(root/name, dict(status='PASS') if name == 'development_audit.json' else dict(fixture=True))
        write_new(root/'audit_control/exit.json', dict(returncode=0))
        write_new(root/'audit_decisions/trial_completed.json', dict(status='completed'))
        def inventory():
            return helper['inventory'](root, str(root), DIRECTORIES, OPTIONAL_DIRECTORIES, METADATA)
        files = inventory()
        validate_inventory(files)
        checks.append('zero_run_missing_optional_categories')
        for category in OPTIONAL_DIRECTORIES:
            (root/category).mkdir()
        assert inventory() == files
        checks.append('zero_run_empty_optional_categories')
        def fixture(name, names, special=None):
            path = root/name
            with path.open('xb') as stream, tarfile.open(fileobj=stream, mode='w|gz') as archive:
                for member in names:
                    archive.add(root/member, arcname=member, recursive=False)
                if special is not None:
                    archive.addfile(special)
            return path
        names = list(files)
        good = fixture('good.tar.gz', names)
        assert verify_archive(good, files)['members'] == len(files)
        checks.append('exact_members_sizes_and_hashes')
        def rejected(label, action, errors=(ValueError,)):
            try:
                action()
            except errors:
                checks.append(label + '_rejected')
            else:
                raise AssertionError(label + ' accepted')
        rejected('duplicate', lambda: verify_archive(fixture('duplicate.tar.gz', names+names[:1]), files))
        rejected('missing', lambda: verify_archive(fixture('missing.tar.gz', names[1:]), files))
        rejected('hash', lambda: verify_archive(good, {n:dict(r, sha256='0'*64) for n,r in files.items()}))
        rejected('size', lambda: verify_archive(good, {n:dict(r, bytes=r['bytes']+1) for n,r in files.items()}))
        extra = tarfile.TarInfo('control/unexpected')
        rejected('unexpected', lambda: verify_archive(fixture('unexpected.tar.gz', names, extra), files))
        for label, kind in [('directory', tarfile.DIRTYPE), ('symlink', tarfile.SYMTYPE)]:
            member = tarfile.TarInfo(names[0])
            member.type = kind
            member.linkname = 'outside'
            rejected(label, lambda: verify_archive(fixture(label+'.tar.gz', [], member), files))
        for name in ('../escape', 'runs/../escape', '/runs/absolute', 'runs/back\\slash', 'runs/C:drive', 'runs//double'):
            rejected('unsafe_path', lambda: validate_inventory(dict(files, **{name:files[names[0]]})))
            rejected('remote_unsafe_path', lambda: helper['checked_file'](root, name))
        corrupted = root/'corrupted.tar.gz'
        payload = bytearray(good.read_bytes())
        payload[-8] ^= 1
        corrupted.write_bytes(payload)
        rejected('gzip_crc', lambda: verify_archive(corrupted, files), (OSError, tarfile.TarError, EOFError))
        rejected('exclusive_creation', lambda: good.open('xb'), (FileExistsError,))
        write_new(root/'control/opaque.json', dict(fixture=True))
        assert inventory() != files
        checks.append('changed_inventory_detected')
        for name, payload in [('development_audit.json', dict(status='FAIL')),
                              ('audit_decisions/trial_completed.json', dict(status='interrupted')),
                              ('audit_control/exit.json', dict(returncode=1))]:
            path = root/name
            original = path.read_bytes()
            path.write_text(json.dumps(payload), encoding='utf-8')
            rejected('completion_gate', inventory)
            path.write_bytes(original)
        # Use an artificial lstat/is_symlink facade so Windows needs no symlink privilege.
        real_path = helper['pathlib']
        import types
        class SymlinkPath:
            def __init__(self, path):
                self.path = Path(path)
            def __truediv__(self, name):
                return SymlinkPath(self.path/name)
            def __eq__(self, other):
                return self.path == getattr(other, 'path', other)
            def __getattr__(self, name):
                return getattr(self.path, name)
            def is_symlink(self):
                return self.path.name == 'runs' or self.path.is_symlink()
            def lstat(self):
                if self.path.name == 'opaque.json':
                    return types.SimpleNamespace(st_mode=helper['stat'].S_IFLNK)
                return self.path.lstat()
        helper['pathlib'] = types.SimpleNamespace(Path=SymlinkPath, PurePosixPath=PurePosixPath)
        rejected('optional_category_symlink', lambda: helper['inventory'](SymlinkPath(root), str(root), DIRECTORIES, OPTIONAL_DIRECTORIES, METADATA))
        rejected('member_symlink', lambda: helper['checked_file'](SymlinkPath(root), 'control/opaque.json'))
        helper['pathlib'] = real_path
    print(json.dumps(dict(status='PASS', network_access=False, checks=checks)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--collect', action='store_true', help='Explicitly enable existing-server SSH collection')
    mode.add_argument('--synthetic-check', action='store_true', help='Run offline artificial-file checks (default)')
    args = parser.parse_args()
    collect() if args.collect else synthetic_check()
