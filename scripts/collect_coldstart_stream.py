"""Exclusive, verified r10 result streaming. Default execution is offline only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import time


DIRECTORIES = ('runs', 'queries', 'control', 'audit_control')
METADATA = ('parent_facts.json', 'parent_receipt.json',
            'deployment_receipt.json', 'development_audit.json')

# This source is sent to the existing SSH Python interpreter, never saved remotely.
REMOTE_HELPER = '''import hashlib,json,os,pathlib,stat,sys,tarfile
def checked_file(root, name):
 parts=pathlib.PurePosixPath(name).parts
 if not parts or name != pathlib.PurePosixPath(name).as_posix() or any(v in ('','.','..') for v in parts) or '\\\\' in name:
  raise ValueError('Unsafe relative path: '+name)
 current=root
 for part in parts:
  current=current/part
  info=current.lstat()
  if stat.S_ISLNK(info.st_mode):raise ValueError('Symlink: '+name)
 if not stat.S_ISREG(info.st_mode):raise ValueError('Not a regular file: '+name)
 if not current.resolve().is_relative_to(root):raise ValueError('Path escape: '+name)
 return current
def inventory(root, directories, metadata):
 if not root.is_absolute() or root.resolve()!=root or not root.is_dir():raise ValueError('Unsafe root')
 if root!=pathlib.Path('/root/auto_data_experiments/20261018_cold_start_r10'):raise ValueError('Wrong root')
 audit=json.loads(checked_file(root,'development_audit.json').read_text())
 audit_exit=json.loads(checked_file(root,'audit_control/exit.json').read_text())
 completed=json.loads(checked_file(root,'audit_decisions/trial_completed.json').read_text())
 if audit['status']!='PASS' or audit_exit['returncode']!=0 or completed['status']!='completed':raise ValueError('Completion gates failed')
 names=[]
 for category in directories:
  base=root/category
  if base.is_symlink() or not base.is_dir():raise ValueError('Missing or symlink category: '+category)
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
   digest=hashlib.file_digest(stream,'sha256').hexdigest()
   after=os.fstat(stream.fileno())
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed: '+name)
  result[name]=dict(bytes=after.st_size,sha256=digest)
 return result
'''


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_inventory(files):
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty inventory')
    for name, record in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or path.as_posix() != name or '\\' in name
                or any(part in ('', '.', '..') for part in path.parts)
                or not path.parts or ':' in name):
            raise ValueError('Unsafe inventory name: ' + name)
        if not (name in METADATA or (len(path.parts) > 1 and path.parts[0] in DIRECTORIES)):
            raise ValueError('Asset outside fixed allowlist: ' + name)
        if type(record['bytes']) is not int or record['bytes'] < 0:
            raise ValueError('Invalid inventory size')
        value = record['sha256']
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid inventory digest')


def verify_archive(path, files):
    """Read member bytes without extraction, checkpoints/arrays are never parsed."""
    validate_inventory(files)
    seen = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if name in seen or name not in files or not member.isfile():
                raise ValueError('Duplicate, unexpected or non-file member: ' + name)
            seen.add(name)
            expected = files[name]
            if member.size != expected['bytes']:
                raise ValueError('Member size mismatch: ' + name)
            with archive.extractfile(member) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected['sha256']:
                raise ValueError('Member digest mismatch: ' + name)
    if seen != set(files):
        raise ValueError('Missing archive members: ' + repr(sorted(set(files) - seen)))
    # Consume through gzip EOF as well, checking its trailer/CRC even after tar EOF.
    import gzip
    with gzip.open(path, 'rb') as stream:
        while stream.read(1024 * 1024):
            pass
    return dict(status='PASS',members=len(seen),bytes=path.stat().st_size,sha256=digest(path))


def collect():
    # Import only in the explicit network branch; default checks read no secrets.
    from coldstart_trial import STAGE, REMOTE, read, transport
    if read(STAGE/'trial_completed.json')['status'] != 'completed':
        raise ValueError('Trial is not completed')
    if read(STAGE/'development_audit.json')['status'] != 'PASS':
        raise ValueError('Development audit is not PASS')
    folder = STAGE/'archives'
    folder.mkdir(exist_ok=True)
    target = folder/'results.tar.gz'
    partial = folder/'results.tar.gz.partial'
    inventory_path = folder/'archive_inventory.json'
    receipt_path = folder/'stream_collection_receipt.json'
    stderr_path = folder/'stream_collection_stderr.log'
    for path in (target, partial, inventory_path, receipt_path, stderr_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Collection already attempted; no retry/overwrite: ' + str(path))
    receipt = dict(status='FAILED',remote_root=REMOTE,started=time.time(),archive=str(target))
    try:
        command = transport() + ['/opt/miniconda3/bin/python', '-B', '-']
        setup = REMOTE_HELPER + f'\nr=pathlib.Path({REMOTE!r})\n'
        listing = setup + f'print(json.dumps(dict(files=inventory(r,{DIRECTORIES!r},{METADATA!r}))))\n'
        result = subprocess.run(command,input=listing.encode('utf-8'),capture_output=True,check=False)
        with stderr_path.open('xb') as stderr:
            stderr.write(result.stderr)
        if result.returncode:
            raise RuntimeError('Remote inventory failed: SSH return code ' + str(result.returncode))
        inventory = json.loads(result.stdout.decode('utf-8'))
        files = inventory['files']
        validate_inventory(files)
        if not set(METADATA).issubset(files) or any(not any(n.startswith(d + '/') for n in files) for d in DIRECTORIES):
            raise ValueError('Required result categories are missing')
        write_new(inventory_path,dict(status='FROZEN',remote_root=REMOTE,files=files))
        script = setup + f'''
expected=json.loads({json.dumps(files)!r})
if inventory(r,{DIRECTORIES!r},{METADATA!r})!=expected:raise ValueError('Assets changed after frozen inventory')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,record in expected.items():
  path=checked_file(r,name)
  with path.open('rb') as source:
   info=archive.gettarinfo(str(path),arcname=name)
   if not info.isfile() or info.size!=record['bytes']:raise ValueError('File changed: '+name)
   archive.addfile(info,source)
'''
        with partial.open('xb') as destination, stderr_path.open('ab') as stderr:
            process = subprocess.run(command,input=script.encode('utf-8'),stdout=destination,stderr=stderr,check=False)
            destination.flush()
            os.fsync(destination.fileno())
        receipt['stream_returncode'] = process.returncode
        if process.returncode:
            raise RuntimeError('Remote archive stream failed: SSH return code ' + str(process.returncode))
        receipt.update(verify_archive(partial,files))
        # Hard-link publication refuses to overwrite any concurrent final artifact.
        os.link(partial,target)
        partial.unlink()
        receipt.update(inventory_sha256=digest(inventory_path),finished=time.time())
    except Exception as exc:
        receipt.update(status='FAILED',error=repr(exc),finished=time.time(),partial=str(partial),
                       partial_exists=partial.exists())
        if partial.exists():
            receipt.update(partial_bytes=partial.stat().st_size,partial_sha256=digest(partial))
        write_new(receipt_path,receipt)
        raise
    write_new(receipt_path,receipt)
    print(json.dumps(receipt))


def synthetic_check():
    """Offline fixtures test exact members, hashes, sizes and rejection cases."""
    with tempfile.TemporaryDirectory(prefix='coldstart_stream_') as temporary:
        root = Path(temporary)
        files = {}
        for name, payload in [('runs/PARENT/1/result.json',b'{"status":"completed"}\n'),
                              ('queries/STRONG/01/result.json',b'fixture\x00\xff')]:
            path = root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(payload)
            files[name] = dict(bytes=len(payload),sha256=digest(path))
        def fixture(name, members):
            path = root/name
            with path.open('xb') as stream, tarfile.open(fileobj=stream,mode='w|gz') as archive:
                for member in members:
                    archive.add(root/member,arcname=member,recursive=False)
            return path
        names = list(files)
        good = fixture('good.tar.gz',names)
        assert verify_archive(good,files)['members'] == 2
        cases = [('duplicate',fixture('duplicate.tar.gz',names + names[:1]),files),
                 ('missing',fixture('missing.tar.gz',names[:1]),files),
                 ('unexpected',good,{names[0]:files[names[0]]}),
                 ('hash',good,{n:dict(r,sha256='0'*64) for n,r in files.items()}),
                 ('size',good,{n:dict(r,bytes=r['bytes']+1) for n,r in files.items()})]
        checks = ['exact_members_and_hashes']
        for label, path, expected in cases:
            try:
                verify_archive(path,expected)
            except ValueError:
                checks.append(label + '_rejected')
            else:
                raise AssertionError(label + ' accepted')
        for name in ('../escape','runs/../escape','/runs/absolute','runs/back\\slash'):
            try:
                validate_inventory({name:files[names[0]]})
            except ValueError:
                pass
            else:
                raise AssertionError('Unsafe path accepted')
        checks.append('unsafe_paths_rejected')
        for label, kind in [('directory',tarfile.DIRTYPE),('symlink',tarfile.SYMTYPE)]:
            path = root/(label + '.tar.gz')
            with path.open('xb') as stream, tarfile.open(fileobj=stream,mode='w|gz') as archive:
                member = tarfile.TarInfo(names[0])
                member.type = kind
                member.linkname = 'outside'
                archive.addfile(member)
            try:
                verify_archive(path,files)
            except ValueError:
                checks.append(label + '_rejected')
            else:
                raise AssertionError(label + ' accepted')
        corrupted = root/'corrupted.tar.gz'
        payload = bytearray(good.read_bytes())
        payload[-8] ^= 1  # Corrupt gzip CRC, including after the tar end marker.
        corrupted.write_bytes(payload)
        try:
            verify_archive(corrupted,files)
        except (OSError,tarfile.TarError,EOFError):
            checks.append('gzip_crc_rejected')
        else:
            raise AssertionError('Corrupt gzip accepted')
        try:
            with good.open('xb'):
                pass
        except FileExistsError:
            checks.append('exclusive_creation_preserves_existing')
        else:
            raise AssertionError('Existing archive overwritten')
        # Validate trusted helper syntax without executing any remote operations.
        compile(REMOTE_HELPER,'remote_inventory_helper','exec')
        checks.append('remote_helper_compiles')
        print(json.dumps(dict(status='PASS',network_access=False,checks=checks)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--collect',action='store_true',help='Explicitly enable existing-server SSH collection')
    mode.add_argument('--synthetic-check',action='store_true',help='Run offline artificial-file checks (default)')
    args = parser.parse_args()
    collect() if args.collect else synthetic_check()
