"""Create a closed-phase M1 delivery without loading scientific array values."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tarfile
import time

ROOT = Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')
RESERVE_BYTES = 64 * 1024 * 1024


def safe_path(path):
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('Exact absolute paths required')
    for part in [*reversed(path.parents), path]:
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Symlinks and reparse points are forbidden')
    return path


def safe_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe relative name')
    if PurePosixPath(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe relative name')
    return name


def signature(info):
    # Windows fstat can report creation time where lstat reports change time.
    changed = info.st_ctime_ns if os.name == 'posix' else None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, changed)


def digest(path):
    safe_path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Ordinary file required')
    with path.open('rb') as stream:
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError('File changed before reading')
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
        if signature(os.fstat(stream.fileno())) != signature(before):
            raise ValueError('File changed while reading')
    if signature(path.lstat()) != signature(before):
        raise ValueError('File changed after reading')
    return value


def read(path):
    safe_path(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def enumerate_files(root):
    files = []
    def visit(directory):
        for path in sorted(directory.iterdir()):
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
                raise ValueError('Symlink/reparse point in study root')
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError('Special file in study root')
            if path.parent == root and path.name in ('inputs.tar.gz', 'delivery'):
                if path.name == 'delivery' and not stat.S_ISDIR(info.st_mode):
                    raise ValueError('delivery must be a directory')
                continue
            if stat.S_ISDIR(info.st_mode):
                visit(path)
            else:
                safe_name(path.relative_to(root).as_posix())
                files.append(path)
    visit(root)
    return files


def closed_phases(root):
    control = root / 'control'
    if not control.exists():
        return []
    phases = []
    for phase in sorted(control.iterdir()):
        if not phase.is_dir():
            raise ValueError('Unexpected control entry')
        exit_path = phase / 'exit.json'
        if not exit_path.is_file() or type(read(exit_path).get('exit_code')) is not int:
            raise ValueError('Phase has no integer exit code: ' + phase.name)
        phases.append(phase.name)
    return phases


def check_locks(root, items):
    plan = read(root / 'plan.json')
    lock = read(root / 'input_lock.json')
    source = read(root / 'source_sync_receipt.json')
    hashes = {i['path']: i['sha256'] for i in items}
    if plan['output_root'] != str(root) or source['status'] != 'exact_commit_and_tree_verified':
        raise ValueError('Invalid source binding')
    if source['plan_sha256'] != hashes['plan.json']:
        raise ValueError('Plan hash mismatch')
    if source['input_lock_sha256'] != plan['input_lock_sha256'] or plan['input_lock_sha256'] != hashes['input_lock.json']:
        raise ValueError('Input lock hash mismatch')
    if not re.fullmatch('[0-9a-f]{40}', source['source_commit']):
        raise ValueError('Invalid source commit')
    for name, expected in plan['code_sha256'].items():
        if hashes[safe_name(name)] != expected:
            raise ValueError('Code hash mismatch: ' + name)
    indexed = {i['path']: i for i in items}
    seen = set()
    for item in lock['files']:
        name = safe_name(item['path'])
        if not name.startswith('inputs/') or name in seen:
            raise ValueError('Invalid input lock member')
        seen.add(name)
        if any(indexed[name][key] != item[key] for key in ('bytes', 'sha256')):
            raise ValueError('Input mismatch: ' + name)
    return dict(plan_sha256=hashes['plan.json'], input_lock_sha256=hashes['input_lock.json'],
                source_commit=source['source_commit'])


def create_archive(root, name):
    if not __debug__:
        raise RuntimeError('Optimization mode is forbidden')
    if root != ROOT or safe_path(root) != ROOT:
        raise ValueError('Unexpected study root')
    if not re.fullmatch('[a-z0-9_]+', name):
        raise ValueError('Invalid delivery name')
    started = time.perf_counter()
    paths = enumerate_files(root)
    phases = closed_phases(root)
    items = []
    signatures = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        signatures[relative] = signature(path.lstat())
        items.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path)))
        if signature(path.lstat()) != signatures[relative]:
            raise ValueError('File changed during manifest')
    binding = check_locks(root, items)
    total = sum(i['bytes'] for i in items)
    # Include tar headers, block padding, gzip expansion, and a fixed safety reserve.
    upper = total + sum(4096 + len(i['path'].encode('utf-8')) * 4 for i in items)
    upper += upper // 1000 + RESERVE_BYTES
    if shutil.disk_usage(root).free < upper:
        raise ValueError('Insufficient delivery capacity')
    delivery = safe_path(root / 'delivery')
    delivery.mkdir(exist_ok=True)
    destination = safe_path(delivery / name)
    destination.mkdir()  # create-only, including failed previous attempts
    archive_path = destination / (name + '.tar.gz')
    try:
        with archive_path.open('xb') as raw, tarfile.open(fileobj=raw, mode='w:gz', dereference=False) as archive:
            for item in items:
                path = safe_path(root / item['path'])
                if signature(path.lstat()) != signatures[item['path']] or digest(path) != item['sha256']:
                    raise ValueError('Member changed before tar add')
                info = tarfile.TarInfo(item['path'])
                info.size, info.mode = item['bytes'], 0o600
                with path.open('rb') as stream:
                    if signature(os.fstat(stream.fileno())) != signatures[item['path']]:
                        raise ValueError('Member changed before tar read')
                    archive.addfile(info, stream)
                    if signature(os.fstat(stream.fileno())) != signatures[item['path']]:
                        raise ValueError('Member changed during tar read')
        if enumerate_files(root) != paths or closed_phases(root) != phases:
            raise ValueError('Study membership changed')
        for item in items:
            path = root / item['path']
            if signature(path.lstat()) != signatures[item['path']] or digest(path) != item['sha256']:
                raise ValueError('Study changed during delivery')
        expected = {i['path']: i for i in items}
        seen = set()
        with tarfile.open(archive_path, 'r:gz') as archive:
            for member in archive:
                name_in_tar = safe_name(member.name)
                if member.type != tarfile.REGTYPE or name_in_tar in seen or name_in_tar not in expected:
                    raise ValueError('Invalid tar member')
                seen.add(name_in_tar)
                item = expected[name_in_tar]
                with archive.extractfile(member) as stream:
                    actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                if member.size != item['bytes'] or actual != item['sha256']:
                    raise ValueError('Tar content mismatch')
        if seen != set(expected):
            raise ValueError('Tar member set mismatch')
        receipt = dict(status='PASS', members=items, archive_name=archive_path.name,
                       archive_bytes=archive_path.stat().st_size, archive_sha256=digest(archive_path),
                       **binding, counts=dict(files=len(items), total_bytes=total, closed_phases=len(phases)),
                       phases=phases, elapsed_seconds=time.perf_counter() - started)
        with (destination / 'receipt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except Exception as error:
        with (destination / 'archive_failure.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(status='failed', error=str(error), elapsed_seconds=time.perf_counter() - started), stream, indent=2)
            stream.write('\n')
        raise
    return dict(status='PASS', directory=str(destination), **receipt['counts'],
                archive_bytes=receipt['archive_bytes'], archive_sha256=receipt['archive_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    print(json.dumps(create_archive(args.output, args.name)))


if __name__ == '__main__':
    main()
