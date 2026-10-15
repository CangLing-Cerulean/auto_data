"""Verify a downloaded delivery and restore inert ordinary files, create-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import time


def safe_path(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('An exact absolute path without aliases is required')
    for part in [*reversed(path.parents), path]:
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 1024:
            raise ValueError('Symlinks and reparse points are forbidden')
    return path


def member_name(name):
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Unsafe member name')
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe member name')
    for part in rel.parts:
        if part.endswith((' ', '.')) or any(ord(c) < 32 for c in part):
            raise ValueError('Unsafe portable member name')
        if re.fullmatch(r'(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])', part.split('.')[0]):
            raise ValueError('Reserved portable member name')
    return name


def stream_hash(stream):
    digest = hashlib.sha256()
    size = 0
    while block := stream.read(1024 * 1024):
        size += len(block)
        digest.update(block)
    return size, digest.hexdigest()


def file_hash(path):
    safe_path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('An ordinary file is required')
    with path.open('rb') as stream:
        return stream_hash(stream)


def expected_members(receipt):
    expected = {}
    portable = set()
    for item in receipt['members']:
        name = member_name(item['path'])
        if name in expected or name.casefold() in portable:
            raise ValueError('Duplicate receipt member')
        if type(item['bytes']) is not int or item['bytes'] < 0:
            raise ValueError('Invalid member size')
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError('Invalid member SHA256')
        expected[name] = item
        portable.add(name.casefold())
    return expected


def check_tar(archive, expected):
    seen = set()
    for member in archive:
        name = member_name(member.name)
        if not member.isfile() or member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise ValueError('Only ordinary tar files are permitted')
        if name in seen or name not in expected or member.size != expected[name]['bytes']:
            raise ValueError('Duplicate, unexpected, or incorrectly sized member')
        seen.add(name)
        with archive.extractfile(member) as stream:
            actual = stream_hash(stream)
        if actual != (member.size, expected[name]['sha256']):
            raise ValueError('Member content hash mismatch: ' + name)
    if seen != set(expected):
        raise ValueError('Missing archive members')


def verify(archive_path, receipt_path, output, verification=None):
    if not __debug__:
        raise RuntimeError('Optimization mode is forbidden')
    started = time.perf_counter()
    archive_path, receipt_path, output = map(safe_path, (archive_path, receipt_path, output))
    base = archive_path.parent
    if output == base or not output.is_relative_to(base):
        raise ValueError('Output must be a new directory beneath the archive directory')
    verification = safe_path(verification or base / 'download_verification.json')
    if not verification.is_relative_to(base) or verification == output or verification.is_relative_to(output):
        raise ValueError('Verification log must be beneath archive directory and outside output')
    if not verification.parent.is_dir():
        raise ValueError('Verification log parent must already exist')
    # Reserve the log before creating any restored asset. Failures remain inspectable.
    with verification.open('x', encoding='utf-8') as log:
        result = dict(status='failed', archive=str(archive_path), output=str(output))
        try:
            if output.exists():
                raise FileExistsError('Output already exists')
            if not stat.S_ISREG(receipt_path.lstat().st_mode):
                raise ValueError('Receipt must be an ordinary file')
            receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
            if archive_path.name != receipt['archive_name']:
                raise ValueError('Archive name mismatch')
            if file_hash(archive_path) != (receipt['archive_bytes'], receipt['archive_sha256']):
                raise ValueError('Archive size or SHA256 mismatch')
            expected = expected_members(receipt)
            with tarfile.open(archive_path, 'r:gz') as archive:
                check_tar(archive, expected)
            output.mkdir()  # Parent must exist; never create unrelated trees.
            with tarfile.open(archive_path, 'r:gz') as archive:
                restored = set()
                for member in archive:
                    name = member_name(member.name)
                    if not member.isfile() or name in restored or name not in expected:
                        raise ValueError('Archive changed during restoration')
                    target = safe_path(output / name)
                    if not target.is_relative_to(output):
                        raise ValueError('Escaping target')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    safe_path(target.parent)
                    with archive.extractfile(member) as src, target.open('xb') as dst:
                        while block := src.read(1024 * 1024):
                            dst.write(block)
                        dst.flush()
                        os.fsync(dst.fileno())
                    target.chmod(0o600)
                    if file_hash(target) != (expected[name]['bytes'], expected[name]['sha256']):
                        raise ValueError('Restored file mismatch: ' + name)
                    restored.add(name)
                if restored != set(expected):
                    raise ValueError('Restored set mismatch')
            if file_hash(archive_path) != (receipt['archive_bytes'], receipt['archive_sha256']):
                raise ValueError('Archive changed during restoration')
            result.update(status='PASS', files=len(expected),
                          total_bytes=sum(i['bytes'] for i in expected.values()))
        except Exception as error:
            result['error'] = str(error)
            raise
        finally:
            result['elapsed_seconds'] = time.perf_counter() - started
            json.dump(result, log, indent=2, allow_nan=False)
            log.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verification', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive, args.receipt, args.output, args.verification)))


if __name__ == '__main__':
    main()
