"""Execute only the user-approved A/B/C file list after complete identity checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

STAGE1 = Path('/tmp/auto_data_intervention_stage1_20261012_r1')
PHASE0 = Path('/tmp/auto_data_intervention_phase0_20261012')
PAIRED = PHASE0 / 'runtime_r1/paired_evaluation'
ARCHIVE_NAMES = {
    'stage1-fixed-references-20261012-r1.tar.gz',
    'stage1-construction-initial-20261012-r1.tar.gz',
    'stage1-confirmation-interrupted-20261012-r1.tar.gz',
    'stage1-confirmation-recovered-20261012-r1.tar.gz',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise RuntimeError('Preserve existing execution receipt')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    auth = json.loads(args.authorization.read_text(encoding='utf-8-sig'))
    if (auth.get('user_instruction') != '清理并继续实验'
            or auth.get('approved_groups') != ['A', 'B', 'C']
            or auth.get('plan_sha256') != digest(args.plan)):
        raise RuntimeError('Explicit authorization and immutable plan must match')
    expected = {STAGE1 / name for name in ARCHIVE_NAMES}
    expected.add(PHASE0 / 'intervention-phase0-20261012-runtime-r1.tar')
    paired = sorted(PAIRED.rglob('*'))
    if len(paired) != 13 or any(not p.is_file() or p.is_symlink() for p in paired):
        raise RuntimeError('Paired directory changed; preserve all files')
    expected.update(paired)
    records = plan['files']
    if len(records) != 18 or {Path(r['path']) for r in records} != expected:
        raise RuntimeError('Plan differs from the exact A/B/C boundary')
    stamps = {}
    for record in records:
        path = Path(record['path'])
        item = path.lstat()
        if path.resolve() != path or not stat.S_ISREG(item.st_mode):
            raise RuntimeError(f'Non-regular or indirect path: {path}')
        if item.st_size != record['bytes'] or digest(path) != record['sha256']:
            raise RuntimeError(f'Identity changed: {path}')
        after = path.stat()
        if (item.st_size, item.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Concurrent change: {path}')
        stamps[path] = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    # Check open file descriptors without reading process command lines or secrets.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    if Path(os.readlink(fd)) in expected:
                        raise RuntimeError(f'Candidate still open by PID {process.name}')
                except FileNotFoundError:
                    pass
        except (FileNotFoundError, PermissionError):
            pass
    receipt = dict(status='deleting', started_unix=time.time(),
                   plan_sha256=digest(args.plan), authorization_sha256=digest(args.authorization),
                   all_18_identities_verified=True, root_free_before=shutil.disk_usage('/').free,
                   removed_files=[], removed_directories=[], fs_access=False, fs_public_access=False,
                   model_training_started=False)

    def save():
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')

    save()
    try:
        for record in records:
            path = Path(record['path'])
            now = path.lstat()
            if (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns) != stamps[path]:
                raise RuntimeError(f'Changed immediately before unlink: {path}')
            path.unlink()
            receipt['removed_files'].append(record)
            save()
        PAIRED.rmdir()
        receipt['removed_directories'].append(str(PAIRED))
        receipt.update(status='PASS', completed_unix=time.time(),
                       root_free_after=shutil.disk_usage('/').free,
                       actual_files_removed=len(receipt['removed_files']),
                       top_level_experiment_directories_preserved=STAGE1.is_dir() and PHASE0.is_dir())
    except BaseException as error:
        receipt.update(status='FAILED_PARTIAL_PRESERVED', error=str(error))
        save()
        raise
    save()
    print(json.dumps({k: v for k, v in receipt.items() if k != 'removed_files'}))


if __name__ == '__main__':
    main()
