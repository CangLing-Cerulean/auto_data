"""Read-only identities for narrowly scoped, already-archived cleanup candidates."""
import hashlib
import json
from pathlib import Path
import time

BASE = Path('/tmp/auto_data_intervention_stage1_20261012_r1')
PHASE0 = Path('/tmp/auto_data_intervention_phase0_20261012')
FILES = [BASE / name for name in (
    'stage1-fixed-references-20261012-r1.tar.gz',
    'stage1-construction-initial-20261012-r1.tar.gz',
    'stage1-confirmation-interrupted-20261012-r1.tar.gz',
    'stage1-confirmation-recovered-20261012-r1.tar.gz')]
FILES.append(PHASE0 / 'intervention-phase0-20261012-runtime-r1.tar')
DIRECTORY = PHASE0 / 'runtime_r1/paired_evaluation'


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected ordinary file: {path}')
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(path=str(path), bytes=after.st_size, allocated_bytes=after.st_blocks*512,
                sha256=h.hexdigest(), mtime_ns=after.st_mtime_ns)


def main():
    assert DIRECTORY.resolve() == DIRECTORY and DIRECTORY.is_dir()
    paths = sorted(DIRECTORY.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlink in candidate directory')
    report = dict(status='READ_ONLY_INVENTORY', observed_unix=time.time(),
                  archives=[identity(p) for p in FILES],
                  directory=str(DIRECTORY),
                  directory_files=[identity(p) for p in paths if p.is_file()],
                  deletion_authorized=False, deleted_files=0, training_started=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
