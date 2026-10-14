"""Create an isolated persistent research directory and retain a write probe."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def main():
    root = Path('/root/fs/auto_data_research_20261014')
    if root.parent.resolve() != Path('/root/fs') or root.exists():
        raise RuntimeError('Unexpected or existing destination; preserve it')
    start = time.perf_counter()
    root.mkdir(mode=0o700)
    probe = root / 'storage_preflight_64MiB.bin'
    block = bytes(1024 * 1024)
    h = hashlib.sha256()
    with probe.open('xb') as output:
        for _ in range(64):
            output.write(block)
            h.update(block)
        output.flush()
        os.fsync(output.fileno())
    actual = hashlib.sha256()
    with probe.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            actual.update(chunk)
    assert h.hexdigest() == actual.hexdigest()
    usage = shutil.disk_usage(root)
    receipt = dict(status='PASS', root=str(root), directory_mode=oct(root.stat().st_mode & 0o777),
                   probe_bytes=probe.stat().st_size, probe_sha256=actual.hexdigest(),
                   mounted_filesystem_free_bytes=usage.free, seconds=time.perf_counter()-start,
                   scope='64 MiB fsync and readback; mount-level free space does not establish per-user quota',
                   original_experiments_changed=False, deleted_files=0, new_training=0,
                   D_access=False, test_access=False)
    (root / 'storage_preflight.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
