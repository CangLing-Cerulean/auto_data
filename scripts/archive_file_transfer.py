"""Snapshot terminal FTR1 phase artifacts without changing scientific inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path('/root/auto_data_experiments/20261014_file_transfer_r1')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('parents', 'forecast', 'final', 'failure'))
    args = parser.parse_args(); root = ROOT
    assert root.is_dir() and not root.is_symlink() and root.resolve() == root
    started = time.perf_counter()
    controls = {}
    for directory in sorted((root / 'control').iterdir()):
        assert directory.is_dir() and not directory.is_symlink()
        for name in ('launch.json', 'child.json'):
            if (directory / name).exists():
                info = read(directory / name)
                assert not Path('/proc', str(info['pid'])).exists(), (directory.name, name, info['pid'])
        controls[directory.name] = read(directory / 'exit.json') if (directory / 'exit.json').exists() else None
    if args.phase != 'failure':
        audit_name = {'parents': 'parents_audit.json', 'forecast': 'forecast_audit.json', 'final': 'final_audit.json'}[args.phase]
        assert read(root / audit_name)['status'] == 'PASS'
        assert all(info is not None and info['exit_code'] == 0 for info in controls.values())
    else:
        assert any(root.glob('*failure.json')) or any(info is None or info['exit_code'] != 0 for info in controls.values())
    archive_dir = root / 'archives'
    assert archive_dir.resolve() == archive_dir and not archive_dir.is_symlink()
    archive_dir.mkdir(exist_ok=True)
    target = archive_dir / f'file_transfer_r1_{args.phase}.tar.gz'
    assert not target.exists()
    records = []
    for path in sorted(root.rglob('*')):
        if archive_dir in path.parents or path == archive_dir:
            continue
        assert not path.is_symlink()
        if path.is_file():
            records.append(dict(path=path.relative_to(root).as_posix(), size=path.stat().st_size, sha256=sha(path)))
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for record in records:
                path = root / record['path']
                assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
                archive.add(path, arcname=record['path'], recursive=False)
    for record in records:
        assert sha(root / record['path']) == record['sha256']
    write(root / f'{args.phase}_archive_members.json', dict(phase=args.phase, members=records))
    receipt = dict(status='terminal_snapshot_created', phase=args.phase, archive=str(target), size=target.stat().st_size,
        sha256=sha(target), members=len(records), source_plan_sha256=sha(root / 'plan.json'),
        controls=controls, seconds=time.perf_counter() - started)
    write(root / f'{args.phase}_archive_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
