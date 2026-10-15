"""Archive a completed audit recovery while retaining its interrupted predecessor."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time


ROOT = Path('/root/auto_data_experiments/20261014_file_transfer_r1')
RECOVERY = 'audit_forecast_recovery_20261015'
PHASE = 'recovered_forecast'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    started = time.perf_counter()
    assert ROOT.is_dir() and ROOT.resolve() == ROOT
    integrity = read(ROOT / 'recovery_integrity_20261015.json')
    assert integrity['status'] == 'PASS'
    audit = read(ROOT / 'forecast_audit.json')
    assert audit['status'] == 'PASS' and not audit['eligible_to_label']
    assert audit['predictions_sha256'] == sha(ROOT / 'predictions.json')
    assert audit['plan_sha256'] == sha(ROOT / 'plan.json')
    for name, expected in integrity['checked_sha256'].items():
        path = ROOT / name
        assert ROOT in path.parents and path.resolve() == path and sha(path) == expected, name
    for name, item in integrity['original_audit_controls'].items():
        assert sha(ROOT / 'control' / 'audit_forecast' / name) == item['sha256'], name
    controls = {}
    for directory in sorted((ROOT / 'control').iterdir()):
        assert directory.is_dir() and directory.resolve() == directory
        for name in ('launch.json', 'child.json'):
            if (directory / name).exists():
                assert not Path('/proc', str(read(directory / name)['pid'])).exists(), (directory.name, name)
        status_path = directory / 'exit.json'
        status = read(status_path) if status_path.exists() else None
        if directory.name == 'audit_forecast':
            assert status is None
        else:
            assert status is not None and status['exit_code'] == 0, directory.name
        controls[directory.name] = status
    assert 'audit_forecast' in controls and RECOVERY in controls
    assert not (ROOT / 'labels.lock').exists() and not (ROOT / 'labels.json').exists()
    archive_dir = ROOT / 'archives'
    assert archive_dir.resolve() == archive_dir
    archive_dir.mkdir(exist_ok=True)
    target = archive_dir / f'file_transfer_r1_{PHASE}.tar.gz'
    assert not target.exists()
    records = []
    for path in sorted(ROOT.rglob('*')):
        if path == archive_dir or archive_dir in path.parents:
            continue
        assert not path.is_symlink()
        if path.is_file():
            records.append(dict(path=path.relative_to(ROOT).as_posix(), size=path.stat().st_size, sha256=sha(path)))
    assert shutil.disk_usage(ROOT).free > sum(item['size'] for item in records) * 1.05 + 16777216
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for item in records:
                path = ROOT / item['path']
                assert path.stat().st_size == item['size'] and sha(path) == item['sha256']
                archive.add(path, arcname=item['path'], recursive=False)
    for item in records:
        assert sha(ROOT / item['path']) == item['sha256']
    write(ROOT / f'{PHASE}_archive_members.json', dict(phase=PHASE, members=records))
    receipt = dict(status='terminal_recovered_audit_snapshot_created', phase=PHASE, archive=str(target),
        size=target.stat().st_size, sha256=sha(target), members=len(records),
        source_plan_sha256=sha(ROOT / 'plan.json'), forecast_audit_sha256=sha(ROOT / 'forecast_audit.json'),
        controls=controls, original_audit_interruption_preserved=True,
        seconds=time.perf_counter() - started)
    write(ROOT / f'{PHASE}_archive_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
