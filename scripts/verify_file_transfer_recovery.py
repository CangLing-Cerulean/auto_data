"""Validate every byte and safely extract the recovered FTR1 snapshot locally."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261014_file_transfer_r1'
PHASE = 'recovered_forecast'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.perf_counter()
    receipt = read(STAGE / f'{PHASE}_archive_receipt.json')
    records = read(STAGE / f'{PHASE}_archive_members.json')['members']
    assert receipt['status'] == 'terminal_recovered_audit_snapshot_created'
    expected = {row['path']: row for row in records}
    assert len(expected) == len(records) == receipt['members']
    assert len({name.casefold() for name in expected}) == len(expected)
    archive_path = STAGE / 'archives' / f'file_transfer_r1_{PHASE}.tar.gz'
    assert archive_path.stat().st_size == receipt['size'] and sha(archive_path) == receipt['sha256']
    destination = STAGE / 'archives' / 'unpacked_recovered_forecast'
    assert not destination.exists() and not destination.is_symlink()
    assert destination.parent.resolve() == destination.parent
    with tarfile.open(archive_path, mode='r:gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(expected)
        assert len({m.name for m in members}) == len(members)
        assert {m.name for m in members} == set(expected)
        for member in members:
            name = member.name
            parts = name.split('/')
            assert not PurePosixPath(name).is_absolute()
            assert '\\' not in name and ':' not in name and '\0' not in name
            assert all(part not in ('', '.', '..') for part in parts)
            assert member.isfile() and not member.issparse()
            assert member.size == expected[name]['size']
            target = destination.joinpath(*parts)
            assert target.is_relative_to(destination) and target.resolve() == target
        destination.mkdir()
        for member in members:
            target = destination.joinpath(*member.name.split('/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            assert target.parent.resolve() == target.parent
            digest = hashlib.sha256()
            count = 0
            with archive.extractfile(member) as source, target.open('xb') as output:
                while block := source.read(1048576):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
            assert count == member.size and digest.hexdigest() == expected[member.name]['sha256'], member.name
    for row in records:
        path = destination / row['path']
        assert path.stat().st_size == row['size'] and sha(path) == row['sha256'], row['path']
    bindings = ('plan.json', 'predictions.json', 'forecast_audit.json', 'recovery_integrity_20261015.json',
                'recovery_source_sync_receipt.json')
    for name in bindings:
        assert sha(STAGE / name) == sha(destination / name), name
    result = dict(status='PASS', phase=PHASE, archive_bytes=receipt['size'], archive_sha256=receipt['sha256'],
        individually_verified_regular_members=len(records), safe_unique_member_paths=True,
        local_key_receipts_equal_archive=list(bindings), seconds=time.perf_counter() - started,
        new_training_runs=0, test_access=False)
    with (STAGE / 'recovered_forecast_download_verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
