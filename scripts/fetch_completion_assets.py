"""Verify and unpack a completed-study archive without replacing differing files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True); args = parser.parse_args()
    assert digest(args.archive) == args.sha256, 'Downloaded archive hash mismatch'
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names)) and all(member.isfile() for member in members)
        receipts = [name for name in names if name.startswith('research_runs/completion_study/') and name.endswith('_receipt.json')]
        assert len(receipts) == 1
        receipt_name = receipts[0]
        receipt = json.load(archive.extractfile(receipt_name))
        expected = receipt['files']
        assert set(names) == set(expected) | {receipt_name}
        for member in members:
            target = (ROOT / member.name).resolve()
            assert target.is_relative_to(ROOT.resolve())
            assert member.name.startswith('research_runs/results/') or member.name == receipt_name
            if member.name == receipt_name:
                source_hash = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            else:
                source_hash = expected[member.name]
                with archive.extractfile(member) as handle:
                    assert hashlib.file_digest(handle, 'sha256').hexdigest() == source_hash
            if target.exists():
                assert target.is_file() and digest(target) == source_hash, 'Refusing to replace differing local asset: ' + member.name
            else:
                archive.extract(member, path=ROOT, filter='data')
                assert digest(target) == source_hash
    record = {'archive': str(args.archive.resolve()), 'archive_sha256': args.sha256,
        'runs': receipt['runs'], 'verified_assets': len(expected), 'all_hashes_match': True}
    destination = ROOT / 'research_runs/completion_study' / (args.archive.name.removesuffix('.tar.gz') + '_verification.json')
    assert not destination.exists()
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__': main()
