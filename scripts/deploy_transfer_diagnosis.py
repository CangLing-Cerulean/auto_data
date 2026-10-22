"""Verify two authorized payloads, then extract only their new stage prefixes."""
import argparse
import hashlib
from pathlib import Path
import tarfile


def extract(base, archive, digest, stage):
    destination = base / stage
    assert base.resolve() == Path('/root/data/auto_data_research')
    assert not destination.exists(), destination
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
    with tarfile.open(archive, 'r:gz') as source:
        members = []
        for member in source.getmembers():
            if member.name == 'MANIFEST.json':
                continue
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == stage
            assert member.isfile() or member.isdir()
            members.append(member)
        assert members
        source.extractall(base, members=members, filter='data')
    print(f'Verified and deployed {stage}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--payloads', type=Path, required=True)
    p.add_argument('--source-sha', required=True)
    a = p.parse_args()
    base = a.payloads.parent
    extract(base, a.payloads / '20261022_m1_shrinkage_r1_complete.tar.gz',
            'f13b4fc53d709c58db1de8f219568adb8b577220cf99d25ad403bd0177de3f60', '20261022_m1_shrinkage_r1')
    extract(base, a.payloads / 'source.tar.gz', a.source_sha, '20261022_transfer_diagnosis_r1')
