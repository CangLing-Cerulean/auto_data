"""Verify frozen files and reconstruct the exact source checksum, without fitting anything."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/metropt3_v1'


def main():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    combined = hashlib.sha256()
    previous = None
    offset = 0
    header = None
    for split in manifest['splits']:
        checksum = hashlib.sha256()
        with (OUT / split['file']).open('rb') as stream:
            current = stream.readline()
            checksum.update(current)
            if header is None:
                header = current
                combined.update(header)
            assert current == header
            count = 0
            for line in stream:
                checksum.update(line)
                combined.update(line)
                row = next(csv.reader([line.decode('utf-8')]))
                assert previous is None or row[1] > previous
                previous = row[1]
                if count == 0:
                    assert row[1] == split['first_timestamp']
                count += 1
            assert previous == split['last_timestamp']
        assert count == split['rows']
        assert checksum.hexdigest() == split['sha256']
        assert split['row_start_inclusive'] == offset
        offset += count
        assert split['row_end_exclusive'] == offset
    assert offset == manifest['rows']
    assert [s['rows'] for s in manifest['splits']] == [offset * 7 // 10, offset * 2 // 10, offset - offset * 7 // 10 - offset * 2 // 10]
    assert combined.hexdigest() == manifest['source_sha256']
    print('PASS: exact source reconstruction, checksums, 7:2:1 counts, chronological disjoint boundaries.')


if __name__ == '__main__':
    main()
