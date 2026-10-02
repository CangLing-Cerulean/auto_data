"""Freeze real MetroPT3 rows without altering their bytes (Python standard library)."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source/metropt3/MetroPT3(AirCompressor).csv'
OUT = ROOT / 'data/metropt3_v1'
EXPECTED = 'db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert sha(SOURCE) == EXPECTED, 'Source checksum mismatch'
    assert not OUT.exists(), 'Frozen dataset already exists; do not overwrite'
    with SOURCE.open('rb') as stream:
        n = sum(1 for _ in stream) - 1
    sizes = [n * 7 // 10, n * 2 // 10]
    sizes.append(n - sum(sizes))
    OUT.mkdir(parents=True)
    profiles = []
    previous = None
    previous_id = None
    total = 0
    with SOURCE.open('rb') as stream:
        header = stream.readline()
        fields = next(csv.reader([header.decode('utf-8').rstrip()]))
        stats = {f: {'min': math.inf, 'max': -math.inf, 'sum': 0.0, 'missing': 0} for f in fields[2:]}
        for name, count in zip(('train', 'validation', 'test'), sizes):
            gaps = Counter()
            first = last = None
            with (OUT / f'{name}.csv').open('wb') as dest:
                dest.write(header)
                for _ in range(count):
                    line = stream.readline()
                    row = next(csv.reader([line.decode('utf-8')]))
                    assert len(row) == len(fields)
                    stamp = datetime.fromisoformat(row[1])
                    source_id = int(row[0])
                    assert previous is None or stamp > previous, 'Non-increasing timestamp'
                    assert previous_id is None or source_id > previous_id, 'Non-increasing source ID'
                    if last is not None:
                        gaps[int((stamp - last).total_seconds())] += 1
                    first = stamp if first is None else first
                    last = previous = stamp
                    previous_id = source_id
                    for field, value in zip(fields[2:], row[2:]):
                        assert value and math.isfinite(float(value)), 'Missing or nonfinite value'
                        if name == 'train':
                            v = float(value)
                            s = stats[field]
                            s['min'], s['max'] = min(s['min'], v), max(s['max'], v)
                            s['sum'] += v
                    dest.write(line)
            profiles.append({'name': name, 'file': f'{name}.csv', 'rows': count,
                             'row_start_inclusive': total, 'row_end_exclusive': total + count,
                             'first_timestamp': first.isoformat(' '), 'last_timestamp': last.isoformat(' '),
                             'sha256': sha(OUT / f'{name}.csv'),
                             'interval_seconds_counts': dict(sorted(gaps.items()))})
            total += count
        assert not stream.read(), 'Unconsumed source rows'
    for s in stats.values():
        s['mean'] = s.pop('sum') / sizes[0]
    manifest = {'dataset_id': 'metropt3_v1', 'source_file': str(SOURCE.relative_to(ROOT)),
                'source_sha256': EXPECTED, 'source_archive': 'industrial_timeseries_eval_120_v0.7.0.zip',
                'source_archive_sha256': 'b0c2489d1103f224c55e070da956f00e7905d4389850f31f5d2a34f42227a940',
                'rows': n, 'columns_raw': fields, 'sensor_columns': fields[2:],
                'split_rule': 'chronological; floor(N*0.7), floor(N*0.2), remainder',
                'transforms': [], 'window_protocol': None, 'splits': profiles,
                'quality': {'strictly_increasing_timestamps': True, 'strictly_increasing_source_ids': True,
                            'missing_or_nonfinite_sensor_values': 0, 'raw_bytes_preserved': True}}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'train_profile.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'rows': n, 'splits': [{k: s[k] for k in ('name','rows','first_timestamp','last_timestamp')} for s in profiles]}, indent=2))


if __name__ == '__main__':
    main()
