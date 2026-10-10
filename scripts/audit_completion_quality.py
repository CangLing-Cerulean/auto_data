"""CPU-only descriptive audit of original training measurements; no labels inferred."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from completion_policies import quality_profile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    begun = time.monotonic()
    folder = ROOT / 'research_runs/completion_study/measurement_audit'
    assert not folder.exists(); folder.mkdir()
    manifest_path = ROOT / 'data/metropt3_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    source = manifest_path.parent / 'train.csv'
    expected = next(row for row in manifest['splits'] if row['name'] == 'train')
    assert digest(source) == expected['sha256']
    raw = np.loadtxt(source, delimiter=',', skiprows=1, usecols=range(2, 17), dtype=np.float64)
    with source.open(newline='', encoding='utf-8') as handle:
        rows = csv.reader(handle); next(rows)
        references = [(row[0], row[1]) for row in rows]
    assert len(raw) == expected['rows'] == len(references)
    times = np.array([row[1] for row in references], dtype='datetime64[s]')
    profile = quality_profile(raw, manifest['sensor_columns'])
    for row in profile['sensors']:
        lo, hi = row['longest_constant_source_rows']
        row['source_index_first_last'] = [references[lo][0], references[hi - 1][0]]
        row['timestamp_first_last'] = [references[lo][1], references[hi - 1][1]]
        row['elapsed_seconds_first_to_last'] = int((times[hi - 1] - times[lo]) / np.timedelta64(1, 's'))
        row['gaps_over_30_seconds_inside_segment'] = int((np.diff(times[lo:hi]) > np.timedelta64(30, 's')).sum())
    change = np.any(np.diff(raw[:, :7], axis=0) != 0, axis=1)
    edges = np.r_[0, np.flatnonzero(change) + 1, len(raw)]
    longest = int(np.diff(edges).argmax()); lo, hi = int(edges[longest]), int(edges[longest + 1])
    profile['longest_joint_analog_constant_segment'] = {'source_rows_zero_based_half_open': [lo, hi], 'rows': hi - lo,
        'source_index_first_last': [references[lo][0], references[hi - 1][0]],
        'timestamp_first_last': [references[lo][1], references[hi - 1][1]],
        'elapsed_seconds_first_to_last': int((times[hi - 1] - times[lo]) / np.timedelta64(1, 's')),
        'digital_change_counts_inside_segment': {name: int((np.diff(raw[lo:hi, column]) != 0).sum())
            for column, name in enumerate(manifest['sensor_columns']) if column >= 7}}
    identity = {'dataset_id': manifest['dataset_id'], 'manifest_sha256': digest(manifest_path),
        'train_sha256': expected['sha256'], 'scope': 'original train only', 'test_or_ground_truth_accessed': False,
        'code_sha256': digest(Path(__file__)), 'helper_sha256': digest(ROOT / 'scripts/completion_policies.py'),
        'numpy_version': np.__version__, 'device': 'CPU', 'processing_triggered': False}
    write(folder / 'identity.json', identity); write(folder / 'measurement_quality.json', profile)
    (folder / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder / 'helper_snapshot.py').write_bytes((ROOT / 'scripts/completion_policies.py').read_bytes())
    elapsed = time.monotonic() - begun
    write(folder / 'result.json', {'phase': 'completed', 'identity': identity, 'elapsed_seconds': elapsed,
        'confirmed_measurement_faults': [], 'processing_triggered': False,
        'conclusion': 'Descriptive candidates were recorded; external acquisition or physical evidence is unavailable, so no sensor-error deletion or downweighting was justified.'})
    write(folder / 'asset_checksums.json', {path.name: digest(path) for path in folder.iterdir() if path.is_file()})
    print(json.dumps({'rows': len(raw), 'nonfinite_values': sum(row['nonfinite'] for row in profile['sensors']),
        'digital_values_outside_0_1': sum(row.get('outside_binary_count', 0) for row in profile['sensors']),
        'joint_analog_constant': profile['longest_joint_analog_constant_segment'], 'elapsed_seconds': elapsed}))


if __name__ == '__main__': main()
