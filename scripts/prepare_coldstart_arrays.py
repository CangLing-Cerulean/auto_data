"""Prepare only frozen A/B/C rows; real numeric access requires a published lock."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10'
DATA = ROOT / 'data/uci_electricity_v1'
PARTS = ('A', 'B', 'C')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def load_selected(path, partitions, channels):
    """Load zero-based, stop-exclusive A/B/C data rows, excluding the header.

    Unselected measurement bytes are never decoded or parsed. The stream stops
    at C's exclusive boundary, before fetching the first sealed D row.
    """
    if tuple(partitions) != PARTS or not isinstance(channels, int) or channels <= 0:
        raise ValueError('Require ordered A/B/C partitions and positive channels')
    previous_stop = 0
    for part in PARTS:
        spec = partitions[part]
        start, stop = spec['row_start'], spec['row_stop']
        if type(start) is not int or type(stop) is not int or start < previous_stop or stop <= start:
            raise ValueError('Invalid or overlapping row coordinates')
        if spec.get('records', stop - start) != stop - start:
            raise ValueError('Record count disagrees with row coordinates')
        previous_stop = stop
    arrays = {part: np.empty((p['row_stop'] - p['row_start'], channels), dtype=np.float64)
              for part, p in partitions.items()}
    counts = dict.fromkeys(PARTS, 0)
    with Path(path).open('rb') as stream:
        header = next(csv.reader([stream.readline().decode('utf-8-sig').rstrip('\r\n')], delimiter=';'))
        if header != [''] + [f'MT_{i:03d}' for i in range(1, channels + 1)]:
            raise ValueError('Unexpected channel header')
        for row, line in enumerate(itertools.islice(stream, previous_stop)):
            time_bytes, separator, value_bytes = line.partition(b';')
            part = next((p for p in PARTS if partitions[p]['row_start'] <= row < partitions[p]['row_stop']), None)
            if part is None:
                continue
            if not separator:
                raise ValueError(f'Missing time separator at selected row {row}')
            spec = partitions[part]
            stamp = datetime.strptime(time_bytes.strip(b'"').decode('ascii'), '%Y-%m-%d %H:%M:%S').isoformat()
            if row == spec['row_start'] and spec.get('first_date', stamp) != stamp:
                raise ValueError(f'First timestamp mismatch for {part}')
            if row == spec['row_stop'] - 1 and spec.get('last_date', stamp) != stamp:
                raise ValueError(f'Last timestamp mismatch for {part}')
            fields = value_bytes.rstrip(b'\r\n').split(b';')
            if len(fields) != channels:
                raise ValueError(f'Incorrect channel count at selected row {row}')
            values = np.array([float(field.strip(b'"').replace(b',', b'.').decode('ascii'))
                               for field in fields], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite measurement at selected row {row}')
            arrays[part][counts[part]] = values
            counts[part] += 1
    if any(counts[p] != len(arrays[p]) for p in PARTS):
        raise ValueError('Source ended before all selected rows were loaded')
    return arrays


def standardize(arrays):
    mean = arrays['A'].mean(axis=0, dtype=np.float64)
    raw_std = arrays['A'].std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(raw_std == 0., 1., raw_std)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Nonfinite A normalization statistics')
    normalized = {p: ((arrays[p] - mean) / scale).astype(np.float32) for p in PARTS}
    if not all(np.isfinite(value).all() for value in normalized.values()):
        raise ValueError('Nonfinite standardized float32 rows')
    summary = dict(A_raw_mean=mean.tolist(), A_raw_std=raw_std.tolist(),
                   A_zero_fraction=(arrays['A'] == 0.).mean(axis=0, dtype=np.float64).tolist(),
                   statistics_partition='A', std_ddof=0, zero_std_scale=1.)
    return normalized, mean, scale, summary


def git_bytes(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def validate_lock():
    """Validate all publication identities before opening the real raw file."""
    lock = read(STUDY / 'execution_lock.json')
    publication = read(STUDY / 'source_publication_receipt.json')
    commit, tree = publication['source_commit'], publication['tree_sha']
    if not all(re.fullmatch(r'[0-9a-f]{40}', value) for value in (commit, tree)):
        raise ValueError('Require full commit and tree identities')
    if publication['status'] != 'exact_commit_and_tree_verified':
        raise ValueError('Exact source publication receipt does not match lock')
    if git_bytes('rev-parse', f'{commit}^{{tree}}').decode('ascii').strip() != tree:
        raise ValueError('Locked commit tree mismatch')
    expected = {
        STUDY / 'execution_lock.json': sha(STUDY / 'execution_lock.json'),
        STUDY / 'source_index.json': lock['source_index_sha256'],
        DATA / 'manifest.json': lock['manifest_sha256'],
        STUDY / 'SOURCE_PROTOCOL.md': lock['source_protocol_sha256'],
        STUDY / 'PROTOCOL.md': lock['protocol_sha256'],
    }
    code = lock['runtime_code_sha256']
    if Path(__file__).name not in code:
        raise ValueError('Preparation code is absent from execution lock')
    for basename, digest in code.items():
        if Path(basename).name != basename or '/' in basename or '\\' in basename:
            raise ValueError('Runtime code keys must be script basenames')
        expected[ROOT / 'scripts' / basename] = digest
    for path, digest in expected.items():
        if not re.fullmatch(r'[0-9a-f]{64}', digest) or sha(path) != digest:
            raise ValueError(f'Local frozen identity mismatch: {path.name}')
        committed = git_bytes('show', f'{commit}:{path.relative_to(ROOT).as_posix()}')
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f'Committed frozen identity mismatch: {path.name}')
    index, manifest = read(STUDY / 'source_index.json'), read(DATA / 'manifest.json')
    if (index['manifest_sha256'] != lock['manifest_sha256'] or
            index['source_protocol_sha256'] != lock['source_protocol_sha256'] or
            manifest['text']['sha256'] != lock['raw_text_sha256'] or
            index['partitions'] != manifest['time_index']['partitions'] or
            index['columns'] != [''] + [f'MT_{i:03d}' for i in range(1, 371)]):
        raise ValueError('Source identity or coordinates disagree with frozen metadata')
    if manifest['text']['path'] != 'raw/LD2011_2014.txt':
        raise ValueError('Unexpected original raw source path')
    return lock, index, manifest, publication


def prepare():
    started = time.perf_counter()
    lock, index, manifest, publication = validate_lock()
    outputs = STUDY / 'inputs'
    for name in ('ABC.npz', 'data_summary.json', 'input_receipt.json'):
        if (outputs / name).exists():
            raise FileExistsError(f'Create-only output already exists: {name}')
    raw = DATA / 'raw/LD2011_2014.txt'
    if raw.stat().st_size != manifest['text']['bytes'] or sha(raw) != lock['raw_text_sha256']:
        raise ValueError('Original raw bytes disagree with frozen source identity')
    hash_seconds = time.perf_counter() - started
    selected = {p: index['partitions'][p] for p in PARTS}
    arrays = load_selected(raw, selected, 370)
    normalized, mean, scale, summary = standardize(arrays)
    outputs.mkdir(parents=True, exist_ok=True)
    with (outputs / 'ABC.npz').open('xb') as stream:
        np.savez(stream, **normalized, mean=mean, std=scale,
                 source_row_starts=np.array([selected[p]['row_start'] for p in PARTS], dtype=np.int64))
    save(outputs / 'data_summary.json', summary)
    receipt = dict(status='PASS', created_at_utc=datetime.now(timezone.utc).isoformat(),
                   arrays_sha256=sha(outputs / 'ABC.npz'), data_summary_sha256=sha(outputs / 'data_summary.json'),
                   source_sha256=lock['raw_text_sha256'], source_index_sha256=lock['source_index_sha256'],
                   execution_lock_sha256=sha(STUDY / 'execution_lock.json'),
                   source_publication_receipt_sha256=sha(STUDY / 'source_publication_receipt.json'),
                   source_commit=publication['source_commit'], source_tree=publication['tree_sha'],
                   shapes={p: list(normalized[p].shape) for p in PARTS},
                   dtype='float32', normalization_dtype='float64', normalization_fit='A', std_ddof=0,
                   source_row_starts={p: selected[p]['row_start'] for p in PARTS},
                   source_row_stops={p: selected[p]['row_stop'] for p in PARTS},
                   windows_expanded=False, D_values_decoded=False, D_values_parsed=False,
                   seconds=dict(lock_and_source_hash=hash_seconds, total=time.perf_counter() - started))
    save(outputs / 'input_receipt.json', receipt)
    return receipt


def synthetic_check():
    checks = []
    with tempfile.TemporaryDirectory(prefix='coldstart_synthetic_') as folder:
        path = Path(folder) / 'fixture.txt'
        path.write_bytes(b'"";"MT_001";"MT_002"\n'
                         b'ignored;\xff;INVALID\n'
                         b'"2013-01-01 00:15:00";"0,0";"5,0"\n'
                         b'"2013-01-01 00:30:00";"2,0";"5,0"\n'
                         b'"2013-01-01 00:45:00";"4,0";"7,0"\n'
                         b'"2013-01-01 01:00:00";"6,0";"9,0"\n'
                         b'sealed_D;\xff;INVALID\n')
        parts = {p: dict(row_start=a, row_stop=b) for p, a, b in [('A', 1, 3), ('B', 3, 4), ('C', 4, 5)]}
        arrays = load_selected(path, parts, 2)
        assert all(v.dtype == np.float64 for v in arrays.values())
        assert arrays['A'].tolist() == [[0., 5.], [2., 5.]]
        assert arrays['B'].tolist() == [[4., 7.]] and arrays['C'].tolist() == [[6., 9.]]
        checks.append('row_coordinates_decimal_comma_float64')
        checks.append('excluded_pre_A_and_sealed_D_invalid_measurements_not_decoded')
        normalized, mean, scale, summary = standardize(arrays)
        assert mean.tolist() == [1., 5.] and scale.tolist() == [1., 1.]
        assert summary['A_raw_std'] == [1., 0.] and summary['A_zero_fraction'] == [.5, 0.]
        assert normalized['C'].dtype == np.float32 and normalized['C'].tolist() == [[5., 4.]]
        checks.append('A_only_population_statistics_zero_std_replaced_only_in_scale')
        for invalid in (b'NaN', b'Inf', b'INVALID', b'\xff', b'1;2;3'):
            bad = Path(folder) / 'invalid.txt'
            bad.write_bytes(path.read_bytes().replace(b'"0,0";"5,0"', invalid + b';5'))
            try:
                load_selected(bad, parts, 2)
            except (ValueError, UnicodeError):
                pass
            else:
                raise AssertionError('Selected invalid value was accepted')
        checks.append('selected_invalid_nonfinite_and_channel_count_rejected')
        truncated = Path(folder) / 'truncated.txt'
        truncated.write_bytes(b'\n'.join(path.read_bytes().splitlines()[:4]) + b'\n')
        try:
            load_selected(truncated, parts, 2)
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated selected interval was accepted')
        checks.append('truncated_interval_rejected')
        target = Path(folder) / 'test.npz'
        with target.open('xb') as stream:
            np.savez(stream, **normalized, mean=mean, std=scale,
                     source_row_starts=np.array([1, 3, 4], dtype=np.int64))
        with np.load(target, allow_pickle=False) as saved:
            assert saved.files == ['A', 'B', 'C', 'mean', 'std', 'source_row_starts']
            assert saved['source_row_starts'].tolist() == [1, 3, 4]
            assert saved['mean'].dtype == saved['std'].dtype == np.float64
        checks.append('npz_schema_dtypes_and_row_starts')
        try:
            with target.open('xb'):
                pass
        except FileExistsError:
            pass
        else:
            raise AssertionError('Create-only output protection failed')
        checks.append('create_only_output')
    result = dict(status='PASS', checks=checks, count=len(checks), real_source_opened=False,
                  prepare_executed=False, code_sha256=sha(Path(__file__)))
    save(STUDY / 'data_prepare_unit_checks.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--synthetic-check', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare() if args.prepare else synthetic_check(), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
