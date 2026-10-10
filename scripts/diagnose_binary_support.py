"""Observed binary-state support and preserved baseline errors; no fitting."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE = Path('/root/data/auto_data_research/metropt3_v1/session_20260926')
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'research_runs/direct_processing/20261009/binary_support.json'
EVIDENCE = PROJECT / 'research_runs/direct_diagnostics/20261009/baseline_features_v1'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    if OUT.exists():
        raise ValueError('Immutable diagnostic already exists')
    manifest = json.loads((BASE / 'data/metropt3_v1/manifest.json').read_text())
    baseline = json.loads((EVIDENCE / 'result.json').read_text())
    if sha(EVIDENCE / 'window_evidence.npz') != baseline['assets']['window_evidence.npz']:
        raise ValueError('Baseline evidence changed')
    spec = importlib.util.spec_from_file_location('frozen_core', BASE / 'scripts/run_research.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    state_tables, source_hashes = {}, {}
    for split in ('train', 'validation'):
        path = BASE / 'data/metropt3_v1' / f'{split}.csv'
        expected = next(s['sha256'] for s in manifest['splits'] if s['name'] == split)
        if sha(path) != expected:
            raise ValueError('Raw source hash mismatch')
        source_hashes[split] = expected
        raw = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(9, 17))
        if not np.isin(raw, (0, 1)).all():
            raise ValueError('Unexpected binary states')
        with path.open(encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle)
            next(reader)
            times = np.array([r[1] for r in reader], dtype='datetime64[s]')
        starts = core.valid_starts(times, 40, 30)
        prefix = np.concatenate([np.zeros((1, 8), np.int64), np.cumsum(np.diff(raw, axis=0) != 0, axis=0)], axis=0)
        constant = prefix[starts + 31] == prefix[starts]
        state_tables[split] = {'starts': starts, 'constant': constant, 'last': raw[starts + 31], 'rows': raw}
    with np.load(EVIDENCE / 'window_evidence.npz', allow_pickle=False) as saved:
        if not np.array_equal(saved['start_row'], state_tables['validation']['starts']):
            raise ValueError('Development window mapping mismatch')
        sq = np.stack([saved[f'{r}_sq'] for r in ('S000_uniform_s2026', 'S005_uniform_s2027', 'S010_uniform_s2028')]).mean(0)
        absolute = np.stack([saved[f'{r}_abs'] for r in ('S000_uniform_s2026', 'S005_uniform_s2027', 'S010_uniform_s2028')]).mean(0)
        persistence_sq = saved['persistence_sq'].copy()
        future_changed = saved['future_change_POSTHOC'].copy()
    records = {}
    for c, name in enumerate(manifest['sensor_columns'][7:]):
        states = {}
        for level in (0, 1):
            t = state_tables['train']
            d = state_tables['validation']
            tmask = t['constant'][:, c] & (t['last'][:, c] == level)
            dmask = d['constant'][:, c] & (d['last'][:, c] == level)
            states[str(level)] = {'T_row_share': float((t['rows'][:, c] == level).mean()), 'D_row_share': float((d['rows'][:, c] == level).mean()),
                'T_constant_windows': int(tmask.sum()), 'T_constant_window_share': float(tmask.mean()),
                'D_constant_windows': int(dmask.sum()), 'D_constant_window_share': float(dmask.mean()),
                'D_NMSE': float(sq[dmask, c + 7].mean()) if dmask.any() else None,
                'D_NMAE': float(absolute[dmask, c + 7].mean()) if dmask.any() else None,
                'D_persistence_NMSE': float(persistence_sq[dmask, c + 7].mean()) if dmask.any() else None,
                'D_future_changes_POSTHOC_windows': int((dmask & future_changed[:, c + 7]).sum())}
        records[name] = states
    OUT.write_text(json.dumps({'scope': 'descriptive_T_and_full_D_development', 'source_hashes': source_hashes,
        'baseline_evidence_sha256': baseline['assets']['window_evidence.npz'], 'code_sha256': sha(__file__), 'training_calls': 0,
        'API_calls': 0, 'test_access': False, 'binary_states': records,
        'limits': ['distribution shift and error association do not prove causal usefulness of upweighting', 'future changes are post-hoc only']}, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'binary_support_completed', 'channels': len(records)}))


if __name__ == '__main__':
    main()
