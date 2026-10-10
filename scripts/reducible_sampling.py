"""Shared data and budget boundaries for train-only reference-gap selection."""
import csv
import json
from pathlib import Path
import numpy as np
from run_research import ROOT, digest, valid_starts

FOLDER = ROOT / 'research_runs/reducible_sampling'
ROUND_START = 4789.922976038419
ROUND_LIMIT = min(7200., ROUND_START + 1800.)
WARMUP_STEPS = 1000
STEPS = 3000
BATCH_SIZE = 256
CANDIDATE_SIZE = 512
REFERENCE_BATCH_SIZE = 128
REFERENCE_LR_RATIO = 0.05
REFERENCE_SEED = 2029
SHUFFLE_SEED = 20261010


def used_seconds():
    return sum(json.loads(p.read_text()).get('elapsed_seconds', 0)
               for p in (ROOT / 'research_runs/results').glob('*/status.json'))


def load_data(origin, reference_only=False):
    manifest_path = ROOT / 'data/metropt3_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = manifest_path.parent / 'train.csv'
    expected = next(s for s in manifest['splits'] if s['name'] == 'train')
    assert digest(source) == expected['sha256']
    raw = np.loadtxt(source, delimiter=',', skiprows=1, usecols=range(2, 17), dtype=np.float64)
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle); next(reader)
        times = np.array([row[1] for row in reader], dtype='datetime64[s]')
    assert len(raw) == expected['rows'] and np.isfinite(raw).all()
    cut = int(len(raw) * origin)
    end = min(len(raw), int(round(len(raw) * (origin + .2))))
    all_starts = valid_starts(times, 40, 30)
    with np.load(ROOT / 'research_runs/selection/window_table.npz') as z:
        assert np.array_equal(all_starts, z['start_row_zero_based'])
    starts = all_starts[all_starts + 40 <= cut]
    mean = raw[:cut].mean(0); std = raw[:cut].std(0); scale = np.where(std == 0, 1, std)
    train = ((raw[:cut if reference_only else end] - mean) / scale).astype(np.float32)
    days = times.astype('datetime64[D]').astype(np.int64)
    reference_mask = ((days[starts] - days[0]) % 5 == 0) & (days[starts] == days[starts + 39])
    with np.load(ROOT / 'research_runs/cluster_selection/window_groups.npz') as z:
        indices = np.searchsorted(z['start_row'], starts)
        assert np.array_equal(z['start_row'][indices], starts)
        clusters = z['group'][indices]
    months = times[starts + 31].astype('datetime64[M]').astype(str)
    unique_months, month_id = np.unique(months, return_inverse=True)
    cells = clusters * len(unique_months) + month_id
    evaluation = None; groups = None; future = None
    if not reference_only and origin < 1:
        future = all_starts[(all_starts >= cut) & (all_starts + 40 <= end)]
        assert starts.max() + 40 <= future.min()
        evaluation = train; groups = {'all': np.ones(len(future), dtype=bool)}
    elif not reference_only:
        validation = source.parent / 'validation.csv'
        validation_meta = next(s for s in manifest['splits'] if s['name'] == 'validation')
        assert digest(validation) == validation_meta['sha256']
        value = np.loadtxt(validation, delimiter=',', skiprows=1, usecols=range(2, 17), dtype=np.float64)
        with validation.open(newline='', encoding='utf-8') as handle:
            reader = csv.reader(handle); next(reader)
            val_times = np.array([row[1] for row in reader], dtype='datetime64[s]')
        assert len(value) == validation_meta['rows'] and np.isfinite(value).all()
        evaluation = ((value - mean) / scale).astype(np.float32)
        future = valid_starts(val_times, 40, 30)
        groups = {'all': np.ones(len(future), dtype=bool)}
        def variation(array, windows):
            delta = np.abs(np.diff(array.astype(np.float64), axis=0)).mean(1)
            prefix = np.r_[0, np.cumsum(delta)]
            return (prefix[windows + 31] - prefix[windows]) / 31
        threshold = float(np.quantile(variation(train, starts), .9))
        high = variation(evaluation, future) > threshold
        groups.update(high_input_variation=high, ordinary_input_variation=~high)
        for month in np.unique(val_times[future + 32].astype('datetime64[M]').astype(str)):
            groups['month_' + month] = val_times[future + 32].astype('datetime64[M]').astype(str) == month
    audit = {'dataset_id': manifest['dataset_id'], 'manifest_sha256': digest(manifest_path),
             'source_train_sha256': expected['sha256'], 'origin': origin,
             'train_raw_rows': [0, cut], 'train_windows': len(starts),
             'mean': mean.tolist(), 'scale': scale.tolist(),
             'reference_eligible_windows': int(reference_mask.sum()),
             'reference_eligible_fraction': float(reference_mask.mean()),
             'evaluation_windows': 0 if future is None else len(future),
             'evaluation_scope': 'none' if reference_only else 'original_validation' if origin == 1 else 'next_20_percent_of_train',
             'probe_raw_rows': None if reference_only or origin == 1 else [cut, end]}
    return manifest, train, starts, reference_mask, cells, months, evaluation, future, groups, audit
