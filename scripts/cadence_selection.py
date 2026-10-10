"""Training-only timestamp cadence flags and matched intervention masks."""
import csv
import json
from pathlib import Path
import numpy as np
from run_research import ROOT, digest, write_json

FOLDER = ROOT / 'research_runs/cadence_selection'
ROUND_START = 6015.669168795459
ROUND_LIMIT = ROUND_START + 1000
RMS_THRESHOLD = .10
SOFT_WEIGHT = .25
SHUFFLE_SEED = 20261003


def build(starts, cells, origin, policy, selected, run):
    manifest = json.loads((ROOT / 'data/metropt3_v1/manifest.json').read_text())
    source = ROOT / 'data/metropt3_v1/train.csv'
    count = next(s['rows'] for s in manifest['splits'] if s['name'] == 'train')
    cut = int(count * origin)
    with source.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle); next(reader)
        times = np.array([next(reader)[1] for _ in range(cut)], dtype='datetime64[s]')
    assert starts.min() >= 0 and starts.max() + 40 <= cut
    delta = np.diff(times).astype('timedelta64[s]').astype(np.int64)
    assert (delta > 0).all()
    median = float(np.median(delta[delta <= 30])); assert median > 0
    prefix = np.r_[0., np.cumsum((delta.astype(np.float64) - median) ** 2)]
    score = np.sqrt((prefix[starts + 39] - prefix[starts]) / 39) / median
    flagged = score > RMS_THRESHOLD
    assert 0 < flagged.sum() < .5 * len(starts), 'Rule is ineligible; do not retune threshold'
    applied = flagged.copy()
    if policy == 'shuffle':
        assert selected in ('drop', 'soft')
        rng = np.random.default_rng(SHUFFLE_SEED)
        for cell in np.unique(cells):
            indices = np.flatnonzero(cells == cell)
            applied[indices] = flagged[rng.permutation(indices)]
    else:
        assert selected is None
    actual = selected if policy == 'shuffle' else policy
    q = np.full(len(starts), 1. / len(starts)); loss_weight = np.ones(len(starts))
    if actual == 'drop':
        q = (~applied).astype(np.float64); q /= q.sum()
    elif actual == 'soft':
        loss_weight[applied] = SOFT_WEIGHT
    else:
        raise ValueError(actual)
    by_cell = {str(c): {'windows': int((cells == c).sum()),
        'flagged': int(flagged[cells == c].sum()), 'applied': int(applied[cells == c].sum())} for c in np.unique(cells)}
    assert all(v['flagged'] == v['applied'] for v in by_cell.values())
    expected_loss_weight = float(q @ loss_weight)
    audit = {'origin': origin, 'history_rows': cut, 'history_last_timestamp': str(times[-1]),
        'median_interval_seconds': median, 'relative_RMS_threshold': RMS_THRESHOLD,
        'score_quantiles': dict(zip(['min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'],
            np.quantile(score, [0, .25, .5, .75, .9, .95, .99, 1]).tolist())),
        'eligible_windows': len(starts), 'flagged_windows': int(flagged.sum()), 'flagged_fraction': float(flagged.mean()),
        'retained_windows': int((q > 0).sum()), 'policy': policy, 'applied_policy': actual,
        'expected_loss_weight': expected_loss_weight, 'cell_counts': by_cell,
        'shuffle_seed': SHUFFLE_SEED if policy == 'shuffle' else None,
        'group_asset_sha256': digest(ROOT / 'research_runs/cluster_selection/window_groups.npz'),
        'no_test_access': True, 'timestamp_score_scope': 'current training prefix only; all 39 training-window intervals'}
    np.savez_compressed(run / 'cadence_policy.npz', train_start=starts, cell=cells, score=score,
        flagged=flagged, applied=applied, probability=q, loss_weight=loss_weight)
    write_json(run / 'cadence_audit.json', audit)
    return q, loss_weight, expected_loss_weight
