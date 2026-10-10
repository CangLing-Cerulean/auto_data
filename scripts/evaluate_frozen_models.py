"""Local CPU validation, immutable freeze, and one final test evaluation."""
import argparse
import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/model_exploration'
TOLERANCE = 1e-5


def digest(path):
    with path.open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_new(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def predict(kind, weights, x):
    flat = x.reshape(len(x), 480)
    if kind in ('cross_linear', 'dlinear'):
        pred = flat @ weights['w'].T + weights['b']
    elif kind == 'residual_mlp':
        hidden = np.maximum(flat @ weights['w1'].T + weights['b1'], 0)
        pred = hidden @ weights['w2'].T + weights['b2']
    else:
        raise ValueError(kind)
    pred = pred.reshape(-1, 8, 15)
    return pred if kind == 'dlinear' else pred + x[:, -1:, :]


def identity():
    manifest_path = ROOT / 'data/metropt3_v1/manifest.json'
    export_path = FOLDER / 'exports/export_manifest.json'
    meta = json.loads(export_path.read_text(encoding='utf-8'))
    assert digest(manifest_path) == meta['manifest_sha256']
    assert digest(FOLDER / 'PROTOCOL.md') == meta['study_protocol_sha256']
    for m in meta['models']:
        assert digest(FOLDER / 'exports' / m['export_file']) == m['export_sha256']
        assert digest(ROOT / 'research_runs/results' / m['run_id'] / 'best.pt') == m['checkpoint_sha256']
    return json.loads(manifest_path.read_text(encoding='utf-8')), meta


def data_for(split, manifest, meta):
    path = ROOT / 'data/metropt3_v1' / (split + '.csv')
    info = next(s for s in manifest['splits'] if s['name'] == split)
    assert digest(path) == info['sha256']
    raw = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(2, 17), dtype=np.float64)
    assert len(raw) == info['rows'] and np.isfinite(raw).all()
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle); next(reader)
        times = np.array([r[1] for r in reader], dtype='datetime64[s]')
    delta = np.diff(times).astype('timedelta64[s]').astype(np.int64); assert (delta > 0).all()
    gaps = np.r_[0, np.cumsum(delta > 30)]
    starts = np.arange(len(raw) - 39)
    starts = starts[gaps[starts + 39] == gaps[starts]]
    return ((raw - np.array(meta['mean'])) / np.array(meta['scale'])).astype(np.float32), starts


def evaluate_all(split, manifest, meta):
    data, starts = data_for(split, manifest, meta)
    rows = []; scale = np.array(meta['scale'])
    for m in meta['models']:
        with np.load(FOLDER / 'exports' / m['export_file']) as z:
            weights = {name: z[name] for name in z.files}
        probe_prediction = predict(m['kind'], weights, weights['probe_x'])
        sample_error = float(np.max(np.abs(probe_prediction - weights['probe_expected'])))
        np.testing.assert_allclose(probe_prediction, weights['probe_expected'], atol=TOLERANCE, rtol=TOLERANCE)
        square = np.zeros(15); absolute = np.zeros(15)
        for lo in range(0, len(starts), 2048):
            batch = data[starts[lo:lo + 2048, None] + np.arange(40)]
            error = (predict(m['kind'], weights, batch[:, :32]) - batch[:, 32:]).astype(np.float64)
            square += np.square(error).sum(axis=(0, 1)); absolute += np.abs(error).sum(axis=(0, 1))
        per_mse = square / (len(starts) * 8); per_mae = absolute / (len(starts) * 8)
        row = {'run_id': m['run_id'], 'seed': m['seed'], 'kind': m['kind'], 'windows': len(starts),
            'NMSE': float(per_mse.mean()), 'NMAE': float(per_mae.mean()),
            'raw_unit_MSE': float(np.mean(per_mse * scale ** 2)),
            'per_channel_NMSE': per_mse.tolist(), 'per_channel_NMAE': per_mae.tolist(),
            'probe_max_absolute_error': sample_error}
        if split == 'validation':
            assert len(starts) == m['reference_validation']['windows']
            for metric in ('NMSE', 'NMAE'):
                assert abs(row[metric] - m['reference_validation'][metric]) <= TOLERANCE, (m['run_id'], metric)
        rows.append(row); print(json.dumps({k: row[k] for k in ('run_id', 'NMSE', 'NMAE', 'raw_unit_MSE')}), flush=True)
    return {'split': split, 'rows': rows, 'valid_windows': len(starts), 'sensor_columns': manifest['sensor_columns'],
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'CPU'},
            'probe_tolerance': {'atol': TOLERANCE, 'rtol': TOLERANCE}, 'full_validation_metric_absolute_tolerance': TOLERANCE}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['validation', 'freeze', 'test'], required=True)
    args = parser.parse_args(); begun = time.monotonic()
    manifest, meta = identity()
    if args.stage == 'validation':
        assert not (FOLDER / 'freeze.json').exists() and not (FOLDER / 'validation_cpu.json').exists()
        result = evaluate_all('validation', manifest, meta)
        result['cpu_wall_seconds'] = time.monotonic() - begun
        save_new(FOLDER / 'validation_cpu.json', result)
    elif args.stage == 'freeze':
        progress = json.loads((FOLDER / 'progress.json').read_text(encoding='utf-8'))
        assert progress['test_eligible'] and progress['phase'] == 'completed_confirmation'
        validation = json.loads((FOLDER / 'validation_cpu.json').read_text(encoding='utf-8'))
        assert {r['run_id'] for r in validation['rows']} == {m['run_id'] for m in meta['models']}
        assert not (FOLDER / 'test_result.json').exists() and not (FOLDER / 'test_started.json').exists()
        freeze = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_family': progress['winner'],
            'selected_runs': progress['selected_runs'], 'seeds': [2026, 2027, 2028],
            'primary_metric': 'NMSE', 'rule': 'report all three paired seeds and arithmetic means; no test selection or tuning',
            'export_manifest_sha256': digest(FOLDER / 'exports/export_manifest.json'),
            'validation_cpu_sha256': digest(FOLDER / 'validation_cpu.json'),
            'evaluator_sha256': digest(Path(__file__)), 'progress_sha256': digest(FOLDER / 'progress.json'),
            'data_and_models': meta, 'test_feedback_previously_used': False}
        save_new(FOLDER / 'freeze.json', freeze)
        print(json.dumps({'freeze_sha256': digest(FOLDER / 'freeze.json'), 'selected_runs': progress['selected_runs']}))
    else:
        freeze = json.loads((FOLDER / 'freeze.json').read_text(encoding='utf-8'))
        for key, path in [('evaluator_sha256', Path(__file__)), ('export_manifest_sha256', FOLDER / 'exports/export_manifest.json'),
                          ('validation_cpu_sha256', FOLDER / 'validation_cpu.json'), ('progress_sha256', FOLDER / 'progress.json')]:
            assert digest(path) == freeze[key]
        assert meta == freeze['data_and_models']
        assert not (FOLDER / 'test_result.json').exists()
        save_new(FOLDER / 'test_started.json', {'started_utc': datetime.now(timezone.utc).isoformat(),
            'freeze_sha256': digest(FOLDER / 'freeze.json')})
        result = evaluate_all('test', manifest, meta)
        pairs = []
        for seed in freeze['seeds']:
            chosen = next(r for r in result['rows'] if r['seed'] == seed and r['kind'] != 'dlinear')
            baseline = next(r for r in result['rows'] if r['seed'] == seed and r['kind'] == 'dlinear')
            pairs.append({'seed': seed, 'candidate': chosen['run_id'], 'baseline': baseline['run_id'],
                **{metric + '_reduction_pct': 100 * (1 - chosen[metric] / baseline[metric]) for metric in ('NMSE', 'NMAE', 'raw_unit_MSE')}})
        result['paired_comparisons'] = pairs
        result['means'] = {group: {metric: float(np.mean([r[metric] for r in result['rows'] if (r['kind'] == 'dlinear') == (group == 'baseline')]))
            for metric in ('NMSE', 'NMAE', 'raw_unit_MSE')} for group in ('baseline', 'candidate')}
        result['mean_reduction_pct'] = {metric: 100 * (1 - result['means']['candidate'][metric] / result['means']['baseline'][metric])
            for metric in ('NMSE', 'NMAE', 'raw_unit_MSE')}
        result.update(cpu_wall_seconds=time.monotonic() - begun, freeze_sha256=digest(FOLDER / 'freeze.json'),
            completed_utc=datetime.now(timezone.utc).isoformat(), test_feedback_used_for_tuning=False)
        save_new(FOLDER / 'test_result.json', result)
        print(json.dumps({'means': result['means'], 'reductions': result['mean_reduction_pct']}))


if __name__ == '__main__': main()
