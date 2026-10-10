"""Direct, inference-only T-fitted diagnostics on full development validation."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

BASE = Path('/root/data/auto_data_research/metropt3_v1/session_20260926')
BASELINES = {
    'S000_uniform_s2026': ('dac0dda87e6694a13cc8d867b733ba7aeefc0b4457df0ac11b420af490f34998', 'f2a42f47cdc622eeaea15a756ed6743c60e5e5fcc6b016b0fee0fbc88d8588a5'),
    'S005_uniform_s2027': ('306e916c185bbf2521dbceb60249408947fdde4d49566bfb464e23eadfe9bd3b', '9e995ed5326dadd6b8e4f9f1689b487a6a449e7ba876b4e91989c33c39b7e6c7'),
    'S010_uniform_s2028': ('271959880402aa5a32d789883762c1b91943a659648c783bc78e8c7e4ab47860', '48f4f039a08565bb1ad70307aa6bcfd90e8573600844847b3925a0b2503ac74e'),
}
RECENT_INTERVALS = 4


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metadata(path):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [(int(row[0]), row[1]) for row in reader]
    return header, np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype='datetime64[s]')


def features(values, starts, times):
    """Never inspect targets; all prefix sums end at the last input record."""
    change = np.diff(values.astype(np.float64), axis=0)
    prefix = np.r_[0., np.cumsum(np.abs(change).mean(1))]
    variation = (prefix[starts + 31] - prefix[starts]) / 31
    counts = np.concatenate([np.zeros((1, 15), dtype=np.int64), np.cumsum(change != 0, axis=0)], axis=0)
    constant = counts[starts + 31] == counts[starts]
    switches = np.r_[0, np.cumsum(np.any(change[:, 7:] != 0, axis=1))]
    total = switches[starts + 31] - switches[starts]
    recent = switches[starts + 31] - switches[starts + 31 - RECENT_INTERVALS]
    duration = (times[starts + 31] - times[starts]).astype('timedelta64[s]').astype(float)
    return {'variation': variation, 'constant': constant, 'switches': total, 'recent': recent, 'duration': duration}


def masks(f, thresholds):
    return {
        'all': np.ones(len(f['variation']), dtype=bool),
        'variation_low': f['variation'] <= thresholds['variation_q25'],
        'variation_middle': (f['variation'] > thresholds['variation_q25']) & (f['variation'] <= thresholds['variation_q90']),
        'variation_high': f['variation'] > thresholds['variation_q90'],
        'digital_no_switch': f['switches'] == 0,
        'digital_early_switch_only': (f['switches'] > 0) & (f['recent'] == 0),
        'digital_recent_switch': f['recent'] > 0,
        'analog_all_constant': f['constant'][:, :7].all(1),
        'analog_any_changes': ~f['constant'][:, :7].all(1),
        'duration_short': f['duration'] <= thresholds['duration_q25'],
        'duration_middle': (f['duration'] > thresholds['duration_q25']) & (f['duration'] <= thresholds['duration_q90']),
        'duration_long': f['duration'] > thresholds['duration_q90'],
    }


def infer(core, model, data, starts, p):
    import torch
    offsets = torch.arange(40, device='cuda')
    squared, absolute, signed, future_change = [], [], [], []
    with torch.inference_mode():
        for lo in range(0, len(starts), p['evaluation_batch_size']):
            batch = data[starts[lo:lo + p['evaluation_batch_size'], None] + offsets]
            x, y = batch[:, :32], batch[:, 32:]
            prediction = x[:, -1:, :].expand(-1, 8, -1) if model is None else core.predict(model, x, 'baseline')
            error = (prediction - y).double()
            squared.append(error.square().mean(1).cpu().numpy())
            absolute.append(error.abs().mean(1).cpu().numpy())
            signed.append(error.mean(1).cpu().numpy())
            if model is None:
                future_change.append((y != x[:, -1:, :]).any(1).cpu().numpy())
    output = {'sq': np.concatenate(squared), 'abs': np.concatenate(absolute), 'signed': np.concatenate(signed)}
    if model is None:
        output['future_change'] = np.concatenate(future_change)
    return output


def metric(error, mask, channel=None):
    if not mask.any():
        return None
    return {'NMSE': float(error['sq'][mask].mean() if channel is None else error['sq'][mask, channel].mean()),
            'NMAE': float(error['abs'][mask].mean() if channel is None else error['abs'][mask, channel].mean()),
            'signed_error': (error['signed'][mask].mean(0).tolist() if channel is None else float(error['signed'][mask, channel].mean()))}


def main(output):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    import hashlib
    def sha(path):
        with Path(path).open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('A new diagnostic directory is required; do not replay an existing ID')
    protected = {
        BASE / 'data/metropt3_v1/manifest.json': '672c0ecf55bc94a52abc6b256c0bfe7a4da02140e57dd0527375bf859a22c087',
        BASE / 'research_runs/protocol.json': '573ee4e87ef442b7140afb26f4befe80a8de1a43fec3b5c1fe6b878e527f8afe',
        BASE / 'scripts/run_research.py': '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052',
        Path(__file__).resolve(): sha(__file__),
        Path(__file__).resolve().parents[1] / 'research_runs/direct_diagnostics/20261009/PROTOCOL.md': sha(Path(__file__).resolve().parents[1] / 'research_runs/direct_diagnostics/20261009/PROTOCOL.md'),
    }
    manifest = read_json(BASE / 'data/metropt3_v1/manifest.json')
    for split in manifest['splits']:
        if split['name'] in ('train', 'validation'):
            protected[BASE / 'data/metropt3_v1' / split['file']] = split['sha256']
    for run_id, hashes in BASELINES.items():
        for name, expected in zip(('best.pt', 'result.json'), hashes):
            protected[BASE / 'research_runs/results' / run_id / name] = expected
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f'Asset changed: {path}')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
        raise RuntimeError('Use the designated 4090 server')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('trusted_original_core', BASE / 'scripts/run_research.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    p = read_json(BASE / 'research_runs/protocol.json')
    if (p['input_length'], p['prediction_length'], p['channels']) != (32, 8, 15):
        raise ValueError('Frozen task required')
    output.mkdir(parents=True)
    core.write_json(output / 'identity.json', {'protected_assets': {str(k): v for k, v in protected.items()}, 'scope': 'researcher_guided_development', 'training_calls': 0, 'api_calls': 0, 'test_access': False})
    started = time.monotonic()
    arrays, starts, original_groups, audit = core.prepare(p, manifest)
    f, timestamps, source_indices = {}, {}, {}
    for split in ('train', 'validation'):
        header, ids, times = metadata(BASE / 'data/metropt3_v1' / f'{split}.csv')
        entry = next(s for s in manifest['splits'] if s['name'] == split)
        if header != manifest['columns_raw'] or len(ids) != entry['rows'] or str(times[0]).replace('T', ' ') != entry['first_timestamp'] or str(times[-1]).replace('T', ' ') != entry['last_timestamp']:
            raise ValueError('Manifest metadata mismatch')
        f[split] = features(arrays[split], starts[split], times)
        timestamps[split], source_indices[split] = times, ids
    thresholds = {f'{name}_q{int(q * 100)}': float(np.quantile(f['train'][name], q)) for name in ('variation', 'duration') for q in (.25, .9)}
    split_masks = {s: masks(f[s], thresholds) for s in f}
    data = torch.from_numpy(arrays['validation']).cuda()
    indices = torch.from_numpy(starts['validation']).cuda()
    errors = {'persistence': infer(core, None, data, indices, p)}
    for run_id in BASELINES:
        model = core.DLinear(p).cuda()
        if sum(v.numel() for v in model.parameters()) != 7920:
            raise ValueError('Architecture changed')
        model.load_state_dict(torch.load(BASE / 'research_runs/results' / run_id / 'best.pt', weights_only=True))
        model.eval()
        errors[run_id] = infer(core, model, data, indices, p)
        old = read_json(BASE / 'research_runs/results' / run_id / 'result.json')
        for name, mask in original_groups.items():
            current = metric(errors[run_id], mask)
            for key in ('NMSE', 'NMAE'):
                if not np.isclose(current[key], old['groups'][name][key], rtol=0, atol=1e-10):
                    raise ValueError(f'Original score mismatch: {run_id}/{name}/{key}')
        for name, key in (('sq', 'per_channel_NMSE'), ('abs', 'per_channel_NMAE')):
            if not np.allclose(errors[run_id][name].mean(0), old[key], rtol=0, atol=1e-10):
                raise ValueError(f'Channel score mismatch: {run_id}')
    days = timestamps['validation'][starts['validation'] + 32].astype('datetime64[D]').astype(str)
    slices = {}
    for name, mask in split_masks['validation'].items():
        tmask = split_masks['train'][name]
        slices[name] = {'T_windows': int(tmask.sum()), 'T_share': float(tmask.mean()), 'D_windows': int(mask.sum()), 'D_share': float(mask.mean()),
            'metrics': {run_id: metric(error, mask) for run_id, error in errors.items()},
            'daily': {str(day): {'windows': int((mask & (days == day)).sum()), 'metrics': {run_id: metric(error, mask & (days == day)) for run_id, error in errors.items()}} for day in np.unique(days[mask])}}
    channels = {}
    for c, name in enumerate(manifest['sensor_columns']):
        constant = f['validation']['constant'][:, c]
        changed = errors['persistence']['future_change'][:, c]
        groups = {'all': np.ones(len(constant), bool), 'input_constant': constant, 'input_changes': ~constant,
            'constant_then_future_constant_POSTHOC': constant & ~changed, 'constant_then_future_changes_POSTHOC': constant & changed,
            'changes_then_future_constant_POSTHOC': ~constant & ~changed, 'changes_then_future_changes_POSTHOC': ~constant & changed}
        channels[name] = {'T_input_constant_share': float(f['train']['constant'][:, c].mean()),
            'groups': {g: {'windows': int(mask.sum()), 'share': float(mask.mean()), 'metrics': {run_id: metric(error, mask, c) for run_id, error in errors.items()}} for g, mask in groups.items()}}
    stored = {'start_row': starts['validation'], 'source_index_start': source_indices['validation'][starts['validation']],
        'source_index_target_start': source_indices['validation'][starts['validation'] + 32],
        'source_index_end_inclusive': source_indices['validation'][starts['validation'] + 39], 'input_constant': f['validation']['constant'],
        'future_change_POSTHOC': errors['persistence']['future_change'], 'target_start_day': days,
        'input_variation': f['validation']['variation'], 'input_duration_seconds': f['validation']['duration']}
    for name, mask in split_masks['validation'].items():
        stored[f'slice_{name}'] = mask
    for run_id, error in errors.items():
        for name in ('sq', 'abs', 'signed'):
            stored[f'{run_id}_{name}'] = error[name]
    np.savez_compressed(output / 'window_evidence.npz', **stored)
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f'Asset changed during diagnostics: {path}')
    result = {'scope': 'researcher_guided_development_only', 'training_calls': 0, 'api_calls': 0, 'independent_E': False,
        'existing_score_reproduction': 'all_three_models_overall_groups_and_channels_passed_atol_1e-10',
        'thresholds_fitted_on_T': thresholds, 'data_audit': audit, 'slices': slices, 'channels': channels,
        'limitations': ['D repeatedly used for development and checkpoint selection', 'overlapping windows and seeds are not independent tasks', 'future_change_POSTHOC is unavailable at decision time', 'descriptive groups do not establish causal processing value'],
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name()},
        'elapsed_seconds': time.monotonic() - started, 'assets': {'window_evidence.npz': sha(output / 'window_evidence.npz')}}
    core.write_json(output / 'result.json', result)
    print(json.dumps({'status': 'completed', 'D_windows': len(starts['validation']), 'training_calls': 0, 'api_calls': 0, 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
