"""Registered, researcher-guided fixed-DLinear processing trials on the 4090."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

BASE = Path('/root/data/auto_data_research/metropt3_v1/session_20260926')
PROJECT = Path(__file__).resolve().parents[1]
STUDY = PROJECT / 'research_runs/direct_processing/20261009'
MAPPING = np.array([1, 8, 1, 3, 8, 6, 6, 7, 1, 9, 10, 11, 12, 13, 14])
CHANGED = np.flatnonzero(MAPPING != np.arange(15))
POLICIES = ('baseline', 'source_prior', 'source_half', 'source_half_scramble')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def raw_split(split, manifest):
    path = BASE / 'data/metropt3_v1' / f'{split}.csv'
    entry = next(s for s in manifest['splits'] if s['name'] == split)
    if sha(path) != entry['sha256']:
        raise ValueError('Dataset hash changed')
    values = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(2, 17))
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [(int(row[0]), row[1]) for row in reader]
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    times = np.array([r[1] for r in rows], dtype='datetime64[s]')
    if header != manifest['columns_raw'] or len(values) != entry['rows'] or not np.isfinite(values).all():
        raise ValueError('Dataset schema changed')
    if str(times[0]).replace('T', ' ') != entry['first_timestamp'] or str(times[-1]).replace('T', ' ') != entry['last_timestamp']:
        raise ValueError('Dataset time boundary changed')
    return values, ids, times


def shifts(ids):
    return (ids * 2654435761 + 20261009) % 31 + 1


def transform(x, start_ids, policy):
    import torch
    if policy == 'baseline':
        return x
    sources = x[:, :, MAPPING.tolist()]
    if policy == 'source_half_scramble':
        rotation = (torch.arange(32, device=x.device)[None, :] + shifts(start_ids)[:, None]) % 32
        sources = sources.gather(1, rotation[:, :, None].expand(-1, -1, 15))
    alpha = 1. if policy == 'source_prior' else .5
    anchor = x[:, -1:, :]
    mixed = anchor + (1 - alpha) * (x - anchor) + alpha * (sources - sources[:, -1:, :])
    selected = torch.tensor(MAPPING != np.arange(15), device=x.device)
    return torch.where(selected[None, None, :], mixed, x)


def group_definitions(train_values, train_starts, values, starts, times):
    def variation(a, s):
        prefix = np.r_[0., np.cumsum(np.abs(np.diff(a.astype(np.float64), axis=0)).mean(1))]
        return (prefix[s + 31] - prefix[s]) / 31
    thresholds = np.quantile(variation(train_values, train_starts), [.25, .9])
    observed_variation = variation(values, starts)
    change = np.diff(values, axis=0) != 0
    counts = np.concatenate([np.zeros((1, 15), np.int64), np.cumsum(change, axis=0)], axis=0)
    constant = counts[starts + 31] == counts[starts]
    digital = np.r_[0, np.cumsum(change[:, 7:].any(1))]
    recent = digital[starts + 31] > digital[starts + 27]
    months = times[starts + 32].astype('datetime64[M]').astype(str)
    groups = {'all': np.ones(len(starts), bool), 'variation_low': observed_variation <= thresholds[0],
        'variation_high': observed_variation > thresholds[1], 'digital_recent_switch': recent,
        'digital_not_recent_switch': ~recent}
    for month in np.unique(months):
        groups[f'month_{month}'] = months == month
    discordant = constant & ~constant[:, MAPPING]
    discordant[:, MAPPING == np.arange(15)] = False
    return groups, constant, discordant, thresholds.tolist()


def evaluate(model, data, starts, ids, policy, p):
    import torch
    offsets = torch.arange(40, device='cuda')
    squared, absolute = [], []
    model.eval()
    with torch.inference_mode():
        for lo in range(0, len(starts), p['evaluation_batch_size']):
            selected = starts[lo:lo + p['evaluation_batch_size']]
            batch = data[selected[:, None] + offsets]
            pred = model(transform(batch[:, :32], ids[selected], policy))
            error = (pred - batch[:, 32:]).double()
            squared.append(error.square().mean(1).cpu().numpy())
            absolute.append(error.abs().mean(1).cpu().numpy())
    model.train()
    return np.concatenate(squared), np.concatenate(absolute)


def metrics(sq, absolute, groups, constant, discordant, times, starts):
    def summarize(mask, columns=None):
        if not mask.any():
            return None
        selected_sq, selected_abs = sq[mask], absolute[mask]
        if columns is not None:
            selected_sq, selected_abs = selected_sq[:, columns], selected_abs[:, columns]
        return {'NMSE': float(selected_sq.mean()), 'NMAE': float(selected_abs.mean()), 'windows': int(mask.sum())}
    result = {'groups': {name: summarize(mask) for name, mask in groups.items()},
        'transferred_channels': summarize(groups['all'], CHANGED),
        'per_channel_NMSE': sq.mean(0).tolist(), 'per_channel_NMAE': absolute.mean(0).tolist(),
        'input_constant_per_channel': [{'windows': int(constant[:, c].sum()), 'NMSE': float(sq[constant[:, c], c].mean()) if constant[:, c].any() else None,
            'NMAE': float(absolute[constant[:, c], c].mean()) if constant[:, c].any() else None} for c in range(15)],
        'discordant_target_cells': {'cells': int(discordant.sum()), 'NMSE': float(sq[discordant].mean()) if discordant.any() else None,
            'NMAE': float(absolute[discordant].mean()) if discordant.any() else None}}
    days = times[starts + 32].astype('datetime64[D]').astype(str)
    result['daily'] = {str(day): {'windows': int((days == day).sum()),
        'squared_sum': float(sq[days == day].mean(1).sum()), 'absolute_sum': float(absolute[days == day].mean(1).sum()),
        'transferred_squared_sum': float(sq[days == day][:, CHANGED].mean(1).sum())} for day in np.unique(days)}
    return result


def run(run_id, origin, seed, policy):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    if Path(run_id).name != run_id or policy not in POLICIES:
        raise ValueError('Invalid registered trial')
    folder = STUDY / 'runs' / run_id
    protected = {BASE / 'data/metropt3_v1/manifest.json': '672c0ecf55bc94a52abc6b256c0bfe7a4da02140e57dd0527375bf859a22c087'}
    protected[BASE / 'research_runs/protocol.json'] = '573ee4e87ef442b7140afb26f4befe80a8de1a43fec3b5c1fe6b878e527f8afe'
    protected[BASE / 'scripts/run_research.py'] = '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'
    protected[Path(__file__).resolve()] = sha(__file__)
    protected[STUDY / 'EVIDENCE_PROTOCOL.md'] = sha(STUDY / 'EVIDENCE_PROTOCOL.md')
    manifest = load(BASE / 'data/metropt3_v1/manifest.json')
    for entry in manifest['splits']:
        if entry['name'] == 'train' or (origin == 'full' and entry['name'] == 'validation'):
            protected[BASE / 'data/metropt3_v1' / entry['file']] = entry['sha256']
    identity = {'run_id': run_id, 'origin': origin, 'seed': seed, 'policy': policy,
        'mapping': MAPPING.tolist(), 'protected_assets': {str(k): v for k, v in protected.items()},
        'steps': 3000, 'scope': 'researcher_guided_development', 'independent_E': False}
    if (folder / 'result.json').exists():
        cached = load(folder / 'result.json')
        if cached['identity'] != identity or any(sha(folder / name) != expected for name, expected in cached['assets'].items()):
            raise ValueError('Completed trial identity changed')
        return cached
    if folder.exists():
        raise ValueError('Unfinished trial must retain failure record; use a new ID after repair')
    for path, expected in protected.items():
        if sha(path) != expected:
            raise ValueError(f'Protected source changed: {path}')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
        raise RuntimeError('Designated 4090 required')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('original_core', BASE / 'scripts/run_research.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    p = load(BASE / 'research_runs/protocol.json')
    folder.mkdir(parents=True)
    write(folder / 'identity.json', identity)
    (folder / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder / 'protocol_snapshot.md').write_bytes((STUDY / 'EVIDENCE_PROTOCOL.md').read_bytes())
    begun = time.monotonic()
    def status(phase, step):
        write(folder / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})
    status('preparing', 0)
    try:
        raw, ids, times = raw_split('train', manifest)
        all_starts = core.valid_starts(times, 40, 30)
        if origin == 'full':
            cut = len(raw)
            eval_raw, eval_ids, eval_times = raw_split('validation', manifest)
            train_starts, eval_starts = all_starts, core.valid_starts(eval_times, 40, 30)
        else:
            fraction = float(origin)
            cut, end = int(len(raw) * fraction), min(len(raw), int(round(len(raw) * (fraction + .2))))
            train_starts = all_starts[all_starts + 40 <= cut]
            eval_starts = all_starts[(all_starts >= cut) & (all_starts + 40 <= end)]
            eval_raw, eval_ids, eval_times = raw[:end], ids[:end], times[:end]
            if train_starts.max() + 40 > eval_starts.min():
                raise ValueError('Overlapping training and evaluation support')
        mean, std = raw[:cut].mean(0), raw[:cut].std(0)
        scale = np.where(std == 0, 1, std)
        train_values = ((raw[:cut] - mean) / scale).astype(np.float32)
        eval_values = ((eval_raw - mean) / scale).astype(np.float32)
        groups, constant, discordant, thresholds = group_definitions(train_values, train_starts, eval_values, eval_starts, eval_times)
        write(folder / 'data_audit.json', {'dataset_id': manifest['dataset_id'], 'train_rows': [0, cut], 'train_windows': len(train_starts),
            'evaluation_windows': len(eval_starts), 'mean': mean.tolist(), 'scale': scale.tolist(), 'T_variation_thresholds': thresholds,
            'target_columns': manifest['sensor_columns'], 'input_preprocessing': policy, 'new_sensor_records': 0})
        data = torch.from_numpy(train_values).cuda()
        evaluation_data = torch.from_numpy(eval_values).cuda()
        positions, evaluation_positions = torch.from_numpy(train_starts).cuda(), torch.from_numpy(eval_starts).cuda()
        train_ids, evaluation_ids = torch.from_numpy(ids[:cut]).cuda(), torch.from_numpy(eval_ids).cuda()
        probability = np.full(len(train_starts), 1 / len(train_starts), dtype=np.float64)
        cdf = torch.from_numpy(np.cumsum(probability)).cuda()
        cdf[-1] = 1
        rng = torch.Generator(device='cuda').manual_seed(seed + 1000)
        offsets = torch.arange(40, device='cuda')
        counts = torch.zeros(len(train_starts), dtype=torch.int64, device='cuda')
        model = core.DLinear(p).cuda()
        if sum(t.numel() for t in model.parameters()) != 7920:
            raise ValueError('Changed model architecture')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
        history, best, best_step = [], float('inf'), 0
        for step in range(1, 3001):
            if time.monotonic() - begun > 360:
                raise RuntimeError('Per-trial execution timeout')
            selected = torch.searchsorted(cdf, torch.rand(256, dtype=torch.float64, device='cuda', generator=rng), right=True)
            counts.scatter_add_(0, selected, torch.ones_like(selected))
            batch = data[positions[selected, None] + offsets]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(transform(batch[:, :32], train_ids[positions[selected]], policy))
            loss = (prediction - batch[:, 32:]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                if origin == 'full':
                    sq, absolute = evaluate(model, evaluation_data, evaluation_positions, evaluation_ids, policy, p)
                    score = float(sq.mean())
                    history.append({'step': step, 'NMSE': score, 'NMAE': float(absolute.mean())})
                    if score < best:
                        best, best_step = score, step
                        torch.save(model.state_dict(), folder / 'best.pt')
                status('training', step)
                write(folder / 'history.json', history)
        torch.save(model.state_dict(), folder / 'final.pt')
        if origin == 'full':
            model.load_state_dict(torch.load(folder / 'best.pt', weights_only=True))
        else:
            best_step = 3000
            torch.save(model.state_dict(), folder / 'best.pt')
        sq, absolute = evaluate(model, evaluation_data, evaluation_positions, evaluation_ids, policy, p)
        report = metrics(sq, absolute, groups, constant, discordant, eval_times, eval_starts)
        draws = counts.cpu().numpy()
        if draws.sum() != 768000:
            raise ValueError('Training exposure mismatch')
        np.savez_compressed(folder / 'sampling.npz', start_row=train_starts, source_index_start=ids[train_starts],
            probability=probability, draw_count=draws)
        np.savez_compressed(folder / 'window_metrics.npz', start_row=eval_starts, source_index_start=eval_ids[eval_starts],
            source_index_target_start=eval_ids[eval_starts + 32], source_index_end_inclusive=eval_ids[eval_starts + 39],
            squared_error=sq, absolute_error=absolute, input_constant=constant, discordant=discordant)
        for path, expected in protected.items():
            if sha(path) != expected:
                raise ValueError(f'Protected source changed during trial: {path}')
        result = {'identity': identity, 'metrics': report, 'steps': 3000, 'parameters': 7920,
            'best_step': best_step, 'samples_presented': int(draws.sum()), 'draw_count_sha256': hashlib.sha256(draws.tobytes()).hexdigest(),
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name()},
            'elapsed_seconds': time.monotonic() - begun,
            'assets': {name: sha(folder / name) for name in ('sampling.npz', 'window_metrics.npz', 'best.pt', 'final.pt', 'code_snapshot.py', 'protocol_snapshot.md', 'data_audit.json', 'history.json')}}
        write(folder / 'result.json', result)
        status('completed', 3000)
        print(json.dumps({'completed': run_id, 'policy': policy, 'origin': origin, 'seed': seed, 'metrics': report['groups']['all'], 'transferred': report['transferred_channels'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        return result
    except BaseException as exc:
        write(folder / 'failure.json', {'error': type(exc).__name__, 'message': str(exc), 'elapsed_seconds': time.monotonic() - begun})
        status('failed', 0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', choices=('0.6', '0.8', 'full'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--policy', choices=POLICIES, required=True)
    args = parser.parse_args()
    run(args.run_id, args.origin, args.seed, args.policy)
