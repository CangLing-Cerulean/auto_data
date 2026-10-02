"""Reproducible, resumable development experiments. Never opens test.csv."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SELECTIVE_CENTER_COLUMNS = {'Oil_temperature': 5, 'Oil_level': 13}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def valid_starts(times, length, gap):
    delta = np.diff(times).astype('timedelta64[s]').astype(np.int64)
    if np.any(delta <= 0):
        raise ValueError('Timestamps must increase strictly')
    bad = np.r_[0, np.cumsum(delta > gap)]
    starts = np.arange(len(times) - length + 1)
    return starts[(bad[starts + length - 1] - bad[starts]) == 0]


class DLinear(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.kernel = p['moving_average_kernel']
        self.seasonal = nn.ModuleList([nn.Linear(p['input_length'], p['prediction_length']) for _ in range(p['channels'])])
        self.trend = nn.ModuleList([nn.Linear(p['input_length'], p['prediction_length']) for _ in range(p['channels'])])

    def forward(self, x):
        x = x.transpose(1, 2)
        trend = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(x, (self.kernel // 2,) * 2, mode='replicate'), self.kernel, stride=1)
        seasonal = x - trend
        return torch.stack([s(seasonal[:, i]) + t(trend[:, i]) for i, (s, t) in enumerate(zip(self.seasonal, self.trend))], dim=-1)


def prepare(p, manifest):
    if any(manifest['sensor_columns'][index] != name for name, index in SELECTIVE_CENTER_COLUMNS.items()):
        raise ValueError('Selective recipe sensor mapping mismatch')
    arrays, times, starts = {}, {}, {}
    audit = {}
    for split in ('train', 'validation'):
        path = ROOT / 'data/metropt3_v1' / f'{split}.csv'
        expected = next(s for s in manifest['splits'] if s['name'] == split)
        if digest(path) != expected['sha256']:
            raise ValueError(f'{split} checksum mismatch')
        arrays[split] = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(2, 17), dtype=np.float64)
        with path.open(newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            times[split] = np.array([row[1] for row in reader], dtype='datetime64[s]')
        if len(arrays[split]) != expected['rows'] or not np.isfinite(arrays[split]).all():
            raise ValueError('Invalid source values or row count')
        starts[split] = valid_starts(times[split], p['input_length'] + p['prediction_length'], p['max_gap_seconds'])
        audit[split] = {'rows': len(arrays[split]), 'valid_windows': len(starts[split]),
                        'gap_excluded_windows': len(arrays[split]) - p['input_length'] - p['prediction_length'] + 1 - len(starts[split])}
    mean = arrays['train'].mean(axis=0)
    std = arrays['train'].std(axis=0)
    scale = np.where(std == 0, 1, std)
    for split in arrays:
        arrays[split] = ((arrays[split] - mean) / scale).astype(np.float32)
    variations = {}
    for split, values in arrays.items():
        diffs = np.abs(np.diff(values.astype(np.float64), axis=0)).mean(axis=1)
        prefix = np.r_[0, np.cumsum(diffs)]
        s = starts[split]
        variations[split] = (prefix[s + p['input_length'] - 1] - prefix[s]) / (p['input_length'] - 1)
    threshold = float(np.quantile(variations['train'], .9))
    groups = {'all': np.ones(len(starts['validation']), dtype=bool),
              'high_input_variation': variations['validation'] > threshold,
              'ordinary_input_variation': variations['validation'] <= threshold}
    months = times['validation'][starts['validation'] + p['input_length']].astype('datetime64[M]').astype(str)
    for month in np.unique(months):
        groups['month_' + month] = months == month
    audit.update({'mean': mean.tolist(), 'scale': scale.tolist(), 'zero_std_channels': np.where(std == 0)[0].tolist(),
                  'input_variation_threshold': threshold})
    return arrays, starts, groups, audit


def predict(model, x, recipe):
    if recipe == 'baseline':
        return model(x)
    if recipe == 'last_center':
        anchor = x[:, -1:, :]
        return model(x - anchor) + anchor
    if recipe == 'mean_center':
        anchor = x.mean(dim=1, keepdim=True)
        return model(x - anchor) + anchor
    if recipe == 'selective_center':
        anchor = torch.zeros_like(x[:, -1:, :])
        selected = list(SELECTIVE_CENTER_COLUMNS.values())
        anchor[:, :, selected] = x[:, -1:, selected]
        return model(x - anchor) + anchor
    raise ValueError(f'Unknown recipe {recipe}')


@torch.no_grad()
def evaluate(model, data, starts, groups, p, recipe, persistence=False):
    model.eval()
    channel_abs = torch.zeros(p['channels'], dtype=torch.float64, device=data.device)
    channel_sq = torch.zeros_like(channel_abs)
    sums = {g: [0., 0., 0] for g in groups}
    length, horizon = p['input_length'], p['prediction_length']
    offsets = torch.arange(length + horizon, device=data.device)
    for lo in range(0, len(starts), p['evaluation_batch_size']):
        indices = starts[lo:lo+p['evaluation_batch_size']]
        batch = data[indices[:, None] + offsets]
        x, y = batch[:, :length], batch[:, length:]
        pred = x[:, -1:, :].expand(-1, horizon, -1) if persistence else predict(model, x, recipe)
        error = (pred - y).double()
        absolute, squared = error.abs(), error.square()
        channel_abs += absolute.sum((0, 1))
        channel_sq += squared.sum((0, 1))
        wa, ws = absolute.mean((1, 2)).cpu().numpy(), squared.mean((1, 2)).cpu().numpy()
        for group, mask in groups.items():
            keep = mask[lo:lo+len(indices)]
            sums[group][0] += float(wa[keep].sum())
            sums[group][1] += float(ws[keep].sum())
            sums[group][2] += int(keep.sum())
    model.train()
    return {'groups': {g: {'NMAE': a/n, 'NMSE': s/n, 'windows': n} for g, (a, s, n) in sums.items() if n},
            'per_channel_NMAE': (channel_abs / (len(starts) * horizon)).cpu().tolist(),
            'per_channel_NMSE': (channel_sq / (len(starts) * horizon)).cpu().tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--recipe', default='baseline', choices=['baseline', 'last_center', 'mean_center', 'selective_center'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id must be a directory name')
    p = json.loads((ROOT / 'research_runs/protocol.json').read_text())
    manifest_path = ROOT / 'data/metropt3_v1/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    seed = p['seed'] if args.seed is None else args.seed
    run = ROOT / 'research_runs/results' / args.run_id
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'result.json').exists():
        raise ValueError('Completed run cannot be overwritten')
    identity = {'protocol_sha256': digest(ROOT / 'research_runs/protocol.json'),
                'manifest_sha256': digest(manifest_path), 'code_sha256': digest(Path(__file__)),
                'recipe': args.recipe, 'seed': seed}
    if (run / 'identity.json').exists():
        if json.loads((run / 'identity.json').read_text()) != identity:
            raise ValueError('Resume identity mismatch')
    else:
        write_json(run / 'identity.json', identity)
        (run / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
        write_json(run / 'protocol.json', p)
    consumed = 0.
    for other in run.parent.iterdir():
        status = other / 'status.json'
        if status.exists():
            consumed += json.loads(status.read_text()).get('elapsed_seconds', 0.)
    old_elapsed = json.loads((run / 'status.json').read_text()).get('elapsed_seconds', 0.) if (run / 'status.json').exists() else 0.
    remaining = p['session_gpu_wall_budget_seconds'] - consumed
    if remaining <= 0:
        raise ValueError('Session compute budget exhausted')
    launched = time.monotonic()
    def status(phase, step):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': old_elapsed + time.monotonic() - launched})
    status('preparing', 0)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; refusing unintended CPU training')
    arrays, indices, groups, audit = prepare(p, manifest)
    write_json(run / 'data_audit.json', audit)
    write_json(run / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
                                         'numpy': np.__version__, 'cuda': torch.version.cuda,
                                         'device': torch.cuda.get_device_name(), 'platform': platform.platform()})
    data = {s: torch.from_numpy(a).cuda() for s, a in arrays.items()}
    starts = {s: torch.from_numpy(a).cuda() for s, a in indices.items()}
    model = DLinear(p).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    rng = torch.Generator(device='cuda').manual_seed(seed + 1000)
    start_step, best_score, best_step = 0, float('inf'), 0
    history = []
    checkpoint = run / 'latest.pt'
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cuda', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.set_state(state['sampling_rng'].cpu())
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        start_step, best_score, best_step, history = state['step'], state['best_score'], state['best_step'], state['history']
    else:
        persistence = evaluate(model, data['validation'], starts['validation'], groups, p, args.recipe, True)
        write_json(run / 'persistence.json', persistence)
    offsets = torch.arange(p['input_length'] + p['prediction_length'], device='cuda')
    losses = []
    for step in range(start_step + 1, p['steps'] + 1):
        if time.monotonic() - launched >= remaining:
            status('budget_exhausted', step - 1)
            raise RuntimeError('Compute budget exhausted; last periodic checkpoint retained')
        sample = torch.randint(len(starts['train']), (p['batch_size'],), generator=rng, device='cuda')
        chosen = starts['train'][sample]
        batch = data['train'][chosen[:, None] + offsets]
        x, y = batch[:, :p['input_length']], batch[:, p['input_length']:]
        optimizer.zero_grad(set_to_none=True)
        loss = (predict(model, x, args.recipe) - y).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % p['validation_every'] == 0 or step == p['steps']:
            metrics = evaluate(model, data['validation'], starts['validation'], groups, p, args.recipe)
            score = metrics['groups']['all']['NMSE']
            entry = {'step': step, 'train_loss': float(np.mean(losses)), 'validation': metrics['groups']['all']}
            losses.clear()
            history.append(entry)
            print(json.dumps(entry), flush=True)
            if score < best_score:
                best_score, best_step = score, step
                torch.save(model.state_dict(), run / 'best.tmp')
                (run / 'best.tmp').replace(run / 'best.pt')
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'sampling_rng': rng.get_state(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                     'step': step, 'best_score': best_score, 'best_step': best_step, 'history': history}
            torch.save(state, run / 'latest.tmp')
            (run / 'latest.tmp').replace(checkpoint)
            write_json(run / 'history.json', history)
            status('training', step)
    model.load_state_dict(torch.load(run / 'best.pt', weights_only=True))
    final = evaluate(model, data['validation'], starts['validation'], groups, p, args.recipe)
    final.update({'run_id': args.run_id, 'identity': identity, 'best_step': best_step,
                  'steps_completed': p['steps'], 'samples_presented': p['steps'] * p['batch_size'],
                  'elapsed_seconds': old_elapsed + time.monotonic() - launched, 'history': history,
                  'sensor_columns': manifest['sensor_columns']})
    write_json(run / 'result.json', final)
    status('completed', p['steps'])
    print(json.dumps({'completed': args.run_id, 'best_step': best_step, **final['groups']['all']}), flush=True)


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
