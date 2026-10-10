"""No-donor control with identical own-history scaling and fixed core model."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from run_direct_processing import BASE, STUDY, PROJECT, CHANGED, load, sha, write, raw_split, group_definitions, evaluate, metrics


def own_transform(x):
    import torch
    selected = torch.zeros(15, dtype=torch.bool, device=x.device)
    selected[CHANGED.tolist()] = True
    mixed = x[:, -1:] + .5 * (x - x[:, -1:])
    return torch.where(selected[None, None, :], mixed, x)


def run(origin, seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    origin_tag = 'full' if origin == 'full' else f'o{int(float(origin) * 100)}'
    run_id = f'DP04_{origin_tag}_own_half_s{seed}'
    folder = STUDY / 'runs' / run_id
    paths = [BASE / 'data/metropt3_v1/manifest.json', BASE / 'research_runs/protocol.json', BASE / 'scripts/run_research.py',
        PROJECT / 'scripts/run_direct_processing.py', Path(__file__).resolve(), STUDY / 'MECHANISM_ABLATION.md',
        BASE / 'data/metropt3_v1/train.csv']
    if origin == 'full':
        paths.append(BASE / 'data/metropt3_v1/validation.csv')
    guards = {str(p): sha(p) for p in paths}
    frozen = load(STUDY / 'freeze_wave1.json')['identity']
    if guards[str(BASE / 'data/metropt3_v1/manifest.json')] != '672c0ecf55bc94a52abc6b256c0bfe7a4da02140e57dd0527375bf859a22c087' or guards[str(BASE / 'scripts/run_research.py')] != '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052' or guards[str(BASE / 'research_runs/protocol.json')] != '573ee4e87ef442b7140afb26f4befe80a8de1a43fec3b5c1fe6b878e527f8afe' or guards[str(PROJECT / 'scripts/run_direct_processing.py')] != frozen['worker_sha256']:
        raise ValueError('Immutable base changed')
    identity = {'run_id': run_id, 'origin': origin, 'seed': seed, 'policy': 'own_half', 'protected_assets': guards,
        'scope': 'mechanism_ablation_development_only', 'independent_E': False}
    if (folder / 'result.json').exists():
        cached = load(folder / 'result.json')
        if cached['identity'] != identity or any(sha(folder / k) != v for k, v in cached['assets'].items()):
            raise ValueError('Completed ablation changed')
        return cached
    if folder.exists():
        raise ValueError('Do not overwrite an incomplete ablation')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
        raise RuntimeError('Designated GPU required')
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('core', BASE / 'scripts/run_research.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    p = load(BASE / 'research_runs/protocol.json')
    manifest = load(BASE / 'data/metropt3_v1/manifest.json')
    folder.mkdir(parents=True)
    write(folder / 'identity.json', identity)
    (folder / 'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (folder / 'protocol_snapshot.md').write_bytes((STUDY / 'MECHANISM_ABLATION.md').read_bytes())
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
        mean, std = raw[:cut].mean(0), raw[:cut].std(0)
        scale = np.where(std == 0, 1, std)
        train_values = ((raw[:cut] - mean) / scale).astype(np.float32)
        eval_values = ((eval_raw - mean) / scale).astype(np.float32)
        groups, constant, discordant, thresholds = group_definitions(train_values, train_starts, eval_values, eval_starts, eval_times)
        write(folder / 'data_audit.json', {'dataset_id': manifest['dataset_id'], 'train_rows': [0, cut],
            'train_windows': len(train_starts), 'evaluation_windows': len(eval_starts), 'mean': mean.tolist(), 'scale': scale.tolist(),
            'T_variation_thresholds': thresholds, 'target_columns': manifest['sensor_columns'], 'input_preprocessing': 'own_half', 'new_sensor_records': 0})
        data, evaluation_data = torch.from_numpy(train_values).cuda(), torch.from_numpy(eval_values).cuda()
        positions, evaluation_positions = torch.from_numpy(train_starts).cuda(), torch.from_numpy(eval_starts).cuda()
        evaluation_ids = torch.from_numpy(eval_ids).cuda()
        probe = data[positions[:128, None] + torch.arange(32, device='cuda')]
        processed = own_transform(probe)
        if not torch.equal(processed[:, -1], probe[:, -1]) or not torch.equal(processed[:, :, [c for c in range(15) if c not in CHANGED]], probe[:, :, [c for c in range(15) if c not in CHANGED]]):
            raise ValueError('Own-history preprocessing implementation failed')
        class FixedPreprocessor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = core.DLinear(p).cuda()
            def forward(self, x):
                return self.core(own_transform(x))
        model = FixedPreprocessor()
        if sum(v.numel() for v in model.parameters()) != 7920:
            raise ValueError('Changed architecture')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0)
        probability = np.full(len(train_starts), 1 / len(train_starts), dtype=np.float64)
        cdf = torch.from_numpy(np.cumsum(probability)).cuda()
        cdf[-1] = 1
        rng = torch.Generator(device='cuda').manual_seed(seed + 1000)
        offsets = torch.arange(40, device='cuda')
        counts = torch.zeros(len(train_starts), dtype=torch.int64, device='cuda')
        best, best_step, history = float('inf'), 3000, []
        for step in range(1, 3001):
            if time.monotonic() - begun > 360:
                raise RuntimeError('Ablation timeout')
            chosen = torch.searchsorted(cdf, torch.rand(256, dtype=torch.float64, device='cuda', generator=rng), right=True)
            counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            batch = data[positions[chosen, None] + offsets]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                if origin == 'full':
                    sq, absolute = evaluate(model, evaluation_data, evaluation_positions, evaluation_ids, 'baseline', p)
                    score = float(sq.mean())
                    history.append({'step': step, 'NMSE': score, 'NMAE': float(absolute.mean())})
                    if score < best:
                        best, best_step = score, step
                        torch.save(model.core.state_dict(), folder / 'best.pt')
                status('training', step)
        torch.save(model.core.state_dict(), folder / 'final.pt')
        if origin == 'full':
            model.core.load_state_dict(torch.load(folder / 'best.pt', weights_only=True))
        else:
            torch.save(model.core.state_dict(), folder / 'best.pt')
        sq, absolute = evaluate(model, evaluation_data, evaluation_positions, evaluation_ids, 'baseline', p)
        report = metrics(sq, absolute, groups, constant, discordant, eval_times, eval_starts)
        draws = counts.cpu().numpy()
        if draws.sum() != 768000:
            raise ValueError('Exposure mismatch')
        np.savez_compressed(folder / 'sampling.npz', start_row=train_starts, source_index_start=ids[train_starts], probability=probability, draw_count=draws)
        np.savez_compressed(folder / 'window_metrics.npz', start_row=eval_starts, source_index_start=eval_ids[eval_starts],
            source_index_target_start=eval_ids[eval_starts + 32], source_index_end_inclusive=eval_ids[eval_starts + 39], squared_error=sq, absolute_error=absolute)
        write(folder / 'history.json', history)
        for path, expected in guards.items():
            if sha(path) != expected:
                raise ValueError('Source changed during ablation')
        result = {'identity': identity, 'metrics': report, 'steps': 3000, 'parameters': 7920, 'best_step': best_step,
            'draw_count_sha256': hashlib.sha256(draws.tobytes()).hexdigest(), 'samples_presented': int(draws.sum()), 'elapsed_seconds': time.monotonic() - begun,
            'assets': {name: sha(folder / name) for name in ('best.pt', 'final.pt', 'sampling.npz', 'window_metrics.npz', 'code_snapshot.py', 'protocol_snapshot.md', 'history.json', 'data_audit.json')}}
        write(folder / 'result.json', result)
        status('completed', 3000)
        print(json.dumps({'completed': run_id, 'metrics': report['groups']['all'], 'transferred': report['transferred_channels']}), flush=True)
        return result
    except BaseException as exc:
        write(folder / 'failure.json', {'error': type(exc).__name__, 'message': str(exc), 'elapsed_seconds': time.monotonic() - begun})
        status('failed', 0)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin', choices=('0.6', '0.8', 'full'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    run(args.origin, args.seed)
