"""Create-only runtime for the frozen four-group M1 development experiment.

Only explicitly copied inputs under this study root are loaded. Evaluation
stores residual identities and metrics; saved states reconstruct every residual.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import platform
import time
import traceback

import numpy as np
import torch
import response_proxy as rp
import run_research as core

ROOT = Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')
GRID = (500, 1000, 1500, 2000, 2500, 3000)
SEEDS = (2026101701, 2026101702, 2026101703)
TASKS = ('G1', 'G2', 'G3')
ACTIONS = tuple(f'G{g}_{sign}' for g in range(4) for sign in ('UP', 'DOWN'))
METHODS = ('M1', 'RAW', 'SOURCE', 'SHORT', 'PARENT')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, update=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if update:
        temporary = Path(str(path) + '.tmp')
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(text)
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def arrays(path, **values):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **values)


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def configure():
    if not __debug__ or os.name != 'posix':
        raise RuntimeError('Unoptimized dedicated Linux runtime required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__,
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0), platform=platform.system())


def safe_file(root, relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Input path must be a confined relative path')
    path = root / rel
    if path.resolve() != path or not path.is_file():
        raise ValueError(f'Nonregular or symlinked input: {relative}')
    return path


def verify(out):
    if not __debug__ or os.name != 'posix':
        raise RuntimeError('Unoptimized Linux runtime required')
    if out != ROOT or out.resolve() != ROOT:
        raise ValueError('Unexpected study root')
    plan = read(out / 'plan.json')
    receipt = read(out / 'source_sync_receipt.json')
    assert receipt['status'] == 'exact_commit_and_tree_verified'
    assert receipt['plan_sha256'] == sha(out / 'plan.json')
    assert plan['output_root'] == str(ROOT)
    assert plan['seeds'] == list(SEEDS) and plan['grid'] == list(GRID)
    assert plan['actions'] == list(ACTIONS)
    assert [t['id'] for t in plan['tasks']] == list(TASKS)
    assert plan['cumulative_before'] == 418 and not plan['N1_N2_enabled']
    assert plan['training'] == dict(input_length=32, prediction_length=8,
        moving_average_kernel=25, batch_size=256, steps=3000, learning_rate=.001, weight_decay=0.)
    assert sha(safe_file(out, 'storage_estimate.json')) == plan['storage_estimate_sha256']
    storage = read(out / 'storage_preflight.json')
    assert storage['status'] == 'PASS' and storage['plan_sha256'] == sha(out / 'plan.json')
    assert storage['required_free_bytes'] == plan['preflight_min_free_bytes']
    assert storage['free_before_bytes'] >= storage['required_free_bytes']
    assert storage['probe_bytes'] == 64 * 1024 * 1024 and storage['probe_removed']
    assert storage['write_sha256'] == storage['read_sha256']
    runtime = read(out / 'runtime_environment.json')
    assert runtime['environment'] == plan['expected_environment'] == environment()
    activation = read(out / 'execution_activation.json')
    assert activation['status'] == 'inputs_environment_storage_verified'
    for key, name in (('plan_sha256', 'plan.json'), ('input_lock_sha256', 'input_lock.json'),
                      ('storage_preflight_sha256', 'storage_preflight.json'),
                      ('environment_sha256', 'runtime_environment.json')):
        assert activation[key] == sha(safe_file(out, name)), key
    assert activation['source_commit'] == receipt['source_commit']
    for name, digest in plan['code_sha256'].items():
        assert sha(safe_file(out, name)) == digest, name
    lock = read(out / 'input_lock.json')
    assert sha(out / 'input_lock.json') == plan['input_lock_sha256']
    for item in lock['files']:
        path = safe_file(out, item['path'])
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
    return plan


def snapshot(model, optimizer, rng, step):
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
        rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state(),
        rng_cuda=torch.cuda.get_rng_state(), original_coordinates=rp.parameters(model).detach().cpu())


def save_snapshot(path, model, optimizer, rng, step):
    with Path(path).open('xb') as stream:
        torch.save(snapshot(model, optimizer, rng, step), stream)


def load_snapshot(directory, step):
    path = Path(directory) / ('initial.pt' if step == 0 else f'step_{step}.pt')
    return torch.load(path, map_location='cpu', weights_only=True)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)


def metric(error):
    if error.dtype != np.float32 or not np.isfinite(error).all():
        raise ValueError('Invalid residual')
    value = error.astype(np.float64)
    return dict(nmse=float(np.mean(value ** 2)), nmae=float(np.mean(np.abs(value))),
        channel_nmse=np.mean(value ** 2, axis=(0, 1)).tolist(),
        channel_nmae=np.mean(np.abs(value), axis=(0, 1)).tolist(),
        shape=list(error.shape), dtype=str(error.dtype),
        byte_sha256=hashlib.sha256(error.tobytes()).hexdigest())


class Context:
    def __init__(self, out, plan, task):
        self.out, self.plan, self.task = out, plan, task
        with np.load(out / 'inputs' / 'prepared' / f'{task}_ABC.npz', allow_pickle=False) as source:
            self.values = {s: torch.tensor(source[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
            self.starts = {s: torch.tensor(source[f'{s}_starts'], device='cuda') for s in self.values}
        self.n = len(self.starts['A'])
        assert self.n == 6468 and len(self.starts['B']) == len(self.starts['C']) == 1617
        self.offset = torch.arange(40, device='cuda')
        self.probability = np.full(self.n, 1. / self.n)
        self.cdf = np.cumsum(self.probability)
        self.cdf[-1] = 1.
        self.weights = {'PARENT': torch.ones(self.n, dtype=torch.float32, device='cuda')}
        if (out / 'prepared').is_dir():
            receipt = read(out / 'preparation.json')
            for action in ACTIONS:
                path = out / 'prepared' / f'{task}_{action}.npz'
                assert sha(path) == receipt['assets_sha256'][path.relative_to(out).as_posix()]
                with np.load(path, allow_pickle=False) as source:
                    value = source['weights']
                    assert value.dtype == np.float32 and value.shape == (self.n,)
                    self.weights[action] = torch.tensor(value, device='cuda')

    def parent(self, engine, seed):
        assert engine in ('full', 'proxy') and seed in SEEDS
        return self.out / 'inputs' / 'runs' / f'DEV_{self.task}_{engine}_PARENT_s{seed}'

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return core.DLinear(dict(self.plan['training'], channels=8)).cuda()

    def batch(self, split, ids):
        assert split in ('A', 'B', 'C')
        return self.values[split][self.starts[split][ids, None] + self.offset]

    @torch.no_grad()
    def evaluate(self, model, split):
        assert split in ('B', 'C')
        model.eval()
        chunks = []
        for lo in range(0, len(self.starts[split]), 2048):
            ids = torch.arange(lo, min(lo + 2048, len(self.starts[split])), device='cuda')
            batch = self.batch(split, ids)
            chunks.append((model(batch[:, :32]) - batch[:, 32:]).cpu().numpy())
        return metric(np.concatenate(chunks))

    @torch.no_grad()
    def moments(self, model, action):
        result = None
        for lo in range(0, self.n, 2048):
            ids = torch.arange(lo, min(lo + 2048, self.n), device='cuda')
            batch = self.batch('A', ids)
            # Multiply the exact stored float32 weights in float64. No renormalization.
            mass = self.weights[action][ids].double() * (1. / self.n)
            item = rp.batch_moments(rp.features(model, batch[:, :32]), batch[:, 32:], mass)
            if result is None:
                result = item
            else:
                result.add_(item)
        return result


def train(ctx, engine, action, seed, *, steps, replay=False):
    assert engine in ('full', 'proxy', 'short') and action in ('PARENT',) + ACTIONS
    assert steps in GRID and (engine == 'short') == (steps < 3000)
    assert not replay or (action == 'PARENT' and engine in ('full', 'proxy'))
    assert shutil.disk_usage(ctx.out).free >= ctx.plan['runtime_min_free_bytes']
    ident = f'FGM1_{ctx.task}_{engine}_{action}_s{seed}'
    directory = ctx.out / ('calibration_runs' if replay else 'runs') / ident
    directory.mkdir(parents=True)
    started = tick()
    status = dict(status='running', id=ident, task=ctx.task, engine=engine,
        action=action, seed=seed, steps_completed=0, replay=replay)
    write(directory / 'status.json', status)
    try:
        model = ctx.model(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        rng = np.random.default_rng(seed + 1000)
        parent = ctx.parent('proxy' if engine == 'proxy' else 'full', seed)
        initial = load_snapshot(parent, 0)
        equal(snapshot(model, optimizer, rng, 0), initial)
        save_snapshot(directory / 'initial.pt', model, optimizer, rng, 0)
        moment = ctx.moments(model, action) if engine == 'proxy' else None
        if moment is not None:
            arrays(directory / 'moments.npz', gram=moment.gram.cpu().numpy(),
                cross=moment.cross.cpu().numpy(), target_square=moment.target_square.cpu().numpy())
        draws = np.zeros(ctx.n, dtype=np.int64)
        curves, prefixes, checkpoints, replay_matches = {}, {}, [], []
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if engine == 'proxy':
                rp.assign_gradient(model, moment)
            else:
                ids = np.searchsorted(ctx.cdf, rng.random(256), side='right')
                np.add.at(draws, ids, 1)
                index = torch.tensor(ids, device='cuda')
                batch = ctx.batch('A', index)
                squared = (model(batch[:, :32]) - batch[:, 32:]).square()
                loss = squared.mean() if action == 'PARENT' else (squared * ctx.weights[action][index, None, None]).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
            optimizer.step()
            status['steps_completed'] = step
            if step in (1, 501):
                save_snapshot(directory / f'witness_{step}.pt', model, optimizer, rng, step)
            if step in GRID:
                save_snapshot(directory / f'step_{step}.pt', model, optimizer, rng, step)
                checkpoints.append(step)
                if engine != 'proxy' or replay:
                    curves[step] = {s: ctx.evaluate(model, s) for s in ('B', 'C')}
                    write(directory / f'step_{step}_metrics.json', curves[step])
                write(directory / 'status.json', status, update=True)
                # Record actual prefix work, including all previous checkpoint writes and hashes.
                hashes = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.pt', '.npz')}
                prefixes[step] = tick() - started
                write(directory / f'prefix_{step}.json', dict(seconds=prefixes[step], assets_sha256=hashes))
                if replay:
                    # Verification overhead is separate from the timed computational prefix.
                    verify_start = tick()
                    equal(load_snapshot(directory, step), load_snapshot(parent, step))
                    replay_matches.append(dict(step=step, bitexact=True, seconds=tick() - verify_start))
        selected = rp.select_checkpoint({k: curves[k]['B']['nmse'] for k in checkpoints}, checkpoints) if curves else None
        if engine != 'proxy':
            arrays(directory / 'sampling.npz', draw_count=draws)
        assets = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.name not in ('status.json', 'result.json', 'timing.json')}
        result = dict(status, status='completed', steps=steps, checkpoints=checkpoints,
            selected_step=selected, curves=curves, selected_metrics=curves[selected] if selected is not None else None,
            assets_sha256=assets, prefix_seconds=prefixes, replay_matches=replay_matches,
            plan_sha256=sha(ctx.out / 'plan.json'))
        write(directory / 'result.json', result)
        status['status'] = 'completed'
        write(directory / 'status.json', status, update=True)
        elapsed = tick() - started
        verification = sum(item['seconds'] for item in replay_matches)
        # Prefix calibration excludes all prior replay comparisons. Writes and hashing remain charged.
        adjusted_prefix = {k: prefixes[k] - sum(x['seconds'] for x in replay_matches if x['step'] < k) for k in prefixes}
        timing = dict(seconds=elapsed - verification, actual_elapsed_seconds=elapsed,
            replay_verification_seconds=verification, prefix_seconds=adjusted_prefix,
            boundary='initialization through all B/C metrics, checkpoints, result, status and asset hashes; clock receipt excluded')
        write(directory / 'timing.json', timing)
        return dict(result, timing=timing, directory=directory.relative_to(ctx.out).as_posix())
    except Exception:
        status.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write(directory / 'status.json', status, update=True)
        raise


def score_proxy(ctx, action, seed, method, *, identity=False):
    assert method in ('M1', 'RAW') and (action == 'PARENT') == identity
    directory = ctx.out / ('identity_scores' if identity else 'forecasts') / f'{ctx.task}_{action}_{seed}_{method}'
    directory.mkdir(parents=True)
    started = tick()
    model = ctx.model(seed)
    parent = ctx.parent('full', seed)
    baseline = read(parent / 'result.json')['selected_losses']['C']
    proxy_parent = ctx.parent('proxy', seed)
    child = proxy_parent if identity else ctx.out / 'runs' / f'FGM1_{ctx.task}_proxy_{action}_s{seed}'
    curves = {}
    for step in GRID:
        proxy = load_snapshot(child, step)
        if method == 'RAW':
            model.load_state_dict(proxy['model'])
        else:
            real = load_snapshot(parent, step)['original_coordinates'].cuda()
            other = load_snapshot(proxy_parent, step)['original_coordinates'].cuda()
            rp.load_parameters(model, rp.paired_parameters(real, proxy['original_coordinates'].cuda(), other))
        curves[step] = {s: ctx.evaluate(model, s) for s in ('B', 'C')}
    selected = rp.select_checkpoint({k: curves[k]['B']['nmse'] for k in GRID}, list(GRID))
    result = dict(method=method, task=ctx.task, action=action, seed=seed, curves=curves,
        selected_step=selected, gain_percent=100. * (1. - curves[selected]['C']['nmse'] / baseline),
        baseline_C=baseline, selected_metrics=curves[selected])
    write(directory / 'result.json', result)
    digest = sha(directory / 'result.json')
    elapsed = tick() - started
    write(directory / 'timing.json', dict(seconds=elapsed, result_sha256=digest))
    return dict(result, seconds=elapsed, directory=directory.relative_to(ctx.out).as_posix())
