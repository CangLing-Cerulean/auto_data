"""Create-only independent preflight, calibration, forecast and label audit.

No new scientific trajectories are generated; saved trajectories and labels
are independently reconstructed only in their authorized audit phase.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

import fourgroup_runtime as rt


def coordinates(state):
    return torch.stack([torch.cat((state[f'seasonal.{c}.weight'].T,
        state[f'seasonal.{c}.bias'][None], state[f'trend.{c}.weight'].T,
        state[f'trend.{c}.bias'][None])) for c in range(8)])


def features(x):
    z = x.transpose(1, 2)
    trend = F.avg_pool1d(F.pad(z, (12, 12), mode='replicate'), 25, stride=1)
    one = torch.ones_like(z[..., :1])
    return torch.cat((z - trend, one, trend, one), dim=2)


def metric(error):
    assert error.dtype == np.float32 and np.isfinite(error).all()
    value = error.astype(np.float64)
    return dict(nmse=float(np.mean(value ** 2)), nmae=float(np.mean(np.abs(value))),
        channel_nmse=np.mean(value ** 2, axis=(0, 1)).tolist(),
        channel_nmae=np.mean(np.abs(value), axis=(0, 1)).tolist(),
        shape=list(error.shape), dtype=str(error.dtype),
        byte_sha256=hashlib.sha256(error.tobytes()).hexdigest())


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def load_abc(out, tid):
    return arrays(out / 'inputs' / 'prepared' / f'{tid}_ABC.npz')


def parent(out, tid, engine, seed):
    return out / 'inputs' / 'runs' / f'DEV_{tid}_{engine}_PARENT_s{seed}'


def new_model(plan, seed, device='cuda', dtype=torch.float32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return rt.core.DLinear(dict(plan['training'], channels=8)).to(device=device, dtype=dtype)


def state_audit(directory, plan, seed, engine):
    """Check the frozen Adam layout and independently advance only its draw RNG."""
    initial = rt.load_snapshot(directory, 0)
    model = new_model(plan, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rt.equal(initial['model'], {k: v.cpu() for k, v in model.state_dict().items()})
    rt.equal(initial['optimizer'], opt.state_dict())
    rt.equal(initial['rng_torch'], torch.get_rng_state())
    rt.equal(initial['rng_cuda'], torch.cuda.get_rng_state())
    rng = np.random.default_rng(seed + 1000)
    groups = initial['optimizer']['param_groups']
    assert len(groups) == 1
    group = groups[0]
    assert group['lr'] == .001 and group['betas'] == (.9, .999)
    assert group['eps'] == 1e-8 and group['weight_decay'] == 0.
    parameters = list(model.parameters())
    assert len(parameters) == 32 and len(group['params']) == 32
    previous = 0
    for step in (0,) + rt.GRID:
        state = rt.load_snapshot(directory, step)
        assert state['step'] == step
        packed = coordinates(state['model'])
        rt.equal(packed, state['original_coordinates'])
        assert packed.shape == (8, 66, 8) and packed.dtype == torch.float32
        assert bool(torch.isfinite(packed).all())
        rt.equal(state['optimizer']['param_groups'], groups)
        optimizer = state['optimizer']['state']
        if step == 0:
            assert optimizer == {}
        else:
            assert set(optimizer) == set(group['params'])
            for key, parameter in zip(group['params'], parameters):
                entry = optimizer[key]
                assert set(entry) == {'step', 'exp_avg', 'exp_avg_sq'}
                assert entry['step'].item() == step
                for name in ('exp_avg', 'exp_avg_sq'):
                    value = entry[name]
                    assert value.shape == parameter.shape and value.dtype == torch.float32
                    assert bool(torch.isfinite(value).all())
                assert bool((entry['exp_avg_sq'] >= 0).all())
        if engine == 'full':
            for _ in range(previous, step):
                rng.random(256)
        rt.equal(state['rng_numpy'], rng.bit_generator.state)
        rt.equal(state['rng_torch'], initial['rng_torch'])
        rt.equal(state['rng_cuda'], initial['rng_cuda'])
        previous = step
    return 7


def rebuild(out, plan):
    manifest = rt.read(out / 'inputs/data/manifest.json')
    assert manifest['dataset_id'] == 'uci_hydraulic_v1'
    train = next(s for s in manifest['splits'] if s['name'] == 'train')
    path = out / 'inputs/data/train.csv'
    assert train['file'] == path.name and train['sha256'] == rt.sha(path)
    assert manifest['sensor_columns'] == ['CE', 'CP', 'SE', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1']
    destinations = {}
    bounds = {'G1': {'A': [0, 308], 'B': [308, 385], 'C': [385, 462]},
        'G2': {'A': [514, 822], 'B': [822, 899], 'C': [899, 976]},
        'G3': {'A': [1028, 1336], 'B': [1336, 1413], 'C': [1413, 1490]}}
    for task in plan['tasks']:
        assert task['partitions'] == bounds[task['id']]
        for split, (lo, hi) in task['partitions'].items():
            assert split in ('A', 'B', 'C')
            destinations[(task['id'], split)] = dict(lo=lo * 60, hi=hi * 60, rows=[])
    # Choose by row number before parsing any tokens, including the old D gaps.
    with path.open(encoding='utf-8-sig', newline='') as stream:
        assert next(csv.reader([next(stream)])) == manifest['columns_raw']
        count = 0
        for index, line in enumerate(stream):
            count += 1
            matches = [v for v in destinations.values() if v['lo'] <= index < v['hi']]
            assert len(matches) <= 1
            if not matches:
                continue
            row = next(csv.reader([line]))
            assert len(row) == 10 and int(row[0]) == index // 60 and int(row[1]) == index % 60
            values = [float(token) for token in row[2:]]
            assert all(math.isfinite(v) for v in values)
            matches[0]['rows'].append(values)
    assert count == train['rows']
    checked = 0
    for task in plan['tasks']:
        tid = task['id']
        abc = load_abc(out, tid)
        raw = {s: np.array(destinations[(tid, s)]['rows'], dtype=np.float64) for s in ('A', 'B', 'C')}
        mean = raw['A'].mean(0)
        scale = raw['A'].std(0)
        scale = np.where(scale == 0., 1., scale)
        np.testing.assert_array_equal(mean, abc['mean'])
        np.testing.assert_array_equal(scale, abc['scale'])
        for split, (lo, hi) in task['partitions'].items():
            expected = dict(values=((raw[split] - mean) / scale).astype(np.float32),
                cycles=np.repeat(np.arange(lo, hi), 60), offsets=np.tile(np.arange(60), hi-lo),
                base_rows=np.arange(lo * 60, hi * 60),
                starts=(np.arange(hi-lo)[:, None] * 60 + np.arange(21)).reshape(-1))
            assert len(expected['starts']) == (6468 if split == 'A' else 1617)
            for name, value in expected.items():
                np.testing.assert_array_equal(value, abc[f'{split}_{name}'])
                assert value.dtype == abc[f'{split}_{name}'].dtype
                checked += 1
    return checked


def weight_audit(out, plan):
    receipt = rt.read(out / 'preparation.json')
    assert receipt['status'] == 'PASS' and not receipt['D_access'] and not receipt['test_access']
    assert receipt['new_full_trainings'] == 0
    assert receipt['plan_sha256'] == rt.sha(out / 'plan.json')
    assert set(receipt['assets_sha256']) == {f'prepared/{t}_{a}.npz' for t in rt.TASKS for a in rt.ACTIONS}
    for name, digest in receipt['assets_sha256'].items():
        assert rt.sha(rt.safe_file(out, name)) == digest
    policies = []
    for task in plan['tasks']:
        tid = task['id']
        abc = load_abc(out, tid)
        cycles = abc['A_cycles'][abc['A_starts']]
        groups = (cycles - task['partitions']['A'][0]) // 77
        np.testing.assert_array_equal(np.bincount(groups), np.full(4, 1617))
        for action in rt.ACTIONS:
            selected, sign = int(action[1]), 1 if action.endswith('UP') else -1
            group_weights = (1. + sign * .2 * np.where(np.arange(4) == selected, 1., -1./3)).astype(np.float32)
            path = out / 'prepared' / f'{tid}_{action}.npz'
            assert path.relative_to(out).as_posix() in receipt['assets_sha256']
            saved = arrays(path)
            expected = dict(weights=group_weights[groups], group_weights=group_weights,
                groups=groups, window_cycles=cycles,
                window_base_rows=abc['A_base_rows'][abc['A_starts']])
            for key, value in expected.items():
                np.testing.assert_array_equal(value, saved[key])
            assert saved['weights'].dtype == np.float32
            delta = group_weights.astype(np.float64) - 1
            assert abs(delta.mean()) <= 1e-7 and np.abs(delta).max() <= .2 + 1e-7
            for first in (True, False):
                chosen = (groups < 2) == first
                for high, low in ((2., 0.), (1.5, .5)):
                    assert not np.array_equal(expected['weights'], np.where(chosen, high, low))
            policies.append(dict(task=tid, action=action, uniform_probability=1./6468,
                sampling='uniform_with_replacement', loss_denominator='fixed_elements',
                mass_error=float(delta.mean()), radius=float(np.abs(delta).max())))
    return policies


def synthetic(plan):
    torch.manual_seed(719)
    model = rt.core.DLinear(dict(plan['training'], channels=8)).double()
    x, y = torch.randn(12, 32, 8, dtype=torch.float64), torch.randn(12, 8, 8, dtype=torch.float64)
    phi = features(x).permute(1, 0, 2)
    target = y.permute(2, 0, 1)
    maximum = 0.
    for action in ('PARENT',) + rt.ACTIONS:
        w = np.ones(12)
        if action != 'PARENT':
            w += (1 if action.endswith('UP') else -1) * .2 * np.where(np.arange(12)//3 == int(action[1]), 1., -1./3)
        mass = torch.tensor(w / 12, dtype=torch.float64)
        gram = phi.transpose(1, 2) @ (phi * mass[None, :, None])
        cross = (phi * mass[None, :, None]).transpose(1, 2) @ target
        value = coordinates(model.state_dict())
        expected = 2./64 * (gram @ value - cross)
        model.zero_grad(set_to_none=True)
        ((model(x) - y).square() * torch.tensor(w)[:, None, None]).mean().backward()
        actual = coordinates({name: p.grad for name, p in model.named_parameters()})
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
        implementation = rt.rp.batch_moments(features(x), y, mass).gradient(value)
        torch.testing.assert_close(implementation, actual, rtol=1e-9, atol=1e-11)
        maximum = max(maximum, float((actual - expected).abs().max()))
    return dict(actions=9, rtol=1e-9, atol=1e-11, maximum_absolute_error=maximum,
        synthetic_algebra_only=True, generated_sensor_records=False)


def preflight(out, plan):
    rebuilt = rebuild(out, plan)
    policies = weight_audit(out, plan)
    algebra = synthetic(plan)
    states = 0
    for task in plan['tasks']:
        for seed in rt.SEEDS:
            for engine in ('full', 'proxy'):
                states += state_audit(parent(out, task['id'], engine, seed), plan, seed, engine)
    return dict(status='PASS', reconstructed_arrays=rebuilt, states_verified=states,
        policies=policies, synthetic_gradient=algebra)


class Reference:
    def __init__(self, out, plan, tid):
        abc = load_abc(out, tid)
        self.values = {s: torch.tensor(abc[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
        self.starts = {s: torch.tensor(abc[f'{s}_starts'], device='cuda') for s in self.values}
        self.model = rt.core.DLinear(dict(plan['training'], channels=8)).cuda().eval()

    def batch(self, split, lo):
        return self.values[split][self.starts[split][lo:lo+2048, None] + torch.arange(40, device='cuda')]

    @torch.no_grad()
    def errors(self, state, split):
        self.model.load_state_dict(state)
        chunks = []
        for lo in range(0, len(self.starts[split]), 2048):
            batch = self.batch(split, lo)
            chunks.append((self.model(batch[:, :32]) - batch[:, 32:]).cpu().numpy())
        return np.concatenate(chunks)

    def evaluate(self, state, split):
        return metric(self.errors(state, split))

    @torch.no_grad()
    def moments(self, weights=None):
        result = None
        for lo in range(0, len(self.starts['A']), 2048):
            batch = self.batch('A', lo)
            phi = features(batch[:, :32]).double().permute(1, 0, 2)
            target = batch[:, 32:].double().permute(2, 0, 1)
            mass = (torch.ones(len(batch), dtype=torch.float64, device='cuda') if weights is None
                else torch.tensor(weights[lo:lo+len(batch)], dtype=torch.float64, device='cuda')) * (1./6468)
            weighted = phi * mass[None, :, None]
            items = [phi.transpose(1, 2) @ weighted, weighted.transpose(1, 2) @ target,
                (target.square() * mass[None, :, None]).sum(1)]
            if result is None:
                result = items
            else:
                for a, b in zip(result, items):
                    a.add_(b)
        return [a.cpu().numpy() for a in result]


def nonnegative(value):
    assert isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
    return value


def calibration(out, plan):
    assert rt.read(out / 'preflight_audit.json')['status'] == 'PASS'
    saved = rt.read(out / 'calibration.json')
    records = {(r['task'], r['engine'], r['seed']): r for r in saved['records']}
    identities = {(r['task'], r['seed']): r for r in saved['identity_scores']}
    assert len(records) == 18 and len(identities) == 9
    states = residuals = moments = 0
    task_results = {}
    for task in plan['tasks']:
        tid = task['id']
        ref = Reference(out, plan, tid)
        expected_moments = ref.moments()
        for seed in rt.SEEDS:
            full_curves = None
            for engine in ('full', 'proxy'):
                record = records[(tid, engine, seed)]
                directory = out / 'calibration_runs' / f'FGM1_{tid}_{engine}_PARENT_s{seed}'
                assert record['directory'] == directory.relative_to(out).as_posix()
                result = rt.read(directory / 'result.json')
                timing = rt.read(directory / 'timing.json')
                assert record == dict(result, timing=timing, directory=record['directory'])
                assert result['status'] == rt.read(directory / 'status.json')['status'] == 'completed'
                assert result['plan_sha256'] == rt.sha(out / 'plan.json') and result['replay']
                assert result['id'] == directory.name and result['action'] == 'PARENT'
                assert result['checkpoints'] == list(rt.GRID) and result['steps'] == 3000
                for name, digest in result['assets_sha256'].items():
                    assert rt.sha(rt.safe_file(directory, name)) == digest
                nonnegative(timing['seconds'])
                nonnegative(timing['actual_elapsed_seconds'])
                overhead = sum(nonnegative(v['seconds']) for v in result['replay_matches'])
                assert timing['replay_verification_seconds'] == overhead
                assert timing['seconds'] == timing['actual_elapsed_seconds'] - overhead
                assert [v['step'] for v in result['replay_matches']] == list(rt.GRID)
                assert all(v['bitexact'] for v in result['replay_matches'])
                assert set(timing['prefix_seconds']) == {str(k) for k in rt.GRID}
                for k in rt.GRID:
                    prefix = rt.read(directory / f'prefix_{k}.json')
                    assert prefix['seconds'] == result['prefix_seconds'][str(k)]
                    assert timing['prefix_seconds'][str(k)] == prefix['seconds'] - sum(v['seconds'] for v in result['replay_matches'] if v['step'] < k)
                    nonnegative(timing['prefix_seconds'][str(k)])
                    for name, digest in prefix['assets_sha256'].items():
                        assert rt.sha(rt.safe_file(directory, name)) == digest
                assert list(timing['prefix_seconds'].values()) == sorted(timing['prefix_seconds'].values())
                assert timing['prefix_seconds']['3000'] <= timing['seconds']
                states += state_audit(directory, plan, seed, engine)
                curves = {}
                for step in (0,) + rt.GRID:
                    state = rt.load_snapshot(directory, step)
                    rt.equal(state, rt.load_snapshot(parent(out, tid, engine, seed), step))
                    if step:
                        curves[str(step)] = {s: ref.evaluate(state['model'], s) for s in ('B', 'C')}
                        assert curves[str(step)] == rt.read(directory / f'step_{step}_metrics.json')
                        residuals += 2
                assert curves == result['curves']
                selected = min(rt.GRID, key=lambda k: curves[str(k)]['B']['nmse'])
                assert result['selected_step'] == selected and result['selected_metrics'] == curves[str(selected)]
                if engine == 'proxy':
                    for source in (directory, parent(out, tid, engine, seed)):
                        stored = arrays(source / 'moments.npz')
                        for name, value in zip(('gram', 'cross', 'target_square'), expected_moments):
                            np.testing.assert_array_equal(value, stored[name])
                        moments += 1
                else:
                    full_curves = curves
                    rng = np.random.default_rng(seed + 1000)
                    cdf = np.cumsum(np.full(6468, 1./6468))
                    cdf[-1] = 1.
                    draws = np.zeros(6468, dtype=np.int64)
                    for _ in range(3000):
                        np.add.at(draws, np.searchsorted(cdf, rng.random(256), side='right'), 1)
                    np.testing.assert_array_equal(draws, arrays(directory / 'sampling.npz')['draw_count'])
            identity = identities[(tid, seed)]
            directory = out / 'identity_scores' / f'{tid}_PARENT_{seed}_M1'
            local = rt.read(directory / 'result.json')
            clock = rt.read(directory / 'timing.json')
            assert identity == dict(local, seconds=clock['seconds'], directory=directory.relative_to(out).as_posix())
            assert clock['result_sha256'] == rt.sha(directory / 'result.json')
            nonnegative(clock['seconds'])
            assert local['method'] == 'M1' and local['action'] == 'PARENT'
            assert local['curves'] == full_curves
            selected = min(rt.GRID, key=lambda k: full_curves[str(k)]['B']['nmse'])
            assert local['selected_step'] == selected
            assert local['selected_metrics'] == full_curves[str(selected)]
            assert local['gain_percent'] == 0. and local['baseline_C'] == full_curves[str(selected)]['C']['nmse']
            old = rt.read(parent(out, tid, 'full', seed) / 'result.json')
            assert old['selected_step'] == selected and old['selected_losses']['C'] == local['baseline_C']
        pmax = max(records[(tid, 'proxy', s)]['timing']['seconds'] for s in rt.SEEDS)
        emax = max(identities[(tid, s)]['seconds'] for s in rt.SEEDS)
        cost = 1.125 * pmax + emax
        short = next((k for k in rt.GRID[:-1] if min(records[(tid, 'full', s)]['timing']['prefix_seconds'][str(k)] for s in rt.SEEDS) >= 1.25 * cost), None)
        minima = {str(k): min(records[(tid, 'full', s)]['timing']['prefix_seconds'][str(k)]
            for s in rt.SEEDS) for k in rt.GRID[:-1]}
        expected = dict(proxy_max_seconds=pmax, identity_max_seconds=emax,
            candidate_seconds=cost, required_prefix_seconds=1.25 * cost,
            prefix_min_seconds=minima, short_steps=short)
        if short is None:
            failed = dict(short_steps=None, failed=True,
                proxy_seconds={str(s): records[(tid, 'proxy', s)]['timing']['seconds'] for s in rt.SEEDS},
                identity_seconds={str(s): identities[(tid, s)]['seconds'] for s in rt.SEEDS},
                prefix_seconds={str(s): {str(k): records[(tid, 'full', s)]['timing']['prefix_seconds'][str(k)]
                    for k in rt.GRID[:-1]} for s in rt.SEEDS})
            assert saved['tasks'][tid] == failed
        else:
            assert saved['tasks'][tid] == expected
        task_results[tid] = dict(expected, eligible=short is not None)
    eligible = all(v['eligible'] for v in task_results.values())
    assert saved['status'] == ('completed' if eligible else 'calibration_budget_failed')
    assert saved['eligible_to_forecast'] == eligible
    assert saved['plan_sha256'] == rt.sha(out / 'plan.json')
    assert saved['new_full_trainings'] == 0 and saved['cumulative_full_trainings'] == 418
    assert saved['replay_update_steps'] == 54000
    assert not saved['D_access'] and not saved['test_access']
    assert {v['task'] for v in saved['failures']} == {t for t, v in task_results.items() if not v['eligible']}
    return dict(status='PASS', eligible_to_forecast=eligible, budget_status=saved['status'],
        states_verified=states, residuals_verified=residuals, moments_verified=moments,
        tasks=task_results, calibration_sha256=rt.sha(out / 'calibration.json'))


def close(a, b, tolerance=1e-9):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b)


def category(gain):
    return 'improve' if gain > .25 else 'hurt' if gain < -.25 else 'unchanged'


def load_coordinates(state, value):
    result = {k: v.clone() for k, v in state.items()}
    for c in range(8):
        for kind, part, val in (('seasonal', 'weight', value[c, :32].T),
                ('seasonal', 'bias', value[c, 32]), ('trend', 'weight', value[c, 33:65].T),
                ('trend', 'bias', value[c, 65])):
            result[f'{kind}.{c}.{part}'] = val.clone()
    return result


def run_record(out, plan, directory, tid, engine, action, seed, steps):
    result = rt.read(directory / 'result.json')
    status = rt.read(directory / 'status.json')
    assert result['status'] == status['status'] == 'completed'
    assert (result['task'], result['engine'], result['action'], result['seed']) == (tid, engine, action, seed)
    assert result['id'] == directory.name and result['steps_completed'] == result['steps'] == steps
    assert status['steps_completed'] == steps and not result['replay']
    assert result['checkpoints'] == [k for k in rt.GRID if k <= steps]
    assert result['plan_sha256'] == rt.sha(out / 'plan.json')
    assert result['replay_matches'] == []
    for name, digest in result['assets_sha256'].items():
        assert rt.sha(rt.safe_file(directory, name)) == digest
    timing = rt.read(directory / 'timing.json')
    assert timing['replay_verification_seconds'] == 0
    assert nonnegative(timing['seconds']) == nonnegative(timing['actual_elapsed_seconds'])
    assert timing['prefix_seconds'] == result['prefix_seconds']
    previous = 0.
    for k in result['checkpoints']:
        prefix = rt.read(directory / f'prefix_{k}.json')
        assert prefix['seconds'] == timing['prefix_seconds'][str(k)]
        assert previous <= prefix['seconds'] <= timing['seconds']
        previous = prefix['seconds']
        for name, digest in prefix['assets_sha256'].items():
            assert rt.sha(rt.safe_file(directory, name)) == digest
    return result, timing


def replay(out, plan, ref, directory, tid, engine, action, seed, steps, terms, weights):
    """Independent in-memory updates; no audit checkpoints or new science IDs."""
    started = rt.tick()
    model = new_model(plan, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rng = np.random.default_rng(seed + 1000)
    initial = rt.load_snapshot(directory, 0)
    rt.equal(initial, rt.load_snapshot(parent(out, tid, 'proxy' if engine == 'proxy' else 'full', seed), 0))
    def check(step, path=None):
        saved = rt.load_snapshot(directory, step) if path is None else torch.load(path, map_location='cpu', weights_only=True)
        actual = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
            rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(), original_coordinates=coordinates(model.state_dict()).detach().cpu())
        rt.equal(actual, saved)
        assert bool(torch.isfinite(actual['original_coordinates']).all())
    check(0)
    if engine == 'proxy':
        gram, cross, _ = [torch.tensor(v, device='cuda') for v in terms]
        stored = arrays(directory / 'moments.npz')
        for name, value in zip(('gram', 'cross', 'target_square'), terms):
            np.testing.assert_array_equal(value, stored[name])
    mass = torch.tensor(weights, device='cuda')
    cdf = np.cumsum(np.full(6468, 1./6468)); cdf[-1] = 1.
    draws = np.zeros(6468, dtype=np.int64)
    checked = 1
    for step in range(1, steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if engine == 'proxy':
            with torch.no_grad():
                gradient = (2./64) * (gram @ coordinates(model.state_dict()).double() - cross)
                for c, (seasonal, trend) in enumerate(zip(model.seasonal, model.trend)):
                    for parameter, value in zip((seasonal.weight, seasonal.bias, trend.weight, trend.bias),
                            (gradient[c, :32].T, gradient[c, 32], gradient[c, 33:65].T, gradient[c, 65])):
                        parameter.grad = value.to(dtype=parameter.dtype).contiguous()
        else:
            ids = np.searchsorted(cdf, rng.random(256), side='right')
            np.add.at(draws, ids, 1)
            index = torch.tensor(ids, device='cuda')
            batch = ref.values['A'][ref.starts['A'][index, None] + torch.arange(40, device='cuda')]
            ((model(batch[:, :32])-batch[:, 32:]).square() * mass[index, None, None]).mean().backward()
        optimizer.step()
        if step in (1, 501):
            check(step, directory / f'witness_{step}.pt'); checked += 1
        if step in rt.GRID:
            check(step); checked += 1
    if engine != 'proxy':
        np.testing.assert_array_equal(draws, arrays(directory / 'sampling.npz')['draw_count'])
    return dict(updates=steps, states=checked, seconds=rt.tick()-started)


def curves_audit(ref, directory, result):
    curves = {str(k): {s: ref.evaluate(rt.load_snapshot(directory, k)['model'], s) for s in ('B', 'C')}
        for k in result['checkpoints']}
    assert curves == result['curves']
    for k, value in curves.items():
        assert rt.read(directory / f'step_{k}_metrics.json') == value
    selected = min(result['checkpoints'], key=lambda k: curves[str(k)]['B']['nmse'])
    assert result['selected_step'] == selected and result['selected_metrics'] == curves[str(selected)]
    return curves


def gamma(n, dtype):
    unit = np.finfo(dtype).eps/2
    return n*unit/(1-n*unit)


def rounding_identity(ref, real, child, proxy, combined, split):
    raw = ref.errors(child['model'], split).astype(np.float64)
    parent_error = ref.errors(real['model'], split).astype(np.float64)
    proxy_error = ref.errors(proxy['model'], split).astype(np.float64)
    paired = ref.errors(combined, split).astype(np.float64)
    pc, pr, pp, combo = [coordinates(v).double().cuda() for v in
        (child['model'], real['model'], proxy['model'], combined)]
    maxima = 0.
    for lo in range(0, len(ref.starts[split]), 2048):
        batch = ref.batch(split, lo)
        phi = features(batch[:, :32]).double().abs()
        target = batch[:, 32:].double().abs()
        coordinate_bound = torch.einsum('nci,cij->njc', phi, (combo-(pr+pc-pp)).abs())
        magnitude = sum(torch.einsum('nci,cij->njc', phi, v.abs()) + target for v in (pc, pr, pp, combo))
        bound = coordinate_bound + gamma(70, np.float32)*magnitude + gamma(256, np.float64)*(magnitude+coordinate_bound)
        gap = paired[lo:lo+len(batch)] - (raw[lo:lo+len(batch)] - (proxy_error[lo:lo+len(batch)]-parent_error[lo:lo+len(batch)]))
        assert np.all(np.abs(gap) <= bound.cpu().numpy())
        maxima = max(maxima, float(np.abs(gap).max()))
    return maxima


def source_audit(out, ref, tid, action, seed, terms, parent_terms, local):
    directory = out / 'forecasts' / f'{tid}_{action}_{seed}_SOURCE'
    for name, expected in (('parent', parent_terms), ('action', terms)):
        stored = arrays(directory / f'{name}_moments.npz')
        for key, value in zip(('gram', 'cross', 'target_square'), expected):
            np.testing.assert_array_equal(value, stored[key])
    delta = np.zeros((8, 66, 8)); curves = {}; probes = []
    for step in rt.GRID:
        saved_parent = rt.load_snapshot(parent(out, tid, 'full', seed), step)
        theta = coordinates(saved_parent['model']).numpy().astype(np.float64)
        variance = {name: saved_parent['optimizer']['state'][i]['exp_avg_sq'].numpy().astype(np.float64)
            for i, name in enumerate(saved_parent['model'])}
        packed_v = coordinates({name: torch.from_numpy(v) for name, v in variance.items()}).numpy()
        correction = -math.expm1(step*math.log(.999))
        p = 1./(np.sqrt(packed_v/correction)+1e-8)
        forcing = (2./64)*(terms[0] @ theta-terms[1]) - (2./64)*(parent_terms[0] @ theta-parent_terms[1])
        previous = delta.copy()
        clipped = 0; max_tolerance = 0.
        for c in range(8):
            h = (2./64)*parent_terms[0][c]
            for j in range(8):
                root = np.sqrt(p[c, :, j])
                sym = root[:, None]*((h+h.T)/2)*root[None, :]
                eigen, q = np.linalg.eigh(sym)
                tolerance = 64*66*np.finfo(np.float64).eps*max(1., float(np.abs(eigen).max()))
                assert eigen.min() >= -tolerance
                clipped += int((eigen < 0).sum()); max_tolerance = max(max_tolerance, tolerance)
                eigen = np.maximum(eigen, 0.)
                decay = np.exp(-.5*eigen)
                response = np.full_like(eigen, .5)
                mask = eigen > 0
                response[mask] = -np.expm1(-.5*eigen[mask])/eigen[mask]
                propagation = root[:, None]*((q*decay)@q.T)/root[None, :]
                integral = root[:, None]*((q*response)@q.T)*root[None, :]
                delta[c, :, j] = propagation@previous[c, :, j]-integral@forcing[c, :, j]
                probe = (tid, seed, action, step, c, j) in (
                    ('G1', rt.SEEDS[0], 'G0_UP', 500, 0, 0),
                    ('G3', rt.SEEDS[2], 'G3_DOWN', 3000, 7, 7))
                if probe:
                    generator = np.zeros((67, 67))
                    generator[:66, :66] = -p[c, :, j, None]*h
                    generator[:66, 66] = -p[c, :, j]*forcing[c, :, j]
                    exponential = torch.matrix_exp(torch.from_numpy(.5*generator)).numpy()
                    reference = (exponential@np.r_[previous[c, :, j], 1.])[:66]
                    np.testing.assert_allclose(delta[c, :, j], reference, rtol=1e-8, atol=1e-7)
                    probes.append(dict(task=tid, seed=seed, action=action, step=step, channel=c,
                        output=j, max_absolute_error=float(np.max(np.abs(reference-delta[c, :, j])))))
        stored = arrays(directory / f'step_{step}.npz')
        np.testing.assert_allclose(stored['delta'], delta, rtol=1e-8, atol=1e-7)
        np.testing.assert_array_equal(stored['coordinates'], (theta+stored['delta']).astype(np.float32))
        state = load_coordinates(saved_parent['model'], torch.tensor(stored['coordinates']))
        curves[str(step)] = {s: ref.evaluate(state, s) for s in ('B', 'C')}
        diagnostic = local['numerical_diagnostics'][rt.GRID.index(step)]
        assert diagnostic == dict(step=step, clipped_negative_eigenvalues=clipped, max_roundoff_tolerance=max_tolerance)
    assert local['curves'] == curves
    return probes


def forecast(out, plan):
    calibration_result = rt.read(out / 'calibration.json')
    calibration_check = rt.read(out / 'calibration_audit.json')
    sync = rt.read(out / 'calibration_sync_receipt.json')
    assert calibration_result['eligible_to_forecast'] and calibration_check['eligible_to_forecast']
    assert calibration_check['status'] == 'PASS' and sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['plan_sha256'] == rt.sha(out / 'plan.json')
    assert sync['calibration_sha256'] == calibration_check['calibration_sha256'] == rt.sha(out / 'calibration.json')
    assert sync['calibration_audit_sha256'] == rt.sha(out / 'calibration_audit.json')
    prediction = rt.read(out / 'predictions.json')
    assert prediction['status'] == 'completed' and prediction['new_full_trainings'] == 0
    assert prediction['cumulative_full_trainings'] == 418 and not prediction['full_children_started']
    assert prediction['plan_sha256'] == rt.sha(out / 'plan.json')
    assert prediction['calibration_sha256'] == rt.sha(out / 'calibration.json')
    assert not prediction['D_access'] and not prediction['test_access']
    assert prediction['scientific_accuracy'] is None and prediction['actual_gain'] is None
    assert prediction['proxy_rollouts'] == prediction['short_runs'] == 72
    actual_assets = {p.relative_to(out).as_posix(): rt.sha(p) for name in ('runs', 'forecasts')
        for p in sorted((out / name).rglob('*')) if p.is_file()}
    assert actual_assets == prediction['assets_sha256']
    for name, digest in prediction['assets_sha256'].items():
        assert rt.sha(rt.safe_file(out, name)) == digest
    records = {(r['task'], r['action'], r['seed']): r for r in prediction['records']}
    expected_keys = {(t, a, s) for t in rt.TASKS for a in rt.ACTIONS for s in rt.SEEDS}
    assert len(prediction['records']) == 72 and set(records) == expected_keys
    expected_runs = {f'FGM1_{t}_{e}_{a}_s{s}' for t, a, s in expected_keys for e in ('proxy', 'short')}
    assert {p.name for p in (out / 'runs').iterdir()} == expected_runs
    assert {p.name for p in (out / 'forecasts').iterdir()} == {f'{t}_{a}_{s}_{m}' for t, a, s in expected_keys for m in ('M1', 'RAW', 'SOURCE')}
    costs = []; replay_costs = []; probes = []; rounding = []; residuals = 0
    recommendations = {}; chosen = {m: {} for m in rt.METHODS}
    for task in plan['tasks']:
        tid = task['id']; ref = Reference(out, plan, tid)
        parent_terms = ref.moments()
        terms_by_action = {a: ref.moments(arrays(out / 'prepared' / f'{tid}_{a}.npz')['weights']) for a in rt.ACTIONS}
        for seed in rt.SEEDS:
            old_parent = parent(out, tid, 'full', seed)
            old_result = rt.read(old_parent / 'result.json')
            baseline = old_result['selected_losses']['C']
            assert baseline > 0 and math.isfinite(baseline)
            cost = dict(task=tid, seed=seed, M1=next(r['timing']['seconds'] for r in calibration_result['records'] if (r['task'], r['engine'], r['seed']) == (tid, 'proxy', seed)),
                RAW=0., SOURCE=0., SHORT=0., PARENT=0., real_parent_seconds=next(r['timing']['seconds'] for r in calibration_result['records'] if (r['task'], r['engine'], r['seed']) == (tid, 'full', seed)))
            for action in rt.ACTIONS:
                record = records[(tid, action, seed)]
                weights = arrays(out / 'prepared' / f'{tid}_{action}.npz')['weights']
                proxy_dir = out / 'runs' / f'FGM1_{tid}_proxy_{action}_s{seed}'
                assert record['proxy_directory'] == proxy_dir.relative_to(out).as_posix()
                short_steps = calibration_result['tasks'][tid]['short_steps']
                short_dir = out / 'runs' / f'FGM1_{tid}_short_{action}_s{seed}'
                proxy_result, proxy_timing = run_record(out, plan, proxy_dir, tid, 'proxy', action, seed, 3000)
                assert proxy_result['curves'] == {} and proxy_result['selected_step'] is None and proxy_result['selected_metrics'] is None
                short_result, short_timing = run_record(out, plan, short_dir, tid, 'short', action, seed, short_steps)
                for engine, directory, steps in (('proxy', proxy_dir, 3000), ('short', short_dir, short_steps)):
                    replay_costs.append(dict(task=tid, action=action, seed=seed, engine=engine,
                        **replay(out, plan, ref, directory, tid, engine, action, seed, steps, terms_by_action[action], weights)))
                curves_audit(ref, short_dir, short_result)
                residuals += 2*len(short_result['checkpoints'])
                assert set(record['methods']) == set(rt.METHODS)
                for method in ('M1', 'RAW', 'SOURCE'):
                    directory = out / 'forecasts' / f'{tid}_{action}_{seed}_{method}'
                    local = rt.read(directory / 'result.json'); timing = rt.read(directory / 'timing.json')
                    assert record['methods'][method] == dict(local, seconds=timing['seconds'], directory=directory.relative_to(out).as_posix())
                    assert timing['result_sha256'] == rt.sha(directory / 'result.json')
                    nonnegative(timing['seconds'])
                    assert (local['task'], local['seed'], local['action'], local['method']) == (tid, seed, action, method)
                    if method == 'SOURCE':
                        for name, digest in local['assets_sha256'].items():
                            assert rt.sha(rt.safe_file(directory, name)) == digest
                        probes += source_audit(out, ref, tid, action, seed, terms_by_action[action], parent_terms, local)
                    else:
                        curves = {}
                        for step in rt.GRID:
                            child = rt.load_snapshot(proxy_dir, step)
                            state = child['model']
                            if method == 'M1':
                                real = rt.load_snapshot(old_parent, step)
                                proxy = rt.load_snapshot(parent(out, tid, 'proxy', seed), step)
                                value = coordinates(real['model']) + (coordinates(child['model'])-coordinates(proxy['model']))
                                state = load_coordinates(real['model'], value)
                                for split in ('B', 'C'):
                                    rounding.append(rounding_identity(ref, real, child, proxy, state, split))
                            curves[str(step)] = {s: ref.evaluate(state, s) for s in ('B', 'C')}
                        assert local['curves'] == curves
                    selected = min(rt.GRID, key=lambda k: local['curves'][str(k)]['B']['nmse'])
                    assert local['selected_step'] == selected
                    assert local['selected_metrics'] == local['curves'][str(selected)]
                    assert local['baseline_C'] == baseline
                    assert local['gain_percent'] == 100.*(1.-local['curves'][str(selected)]['C']['nmse']/baseline)
                    residuals += 12
                short = record['methods']['SHORT']
                assert short == dict(method='SHORT', task=tid, action=action, seed=seed,
                    selected_step=short_result['selected_step'], selected_metrics=short_result['selected_metrics'],
                    curves=short_result['curves'], steps=short_steps, directory=short_dir.relative_to(out).as_posix(),
                    gain_percent=100.*(1.-short_result['selected_metrics']['C']['nmse']/baseline), seconds=short_timing['seconds'])
                assert record['methods']['PARENT'] == dict(method='PARENT', gain_percent=0., seconds=0.)
                for m in ('M1', 'RAW'):
                    cost[m] += proxy_timing['seconds'] + record['methods'][m]['seconds']
                cost['SOURCE'] += record['methods']['SOURCE']['seconds']; cost['SHORT'] += short_timing['seconds']
            cost['budget_pass'] = cost['M1'] <= cost['SHORT']
            historical = rt.read(out / 'inputs/prior_preparation.json')['seconds']
            assert historical == plan['historical_preparation_seconds']
            common = (rt.read(out / 'preparation.json')['seconds']+historical)/9 + next(s['seconds'] for s in prediction['setup'] if s['task'] == tid)/3
            nonnegative(common)
            cost['allocated_common_preparation_seconds'] = common
            cost['cold_allocated_seconds'] = {m: cost[m]+cost['real_parent_seconds']+common for m in rt.METHODS}
            registered = [c for c in prediction['costs'] if (c['task'], c['seed']) == (tid, seed)]
            assert len(registered) == 1
            for key, value in cost.items():
                if key in rt.METHODS or key in ('real_parent_seconds', 'allocated_common_preparation_seconds'):
                    close(value, registered[0][key])
                elif key == 'cold_allocated_seconds':
                    for m in rt.METHODS: close(value[m], registered[0][key][m])
                else: assert value == registered[0][key]
            costs.append(cost)
        recommendations[tid] = {}
        for m in rt.METHODS:
            means = {a: float(np.mean([records[(tid, a, s)]['methods'][m]['gain_percent'] for s in rt.SEEDS])) for a in rt.ACTIONS}
            best = max(rt.ACTIONS, key=lambda a: means[a]); choice = best if means[best] >= 1. else 'PARENT'
            recommendations[tid][m] = dict(action=choice, best_action=best, best_mean_gain=means[best], mean_gains=means)
            assert prediction['recommendations'][tid][m] == recommendations[tid][m]
            chosen[m][tid] = choice
    matches = [m for m in rt.METHODS[1:] if all(chosen['M1'][t] == chosen[m][t] for t in rt.TASKS)]
    budget = all(c['budget_pass'] for c in costs)
    assert prediction['identical_recommendation_stop'] == dict(stop=bool(matches), matching_baselines=matches)
    assert prediction['budget_gate'] == budget and prediction['eligible_to_label'] == (budget and not matches)
    assert len(probes) == 2 and len(costs) == 9 and len(replay_costs) == 144
    assert len(prediction['costs']) == 9 and len(prediction['setup']) == 3
    assert {s['task'] for s in prediction['setup']} == set(rt.TASKS)
    for setup in prediction['setup']:
        nonnegative(setup['seconds'])
    return dict(status='PASS', eligible_to_label=bool(budget and not matches), budget_gate=budget,
        matching_baselines=matches, recommendations=recommendations, costs=costs,
        replay_audit=replay_costs, audit_updates=sum(r['updates'] for r in replay_costs),
        replay_audit_seconds=sum(r['seconds'] for r in replay_costs), residuals_verified=residuals,
        source_matrix_exponential_probes=probes, source_propagation_blocks_verified=72*6*64,
        rounding_identity_max_gap=max(rounding), predictions_sha256=rt.sha(out / 'predictions.json'),
        states_verified=sum(r['states'] for r in replay_costs),
        moment_files_verified=72+72*2)


def final(out, plan):
    prediction = rt.read(out / 'predictions.json'); audited = rt.read(out / 'forecast_audit.json')
    lock = rt.read(out / 'prediction_lock.json'); sync = rt.read(out / 'prediction_sync_receipt.json')
    assert audited['status'] == 'PASS' and audited['eligible_to_label'] and prediction['eligible_to_label']
    digest = rt.sha(out / 'predictions.json')
    assert audited['predictions_sha256'] == lock['predictions_sha256'] == sync['predictions_sha256'] == digest
    assert lock['plan_sha256'] == sync['plan_sha256'] == rt.sha(out / 'plan.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['forecast_audit_sha256'] == rt.sha(out / 'forecast_audit.json')
    assert sync['prediction_lock_sha256'] == rt.sha(out / 'prediction_lock.json')
    assert rt.read(out / 'labels.lock') == dict(plan_sha256=rt.sha(out / 'plan.json'), predictions_sha256=digest)
    labels = rt.read(out / 'labels.json')
    assert labels['status'] == rt.read(out / 'labels_status.json')['status'] == 'completed'
    assert labels['predictions_sha256'] == digest and labels['new_full_trainings'] == 72 and labels['cumulative_full_trainings'] == 490
    assert rt.read(out / 'labels_status.json')['completed'] == 72
    assert not labels['D_access'] and not labels['test_access']
    for name, value in prediction['assets_sha256'].items():
        assert rt.sha(rt.safe_file(out, name)) == value
    expected = {(t, a, s) for t in rt.TASKS for a in rt.ACTIONS for s in rt.SEEDS}
    records = {(r['task'], r['action'], r['seed']): r for r in labels['records']}
    assert len(labels['records']) == 72 and set(records) == expected
    forecasts = {(r['task'], r['action'], r['seed']): r for r in prediction['records']}
    assert set(forecasts) == expected
    expected_runs = {f'FGM1_{t}_{e}_{a}_s{s}' for t, a, s in expected for e in ('proxy', 'short', 'full')}
    assert {p.name for p in (out / 'runs').iterdir()} == expected_runs
    rows = []; replays = []; full_cost = {}
    for task in plan['tasks']:
        tid = task['id']; ref = Reference(out, plan, tid)
        for seed in rt.SEEDS:
            old = rt.read(parent(out, tid, 'full', seed) / 'result.json')
            baseline = {s: ref.evaluate(rt.load_snapshot(parent(out, tid, 'full', seed), old['selected_step'])['model'], s) for s in ('B', 'C')}
            assert baseline['C']['nmse'] == old['selected_losses']['C']
            full_cost[(tid, seed)] = 0.
            for action in rt.ACTIONS:
                directory = out / 'runs' / f'FGM1_{tid}_full_{action}_s{seed}'
                result, timing = run_record(out, plan, directory, tid, 'full', action, seed, 3000)
                assert records[(tid, action, seed)] == dict(result, timing=timing, directory=directory.relative_to(out).as_posix())
                weights = arrays(out / 'prepared' / f'{tid}_{action}.npz')['weights']
                replays.append(dict(task=tid, action=action, seed=seed,
                    **replay(out, plan, ref, directory, tid, 'full', action, seed, 3000, None, weights)))
                curves_audit(ref, directory, result)
                true = result['selected_metrics']['C']
                gain = 100.*(1.-true['nmse']/baseline['C']['nmse'])
                nmae_degradation = 100.*(true['nmae']/baseline['C']['nmae']-1.) if baseline['C']['nmae'] > 0 else (0. if true['nmae'] == 0 else None)
                rows.append(dict(task=tid, action=action, seed=seed, actual_gain=gain,
                    nmae_degradation_percent=nmae_degradation, selected_step=result['selected_step'],
                    selected_metrics=true, parent_metrics=baseline['C'],
                    predicted={m: forecasts[(tid, action, seed)]['methods'][m]['gain_percent'] for m in rt.METHODS}))
                full_cost[(tid, seed)] += timing['seconds']
    cells = []; decisions = []; task_mae = {m: {} for m in rt.METHODS}; realized = {m: [] for m in rt.METHODS}
    for tid in rt.TASKS:
        for action in rt.ACTIONS:
            subset = [r for r in rows if (r['task'], r['action']) == (tid, action)]
            assert len(subset) == 3
            truth = float(np.mean([r['actual_gain'] for r in subset]))
            predicted = {m: float(np.mean([r['predicted'][m] for r in subset])) for m in rt.METHODS}
            classes = [category(r['actual_gain']) for r in subset]
            cells.append(dict(task=tid, action=action, truth=truth, predictions=predicted,
                category=category(truth), seed_categories=classes, unstable=len(set(classes)) > 1,
                qualified=truth >= 1. and all(r['nmae_degradation_percent'] is not None and r['nmae_degradation_percent'] <= 1. for r in subset)))
        task_cells = [c for c in cells if c['task'] == tid]
        raw_oracle = max([0.] + [c['truth'] for c in task_cells])
        qualified_oracle = max([0.] + [c['truth'] for c in task_cells if c['qualified']])
        for m in rt.METHODS:
            task_mae[m][tid] = float(np.mean([abs(c['truth']-c['predictions'][m]) for c in task_cells]))
            choice = prediction['recommendations'][tid][m]['action']
            gain = 0. if choice == 'PARENT' else next(c['truth'] for c in task_cells if c['action'] == choice)
            subset = [] if choice == 'PARENT' else [r for r in rows if (r['task'], r['action']) == (tid, choice)]
            nmae_ok = all(r['nmae_degradation_percent'] is not None and r['nmae_degradation_percent'] <= 1. for r in subset)
            realized[m].append(gain)
            decisions.append(dict(task=tid, method=m, action=choice, gain=gain, raw_oracle=raw_oracle,
                qualified_oracle=qualified_oracle, raw_regret=raw_oracle-gain, qualified_regret=qualified_oracle-gain,
                nmae_guard=nmae_ok, harm=gain < -.25, abstained=choice == 'PARENT',
                qualified_actions=sum(c['qualified'] for c in task_cells),
                covered_qualified_actions=sum(c['qualified'] and c['predictions'][m] >= 1. for c in task_cells)))
    mae = {m: float(np.mean(list(task_mae[m].values()))) for m in rt.METHODS}
    best = min(('RAW', 'SOURCE', 'SHORT'), key=lambda m: mae[m])
    noninferior = sum(task_mae['M1'][t] <= task_mae[best][t] for t in rt.TASKS)
    means = {m: float(np.mean(realized[m])) for m in rt.METHODS}
    budgets = []
    for cost in audited['costs']:
        total = full_cost[(cost['task'], cost['seed'])]
        budgets.append(dict(task=cost['task'], seed=cost['seed'], M1=cost['M1'], SHORT=cost['SHORT'],
            complete_children_seconds=total, short_gate=cost['M1'] <= cost['SHORT'], full_gate=cost['M1'] <= .5*total))
    gates = dict(accuracy_ratio=mae['M1'] <= .8*mae[best], accuracy_difference=mae[best]-mae['M1'] >= .05,
        task_coverage=noninferior >= 2, recommendation_increment=means['M1'] >= max(means[m] for m in rt.METHODS[1:])+.10,
        harm_guard=min(realized['M1']) >= -.25,
        nmae_guard=all(d['nmae_guard'] for d in decisions if d['method'] == 'M1'),
        cost=all(c['short_gate'] and c['full_gate'] for c in budgets), completeness=True)
    assert len(cells) == 24 and len(rows) == 72 and len(replays) == 72
    return dict(status='PASS', scientific_gates=gates, advance=all(gates.values()),
        new_full_trainings=72, cumulative_full_trainings=490, best_dynamic_baseline=best,
        prediction_MAE_pp=mae, task_MAE_pp=task_mae, noninferior_tasks=noninferior,
        mean_recommendation_gain=means, recommendation_gains=realized, cells=cells,
        scalar_records=rows, decisions=decisions, costs=budgets,
        per_seed_MAE_pp={m: {str(s): float(np.mean([abs(r['actual_gain']-r['predicted'][m]) for r in rows if r['seed'] == s])) for s in rt.SEEDS} for m in rt.METHODS},
        replay_audit=replays, audit_updates=sum(r['updates'] for r in replays),
        replay_audit_seconds=sum(r['seconds'] for r in replays), states_verified=sum(r['states'] for r in replays),
        residuals_verified=72*12+18, predictions_sha256=digest, labels_sha256=rt.sha(out / 'labels.json'),
        independent_source_claim=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'calibration', 'forecast', 'final'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    # Reject an unrelated destination before any reads or failure writes.
    if out != rt.ROOT or out.resolve() != rt.ROOT:
        raise ValueError('Unexpected study root')
    destination = out / f'{args.mode}_audit.json'
    failure = out / f'{args.mode}_audit_failure.json'
    if destination.exists() or failure.exists():
        raise FileExistsError('Audit already attempted; preserve its original receipt')
    started = time.perf_counter()
    try:
        rt.configure()
        plan = rt.verify(out)
        result = globals()[args.mode](out, plan)
        rt.verify(out)
        result.update(mode=args.mode, plan_sha256=rt.sha(out / 'plan.json'),
            input_lock_sha256=rt.sha(out / 'input_lock.json'),
            train_sha256=rt.sha(out / 'inputs/data/train.csv'),
            manifest_sha256=rt.sha(out / 'inputs/data/manifest.json'),
            preparation_sha256=rt.sha(out / 'preparation.json'),
            auditor_sha256=rt.sha(Path(__file__)), D_access=False, test_access=False,
            old_D_sensor_tokens_parsed=False, new_scientific_trainings=0,
            audit_seconds=time.perf_counter() - started)
        rt.write(destination, result)
        print(result['status'], flush=True)
    except Exception:
        rt.write(failure, dict(status='FAIL', mode=args.mode, error=traceback.format_exc(),
            audit_seconds=time.perf_counter() - started, rerun_performed=False))
        raise


if __name__ == '__main__':
    main()
