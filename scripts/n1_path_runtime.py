"""Create-only N1 engineering on nine frozen real Adam paths; no children."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import time
import traceback
import hashlib
import os

import numpy as np
import torch
import fourgroup_runtime as rt
import response_tangent_torch as tangent

ROOT = Path('/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment')
STEPS = (0, 1, 2, 3, 500, 501, 1000, 1500, 2000, 2500, 3000)
SHAPE = (8, 66, 8)
PARAMETERS = 8 * 66 * 8


def verify(out):
    if not __debug__ or os.name != 'posix':
        raise RuntimeError('Unoptimized dedicated Linux runtime required')
    assert out == ROOT and out.resolve() == ROOT
    plan = rt.read(out / 'plan.json')
    source = rt.read(out / 'source_sync_receipt.json')
    activation = rt.read(out / 'execution_activation.json')
    assert plan['stage'] == 'n1_path_engineering_r1' and plan['output_root'] == str(ROOT)
    assert plan['seeds'] == list(rt.SEEDS) and plan['grid'] == list(rt.GRID)
    assert [t['id'] for t in plan['tasks']] == list(rt.TASKS)
    assert plan['cumulative_before'] == 418 and not plan['scientific_children_enabled']
    assert plan['training'] == dict(input_length=32, prediction_length=8,
        moving_average_kernel=25, batch_size=256, steps=3000, learning_rate=.001, weight_decay=0.)
    assert source['status'] == 'exact_commit_and_tree_verified'
    assert source['plan_sha256'] == rt.sha(out / 'plan.json')
    assert activation['status'] == 'inputs_environment_storage_verified'
    for key, name in (('plan_sha256', 'plan.json'), ('input_lock_sha256', 'input_lock.json'),
                      ('storage_preflight_sha256', 'storage_preflight.json'),
                      ('environment_sha256', 'runtime_environment.json')):
        assert activation[key] == rt.sha(rt.safe_file(out, name))
    assert activation['source_commit'] == source['source_commit']
    assert rt.read(out / 'runtime_environment.json')['environment'] == plan['expected_environment'] == rt.environment()
    storage = rt.read(out / 'storage_preflight.json')
    assert storage['status'] == 'PASS' and storage['plan_sha256'] == source['plan_sha256']
    assert storage['required_free_bytes'] == plan['preflight_min_free_bytes']
    assert storage['free_before_bytes'] >= storage['required_free_bytes']
    assert storage['probe_removed'] and storage['probe_bytes'] == 64 * 1024 * 1024
    assert storage['write_sha256'] == storage['read_sha256']
    assert rt.sha(out / 'storage_estimate.json') == plan['storage_estimate_sha256']
    assert rt.sha(out / 'input_lock.json') == plan['input_lock_sha256']
    for name, digest in plan['code_sha256'].items():
        assert rt.sha(rt.safe_file(out, name)) == digest, name
    for item in rt.read(out / 'input_lock.json')['files']:
        path = rt.safe_file(out, item['path'])
        assert path.stat().st_size == item['bytes'] and rt.sha(path) == item['sha256']
    assert not (out / 'prepared').exists()  # Do not activate the closed M1 menu.
    return plan


def pack(model, optimizer=None, field=None):
    values = {}
    for name, p in model.named_parameters():
        if field == 'gradient':
            value = p.grad
        elif field is None:
            value = p
        else:
            value = optimizer.state[p].get(field, torch.zeros_like(p))
        values[name] = value.detach()
    return torch.stack([torch.cat((values[f'seasonal.{c}.weight'].T,
        values[f'seasonal.{c}.bias'][None], values[f'trend.{c}.weight'].T,
        values[f'trend.{c}.bias'][None])) for c in range(8)]).double()


class Context(rt.Context):
    def __init__(self, out, plan, task):
        super().__init__(out, plan, task)
        with np.load(out / 'inputs/prepared' / f'{task}_ABC.npz', allow_pickle=False) as data:
            lo = next(t['partitions']['A'][0] for t in plan['tasks'] if t['id'] == task)
            cycles = data['A_cycles'][data['A_starts']]
            groups = (cycles - lo) // 77
            np.testing.assert_array_equal(np.bincount(groups), np.full(4, 1617))
            self.groups = torch.tensor(groups, device='cuda', dtype=torch.int64)
        self.phi, self.targets = {}, {}
        for split in ('A', 'B', 'C'):
            xx, yy = [], []
            for lo in range(0, len(self.starts[split]), 2048):
                ids = torch.arange(lo, min(lo + 2048, len(self.starts[split])), device='cuda')
                batch = self.batch(split, ids)
                xx.append(rt.rp.features(SimpleNamespace(kernel=25), batch[:, :32]).double())
                yy.append(batch[:, 32:].double())
            self.phi[split], self.targets[split] = torch.cat(xx), torch.cat(yy)


def initialization(ctx, seed):
    model = ctx.model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rng = np.random.default_rng(seed + 1000)
    rt.equal(rt.snapshot(model, optimizer, rng, 0), rt.load_snapshot(ctx.parent('full', seed), 0))
    return model, optimizer, rng


def parent_step(ctx, model, optimizer, rng, draws):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    ids = np.searchsorted(ctx.cdf, rng.random(256), side='right')
    np.add.at(draws, ids, 1)
    index = torch.tensor(ids, device='cuda')
    batch = ctx.batch('A', index)
    squared = (model(batch[:, :32]) - batch[:, 32:]).square()
    loss = squared.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite parent loss')
    loss.backward()
    return index


def save_state(directory, step, model, optimizer, rng, state, history):
    name = 'initial' if step == 0 else f'step_{step}'
    rt.save_snapshot(directory / f'{name}.pt', model, optimizer, rng, step)
    rt.arrays(directory / f'{name}_tangent.npz', **{k: getattr(state, k).cpu().numpy() for k in ('S', 'M', 'V')},
        history_all_zero=history.cpu().numpy())


@torch.no_grad()
def response(ctx, model, state, split):
    model.eval()
    residual = []
    for lo in range(0, len(ctx.starts[split]), 2048):
        ids = torch.arange(lo, min(lo + 2048, len(ctx.starts[split])), device='cuda')
        batch = ctx.batch(split, ids)
        residual.append((model(batch[:, :32]) - batch[:, 32:]).cpu().numpy())
    residual = np.concatenate(residual)
    jacobian = torch.einsum('bcd,cdhj->bhcj', ctx.phi[split], state.S.reshape(*SHAPE, 4)).cpu().numpy()
    r = residual.astype(np.float64)
    n = r.size
    values = dict(cross=np.einsum('bhc,bhcj->j', r, jacobian) / n,
        gram=np.einsum('bhcj,bhck->jk', jacobian, jacobian) / n,
        channel_cross=np.einsum('bhc,bhcj->cj', r, jacobian) / (r.shape[0]*r.shape[1]),
        channel_gram=np.einsum('bhcj,bhck->cjk', jacobian, jacobian) / (r.shape[0]*r.shape[1]))
    meta = dict(residual=rt.metric(residual), jacobian_shape=list(jacobian.shape),
        jacobian_dtype=str(jacobian.dtype), jacobian_byte_sha256=hashlib.sha256(jacobian.tobytes()).hexdigest())
    return meta, values


def train_path(ctx, seed):
    assert shutil.disk_usage(ctx.out).free >= ctx.plan['runtime_min_free_bytes']
    directory = ctx.out / 'paths' / f'N1_{ctx.task}_s{seed}'
    directory.mkdir(parents=True)
    started = rt.tick()
    status = dict(status='running', task=ctx.task, seed=seed, steps_completed=0, new_full_trainings=0)
    rt.write(directory / 'status.json', status)
    timings = dict(parent_update_seconds=0., tangent_seconds=0., evaluation_write_hash_seconds=0., comparison_seconds=0., diagnostic_write_seconds=0.)
    try:
        model, optimizer, rng = initialization(ctx, seed)
        state = tangent.TangentState.zeros(PARAMETERS, torch.device('cuda'))
        history = torch.ones(PARAMETERS, dtype=torch.bool, device='cuda')
        draws = np.zeros(ctx.n, dtype=np.int64)
        save_state(directory, 0, model, optimizer, rng, state, history)
        for step in range(1, 3001):
            clock = rt.tick()
            theta = pack(model)
            m, v = pack(model, optimizer, 'exp_avg').flatten(), pack(model, optimizer, 'exp_avg_sq').flatten()
            timings['tangent_seconds'] += rt.tick() - clock
            clock = rt.tick()
            index = parent_step(ctx, model, optimizer, rng, draws)
            timings['parent_update_seconds'] += rt.tick() - clock
            clock = rt.tick()
            gradient = pack(model, field='gradient').flatten()
            _, hs, forcing = tangent.batch_terms(ctx.phi['A'][index], ctx.targets['A'][index],
                theta, ctx.groups[index], state.S)
            timings['tangent_seconds'] += rt.tick() - clock
            clock = rt.tick()
            optimizer.step()
            timings['parent_update_seconds'] += rt.tick() - clock
            clock = rt.tick()
            state = tangent.tangent_step(theta.flatten(), m, v, step-1, gradient, hs, forcing, state,
                history_all_zero=history, updated_parent_m=pack(model, optimizer, 'exp_avg').flatten(),
                updated_parent_v=pack(model, optimizer, 'exp_avg_sq').flatten())
            history &= gradient == 0
            timings['tangent_seconds'] += rt.tick() - clock
            status['steps_completed'] = step
            if step in STEPS:
                clock = rt.tick()
                save_state(directory, step, model, optimizer, rng, state, history)
                if step in rt.GRID:
                    for split in ('B', 'C'):
                        meta, values = response(ctx, model, state, split)
                        rt.write(directory / f'step_{step}_{split}.json', meta)
                        rt.arrays(directory / f'step_{step}_{split}.npz', **values)
                rt.write(directory / 'status.json', status, update=True)
                field = 'evaluation_write_hash_seconds' if step in rt.GRID else 'diagnostic_write_seconds'
                timings[field] += rt.tick() - clock
            if step in rt.GRID:
                clock = rt.tick()
                rt.equal(rt.snapshot(model, optimizer, rng, step), rt.load_snapshot(ctx.parent('full', seed), step))
                timings['comparison_seconds'] += rt.tick() - clock
                print(dict(task=ctx.task, seed=seed, step=step, parent_bitexact=True), flush=True)
        clock = rt.tick()
        rt.arrays(directory / 'sampling.npz', draw_count=draws)
        assets = {p.name: rt.sha(p) for p in sorted(directory.iterdir()) if p.name != 'status.json'}
        result = dict(status, status='completed', saved_steps=list(STEPS), parent_bitexact=True,
            assets_sha256=assets, plan_sha256=rt.sha(ctx.out / 'plan.json'))
        rt.write(directory / 'result.json', result)
        rt.write(directory / 'status.json', dict(status, status='completed'), update=True)
        timings['evaluation_write_hash_seconds'] += rt.tick() - clock
        elapsed = rt.tick() - started
        timings.update(actual_elapsed_seconds=elapsed, method_seconds=elapsed-timings['comparison_seconds']-timings['diagnostic_write_seconds'],
            boundary='initialization through result/status/assets hashes; bitexact comparisons and extra witness writes excluded; clock receipt excluded')
        rt.write(directory / 'timing.json', timings)
        return dict(task=ctx.task, seed=seed, directory=directory.relative_to(ctx.out).as_posix(), **timings)
    except Exception:
        rt.write(directory / 'status.json', dict(status, status='failed', error=traceback.format_exc(),
            actual_elapsed_seconds=time.perf_counter()-started), update=True)
        raise


def paths(out, plan, verification_seconds):
    audit = rt.read(out / 'preflight_audit.json')
    assert audit['status'] == 'PASS' and audit['plan_sha256'] == rt.sha(out / 'plan.json')
    rt.write(out / 'paths.lock', dict(started_unix=time.time(), plan_sha256=rt.sha(out / 'plan.json')))
    started = rt.tick()
    all_runs, preparation = [], {}
    for task in rt.TASKS:
        clock = rt.tick()
        ctx = Context(out, plan, task)
        preparation[task] = rt.tick() - clock
        for seed in rt.SEEDS:
            all_runs.append(train_path(ctx, seed))
        del ctx
    rows = []
    for task in rt.TASKS:
        lower = verification_seconds / 3 + preparation[task] + sum(r['method_seconds'] for r in all_runs if r['task'] == task)
        budget = plan['M1_task_budget_seconds'][task]
        rows.append(dict(task=task, necessary_method_seconds=lower, M1_seconds=budget, lower_bound_pass=lower <= budget))
    result = dict(status='completed_pending_independent_audit', runs=all_runs, preparation_seconds=preparation,
        input_verification_seconds=verification_seconds, costs=rows,
        all_cost_lower_bounds_pass=all(r['lower_bound_pass'] for r in rows),
        new_full_trainings=0, parent_replays=9, cumulative_full_trainings=418,
        actual_phase_seconds=rt.tick()-started+verification_seconds, plan_sha256=rt.sha(out / 'plan.json'))
    rt.write(out / 'paths.json', result)
    return result
