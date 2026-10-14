"""Create-only, registered source-file transfer study with the fixed M1 kernel."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

import response_proxy as rp
import response_proxy_selection as frozen
import response_proxy_existing as existing
from register_file_transfer import registration

ROOT = Path('/root/auto_data_experiments/20261014_file_transfer_r1')
GRID = (500, 1000, 1500, 2000, 2500, 3000)
METHODS = ('M1', 'RAW', 'SOURCE', 'SHORT')
POLICIES = ('PARENT', 'SIZE_DROP', 'LOSS_DROP')
read, sha, write = frozen.read, frozen.sha, frozen.write
tick, configure = frozen.tick, frozen.configure
snapshot, load_state = frozen.snapshot, frozen.load_state
arrays_save = existing.arrays_save


def job_id(task, engine, action, seed):
    return f'FTR1_{task}_{engine}_{action}_s{seed}'


def expected_jobs(plan):
    jobs = []
    for phase in ('parents', 'forecast', 'labels'):
        for task in plan['tasks']:
            actions = [a['id'] for a in task['actions']]
            for seed in plan['seeds']:
                pairs = [('full', 'PARENT'), ('proxy', 'PARENT')] if phase == 'parents' else (
                    [(engine, action) for action in actions for engine in ('proxy', 'short')]
                    if phase == 'forecast' else [('full', action) for action in actions])
                jobs.extend(dict(id=job_id(task['id'], e, a, seed), task=task['id'], seed=seed,
                                 engine=e, action=a, phase=phase) for e, a in pairs)
    return jobs


def verify(out):
    assert os.name == 'posix' and out == ROOT and out.resolve() == ROOT
    plan = read(out / 'plan.json')
    receipt = read(out / 'source_sync_receipt.json')
    assert receipt['status'] == 'exact_commit_and_tree_verified'
    assert receipt['plan_sha256'] == sha(out / 'plan.json')
    assert receipt['registration_sha256'] == sha(out / 'task_registration.json')
    assert plan['output_root'] == str(ROOT) and plan['stage'] == 'file_transfer_r1'
    assert plan['cumulative_before'] == 409 and plan['expected_cumulative_after'] == 487
    for name, digest in plan['code_sha256'].items():
        assert Path(name).name == name and sha(out / name) == digest, name
    data = plan['dataset']
    assert sha(data['manifest']) == data['manifest_sha256']
    assert sha(data['train']) == data['train_sha256']
    registered = read(out / 'task_registration.json')
    assert registered == registration(read(data['manifest']))
    assert data['manifest_sha256'] == registered['manifest_sha256']
    assert data['train_sha256'] == registered['train_sha256']
    assert plan['tasks'] == registered['tasks'] and plan['seeds'] == registered['seeds']
    assert plan['training'] == dict(input_length=32, prediction_length=8, moving_average_kernel=25)
    assert plan['grid'] == list(GRID) and plan['jobs'] == expected_jobs(plan)
    assert len(plan['jobs']) == 225 and len({j['id'] for j in plan['jobs']}) == 225
    assert sha(plan['core']) == plan['core_sha256'] == '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'
    return plan


def load_source(plan):
    data = plan['dataset']
    manifest = read(data['manifest'])
    train = next(s for s in manifest['splits'] if s['name'] == 'train')
    with Path(data['train']).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == manifest['columns_raw']
        rows = list(reader)
    assert len(rows) == train['rows'] == 32802
    sensors = manifest['sensor_columns']
    assert sensors == data['sensors'] and len(sensors) == 8
    raw = np.asarray([[float(row[c]) for c in sensors] for row in rows], dtype=np.float64)
    assert np.isfinite(raw).all()
    files = np.asarray([row['source_file'] for row in rows])
    source_rows = np.asarray([int(row['source_row']) for row in rows], dtype=np.int64)
    times = np.asarray([row['datetime'] for row in rows], dtype='datetime64[s]').astype(np.int64)
    assert sorted(set(files)) == sorted(s['series_id'] for s in train['series_coverage'])
    for s in train['series_coverage']:
        ids = np.flatnonzero(files == s['series_id'])
        ids = ids[np.argsort(source_rows[ids], kind='stable')]
        np.testing.assert_array_equal(source_rows[ids], np.arange(s['source_row_start_inclusive'], s['source_row_end_exclusive']))
        assert len(ids) == s['rows'] and np.all(np.diff(times[ids]) > 0)
        assert times[ids[0]] == np.datetime64(s['first_time'], 's').astype(np.int64)
        assert times[ids[-1]] == np.datetime64(s['last_time'], 's').astype(np.int64)
    return dict(raw=raw, files=files, source_rows=source_rows, times=times)


def partition(source, names):
    indices = [np.flatnonzero(source['files'] == name) for name in names]
    order = np.concatenate([ids[np.argsort(source['source_rows'][ids], kind='stable')] for ids in indices])
    files, rows, times = (source[key][order] for key in ('files', 'source_rows', 'times'))
    valid_edges = (files[:-1] == files[1:]) & (np.diff(rows) == 1) & (np.diff(times) > 0) & (np.diff(times) <= 30)
    invalid_prefix = np.r_[0, np.cumsum(~valid_edges)]
    starts = np.arange(max(0, len(order) - 39), dtype=np.int64)
    starts = starts[invalid_prefix[starts + 39] == invalid_prefix[starts]]
    assert len(starts) >= 256
    assert all(np.count_nonzero(files[starts] == name) > 0 for name in names)
    assert np.all(files[starts] == files[starts + 39]) and np.all(rows[starts + 39] - rows[starts] == 39)
    return dict(base_rows=order, files=files, source_rows=rows, times=times, starts=starts)


def prepare(out):
    started = time.perf_counter()
    plan = verify(out)
    directory = out / 'prepared'
    directory.mkdir()
    source = load_source(plan)
    summaries = []
    for task in plan['tasks']:
        parts = {s: partition(source, task['files'][s]) for s in ('A', 'B', 'C')}
        assert sum(len(p['base_rows']) for p in parts.values()) == len(source['raw'])
        np.testing.assert_array_equal(np.sort(np.concatenate([p['base_rows'] for p in parts.values()])), np.arange(len(source['raw'])))
        for left, right in (('A', 'B'), ('A', 'C'), ('B', 'C')):
            assert set(parts[left]['files']).isdisjoint(parts[right]['files'])
        fitted = source['raw'][parts['A']['base_rows']]
        mean, scale = fitted.mean(0), fitted.std(0)
        scale[scale == 0] = 1.
        packed = dict(mean=mean, scale=scale)
        spans, counts = {}, {}
        for split, part in parts.items():
            packed[f'{split}_values'] = ((source['raw'][part['base_rows']] - mean) / scale).astype(np.float32)
            for key, value in part.items():
                packed[f'{split}_{key}'] = value
            elapsed = part['times'][part['starts'] + 39] - part['times'][part['starts'] + 31]
            spans[split] = dict(min_seconds=int(elapsed.min()), max_seconds=int(elapsed.max()))
            counts[split] = {name: int(np.count_nonzero(part['files'][part['starts']] == name)) for name in task['files'][split]}
        arrays_save(directory / f'{task["id"]}_ABC.npz', **packed)
        window_files = parts['A']['files'][parts['A']['starts']]
        n = len(window_files)
        arrays_save(directory / f'{task["id"]}_PARENT.npz', probability=np.full(n, 1. / n))
        for action in task['actions']:
            removed = window_files == action['removed_source_file']
            assert removed.any() and np.count_nonzero(~removed) >= 256
            probability = np.zeros(n, dtype=np.float64)
            probability[~removed] = 1. / np.count_nonzero(~removed)
            assert abs(probability.sum() - 1.) < 1e-12
            arrays_save(directory / f'{task["id"]}_{action["id"]}.npz', probability=probability, removed=removed)
        summaries.append(dict(task=task['id'], rows={s: len(p['base_rows']) for s, p in parts.items()},
                              windows={s: len(p['starts']) for s, p in parts.items()},
                              source_window_counts=counts, forecast_elapsed=spans,
                              fixed_parent_normalization=True, mean=mean.tolist(), scale=scale.tolist()))
    verify(out)
    result = dict(status='PASS', plan_sha256=sha(out / 'plan.json'),
                  registration_sha256=sha(out / 'task_registration.json'), tasks=summaries,
                  assets_sha256={p.relative_to(out).as_posix(): sha(p) for p in sorted(directory.iterdir())},
                  seconds=time.perf_counter() - started, new_full_trainings=0, test_access=False, D_access=False)
    write(out / 'preparation.json', result)
    return result


class Context(frozen.Context):
    def __init__(self, out, plan, task):
        self.out, self.plan, self.task = out, plan, task
        self.spec = next(t for t in plan['tasks'] if t['id'] == task)
        self.actions = tuple(a['id'] for a in self.spec['actions'])
        receipt = read(out / 'preparation.json')
        assert receipt['status'] == 'PASS' and receipt['plan_sha256'] == sha(out / 'plan.json')
        name = f'prepared/{task}_ABC.npz'
        assert sha(out / name) == receipt['assets_sha256'][name]
        with np.load(out / name, allow_pickle=False) as z:
            self.values = {s: torch.tensor(z[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
            self.starts = {s: torch.tensor(z[f'{s}_starts'], device='cuda') for s in self.values}
            self.window_files = {s: z[f'{s}_files'][z[f'{s}_starts']] for s in self.values}
        self.core = frozen.module('file_transfer_core', plan['core'])
        self.channels = len(plan['dataset']['sensors'])
        self.offset = torch.arange(40, device='cuda')
        self.n = len(self.starts['A'])
        self.probabilities, self.cdfs = {}, {}
        for action in ('PARENT',) + self.actions:
            name = f'prepared/{task}_{action}.npz'
            assert sha(out / name) == receipt['assets_sha256'][name]
            with np.load(out / name, allow_pickle=False) as z:
                self.probabilities[action] = z['probability'].copy()
            cdf = np.cumsum(self.probabilities[action])
            cdf /= cdf[-1]
            cdf[np.flatnonzero(self.probabilities[action])[-1]:] = 1.
            assert np.all(np.diff(cdf) >= 0)
            self.cdfs[action] = cdf

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.core.DLinear(dict(self.plan['training'], channels=self.channels)).cuda()

    def transform(self, x, action):
        assert action in ('PARENT',) + self.actions
        return x

    def parent_dir(self, engine, seed):
        assert engine in ('full', 'proxy')
        return self.out / 'runs' / job_id(self.task, engine, 'PARENT', seed)


def update_step(ctx, model, opt, rng, draw_count, engine, action, moment):
    model.train()
    opt.zero_grad(set_to_none=True)
    if engine == 'proxy':
        rp.assign_gradient(model, moment)
    else:
        ids = np.searchsorted(ctx.cdfs[action], rng.random(256), side='right')
        np.add.at(draw_count, ids, 1)
        batch = ctx.batch('A', torch.tensor(ids, device='cuda'))
        loss = (model(ctx.transform(batch[:, :32], action)) - batch[:, 32:]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('nonfinite training loss')
        loss.backward()
    opt.step()


def train(ctx, engine, action, seed):
    assert engine in ('full', 'proxy', 'short') and action in ('PARENT',) + ctx.actions
    ident = job_id(ctx.task, engine, action, seed)
    assert any(j['id'] == ident for j in ctx.plan['jobs'])
    maximum = 3000
    if engine == 'short':
        calibration = read(ctx.out / 'calibration.json')
        assert calibration['status'] == 'PASS' and calibration['plan_sha256'] == sha(ctx.out / 'plan.json')
        maximum = calibration['tasks'][ctx.task]['short_steps']
        assert maximum in GRID[:-1]
    directory = ctx.out / 'runs' / ident
    directory.mkdir(parents=True)
    state = dict(status='running', id=ident, task=ctx.task, engine=engine, action=action, seed=seed, steps_completed=0)
    write(directory / 'status.json', state)
    started = tick()
    try:
        model = ctx.model(seed)
        opt = ctx.optimizer(model)
        rng = np.random.default_rng(seed + 1000)
        draw_count = np.zeros(ctx.n, dtype=np.int64)
        snapshot(model, opt, rng, 0, directory / 'initial.pt')
        initial_sha = hashlib.sha256(rp.parameters(model).detach().cpu().numpy().tobytes()).hexdigest()
        moment = ctx.moments(model, action) if engine == 'proxy' else None
        if moment is not None:
            arrays_save(directory / 'moments.npz', gram=moment.gram.cpu().numpy(), cross=moment.cross.cpu().numpy(),
                        target_square=moment.target_square.cpu().numpy())
        scores, curves, checkpoint_steps, prefix_seconds = {}, {}, [], {}
        for step in range(1, maximum + 1):
            update_step(ctx, model, opt, rng, draw_count, engine, action, moment)
            state['steps_completed'] = step
            if step in GRID:
                snapshot(model, opt, rng, step, directory / f'step_{step}.pt')
                checkpoint_steps.append(step)
                if engine != 'proxy':
                    curves[step] = {}
                    for split in ('B', 'C'):
                        curves[step][split], error = ctx.evaluate(model, action, split)
                        arrays_save(directory / f'step_{step}_{split}_errors.npz', error=error)
                    scores[step] = curves[step]['B']
                write(directory / 'status.json', state, replace=True)
                prefix_seconds[step] = tick() - started
        selected = None
        selected_losses = None
        if engine != 'proxy':
            selected = rp.select_checkpoint(scores, checkpoint_steps)
            model.load_state_dict(load_state(directory / f'step_{selected}.pt')['model'])
            selected_losses = {}
            for split in ('B', 'C'):
                selected_losses[split], error = ctx.evaluate(model, action, split)
                arrays_save(directory / f'selected_{split}_errors.npz', error=error)
            assert selected_losses['B'] == scores[selected]
            arrays_save(directory / 'sampling.npz', draw_count=draw_count)
        assets = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.pt', '.npz')}
        result = dict(state, status='completed', steps=maximum, selected_step=selected,
                      selection_losses=scores, selected_losses=selected_losses, checkpoints=checkpoint_steps,
                      curves=curves, prefix_seconds=prefix_seconds, assets_sha256=assets,
                      initial_sha256=initial_sha, plan_sha256=sha(ctx.out / 'plan.json'))
        write(directory / 'result.json', result)
        state['status'] = 'completed'
        write(directory / 'status.json', state, replace=True)
        seconds = tick() - started
        write(directory / 'timing.json', dict(seconds=seconds, boundary='initialization through result/status/assets/hash; clock receipt excluded'))
        return dict(result, seconds=seconds)
    except Exception:
        state.update(status='failed', error=traceback.format_exc(), seconds=time.perf_counter() - started,
                     failure_clock='CPU perf_counter; no CUDA synchronization attempted')
        write(directory / 'status.json', state, replace=True)
        raise


def score_proxy(ctx, action, seed, method):
    assert method in ('M1', 'RAW') and action in ('PARENT',) + ctx.actions
    directory = ctx.out / 'forecasts' / f'{ctx.task}_{action}_{seed}_{method}'
    directory.mkdir(parents=True)
    started = tick()
    model = ctx.model(seed)
    parent = ctx.parent_dir('full', seed)
    p_result = read(parent / 'result.json')
    scores, curves = {}, {}
    for step in GRID:
        child = load_state(ctx.out / 'runs' / job_id(ctx.task, 'proxy', action, seed) / f'step_{step}.pt')
        if method == 'RAW':
            model.load_state_dict(child['model'])
        else:
            real = load_state(parent / f'step_{step}.pt')['original_coordinates']
            proxy_parent = load_state(ctx.parent_dir('proxy', seed) / f'step_{step}.pt')['original_coordinates']
            rp.load_parameters(model, rp.paired_parameters(real, child['original_coordinates'], proxy_parent))
        curves[step] = {}
        for split in ('B', 'C'):
            curves[step][split], error = ctx.evaluate(model, action, split)
            arrays_save(directory / f'step_{step}_{split}_errors.npz', error=error)
        scores[step] = curves[step]['B']
    selected = rp.select_checkpoint(scores, GRID)
    baseline = p_result['selected_losses']['C']
    assert baseline > 0
    result = dict(method=method, task=ctx.task, action=action, seed=seed, selected_step=selected,
                  curves=curves, gain_percent=100. * (baseline - curves[selected]['C']) / baseline,
                  assets_sha256={p.name: sha(p) for p in sorted(directory.iterdir()) if p.suffix == '.npz'})
    write(directory / 'result.json', result)
    seconds = tick() - started
    write(directory / 'timing.json', dict(seconds=seconds))
    return dict(result, seconds=seconds)


def phase_status(out, phase, completed, registered, status='running', **extra):
    write(out / f'{phase}_status.json', dict(status=status, phase=phase, model_jobs_completed=completed,
          registered_model_jobs=registered, **extra), replace=True)
    print(json.dumps(dict(phase=phase, status=status, completed=completed, registered=registered)), flush=True)


def collect_assets(out, directories):
    return {p.relative_to(out).as_posix(): sha(p) for directory in directories
            for p in sorted((out / directory).rglob('*')) if p.is_file()}


def check_preflight(out):
    audit = read(out / 'preflight_audit.json')
    assert audit['status'] == 'PASS' and audit['plan_sha256'] == sha(out / 'plan.json')
    assert audit['preparation_sha256'] == sha(out / 'preparation.json')


def check_calibration(out):
    audit = read(out / 'parents_audit.json')
    receipt = read(out / 'calibration_sync_receipt.json')
    assert audit['status'] == 'PASS' and audit['calibration_sha256'] == sha(out / 'calibration.json')
    assert receipt['status'] == 'exact_commit_and_tree_verified'
    assert receipt['calibration_sha256'] == sha(out / 'calibration.json')
    assert receipt['parents_audit_sha256'] == sha(out / 'parents_audit.json')
    assert receipt['plan_sha256'] == sha(out / 'plan.json')
    calibration = read(out / 'calibration.json')
    assert calibration['status'] == 'PASS'
    for name, digest in calibration['assets_sha256'].items():
        assert sha(out / name) == digest, name
    return calibration


def parents(out):
    started = tick()
    plan = verify(out)
    check_preflight(out)
    write(out / 'parents.lock', dict(plan_sha256=sha(out / 'plan.json')))
    assert not (out / 'runs').exists()
    results, measures, budgets, setup_rows = [], [], {}, []
    for task in plan['tasks']:
        begin = tick(); ctx = Context(out, plan, task['id'])
        setup = dict(task=ctx.task, seconds=tick() - begin)
        write(out / f'{ctx.task}_parent_setup.json', setup); setup_rows.append(setup)
        task_full, task_proxy, task_scores = [], [], []
        for seed in plan['seeds']:
            for engine in ('full', 'proxy'):
                result = train(ctx, engine, 'PARENT', seed)
                results.append(result)
                (task_full if engine == 'full' else task_proxy).append(result)
                phase_status(out, 'parents', len(results), 18)
            score = score_proxy(ctx, 'PARENT', seed, 'M1')
            parent = task_full[-1]
            assert score['gain_percent'] == 0. and score['selected_step'] == parent['selected_step']
            assert score['curves'] == parent['curves']
            task_scores.append(score)
            measures.append(dict(task=ctx.task, seed=seed, identity_score_seconds=score['seconds']))
        k = len(ctx.actions)
        p = max(r['seconds'] for r in task_proxy)
        s = max(r['seconds'] for r in task_scores)
        c = (1. + 1. / k) * p + s
        minimum_prefix = {j: min(r['prefix_seconds'][j] for r in task_full) for j in GRID[:-1]}
        valid = [j for j in GRID[:-1] if minimum_prefix[j] >= 1.25 * c]
        budgets[ctx.task] = dict(actions=k, max_proxy_parent_seconds=p, max_identity_score_seconds=s,
                                estimated_m1_seconds_per_candidate=c, calibration_margin=1.25,
                                minimum_parent_prefix_seconds=minimum_prefix,
                                short_steps=valid[0] if valid else None)
        del ctx
    assert len(results) == 18 and len(measures) == 9
    verify(out)
    inputs = collect_assets(out, ('runs', 'forecasts'))
    inputs.update({f'{t["id"]}_parent_setup.json': sha(out / f'{t["id"]}_parent_setup.json') for t in plan['tasks']})
    passed = all(b['short_steps'] is not None for b in budgets.values())
    result = dict(status='PASS' if passed else 'STOP_NO_FEASIBLE_SHORT_PREFIX', plan_sha256=sha(out / 'plan.json'),
                  tasks=budgets, runs=results, measures=measures, setups=setup_rows, assets_sha256=inputs,
                  new_full_trainings=9, cumulative_full_trainings=418, phase_seconds=tick() - started)
    write(out / 'calibration.json', result)
    phase_status(out, 'parents', 18, 18, status='completed' if passed else 'stopped',
                 new_full_trainings=9, cumulative_full_trainings=418, calibration_passed=passed)
    return result


@torch.no_grad()
def cheap_loss(ctx, seed):
    directory = ctx.out / 'cheap' / f'{ctx.task}_s{seed}'
    directory.mkdir(parents=True)
    started = tick()
    parent = ctx.parent_dir('full', seed)
    result = read(parent / 'result.json')
    model = ctx.model(seed)
    model.load_state_dict(load_state(parent / f'step_{result["selected_step"]}.pt')['model'])
    model.eval()
    error = []
    for lo in range(0, ctx.n, 2048):
        batch = ctx.batch('A', torch.arange(lo, min(lo + 2048, ctx.n), device='cuda'))
        error.append((model(batch[:, :32]) - batch[:, 32:]).cpu().numpy())
    error = np.concatenate(error)
    assert np.isfinite(error).all()
    losses = {name: float(np.mean(error[ctx.window_files['A'] == name].astype(np.float64) ** 2))
              for name in ctx.spec['files']['A']}
    arrays_save(directory / 'A_errors.npz', error=error)
    output = dict(task=ctx.task, seed=seed, selected_parent_step=result['selected_step'], file_losses=losses,
                  error_summary_sha256=sha(directory / 'A_errors.npz'))
    write(directory / 'result.json', output)
    seconds = tick() - started
    write(directory / 'timing.json', dict(seconds=seconds))
    return dict(output, seconds=seconds)


def recommend(means, actions):
    assert list(means) == list(actions) and all(math.isfinite(v) for v in means.values())
    best = max(actions, key=lambda a: means[a])
    return best if means[best] >= 1. else 'PARENT'


def forecast(out):
    from file_transfer_source import run_source
    started = tick()
    plan = verify(out)
    check_preflight(out)
    calibration = check_calibration(out)
    write(out / 'forecast.lock', dict(plan_sha256=sha(out / 'plan.json'), calibration_sha256=sha(out / 'calibration.json')))
    records, costs, setups, cheap_rows = [], [], [], []
    policies, recommendations = {}, {}
    completed = 0
    prep_seconds = read(out / 'preparation.json')['seconds']
    for task in plan['tasks']:
        begin = tick(); ctx = Context(out, plan, task['id'])
        setup = dict(task=ctx.task, seconds=tick() - begin)
        write(out / f'{ctx.task}_forecast_setup.json', setup); setups.append(setup)
        task_records = []
        for seed in plan['seeds']:
            row_records = []
            method_costs = {name: 0. for name in METHODS}
            proxy_parent_seconds = read(ctx.parent_dir('proxy', seed) / 'timing.json')['seconds']
            method_costs['M1'] = proxy_parent_seconds
            real_parent = read(ctx.parent_dir('full', seed) / 'result.json')
            parent_seconds = read(ctx.parent_dir('full', seed) / 'timing.json')['seconds']
            for action in ctx.actions:
                proxy = train(ctx, 'proxy', action, seed); completed += 1
                phase_status(out, 'forecast', completed, 138)
                short = train(ctx, 'short', action, seed); completed += 1
                assert short['steps'] < 3000
                phase_status(out, 'forecast', completed, 138)
                predictions = {method: score_proxy(ctx, action, seed, method) for method in ('M1', 'RAW')}
                predictions['SHORT'] = dict(method='SHORT', task=ctx.task, action=action, seed=seed,
                    selected_step=short['selected_step'], curves=short['curves'],
                    gain_percent=100. * (real_parent['selected_losses']['C'] - short['selected_losses']['C']) / real_parent['selected_losses']['C'])
                for method in ('M1', 'RAW'):
                    method_costs[method] += proxy['seconds'] + predictions[method]['seconds']
                method_costs['SHORT'] += short['seconds']
                row_records.append(dict(task=ctx.task, seed=seed, action=action, predictions=predictions))
            source = run_source(ctx, seed)
            assert len(source['records']) == len(ctx.actions)
            for row in row_records:
                row['predictions']['SOURCE'] = next(r for r in source['records'] if r['action'] == row['action'])
                assert set(row['predictions']) == set(METHODS)
            method_costs['SOURCE'] = source['total_seconds']
            loss_info = cheap_loss(ctx, seed); cheap_rows.append(loss_info)
            method_costs.update(PARENT=0., LOSS_DROP=loss_info['seconds'])
            common = prep_seconds / 9 + (setup['seconds'] + read(out / f'{ctx.task}_parent_setup.json')['seconds']) / 3
            identity_seconds = next(m['identity_score_seconds'] for m in calibration['measures'] if m['task'] == ctx.task and m['seed'] == seed)
            costs.append(dict(task=ctx.task, seed=seed, warm=method_costs,
                              common_preparation_allocation_seconds=common, real_parent_seconds=parent_seconds,
                              proxy_parent_seconds=proxy_parent_seconds, identity_calibration_seconds=identity_seconds,
                              m1_not_more_expensive_than_short=method_costs['M1'] <= method_costs['SHORT']))
            records.extend(row_records); task_records.extend(row_records)
        begin = time.perf_counter()
        counts = {a['id']: int(np.count_nonzero(ctx.window_files['A'] == a['removed_source_file'])) for a in task['actions']}
        size_choice = max(ctx.actions, key=lambda a: counts[a])
        size_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        losses = {a['id']: float(np.mean([r['file_losses'][a['removed_source_file']] for r in cheap_rows if r['task'] == ctx.task]))
                  for a in task['actions']}
        loss_choice = max(ctx.actions, key=lambda a: losses[a])
        loss_reduction_seconds = time.perf_counter() - begin
        policies[ctx.task] = dict(PARENT='PARENT', SIZE_DROP=size_choice, LOSS_DROP=loss_choice,
                                  window_counts=counts, mean_file_losses=losses,
                                  size_seconds=size_seconds, loss_reduction_seconds=loss_reduction_seconds)
        recommendations[ctx.task] = {}
        for method in METHODS:
            means = {a: float(np.mean([r['predictions'][method]['gain_percent'] for r in task_records if r['action'] == a])) for a in ctx.actions}
            recommendations[ctx.task][method] = dict(action=recommend(means, ctx.actions), means=means)
        for method in POLICIES:
            recommendations[ctx.task][method] = dict(action=policies[ctx.task][method])
        for cost in costs:
            if cost['task'] != ctx.task:
                continue
            cost['warm']['SIZE_DROP'] = size_seconds / 3
            cost['warm']['LOSS_DROP'] += loss_reduction_seconds / 3
            common = cost['common_preparation_allocation_seconds'] + cost['real_parent_seconds']
            cost['cold_required_computation'] = {m: common + value for m, value in cost['warm'].items()}
            cost['cold_registered_execution'] = {m: common + value + cost['identity_calibration_seconds'] +
                (0. if m == 'M1' else cost['proxy_parent_seconds']) for m, value in cost['warm'].items()}
        del ctx
    assert completed == 138 and len(records) == 69 and len(costs) == 9 and len(cheap_rows) == 9
    agreement = [method for method in ('RAW', 'SOURCE', 'SHORT') + POLICIES
                 if all(recommendations[t['id']]['M1']['action'] == recommendations[t['id']][method]['action'] for t in plan['tasks'])]
    cost_gate = all(c['m1_not_more_expensive_than_short'] for c in costs)
    prediction_gate = cost_gate and not agreement
    verify(out); check_calibration(out)
    assets = collect_assets(out, ('runs', 'forecasts', 'source', 'cheap'))
    assets.update({f'{t["id"]}_forecast_setup.json': sha(out / f'{t["id"]}_forecast_setup.json') for t in plan['tasks']})
    result = dict(status='completed', plan_sha256=sha(out / 'plan.json'), calibration_sha256=sha(out / 'calibration.json'),
                  records=records, costs=costs, setups=setups, cheap_records=cheap_rows, policies=policies,
                  recommendations=recommendations, cost_gate=cost_gate, identical_recommendation_baselines=agreement,
                  eligible_to_label=prediction_gate, assets_sha256=assets, phase_seconds=tick() - started,
                  new_full_trainings=0, cumulative_full_trainings=418, full_candidate_responses_observed=False,
                  cost_definitions='warm includes each method own moments/features/trajectory/scoring; cold_required_computation adds allocated common prep/load and real parent; cold_registered_execution additionally charges this study cost-calibration identity scores and proxy parents without double-counting M1. Allocations are not separate measured process starts.')
    write(out / 'predictions.json', result)
    phase_status(out, 'forecast', completed, 138, status='completed' if prediction_gate else 'stopped',
                 eligible_to_label=prediction_gate, new_full_trainings=0, cumulative_full_trainings=418)
    return result


def labels(out):
    started = tick()
    plan = verify(out)
    check_preflight(out); check_calibration(out)
    predictions = read(out / 'predictions.json')
    audit = read(out / 'forecast_audit.json')
    receipt = read(out / 'prediction_sync_receipt.json')
    assert predictions['eligible_to_label'] and predictions['cost_gate'] and not predictions['identical_recommendation_baselines']
    assert audit['status'] == 'PASS' and audit['predictions_sha256'] == sha(out / 'predictions.json')
    assert receipt['status'] == 'exact_commit_and_tree_verified'
    assert receipt['plan_sha256'] == sha(out / 'plan.json')
    assert receipt['predictions_sha256'] == sha(out / 'predictions.json')
    assert receipt['forecast_audit_sha256'] == sha(out / 'forecast_audit.json')
    for name, digest in predictions['assets_sha256'].items():
        assert sha(out / name) == digest, name
    write(out / 'labels.lock', dict(predictions_sha256=sha(out / 'predictions.json'), source_commit=receipt['source_commit']))
    results, setups = [], []
    for task in plan['tasks']:
        begin = tick(); ctx = Context(out, plan, task['id'])
        setup = dict(task=ctx.task, seconds=tick() - begin)
        write(out / f'{ctx.task}_label_setup.json', setup); setups.append(setup)
        for seed in plan['seeds']:
            for action in ctx.actions:
                results.append(train(ctx, 'full', action, seed))
                phase_status(out, 'labels', len(results), 69)
        del ctx
    assert len(results) == 69 and all(r['steps'] == 3000 for r in results)
    verify(out)
    for name, digest in predictions['assets_sha256'].items():
        assert sha(out / name) == digest, name
    result = dict(status='completed', plan_sha256=sha(out / 'plan.json'), predictions_sha256=sha(out / 'predictions.json'),
                  runs=results, setups=setups, new_full_trainings=69, cumulative_full_trainings=487,
                  phase_seconds=tick() - started, scientific_audit_pending=True)
    write(out / 'labels.json', result)
    phase_status(out, 'labels', 69, 69, status='completed', new_full_trainings=69, cumulative_full_trainings=487)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'parents', 'forecast', 'labels'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configure()
    try:
        globals()[args.phase](args.output)
    except Exception:
        if os.name == 'posix' and args.output == ROOT and args.output.resolve() == ROOT and args.output.is_dir():
            write(args.output / f'{args.phase}_failure.json', dict(status='failed', phase=args.phase, error=traceback.format_exc()))
        raise
