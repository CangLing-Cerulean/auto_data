"""Create-only existing-source M1 verification; frozen update/score semantics."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
import response_proxy as rp
import response_proxy_selection as frozen

GRID, METHODS, ACTIONS = frozen.GRID, frozen.METHODS, frozen.ACTIONS
read, sha, write = frozen.read, frozen.sha, frozen.write
tick, configure, snapshot, load_state = frozen.tick, frozen.configure, frozen.snapshot, frozen.load_state
category, recommendation = frozen.category, frozen.recommendation


def job_id(task, engine, action, seed):
    return f'EXSR1_{task}_{engine}_{action}_s{seed}'


def arrays_save(path, **arrays):
    if set(arrays) == {'error'}:
        error = arrays['error']
        arrays = dict(error_shape=np.asarray(error.shape), error_dtype=np.asarray(error.dtype.str),
            error_sha256=np.asarray(hashlib.sha256(error.tobytes()).hexdigest()),
            nmse=np.asarray(np.mean(error.astype(np.float64) ** 2)),
            nmae=np.asarray(np.mean(np.abs(error.astype(np.float64)))))
    frozen.arrays_save(path, **arrays)


def expected_jobs(plan):
    jobs = []
    for phase in ('parents', 'forecast', 'labels'):
        for task in plan['tasks']:
            for seed in plan['seeds']:
                pairs = [('full', 'PARENT'), ('proxy', 'PARENT')] if phase == 'parents' else (
                    [(e, a) for a in ACTIONS for e in ('proxy', 'short')] if phase == 'forecast' else [('full', a) for a in ACTIONS])
                jobs.extend(dict(id=job_id(task['id'], e, a, seed), task=task['id'], seed=seed,
                                 engine=e, action=a, phase=phase) for e, a in pairs)
    return jobs


def verify(out):
    plan = read(out / 'plan.json')
    assert read(out / 'source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest, name
    for task in plan['tasks']:
        assert sha(task['manifest']) == task['manifest_sha256']
        assert sha(task['train']) == task['train_sha256']
    assert sha(plan['core']) == plan['core_sha256']
    assert plan['grid'] == list(GRID) and plan['actions'] == list(ACTIONS)
    assert plan['jobs'] == expected_jobs(plan) and len(plan['jobs']) == 84
    assert plan['cumulative_before'] == 367
    # Explicit copies keep the old registered runner unchanged; prevent semantic drift.
    new = {n.name: n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    old = {n.name: n for n in ast.parse((out / 'response_proxy_selection.py').read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in ('train', 'score_proxy'):
        assert ast.dump(new[name], include_attributes=False) == ast.dump(old[name], include_attributes=False)
    return plan


def load_source(task):
    manifest = read(task['manifest'])
    entry = next(s for s in manifest['splits'] if s['name'] == 'train')
    assert task['train_sha256'] == entry['sha256'] and sha(task['train']) == entry['sha256']
    with Path(task['train']).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert len(rows) == entry['rows']
    sensors = manifest['sensor_columns']
    assert sensors == task['sensors']
    values = np.asarray([[float(row[c]) for c in sensors] for row in rows], dtype=np.float64)
    assert np.isfinite(values).all()
    metro = manifest['dataset_id'] == 'metropt3_v1'
    times = np.asarray([r['timestamp' if metro else 'datetime'] for r in rows], dtype='datetime64[s]').astype(np.int64)
    files = np.asarray(['metropt3'] * len(rows) if metro else [r['source_file'] for r in rows])
    source_rows = np.arange(len(rows), dtype=np.int64) if metro else np.asarray([int(r['source_row']) for r in rows])
    original_indices = np.asarray([int(r['']) for r in rows]) if metro else source_rows.copy()
    return dict(raw=values, times=times, files=files, source_rows=source_rows, original_indices=original_indices)


def partition_source(source):
    parts = {s: [] for s in ('A', 'B', 'C')}
    groups, bounds = {}, []
    for name in sorted(set(source['files'])):
        ids = np.flatnonzero(source['files'] == name)
        ids = ids[np.argsort(source['source_rows'][ids], kind='stable')]
        assert np.all(np.diff(source['source_rows'][ids]) == 1)
        assert np.all(np.diff(source['times'][ids]) > 0)
        n = len(ids)
        a, b = n * 3 // 5, n * 4 // 5
        assert min(a // 2, a - a // 2, b - a, n - b) >= 40
        for split, chosen in zip(parts, (ids[:a], ids[a:b], ids[b:])):
            parts[split].extend(chosen.tolist())
        groups[name] = set(ids[:a // 2].tolist())
        bounds.append(dict(file=str(name), rows=n, A=[0, a], B=[a, b], C=[b, n], A_midpoint=a // 2))
    output = {}
    for split, raw_ids in parts.items():
        ids = np.asarray(raw_ids, dtype=np.int64)
        files, rows, times = (source[k][ids] for k in ('files', 'source_rows', 'times'))
        first = np.asarray([int(i) in groups[f] for i, f in zip(ids, files)]) if split == 'A' else np.zeros(len(ids), dtype=bool)
        breaks = (files[1:] != files[:-1]) | (np.diff(rows) != 1) | (np.diff(times) <= 0) | (np.diff(times) > 30)
        if split == 'A':
            breaks |= first[1:] != first[:-1]
        prefix = np.r_[0, np.cumsum(breaks)]
        starts = np.arange(max(0, len(ids) - 39), dtype=np.int64)
        starts = starts[prefix[starts + 39] == prefix[starts]]
        assert len(starts) >= 256
        output[split] = dict(raw=source['raw'][ids], times=times, files=files, source_rows=rows,
            original_indices=source['original_indices'][ids], base_rows=ids, starts=starts, first=first)
    assert len(set(np.concatenate([v['base_rows'] for v in output.values()]).tolist())) == len(source['raw'])
    return output, bounds


def prepare(out):
    started = time.perf_counter()
    plan = verify(out)
    assert shutil.disk_usage(out).free >= 2_000_000_000, 'Need 2 GB reserve before preparation'
    destination = out / 'prepared'
    destination.mkdir()
    entries = []
    for task in plan['tasks']:
        source = load_source(task)
        parts, bounds = partition_source(source)
        mean = parts['A']['raw'].mean(0)
        scale = parts['A']['raw'].std(0)
        scale = np.where(scale == 0., 1., scale)
        arrays = dict(mean=mean, scale=scale)
        for split, part in parts.items():
            part['values'] = ((part.pop('raw') - mean) / scale).astype(np.float32)
            assert np.isfinite(part['values']).all()
            arrays.update({f'{split}_{k}': v for k, v in part.items()})
        arrays_save(destination / f'{task["id"]}_ABC.npz', **arrays)
        first = parts['A']['first'][parts['A']['starts']]
        assert 0 < first.sum() < len(first)
        for action in ('PARENT',) + ACTIONS:
            if action == 'PARENT':
                p = np.full(len(first), 1. / len(first))
            else:
                chosen = first if action.endswith('FIRST') else ~first
                total = 1. if action.startswith('KEEP') else .75
                p = np.where(chosen, total / chosen.sum(), (1. - total) / (~chosen).sum())
            assert abs(p.sum() - 1.) < 1e-13
            arrays_save(destination / f'{task["id"]}_{action}.npz', probability=p)
        entries.append(dict(task=task['id'], file_boundaries=bounds,
            windows={s: len(p['starts']) for s, p in parts.items()}, rows={s: len(p['values']) for s, p in parts.items()},
            first_windows=int(first.sum()), second_windows=int((~first).sum()),
            forecast_elapsed_seconds={s: dict(min=int((p['times'][p['starts']+39]-p['times'][p['starts']+31]).min()),
                max=int((p['times'][p['starts']+39]-p['times'][p['starts']+31]).max())) for s, p in parts.items()}))
        del source, parts, arrays
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in sorted(destination.iterdir())}
    verify(out)
    write(out / 'preparation.json', dict(status='PASS', tasks=entries, assets_sha256=hashes,
        seconds=time.perf_counter()-started, plan_sha256=sha(out/'plan.json'), test_access=False,
        base_validation_access=False, labels_inspected=False, new_full_trainings=0))


class Context(frozen.Context):
    def __init__(self, out, plan, task):
        self.out, self.plan, self.task = out, dict(plan), task
        receipt = read(out/'preparation.json')
        assert receipt['status'] == 'PASS' and receipt['plan_sha256'] == sha(out/'plan.json')
        name = f'prepared/{task}_ABC.npz'
        assert sha(out/name) == receipt['assets_sha256'][name]
        with np.load(out/name, allow_pickle=False) as z:
            self.values = {s: torch.tensor(z[f'{s}_values'], device='cuda') for s in ('A','B','C')}
            self.starts = {s: torch.tensor(z[f'{s}_starts'], device='cuda') for s in self.values}
        self.core = frozen.module('core', plan['core'])
        self.channels = len(next(t['sensors'] for t in plan['tasks'] if t['id'] == task))
        self.offset = torch.arange(40, device='cuda')
        self.n = len(self.starts['A'])
        self.probabilities, self.cdfs = {}, {}
        for action in ('PARENT',) + ACTIONS:
            name = f'prepared/{task}_{action}.npz'
            assert sha(out/name) == receipt['assets_sha256'][name]
            with np.load(out/name, allow_pickle=False) as z:
                self.probabilities[action] = z['probability']
            cdf = np.cumsum(self.probabilities[action]); cdf[-1] = 1.
            self.cdfs[action] = cdf
        if (out/'calibration.json').exists():
            budget = read(out/'calibration.json')['budgets'][task]
            self.plan['short_training_boundary_seconds'] = budget['short_training_boundary_seconds']

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.core.DLinear(dict(self.plan['training'], channels=self.channels)).cuda()

    def parent_dir(self, engine, seed):
        assert engine in ('full', 'proxy')
        return self.out/'runs'/job_id(self.task, engine, 'PARENT', seed)


def train(ctx, engine, action, seed):
    ident = job_id(ctx.task, engine, action, seed)
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
            arrays_save(directory / 'moments.npz', gram=moment.gram.cpu().numpy(),
                        cross=moment.cross.cpu().numpy(), target_square=moment.target_square.cpu().numpy())
        scores, checkpoint_steps, short_C = {}, [], {}
        for step in range(1, 3001):
            if engine == 'short' and (step - 1) % 25 == 0 and tick() - started >= ctx.plan['short_training_boundary_seconds']:
                break
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
            state['steps_completed'] = step
            if step in GRID:
                snapshot(model, opt, rng, step, directory / f'step_{step}.pt')
                checkpoint_steps.append(step)
                if engine != 'proxy':
                    scores[step] = ctx.evaluate(model, action, 'B')[0]
                    if engine == 'short':
                        short_C[step], error = ctx.evaluate(model, action, 'C')
                        arrays_save(directory / f'step_{step}_C_errors.npz', error=error)
                write(directory / 'status.json', state, replace=True)
        completed_steps = state['steps_completed']
        if engine == 'short' and completed_steps not in checkpoint_steps:
            assert completed_steps > 0
            snapshot(model, opt, rng, completed_steps, directory / f'step_{completed_steps}.pt')
            checkpoint_steps.append(completed_steps)
            scores[completed_steps] = ctx.evaluate(model, action, 'B')[0]
            short_C[completed_steps], error = ctx.evaluate(model, action, 'C')
            arrays_save(directory / f'step_{completed_steps}_C_errors.npz', error=error)
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
        result = dict(state)
        result.update(status='completed', steps=completed_steps, selected_step=selected,
            selection_losses=scores, selected_losses=selected_losses, checkpoints=checkpoint_steps,
            short_curves={k: dict(B=scores[k], C=short_C[k]) for k in short_C},
            assets_sha256=assets, initial_sha256=initial_sha, plan_sha256=sha(ctx.out / 'plan.json'))
        write(directory / 'result.json', result)
        state['status'] = 'completed'
        write(directory / 'status.json', state, replace=True)
        # The final clock receipt cannot include its own write; all algorithm/result assets above are included.
        wall = tick() - started
        write(directory / 'timing.json', dict(seconds=wall, boundary='initialization through result/status/assets/hash; clock receipt excluded'))
        result['seconds'] = wall
        return result
    except Exception:
        state.update(status='failed', error=traceback.format_exc(), seconds=time.perf_counter() - started,
                     failure_clock='CPU perf_counter; no CUDA synchronization attempted')
        write(directory / 'status.json', state, replace=True)
        raise


def score_proxy(ctx, action, seed, method):
    assert method in ('M1', 'RAW', 'FIXED')
    directory = ctx.out / 'forecasts' / f'{ctx.task}_{action}_{seed}_{method}'
    directory.mkdir(parents=True)
    started = tick()
    model = ctx.model(seed)
    parent = ctx.parent_dir('full', seed)
    p_result = read(parent / 'result.json')
    scores, curves = {}, {}
    steps = [p_result['selected_step']] if method == 'FIXED' else list(GRID)
    for step in steps:
        if method == 'FIXED':
            model.load_state_dict(load_state(parent / f'step_{step}.pt')['model'])
        else:
            child = load_state(ctx.out / 'runs' / job_id(ctx.task, 'proxy', action, seed) / f'step_{step}.pt')
            if method == 'RAW':
                model.load_state_dict(child['model'])
            else:
                real = load_state(parent / f'step_{step}.pt')['original_coordinates']
                proxy_parent = load_state(ctx.parent_dir('proxy', seed) / f'step_{step}.pt')['original_coordinates']
                rp.load_parameters(model, rp.paired_parameters(real, child['original_coordinates'], proxy_parent))
        curves[step] = {}
        for split in ('B', 'C'):
            loss, error = ctx.evaluate(model, action, split)
            curves[step][split] = loss
            arrays_save(directory / f'step_{step}_{split}_errors.npz', error=error)
        scores[step] = curves[step]['B']
    selected = rp.select_checkpoint(scores, steps)
    baseline = p_result['selected_losses']['C']
    assert baseline > 0
    result = dict(method=method, task=ctx.task, action=action, seed=seed, selected_step=selected,
        curves=curves, gain_percent=100. * (baseline - curves[selected]['C']) / baseline,
        assets_sha256={p.name: sha(p) for p in sorted(directory.iterdir()) if p.suffix == '.npz'})
    write(directory / 'result.json', result)
    seconds = tick() - started
    write(directory / 'timing.json', dict(seconds=seconds))
    result['seconds'] = seconds
    return result


def parents(out):
    plan = verify(out)
    write(out/'parents.lock', dict(plan_sha256=sha(out/'plan.json')))
    results, measures, budgets = [], [], {}
    for task in plan['tasks']:
        setup = tick()
        ctx = Context(out, plan, task['id'])
        write(out/f'{ctx.task}_parent_setup.json', dict(seconds=tick()-setup))
        for seed in plan['seeds']:
            for engine in ('full', 'proxy'):
                result = train(ctx, engine, 'PARENT', seed)
                results.append(result)
                write(out/'parents_status.json', dict(status='running', completed=len(results), registered=12, last_id=result['id']), replace=True)
                print(json.dumps(dict(phase='parents', completed=len(results), last_id=result['id'])), flush=True)
            model = ctx.model(seed)
            directory = out/'calibration'/f'{ctx.task}_{seed}'
            directory.mkdir(parents=True)
            started, checkpoints = tick(), []
            for step in GRID:
                t0 = tick()
                model.load_state_dict(load_state(ctx.parent_dir('full',seed)/f'step_{step}.pt')['model'])
                for split in ('B','C'):
                    loss, error = ctx.evaluate(model,'PARENT',split)
                    arrays_save(directory/f'step_{step}_{split}.npz', error=error)
                checkpoints.append(tick()-t0)
            measures.append(dict(task=ctx.task, seed=seed, six_checkpoint_seconds=tick()-started,
                                 checkpoint_seconds=checkpoints))
        subset = [r for r in results if r['task'] == ctx.task]
        timings = [m for m in measures if m['task'] == ctx.task]
        p = max(r['seconds'] for r in subset if r['engine'] == 'proxy')
        s = max(m['six_checkpoint_seconds'] for m in timings)
        e = max(max(m['checkpoint_seconds']) for m in timings)
        b, reserve = math.ceil(1.25*(1.25*p+s)), math.ceil(max(1.,2*e))
        full = min(r['seconds'] for r in subset if r['engine'] == 'full')
        budgets[ctx.task] = dict(short_budget_seconds=b, short_training_boundary_seconds=b-reserve,
            selection_budget_seconds=4*b, reserve_seconds=reserve, max_proxy_parent_seconds=p,
            max_six_checkpoint_seconds=s, max_checkpoint_seconds=e, min_full_parent_seconds=full,
            gate=bool(b>reserve and b<full))
        del ctx
    assets = {p.relative_to(out).as_posix(): sha(p) for base in (out/'runs',out/'calibration') for p in sorted(base.rglob('*')) if p.is_file()}
    result = dict(status='completed', budgets=budgets, budget_gate=all(b['gate'] for b in budgets.values()),
        results=results, measures=measures, assets_sha256=assets, plan_sha256=sha(out/'plan.json'),
        full_parent_trainings=6, proxy_parent_rollouts=6, cumulative_full_trainings=373,
        test_access=False, candidate_responses_access=False)
    verify(out)
    write(out/'calibration.json',result)
    write(out/'parents_status.json',dict(status='completed',completed=12,registered=12,budget_gate=result['budget_gate']),replace=True)


def check_calibration(out):
    result = read(out/'calibration.json')
    sync = read(out/'calibration_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['calibration_sha256'] == sha(out/'calibration.json')
    assert result['plan_sha256'] == sha(out/'plan.json') and result['budget_gate']
    for name,digest in result['assets_sha256'].items():
        assert sha(out/name)==digest
    return result


def forecast(out):
    started = time.perf_counter()
    plan, calibration = verify(out), check_calibration(out)
    write(out/'forecast.lock',dict(plan_sha256=sha(out/'plan.json')))
    results,records,costs=[],[],[]
    for task in plan['tasks']:
        setup=tick()
        ctx=Context(out,plan,task['id'])
        write(out/f'{ctx.task}_setup.json',dict(seconds=tick()-setup))
        for seed in plan['seeds']:
            parent=read(ctx.parent_dir('full',seed)/'result.json')
            cost=dict(task=ctx.task,seed=seed,real_parent_seconds=read(ctx.parent_dir('full',seed)/'timing.json')['seconds'],
                M1=read(ctx.parent_dir('proxy',seed)/'timing.json')['seconds'],RAW=0.,FIXED=0.,SHORT=0.)
            for action in ACTIONS:
                child=train(ctx,'proxy',action,seed);results.append(child)
                methods={}
                for method in ('M1','RAW','FIXED'):
                    methods[method]=score_proxy(ctx,action,seed,method)
                    cost[method]+=methods[method]['seconds']
                cost['M1']+=child['seconds'];cost['RAW']+=child['seconds']
                short=train(ctx,'short',action,seed);results.append(short)
                assert short['steps']<3000,'SHORT exposed full response; preserve and stop without duplicate label run'
                baseline=parent['selected_losses']['C']
                methods['SHORT']=dict(method='SHORT',selected_step=short['selected_step'],
                    gain_percent=100.*(baseline-short['selected_losses']['C'])/baseline,
                    loss_C=short['selected_losses']['C'],steps=short['steps'],seconds=short['seconds'],curves=short['short_curves'])
                for method in METHODS:methods[method]['category']=category(methods[method]['gain_percent'])
                cost['SHORT']+=short['seconds']
                records.append(dict(task=ctx.task,seed=seed,action=action,methods=methods))
                write(out/'forecast_status.json',dict(status='running',completed_runs=len(results),records=len(records),registered_records=24,last_id=short['id']),replace=True)
                print(json.dumps(dict(phase='forecast',records=len(records),last_id=short['id'])),flush=True)
            costs.append(cost)
        del ctx
    assert len(records)==24 and len(results)==48
    assets={p.relative_to(out).as_posix():sha(p) for base in (out/'runs',out/'forecasts') for p in sorted(base.rglob('*')) if p.is_file()}
    for cost in costs:
        common=read(out/'preparation.json')['seconds']/6
        common+=(read(out/f'{cost["task"]}_parent_setup.json')['seconds']+read(out/f'{cost["task"]}_setup.json')['seconds'])/3
        cost['allocated_common_preparation_seconds']=common
        cost['cold_total_seconds']={m:cost[m]+cost['real_parent_seconds']+common for m in METHODS}
    gate=all(c[m]<=calibration['budgets'][c['task']]['selection_budget_seconds'] for c in costs for m in ('M1','SHORT'))
    gate=gate and all(r['seconds']<=calibration['budgets'][r['task']]['short_budget_seconds'] for r in results if r['engine']=='short')
    verify(out);check_calibration(out)
    write(out/'predictions.json',dict(status='locked_locally',records=records,recommendations=recommendation(records),costs=costs,
        budget_gate=gate,assets_sha256=assets,plan_sha256=sha(out/'plan.json'),calibration_sha256=sha(out/'calibration.json'),
        preparation_sha256=sha(out/'preparation.json'),phase_seconds=time.perf_counter()-started,
        full_children_started=False,test_access=False,base_validation_access=False,
        batch_cold_total_seconds={m:sum(c['cold_total_seconds'][m] for c in costs) for m in METHODS}))
    write(out/'forecast_status.json',dict(status='completed',completed_runs=48,records=24,predictions_sha256=sha(out/'predictions.json')),replace=True)


def labels(out):
    plan=verify(out)
    prediction=read(out/'predictions.json')
    lock,sync=read(out/'prediction_lock.json'),read(out/'prediction_sync_receipt.json')
    assert sync['status']=='exact_commit_and_tree_verified'
    assert lock['predictions_sha256']==sync['predictions_sha256']==sha(out/'predictions.json')
    assert sync['prediction_lock_sha256']==sha(out/'prediction_lock.json')
    assert lock['plan_sha256']==prediction['plan_sha256']==sha(out/'plan.json')
    assert prediction['budget_gate'] and not prediction['full_children_started']
    for name,digest in prediction['assets_sha256'].items():assert sha(out/name)==digest
    write(out/'labels.lock',dict(predictions_sha256=sha(out/'predictions.json')))
    results=[];started=time.perf_counter()
    for task in plan['tasks']:
        ctx=Context(out,plan,task['id'])
        for seed in plan['seeds']:
            for action in ACTIONS:
                result=train(ctx,'full',action,seed);results.append(result)
                write(out/'labels_status.json',dict(status='running',completed=len(results),registered=24,last_id=result['id']),replace=True)
                print(json.dumps(dict(phase='labels',completed=len(results),last_id=result['id'])),flush=True)
        del ctx
    verify(out)
    for name,digest in prediction['assets_sha256'].items():assert sha(out/name)==digest
    assert len(results)==24
    write(out/'labels.json',dict(status='completed',results=results,predictions_sha256=sha(out/'predictions.json'),
        phase_seconds=time.perf_counter()-started,test_access=False,base_validation_access=False))
    write(out/'labels_status.json',dict(status='completed',completed=24,registered=24),replace=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','parents','forecast','labels'))
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output)
    try:
        configure()
        globals()[args.mode](out)
    except Exception:
        write(out/f'{args.mode}_failure.json',dict(status='failed',mode=args.mode,error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
