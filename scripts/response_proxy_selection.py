"""Create-only sampling-policy response study, reusing frozen parent trajectories."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import response_proxy as rp


GRID = (500, 1000, 1500, 2000, 2500, 3000)
METHODS = ('M1', 'RAW', 'FIXED', 'SHORT')
ACTIONS = ('KEEP_FIRST', 'KEEP_SECOND', 'FAVOR_FIRST', 'FAVOR_SECOND')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if replace:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def configure():
    assert os.name == 'posix'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def verify(out):
    plan = read(out / 'plan.json')
    assert read(out / 'source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest, name
    old = Path(plan['input_root'])
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest, name
    assert sha(plan['core']) == plan['core_sha256']
    assert plan['grid'] == list(GRID)
    expected = []
    for phase in ('forecast', 'labels'):
        for task in plan['tasks']:
            for seed in plan['seeds']:
                jobs = [(engine, action) for action in ACTIONS for engine in ('proxy', 'short')] if phase == 'forecast' else [('full', a) for a in ACTIONS]
                for engine, action in jobs:
                    expected.append(dict(id=job_id(task['id'], engine, action, seed), task=task['id'],
                                         engine=engine, action=action, seed=seed, phase=phase))
    assert plan['jobs'] == expected and len(expected) == 108
    assert plan['actions'] == list(ACTIONS)
    return plan


def job_id(task, engine, action, seed):
    return f'SELR1_{task}_{engine}_{action}_s{seed}'


def category(value):
    return 'improve' if value > .25 else 'hurt' if value < -.25 else 'unchanged'


def arrays_save(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def prepare(out):
    started = time.perf_counter()
    plan = verify(out)
    destination = out / 'prepared'
    destination.mkdir()
    old = Path(plan['input_root'])
    entries = []
    for task in plan['tasks']:
        with np.load(old / 'prepared' / f'{task["id"]}_ABC.npz', allow_pickle=False) as z:
            starts, cycles = z['A_starts'], z['A_cycles']
            lo, hi = task['partitions']['A']
            assert hi - lo == 308 and len(starts) == 6468
            first = cycles[starts] < lo + (hi - lo) // 2
            assert int(first.sum()) == len(starts) // 2
            probabilities = {}
            for action in ACTIONS:
                favored = first if action.endswith('FIRST') else ~first
                mass = np.where(favored, 1., 0.) if action.startswith('KEEP') else np.where(favored, 3., 1.)
                probability = mass / mass.sum()
                assert np.isfinite(probability).all() and abs(probability.sum() - 1.) < 1e-14
                probabilities[action] = probability
                arrays_save(destination / f'{task["id"]}_{action}.npz', probability=probability,
                            window_base_rows=z['A_base_rows'][starts], window_cycles=cycles[starts])
            entries.append(dict(task=task['id'], A_windows=len(starts), A_cycles=hi - lo,
                pair_mask_overlap=0, half_windows=int(first.sum()),
                prior_input_transform_actions_use_all_windows=True,
                retained={a: int(np.count_nonzero(p)) for a, p in probabilities.items()}))
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in sorted(destination.iterdir())}
    verify(out)
    write(out / 'preparation.json', dict(status='PASS', tasks=entries, assets_sha256=hashes,
        seconds=time.perf_counter() - started, plan_sha256=sha(out / 'plan.json'),
        D_loaded=False, test_access=False, new_full_trainings=0, labels_inspected=False))


class Context:
    def __init__(self, out, plan, task):
        self.out, self.plan, self.task = out, plan, task
        receipt = read(out / 'preparation.json')
        assert receipt['status'] == 'PASS' and receipt['plan_sha256'] == sha(out / 'plan.json')
        for action in ACTIONS:
            name = f'prepared/{task}_{action}.npz'
            assert sha(out / name) == receipt['assets_sha256'][name]
        old = Path(plan['input_root'])
        # D is never opened by the runner, including for preparation verification.
        with np.load(old / 'prepared' / f'{task}_ABC.npz', allow_pickle=False) as z:
            self.values = {s: torch.tensor(z[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
            self.starts = {s: torch.tensor(z[f'{s}_starts'], device='cuda') for s in self.values}
        self.core = module('core', plan['core'])
        self.offset = torch.arange(40, device='cuda')
        self.n = len(self.starts['A'])
        self.probabilities, self.cdfs = {}, {}
        for action in ACTIONS:
            with np.load(out / 'prepared' / f'{task}_{action}.npz', allow_pickle=False) as z:
                self.probabilities[action] = z['probability']
            cdf = np.cumsum(self.probabilities[action]); cdf[-1] = 1.
            self.cdfs[action] = cdf

    def parent_dir(self, engine, seed):
        assert engine in ('full', 'proxy')
        return Path(self.plan['input_root']) / 'runs' / f'DEV_{self.task}_{engine}_PARENT_s{seed}'

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.core.DLinear(dict(self.plan['training'], channels=8)).cuda()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)

    def batch(self, split, ids):
        assert split in ('A', 'B', 'C')
        return self.values[split][self.starts[split][ids, None] + self.offset]

    def transform(self, x, action):
        assert action == 'PARENT' or action in ACTIONS
        return x

    @torch.no_grad()
    def evaluate(self, model, action, split):
        assert split in ('B', 'C')
        model.eval()
        errors = []
        for lo in range(0, len(self.starts[split]), 2048):
            batch = self.batch(split, torch.arange(lo, min(lo + 2048, len(self.starts[split])), device='cuda'))
            errors.append((model(self.transform(batch[:, :32], action)) - batch[:, 32:]).cpu().numpy())
        error = np.concatenate(errors)
        assert np.isfinite(error).all()
        return float(np.mean(error.astype(np.float64) ** 2)), error

    @torch.no_grad()
    def moments(self, model, action):
        result = None
        for lo in range(0, self.n, 2048):
            batch = self.batch('A', torch.arange(lo, min(lo + 2048, self.n), device='cuda'))
            phi = rp.features(model, self.transform(batch[:, :32], action))
            item = rp.batch_moments(phi, batch[:, 32:], torch.tensor(self.probabilities[action][lo:lo + len(batch)],
                                    dtype=torch.float64, device='cuda'))
            if result is None:
                result = item
            else:
                result.add_(item)
        return result


def snapshot(model, opt, rng, step, path):
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
        rng_numpy=rng.bit_generator.state, rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
        original_coordinates=rp.parameters(model).detach().cpu()), path)


def load_state(path):
    return torch.load(path, map_location='cuda', weights_only=True)


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


def recommendation(records):
    result = {}
    for task in sorted({r['task'] for r in records}):
        result[task] = {}
        for method in METHODS:
            means = {action: float(np.mean([r['methods'][method]['gain_percent'] for r in records
                     if r['task'] == task and r['action'] == action])) for action in ACTIONS}
            action = max(ACTIONS, key=lambda a: means[a])
            if means[action] < 1.:
                action = 'PARENT'
            result[task][method] = dict(action=action, means=means, categories={a: category(v) for a, v in means.items()})
    return result


def forecast(out):
    started = time.perf_counter()
    plan = verify(out)
    write(out / 'forecast.lock', dict(plan_sha256=sha(out / 'plan.json')))
    results, records, costs = [], [], []
    try:
        for task in plan['tasks']:
            t0 = tick()
            ctx = Context(out, plan, task['id'])
            write(out / f'{task["id"]}_setup.json', dict(seconds=tick() - t0))
            for seed in plan['seeds']:
                parent = read(ctx.parent_dir('full', seed) / 'result.json')
                parent['seconds'] = read(ctx.parent_dir('full', seed) / 'timing.json')['seconds']
                proxy_parent = read(ctx.parent_dir('proxy', seed) / 'timing.json')
                cost = dict(task=ctx.task, seed=seed, real_parent_seconds=parent['seconds'],
                            M1=proxy_parent['seconds'], RAW=0., FIXED=0., SHORT=0.)
                for action in ACTIONS:
                    child = train(ctx, 'proxy', action, seed)
                    results.append(child)
                    methods = {}
                    for method in ('M1', 'RAW', 'FIXED'):
                        methods[method] = score_proxy(ctx, action, seed, method)
                        cost[method] += methods[method]['seconds']
                    cost['M1'] += child['seconds']
                    cost['RAW'] += child['seconds']
                    short = train(ctx, 'short', action, seed)
                    results.append(short)
                    baseline = parent['selected_losses']['C']
                    methods['SHORT'] = dict(method='SHORT', selected_step=short['selected_step'],
                        gain_percent=100. * (baseline - short['selected_losses']['C']) / baseline,
                        loss_C=short['selected_losses']['C'], steps=short['steps'], seconds=short['seconds'], curves=short['short_curves'])
                    for method in METHODS:
                        methods[method]['category'] = category(methods[method]['gain_percent'])
                    cost['SHORT'] += short['seconds']
                    records.append(dict(task=ctx.task, seed=seed, action=action, methods=methods))
                    write(out / 'forecast_status.json', dict(status='running', completed_runs=len(results),
                        records=len(records), registered_records=36, last_id=short['id']), replace=True)
                    print(json.dumps(dict(stage='forecast', records=len(records), last=short['id'])), flush=True)
                costs.append(cost)
            del ctx
        assert len(records) == 36 and len(results) == 72
        recs = recommendation(records)
        verify(out)
        assets = {p.relative_to(out).as_posix(): sha(p) for base in (out / 'runs', out / 'forecasts')
                  for p in sorted(base.rglob('*')) if p.is_file()}
        for task in plan['tasks']:
            name = f'{task["id"]}_setup.json'
            assets[name] = sha(out / name)
        preparation_seconds = read(out / 'preparation.json')['seconds']
        prior_preparation_seconds = read(Path(plan['input_root']) / 'preparation.json')['seconds']
        assert prior_preparation_seconds == plan['prior_preparation_seconds']
        for cost in costs:
            common = (preparation_seconds + prior_preparation_seconds) / len(costs)
            common += read(out / f'{cost["task"]}_setup.json')['seconds'] / len(plan['seeds'])
            cost['allocated_common_preparation_seconds'] = common
            cost['cold_total_seconds'] = {m: cost[m] + cost['real_parent_seconds'] + common for m in METHODS}
        budget_gate = all(c['M1'] <= plan['selection_budget_seconds'] and c['SHORT'] <= plan['selection_budget_seconds'] for c in costs)
        budget_gate = budget_gate and all(r['seconds'] <= plan['short_budget_seconds'] for r in results if r['engine'] == 'short')
        predictions = dict(status='locked_locally', records=records, recommendations=recs, costs=costs, budget_gate=budget_gate,
            assets_sha256=assets, plan_sha256=sha(out / 'plan.json'),
            preparation_sha256=sha(out / 'preparation.json'), phase_seconds=time.perf_counter() - started,
            full_parent_trainings=0, reused_full_parents=9, reused_proxy_parents=9, proxy_rollouts=36, short_trainings=36,
            short_completed_3000=sum(r['engine'] == 'short' and r['steps'] == 3000 for r in results),
            batch_cold_total_seconds={m: sum(c['cold_total_seconds'][m] for c in costs) for m in METHODS},
            D_loaded=False, full_children_started=False, test_access=False)
        write(out / 'predictions.json', predictions)
        write(out / 'forecast_status.json', dict(status='completed', completed_runs=72, records=36,
              predictions_sha256=sha(out / 'predictions.json')), replace=True)
    except Exception:
        write(out / 'forecast_status.json', dict(status='failed', completed_runs=len(results),
              records=len(records), seconds=time.perf_counter() - started, error=traceback.format_exc()), replace=True)
        raise


def labels(out):
    plan = verify(out)
    prediction = read(out / 'predictions.json')
    receipt = read(out / 'prediction_lock.json')
    assert receipt['predictions_sha256'] == sha(out / 'predictions.json')
    sync = read(out / 'prediction_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['predictions_sha256'] == sha(out / 'predictions.json')
    assert sync['prediction_lock_sha256'] == sha(out / 'prediction_lock.json')
    assert receipt['plan_sha256'] == sha(out / 'plan.json')
    assert len(sync['source_commit']) == 40
    assert read(out / 'forecast_status.json')['status'] == 'completed'
    assert prediction['plan_sha256'] == sha(out / 'plan.json') and not prediction['full_children_started']
    assert prediction['budget_gate'], 'Budget failure stops before full child labels'
    for name, digest in prediction['assets_sha256'].items():
        assert sha(out / name) == digest, name
    write(out / 'labels.lock', dict(predictions_sha256=sha(out / 'predictions.json')))
    results = []
    started = time.perf_counter()
    try:
        for task in plan['tasks']:
            ctx = Context(out, plan, task['id'])
            for seed in plan['seeds']:
                for action in ACTIONS:
                    result = train(ctx, 'full', action, seed)
                    results.append(result)
                    write(out / 'labels_status.json', dict(status='running', completed=len(results),
                        registered=36, last_id=result['id']), replace=True)
                    print(json.dumps(dict(stage='labels', completed=len(results), last=result['id'])), flush=True)
            del ctx
        verify(out)
        assert sha(out / 'predictions.json') == receipt['predictions_sha256']
        for name, digest in prediction['assets_sha256'].items():
            assert sha(out / name) == digest
        assert len(results) == 36
        write(out / 'labels.json', dict(status='completed', results=results, predictions_sha256=receipt['predictions_sha256'],
            phase_seconds=time.perf_counter() - started, D_loaded=False, test_access=False))
        write(out / 'labels_status.json', dict(status='completed', completed=36, registered=36), replace=True)
    except Exception:
        write(out / 'labels_status.json', dict(status='failed', completed=len(results),
              registered=36, error=traceback.format_exc()), replace=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'forecast', 'labels'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    configure()
    globals()[args.mode](Path(args.output))


if __name__ == '__main__':
    main()
