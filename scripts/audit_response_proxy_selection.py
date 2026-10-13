"""Independent audit of frozen sampling-policy response predictions and labels."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch
import torch.nn.functional as F

METHODS = ('M1', 'RAW', 'FIXED', 'SHORT')
ACTIONS = ('KEEP_FIRST', 'KEEP_SECOND', 'FAVOR_FIRST', 'FAVOR_SECOND')
GRID = (500, 1000, 1500, 2000, 2500, 3000)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def coordinates(state):
    return torch.stack([torch.cat([state[f'seasonal.{c}.weight'].T, state[f'seasonal.{c}.bias'][None],
                                  state[f'trend.{c}.weight'].T, state[f'trend.{c}.bias'][None]]) for c in range(8)])


def snap(directory, step):
    return torch.load(directory / ('initial.pt' if step == 0 else f'step_{step}.pt'), weights_only=True, map_location='cpu')


def category(value):
    return 'improve' if value > .25 else 'hurt' if value < -.25 else 'unchanged'


def mse(error):
    return float(np.mean(error.astype(np.float64) ** 2))


def gamma(n, dtype):
    u = np.finfo(dtype).eps / 2
    return n * u / (1 - n * u)


def audit(out):
    started = time.perf_counter()
    plan, prediction, labels = (read(out / n) for n in ('plan.json', 'predictions.json', 'labels.json'))
    lock, sync = (read(out / n) for n in ('prediction_lock.json', 'prediction_sync_receipt.json'))
    assert read(out / 'labels_status.json')['status'] == labels['status'] == 'completed'
    assert lock['predictions_sha256'] == labels['predictions_sha256'] == sync['predictions_sha256'] == sha(out / 'predictions.json')
    assert sync['status'] == 'exact_commit_and_tree_verified' and sync['prediction_lock_sha256'] == sha(out / 'prediction_lock.json')
    assert lock['plan_sha256'] == prediction['plan_sha256'] == sha(out / 'plan.json')
    assert prediction['budget_gate'] and not prediction['D_loaded'] and not labels['D_loaded']
    assert plan['actions'] == list(ACTIONS) and plan['grid'] == list(GRID)
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    old = Path(plan['input_root'])
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    assert sha(plan['core']) == plan['core_sha256']
    for name, digest in prediction['assets_sha256'].items():
        assert sha(out / name) == digest
    prep = read(out / 'preparation.json')
    assert prediction['preparation_sha256'] == sha(out / 'preparation.json')
    for name, digest in prep['assets_sha256'].items():
        assert sha(out / name) == digest
    expected_ids = {j['id'] for j in plan['jobs']}
    assert len(expected_ids) == 108 and {p.name for p in (out / 'runs').iterdir()} == expected_ids
    assert {r['id'] for r in labels['results']} == {j['id'] for j in plan['jobs'] if j['phase'] == 'labels'}
    assert len(labels['results']) == 36
    for item in labels['results']:
        saved = dict(item)
        seconds = saved.pop('seconds')
        directory = out / 'runs' / item['id']
        assert read(directory / 'result.json') == saved and read(directory / 'timing.json')['seconds'] == seconds
        for name, digest in item['assets_sha256'].items():
            assert sha(directory / name) == digest
    destination = out / 'audit'
    destination.mkdir()
    write(destination / 'label_binding.json', dict(labels_sha256=sha(out / 'labels.json'), predictions_sha256=sha(out / 'predictions.json')))
    core = module('core', plan['core'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    records = {(r['task'], r['seed'], r['action']): r for r in prediction['records']}
    assert len(records) == 36
    rows, diagnostics, energies, run_costs, costs, policies = [], [], [], [], [], []
    checked_states = checked_residuals = checked_moments = 0
    for task in plan['tasks']:
        tid = task['id']
        with np.load(old / 'prepared' / f'{tid}_ABC.npz', allow_pickle=False) as z:
            arrays = {name: z[name] for name in z.files}
        values = {s: torch.tensor(arrays[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
        starts = {s: torch.tensor(arrays[f'{s}_starts'], device='cuda') for s in values}
        for split in values:
            lo, hi = task['partitions'][split]
            np.testing.assert_array_equal(arrays[f'{split}_starts'], (np.arange(hi - lo)[:, None] * 60 + np.arange(21)).reshape(-1))
            np.testing.assert_array_equal(arrays[f'{split}_cycles'], np.repeat(np.arange(lo, hi), 60))
            np.testing.assert_array_equal(arrays[f'{split}_offsets'], np.tile(np.arange(60), hi - lo))
            np.testing.assert_array_equal(arrays[f'{split}_base_rows'], np.arange(lo * 60, hi * 60))
        n = len(starts['A'])
        assert n == 6468 and task['partitions']['A'][1] - task['partitions']['A'][0] == 308
        window_cycles = arrays['A_cycles'][arrays['A_starts']]
        first = window_cycles < task['partitions']['A'][0] + 154
        probabilities = {}
        for action in ACTIONS:
            chosen = first if action.endswith('FIRST') else ~first
            # Independently specify group probability totals, then distribute uniformly inside each group.
            total = 1. if action.startswith('KEEP') else .75
            p = np.where(chosen, total / chosen.sum(), (1. - total) / (~chosen).sum())
            with np.load(out / 'prepared' / f'{tid}_{action}.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(p, z['probability'])
                np.testing.assert_array_equal(z['window_cycles'], window_cycles)
                np.testing.assert_array_equal(z['window_base_rows'], arrays['A_base_rows'][arrays['A_starts']])
            probabilities[action] = p
            policies.append(dict(task=tid, action=action, positive_windows=int(np.count_nonzero(p)), probability_sum=float(p.sum()),
                                 effective_sample_size=float(1. / np.sum(p ** 2)), first_group_mass=float(p[first].sum())))
        model = core.DLinear(dict(plan['training'], channels=8)).cuda().eval()

        def features(batch):
            x = batch[:, :32].transpose(1, 2)
            trend = F.avg_pool1d(F.pad(x, (12, 12), mode='replicate'), 25, stride=1)
            one = torch.ones_like(x[..., :1])
            return torch.cat([x - trend, one, trend, one], dim=2).double()

        def errors(state, split):
            model.load_state_dict(state)
            result = []
            with torch.no_grad():
                for lo in range(0, len(starts[split]), 2048):
                    batch = values[split][starts[split][lo:lo + 2048, None] + torch.arange(40, device='cuda')]
                    result.append((model(batch[:, :32]) - batch[:, 32:]).cpu().numpy())
            result = np.concatenate(result)
            assert np.isfinite(result).all()
            return result

        def check_saved(error, path):
            nonlocal checked_residuals
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(error, z['error'])
            checked_residuals += 1

        independent_moments = {}
        with torch.no_grad():
            for action, p in probabilities.items():
                gram = cross = square = None
                for lo in range(0, n, 2048):
                    batch = values['A'][starts['A'][lo:lo + 2048, None] + torch.arange(40, device='cuda')]
                    phi = features(batch).permute(1, 0, 2)
                    target = batch[:, 32:].double().permute(2, 0, 1)
                    mass = torch.tensor(p[lo:lo + len(batch)], device='cuda')
                    weighted = phi * mass[None, :, None]
                    terms = (phi.transpose(1, 2) @ weighted, weighted.transpose(1, 2) @ target,
                             (target.square() * mass[None, :, None]).sum(1))
                    if gram is None:
                        gram, cross, square = terms
                    else:
                        gram += terms[0]; cross += terms[1]; square += terms[2]
                independent_moments[action] = [v.cpu().numpy() for v in (gram, cross, square)]
            c_batch = values['C'][starts['C'][:, None] + torch.arange(40, device='cuda')]
            c_phi, c_target = features(c_batch), c_batch[:, 32:].double()

        for seed in plan['seeds']:
            parent_dir = old / 'runs' / f'DEV_{tid}_full_PARENT_s{seed}'
            proxy_dir = old / 'runs' / f'DEV_{tid}_proxy_PARENT_s{seed}'
            parent = read(parent_dir / 'result.json')
            initial = coordinates(snap(parent_dir, 0)['model'])
            parent_C = mse(errors(snap(parent_dir, parent['selected_step'])['model'], 'C'))
            assert parent_C == parent['selected_losses']['C']
            cost = dict(task=tid, seed=seed, real_parent_seconds=read(parent_dir / 'timing.json')['seconds'],
                        M1=read(proxy_dir / 'timing.json')['seconds'], RAW=0., FIXED=0., SHORT=0.)
            for action in ACTIONS:
                results, truths = {}, {}
                for engine in ('proxy', 'short', 'full'):
                    directory = out / 'runs' / f'SELR1_{tid}_{engine}_{action}_s{seed}'
                    result = read(directory / 'result.json')
                    status = read(directory / 'status.json')
                    assert result['status'] == status['status'] == 'completed'
                    assert result['id'] == directory.name and result['plan_sha256'] == sha(out / 'plan.json')
                    assert (result['task'], result['engine'], result['action'], result['seed']) == (tid, engine, action, seed)
                    for name, digest in result['assets_sha256'].items():
                        assert sha(directory / name) == digest
                    checkpoints = result['checkpoints']
                    assert checkpoints == (list(GRID) if engine != 'short' else sorted(set([k for k in GRID if k <= result['steps']] + [result['steps']])))
                    assert result['steps'] == 3000 if engine != 'short' else 0 < result['steps'] <= 3000
                    for step in [0] + checkpoints:
                        snapshot = snap(directory, step)
                        packed = coordinates(snapshot['model'])
                        assert snapshot['step'] == step and torch.equal(packed, snapshot['original_coordinates'])
                        assert bool(torch.isfinite(packed).all())
                        if step == 0:
                            assert torch.equal(initial, packed) and not snapshot['optimizer']['state']
                            assert hashlib.sha256(packed.numpy().tobytes()).hexdigest() == result['initial_sha256']
                        else:
                            assert len(snapshot['optimizer']['state']) == 32
                            for v in snapshot['optimizer']['state'].values():
                                assert v['step'].item() == step and bool(torch.isfinite(v['exp_avg']).all()) and bool(torch.isfinite(v['exp_avg_sq']).all())
                        checked_states += 1
                    seconds = read(directory / 'timing.json')['seconds']
                    run_costs.append(dict(id=directory.name, engine=engine, steps=result['steps'], seconds=seconds))
                    results[engine] = result
                    if engine == 'proxy':
                        assert result['selected_step'] is None and result['selection_losses'] == {}
                        with np.load(directory / 'moments.npz', allow_pickle=False) as z:
                            for key, val in zip(('gram', 'cross', 'target_square'), independent_moments[action]):
                                np.testing.assert_array_equal(val, z[key])
                        checked_moments += 1
                        cost['M1'] += seconds; cost['RAW'] += seconds
                        continue
                    if engine == 'short':
                        cost['SHORT'] += seconds
                    cdf = np.cumsum(probabilities[action]); cdf[-1] = 1.
                    rng = np.random.default_rng(seed + 1000)
                    draws = np.zeros(n, dtype=np.int64)
                    for _ in range(result['steps']):
                        np.add.at(draws, np.searchsorted(cdf, rng.random(256), side='right'), 1)
                    with np.load(directory / 'sampling.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(draws, z['draw_count'])
                    assert snapshot['rng_numpy'] == rng.bit_generator.state
                    assert draws[probabilities[action] == 0].sum() == 0
                    losses = {}
                    for step in checkpoints:
                        state = snap(directory, step)['model']
                        losses[step] = mse(errors(state, 'B'))
                        assert losses[step] == result['selection_losses'][str(step)]
                        error = errors(state, 'C')
                        if engine == 'short':
                            check_saved(error, directory / f'step_{step}_C_errors.npz')
                            assert result['short_curves'][str(step)] == dict(B=losses[step], C=mse(error))
                        else:
                            truths[step] = error
                            with (destination / f'{tid}_{action}_{seed}_true_{step}_C.npz').open('xb') as f:
                                np.savez_compressed(f, error=error)
                    selected = min(checkpoints, key=losses.get)
                    assert selected == result['selected_step']
                    for split in ('B', 'C'):
                        error = errors(snap(directory, selected)['model'], split)
                        check_saved(error, directory / f'selected_{split}_errors.npz')
                        assert mse(error) == result['selected_losses'][split]
                record = records[(tid, seed, action)]
                true_step = results['full']['selected_step']
                actual = 100. * (parent_C - mse(truths[true_step])) / parent_C
                rows.append(dict(task=tid, action=action, seed=seed, actual_gain=actual,
                                 predicted={m: record['methods'][m]['gain_percent'] for m in METHODS}))
                for method in ('M1', 'RAW', 'FIXED'):
                    directory = out / 'forecasts' / f'{tid}_{action}_{seed}_{method}'
                    local = read(directory / 'result.json')
                    for key in ('method', 'task', 'action', 'seed', 'selected_step', 'gain_percent', 'curves', 'assets_sha256'):
                        assert local[key] == record['methods'][method][key]
                    assert record['methods'][method]['category'] == category(local['gain_percent'])
                    for name, digest in local['assets_sha256'].items():
                        assert sha(directory / name) == digest
                    curves = {}
                    steps = [parent['selected_step']] if method == 'FIXED' else list(GRID)
                    for step in steps:
                        if method == 'FIXED':
                            state = snap(parent_dir, step)['model']
                        else:
                            child = snap(out / 'runs' / f'SELR1_{tid}_proxy_{action}_s{seed}', step)['model']
                            state = child
                            if method == 'M1':
                                real, reference = snap(parent_dir, step)['model'], snap(proxy_dir, step)['model']
                                state = {k: real[k] + (child[k] - reference[k]) for k in child}
                        curves[step] = {}
                        for split in ('B', 'C'):
                            error = errors(state, split)
                            check_saved(error, directory / f'step_{step}_{split}_errors.npz')
                            curves[step][split] = mse(error)
                            assert curves[step][split] == local['curves'][str(step)][split]
                    selected = min(steps, key=lambda k: curves[k]['B'])
                    assert selected == local['selected_step']
                    assert local['gain_percent'] == 100. * (parent_C - curves[selected]['C']) / parent_C
                    if method == 'FIXED':
                        assert local['gain_percent'] == 0.
                    else:
                        diagnostics.append(dict(task=tid, action=action, seed=seed, method=method, hit=selected == true_step,
                            B_selection_regret=results['full']['selection_losses'][str(selected)] - results['full']['selection_losses'][str(true_step)],
                            C_prediction_term=curves[selected]['C'] - mse(truths[selected]),
                            C_selection_term=mse(truths[selected]) - mse(truths[true_step])))
                    cost[method] += read(directory / 'timing.json')['seconds']
                short = record['methods']['SHORT']
                assert short['gain_percent'] == 100. * (parent_C - results['short']['selected_losses']['C']) / parent_C
                assert short['selected_step'] == results['short']['selected_step'] and short['steps'] == results['short']['steps']
                assert short['curves'] == results['short']['short_curves'] and short['category'] == category(short['gain_percent'])
                for step in GRID:
                    child = snap(out / 'runs' / f'SELR1_{tid}_proxy_{action}_s{seed}', step)['model']
                    real, reference = snap(parent_dir, step)['model'], snap(proxy_dir, step)['model']
                    paired_state = {k: real[k] + (child[k] - reference[k]) for k in child}
                    with np.load(out / 'forecasts' / f'{tid}_{action}_{seed}_RAW' / f'step_{step}_C_errors.npz') as z:
                        raw = z['error'].astype(np.float64)
                    with np.load(out / 'forecasts' / f'{tid}_{action}_{seed}_M1' / f'step_{step}_C_errors.npz') as z:
                        paired = z['error'].astype(np.float64)
                    e0 = errors(reference, 'C').astype(np.float64) - errors(real, 'C')
                    eT = raw - truths[step]
                    corrected = paired - truths[step]
                    with torch.no_grad():
                        pc, pr, pp, combo = [coordinates(v).double().cuda() for v in (child, real, reference, paired_state)]
                        parameter_bound = torch.einsum('nci,cij->njc', c_phi.abs(), (combo - (pr + pc - pp)).abs())
                        magnitude = sum(torch.einsum('nci,cij->njc', c_phi.abs(), v.abs()) + c_target.abs() for v in (pc, pr, pp, combo))
                        bound = parameter_bound + gamma(70, np.float32) * magnitude + gamma(256, np.float64) * (magnitude + parameter_bound)
                    gap = paired - (raw - e0)
                    assert np.all(np.abs(gap) <= bound.cpu().numpy())
                    energy_T, energy_0, cross = float(np.mean(eT ** 2)), float(np.mean(e0 ** 2)), float(np.mean(eT * e0))
                    energies.append(dict(task=tid, action=action, seed=seed, step=step, raw_energy=energy_T,
                        parent_energy=energy_0, cross=cross, paired_energy=float(np.mean(corrected ** 2)),
                        alignment=cross / np.sqrt(energy_T * energy_0) if energy_T * energy_0 > 0 else None,
                        rounding_bound_pass=True, identity_gap_max=float(np.abs(gap).max())))
            registered = next(c for c in prediction['costs'] if (c['task'], c['seed']) == (tid, seed))
            for key in ('M1', 'RAW', 'FIXED', 'SHORT', 'real_parent_seconds'):
                assert abs(cost[key] - registered[key]) < 1e-9
            prior_prepare = read(old / 'preparation.json')['seconds']
            assert prior_prepare == plan['prior_preparation_seconds']
            common = (prep['seconds'] + prior_prepare) / 9. + read(out / f'{tid}_setup.json')['seconds'] / 3.
            assert registered['allocated_common_preparation_seconds'] == common
            cost['allocated_common_preparation_seconds'] = common
            cost['cold_total_seconds'] = {m: cost[m] + cost['real_parent_seconds'] + common for m in METHODS}
            assert all(abs(cost['cold_total_seconds'][m] - registered['cold_total_seconds'][m]) < 1e-9 for m in METHODS)
            costs.append(cost)
    assert len(rows) == 36 and len(run_costs) == 108 and len(energies) == 216 and checked_moments == 36
    cells, task_mae, realized, decisions = [], {m: {} for m in METHODS}, {m: [] for m in METHODS}, []
    for task in plan['tasks']:
        tid = task['id']
        for action in ACTIONS:
            subset = [r for r in rows if (r['task'], r['action']) == (tid, action)]
            truth = float(np.mean([r['actual_gain'] for r in subset]))
            predictions = {m: float(np.mean([r['predicted'][m] for r in subset])) for m in METHODS}
            cells.append(dict(task=tid, action=action, truth=truth, predictions=predictions,
                actual_category=category(truth), predicted_categories={m: category(v) for m, v in predictions.items()},
                unstable=min(r['actual_gain'] for r in subset) < -.25 and max(r['actual_gain'] for r in subset) > .25))
        selected_cells = [c for c in cells if c['task'] == tid]
        oracle = max([0.] + [c['truth'] for c in selected_cells])
        for method in METHODS:
            task_mae[method][tid] = float(np.mean([abs(c['truth'] - c['predictions'][method]) for c in selected_cells]))
            means = {c['action']: c['predictions'][method] for c in selected_cells}
            choice = max(ACTIONS, key=lambda a: means[a])
            if means[choice] < 1.:
                choice = 'PARENT'
            assert prediction['recommendations'][tid][method] == dict(action=choice, means=means, categories={a: category(v) for a, v in means.items()})
            gain = 0. if choice == 'PARENT' else next(c['truth'] for c in selected_cells if c['action'] == choice)
            realized[method].append(gain)
            true_eligible = {c['action'] for c in selected_cells if c['truth'] >= 1.}
            predicted_eligible = {a for a in ACTIONS if means[a] >= 1.}
            decisions.append(dict(task=tid, method=method, action=choice, gain=gain, oracle_gain=oracle, regret=oracle - gain,
                effective_actions=len(true_eligible), covered_effective_actions=len(true_eligible & predicted_eligible),
                missed_effective_actions=sorted(true_eligible - predicted_eligible), accepted_harm=choice != 'PARENT' and gain < -.25))
    mae = {m: float(np.mean(list(v.values()))) for m, v in task_mae.items()}
    best = min(('RAW', 'SHORT'), key=lambda m: mae[m])
    noninferior = sum(task_mae['M1'][t] <= task_mae[best][t] for t in task_mae['M1'])
    error_gate = mae['M1'] <= .8 * mae[best] and mae[best] - mae['M1'] >= .05 and noninferior >= 2
    mean_gain = {m: float(np.mean(v)) for m, v in realized.items()}
    selection_gate = mean_gain['M1'] >= max(mean_gain[b] for b in ('RAW', 'SHORT', 'FIXED')) + .10 and min(realized['M1']) >= -.25
    budget_gate = all(c['M1'] <= 60. and c['SHORT'] <= 60. for c in costs) and all(r['seconds'] <= 15. for r in run_costs if r['engine'] == 'short')
    assert budget_gate
    cold_totals = {m: sum(c['cold_total_seconds'][m] for c in costs) for m in METHODS}
    assert all(abs(cold_totals[m] - prediction['batch_cold_total_seconds'][m]) < 1e-8 for m in METHODS)
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    assert sha(out / 'predictions.json') == lock['predictions_sha256']
    new_full = sum(r['steps'] == 3000 and r['engine'] in ('full', 'short') for r in run_costs)
    result = dict(status='PASS', original_inputs_unchanged=True, new_full_trainings=new_full, cumulative_full_trainings=331 + new_full,
        reused_full_parents=9, reused_proxy_parents=9, new_proxy_rollouts=36, short_trainings=36,
        states_verified=checked_states, residuals_verified=checked_residuals, moments_verified=checked_moments,
        policies=policies, cells=cells, scalar_records=rows, prediction_MAE_pp=mae, task_MAE_pp=task_mae,
        best_dynamic_baseline=best, noninferior_tasks=noninferior, decisions=decisions, mean_recommendation_gain=mean_gain,
        recommendation_gains=realized, checkpoint_diagnostics=diagnostics, error_energy=energies,
        run_costs=run_costs, costs=costs, batch_cold_total_seconds=cold_totals,
        budget_gate=budget_gate, error_gate=bool(error_gate), selection_gate=bool(selection_gate),
        advance=bool(budget_gate and error_gate and selection_gate), D_access=False, test_access=False,
        independent_source_claim=False, audit_seconds=time.perf_counter() - started)
    write(out / 'final_audit.json', result)
    archive_started = time.perf_counter()
    allowed = {'.py', '.md', '.json', '.log', '.npz', '.pt', '.lock'}
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in allowed and p.name != 'audit.log')
    assert all(not p.is_symlink() for p in paths)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    archive = out / 'response_proxy_selection_20261013.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in hashes:
            tar.add(out / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(hashes)
        for name, digest in hashes.items():
            assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == digest
    write(out / 'archive_receipt.json', dict(status='PASS', file=archive.name, bytes=archive.stat().st_size,
        sha256=sha(archive), members_sha256=hashes, archive_seconds=time.perf_counter() - archive_started))
    print(json.dumps({k: result[k] for k in ('status', 'prediction_MAE_pp', 'error_gate', 'selection_gate', 'advance')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    audit(Path(parser.parse_args().output))
