"""Separate supplementary verification; original failed audit is immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch


METHODS = ('M1', 'RAW', 'FIXED', 'SHORT')
BASELINES = ('RAW', 'FIXED', 'SHORT')
ACTIONS = ('P1', 'CP1')
GRID = (500, 1000, 1500, 2000, 2500, 3000)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def coordinates(state):
    return torch.stack([torch.cat([state[f'seasonal.{c}.weight'].T,
        state[f'seasonal.{c}.bias'][None], state[f'trend.{c}.weight'].T,
        state[f'trend.{c}.bias'][None]]) for c in range(8)])


def path_for(out, task, engine, action, seed):
    return out / 'runs' / f'DEV_{task}_{engine}_{action}_s{seed}'


def snap(directory, step):
    return torch.load(directory / ('initial.pt' if step == 0 else f'step_{step}.pt'),
                      weights_only=True, map_location='cpu')


def category(value):
    return 'improve' if value > .25 else 'hurt' if value < -.25 else 'unchanged'


def mse(error):
    return float(np.mean(error.astype(np.float64) ** 2))


def recommendation(means):
    action = max(ACTIONS, key=lambda a: means[a])
    return action if means[action] >= 1. else 'PARENT'


def audit(out, supplement, diagnosis_path):
    started = time.perf_counter()
    supplement.mkdir()
    assert sha(diagnosis_path) == '6a656039d2576a19e91f7b2db8cda8bc774d543009a07250bc5bcc1feb6a45c0'
    diagnosis = read(diagnosis_path)
    assert diagnosis['status'] == 'diagnosis_completed' and diagnosis['diagnostic_checks_passed']
    assert len(diagnosis['cells']) == 108 and not diagnosis['D_access'] and not diagnosis['test_access']
    for name, digest in diagnosis['checked_input_hashes'].items():
        assert sha(out / name) == digest
    numeric_cells = {(r['task'], r['seed'], r['action'], r['step']): r for r in diagnosis['cells']}
    assert len(numeric_cells) == 108
    original_log_sha = sha(out / 'audit.log')
    assert not (out / 'final_audit.json').exists()
    plan = read(out / 'plan.json')
    predictions = read(out / 'predictions.json')
    labels = read(out / 'labels.json')
    lock = read(out / 'prediction_lock.json')
    assert labels['status'] == 'completed' and len(labels['results']) == 18
    assert read(out / 'labels_status.json')['status'] == 'completed'
    assert lock['predictions_sha256'] == labels['predictions_sha256'] == sha(out / 'predictions.json')
    sync = read(out / 'prediction_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['predictions_sha256'] == sha(out / 'predictions.json')
    assert sync['prediction_lock_sha256'] == sha(out / 'prediction_lock.json')
    assert lock['plan_sha256'] == sha(out / 'plan.json')
    assert predictions['plan_sha256'] == sha(out / 'plan.json')
    assert predictions['budget_gate'] and not predictions['D_loaded'] and not labels['D_loaded']
    expected_labels = {j['id'] for j in plan['jobs'] if j['phase'] == 'labels'}
    assert {r['id'] for r in labels['results']} == expected_labels
    for locked in labels['results']:
        directory = out / 'runs' / locked['id']
        recorded = dict(locked)
        seconds = recorded.pop('seconds')
        assert read(directory / 'result.json') == recorded
        assert read(directory / 'timing.json')['seconds'] == seconds
        for name, digest in locked['assets_sha256'].items():
            assert sha(directory / name) == digest
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    old = Path(plan['input_root'])
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    assert sha(plan['core']) == plan['core_sha256']
    for name, digest in predictions['assets_sha256'].items():
        assert sha(out / name) == digest, name
    prep = read(out / 'preparation.json')
    assert predictions['preparation_sha256'] == sha(out / 'preparation.json')
    for name, digest in prep['assets_sha256'].items():
        assert sha(out / name) == digest
    destination = supplement / 'audit'
    destination.mkdir()
    # All prediction and label identities are locked before D can be opened below.
    write(supplement / 'labels_lock.json', dict(labels_sha256=sha(out / 'labels.json'),
        predictions_sha256=sha(out / 'predictions.json'), audit_started_utc=datetime.now(timezone.utc).isoformat()))
    h = module('old_h', old / 'hydraulic_signal_validation.py')
    values, cycles, offsets = h.load_split(old, 'train', read(old / 'data' / 'manifest.json'))
    core, proposal = module('core', plan['core']), module('proposal', old / 'proposal_median3.py')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    expected_ids = {j['id'] for j in plan['jobs']}
    assert {p.name for p in (out / 'runs').iterdir()} == expected_ids
    records = {(r['task'], r['seed'], r['action']): r for r in predictions['records']}
    assert len(records) == 18
    scalar_records, checkpoints, energies, tasks, run_costs = [], [], [], {}, []
    checked_states = checked_residuals = 0
    initialization, sampling = {}, {}
    all_row_ids = set()
    full_count = short_count = proxy_count = 0

    for task in plan['tasks']:
        task_id = task['id']
        with np.load(out / 'prepared' / f'{task_id}_ABC.npz', allow_pickle=False) as z:
            saved = {name: z[name] for name in z.files}
        a_lo, a_hi = task['partitions']['A']
        a_ids = np.flatnonzero((cycles >= a_lo) & (cycles < a_hi))
        mean = values[a_ids].mean(0)
        scale = values[a_ids].std(0)
        scale = np.where(scale == 0., 1., scale)
        np.testing.assert_array_equal(mean, saved['mean'])
        np.testing.assert_array_equal(scale, saved['scale'])
        splits = {}
        for split, (lo, hi) in task['partitions'].items():
            ids = np.flatnonzero((cycles >= lo) & (cycles < hi))
            assert not all_row_ids.intersection(ids.tolist())
            all_row_ids.update(ids.tolist())
            if split == 'D':
                with np.load(out / 'audit_only' / f'{task_id}_D.npz', allow_pickle=False) as z:
                    block = {name: z[name] for name in z.files}
            else:
                block = {key: saved[f'{split}_{key}'] for key in ('values', 'cycles', 'offsets', 'starts', 'base_rows')}
            normalized = ((values[ids] - mean) / scale).astype(np.float32)
            np.testing.assert_array_equal(block['values'], normalized)
            np.testing.assert_array_equal(block['base_rows'], ids)
            np.testing.assert_array_equal(block['cycles'], cycles[ids])
            np.testing.assert_array_equal(block['offsets'], offsets[ids])
            # Independent complete-cycle window construction.
            expected_starts = (np.arange(hi - lo)[:, None] * 60 + np.arange(21)).reshape(-1)
            np.testing.assert_array_equal(block['starts'], expected_starts)
            assert len(ids) == 60 * (hi - lo)
            splits[split] = block
        params = read(out / 'prepared' / f'{task_id}_params.json')
        independently_fitted = proposal.fit(splits['A']['values'], splits['A']['starts'],
            splits['A']['cycles'][splits['A']['starts']], plan['seeds'][0])['params']
        assert params == independently_fitted
        model = core.DLinear(dict(plan['training'], channels=8)).cuda().eval()
        tensors = {s: torch.tensor(splits[s]['values'], device='cuda') for s in ('B', 'C', 'D')}

        def errors(state, action, split):
            nonlocal checked_residuals
            model.load_state_dict(state)
            result = []
            with torch.no_grad():
                for lo in range(0, len(splits[split]['starts']), 2048):
                    starts = torch.tensor(splits[split]['starts'][lo:lo + 2048], device='cuda')
                    batch = tensors[split][starts[:, None] + torch.arange(40, device='cuda')]
                    x = batch[:, :32]
                    if action != 'PARENT':
                        x = proposal.transform(x, params, 'P1', action == 'CP1')
                    result.append((model(x) - batch[:, 32:]).cpu().numpy())
            result = np.concatenate(result)
            assert np.isfinite(result).all()
            return result

        def check_saved(error, filename):
            nonlocal checked_residuals
            with np.load(filename, allow_pickle=False) as z:
                np.testing.assert_array_equal(error, z['error'])
            checked_residuals += 1

        task_truth = {s: {} for s in ('C', 'D')}
        for seed in plan['seeds']:
            parent_dir = path_for(out, task_id, 'full', 'PARENT', seed)
            parent_result = read(parent_dir / 'result.json')
            parent_state = snap(parent_dir, parent_result['selected_step'])['model']
            parent_loss = {s: mse(errors(parent_state, 'PARENT', s)) for s in ('C', 'D')}
            assert parent_loss['C'] == parent_result['selected_losses']['C'] and min(parent_loss.values()) > 0
            for action in ('PARENT', 'P1', 'CP1'):
                engines = ('full', 'proxy') if action == 'PARENT' else ('full', 'proxy', 'short')
                for engine in engines:
                    directory = path_for(out, task_id, engine, action, seed)
                    result = read(directory / 'result.json')
                    status = read(directory / 'status.json')
                    assert status['status'] == result['status'] == 'completed'
                    assert result['id'] == directory.name and result['plan_sha256'] == sha(out / 'plan.json')
                    assert (result['task'], result['engine'], result['action'], result['seed']) == (task_id, engine, action, seed)
                    if engine != 'short':
                        assert result['steps'] == 3000 and result['checkpoints'] == list(GRID)
                    else:
                        assert result['steps'] > 0 and result['checkpoints'][-1] == result['steps']
                        assert result['checkpoints'] == sorted(set([k for k in GRID if k <= result['steps']] + [result['steps']]))
                    for name, digest in result['assets_sha256'].items():
                        assert sha(directory / name) == digest
                    for step in [0] + result['checkpoints']:
                        snapshot = snap(directory, step)
                        packed = coordinates(snapshot['model'])
                        assert torch.equal(packed, snapshot['original_coordinates']) and bool(torch.isfinite(packed).all())
                        assert snapshot['step'] == step
                        if step == 0:
                            digest = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
                            assert result['initial_sha256'] == digest
                            assert initialization.setdefault((task_id, seed), digest) == digest
                            assert not snapshot['optimizer']['state']
                        else:
                            assert len(snapshot['optimizer']['state']) == 32
                            for value in snapshot['optimizer']['state'].values():
                                assert value['step'].item() == step
                                assert bool(torch.isfinite(value['exp_avg']).all()) and bool(torch.isfinite(value['exp_avg_sq']).all())
                        checked_states += 1
                    wall = read(directory / 'timing.json')['seconds']
                    assert wall > 0
                    run_costs.append(dict(id=directory.name, engine=engine, steps=result['steps'], seconds=wall))
                    if engine == 'proxy':
                        proxy_count += 1
                        assert result['selected_step'] is None and result['selection_losses'] == {}
                        continue
                    full_count += int(result['steps'] == 3000)
                    short_count += int(engine == 'short')
                    n = len(splits['A']['starts'])
                    cdf = np.cumsum(np.full(n, 1. / n)); cdf[-1] = 1.
                    rng = np.random.default_rng(seed + 1000)
                    draws = np.zeros(n, dtype=np.int64)
                    for _ in range(result['steps']):
                        np.add.at(draws, np.searchsorted(cdf, rng.random(256), side='right'), 1)
                    with np.load(directory / 'sampling.npz', allow_pickle=False) as z:
                        np.testing.assert_array_equal(draws, z['draw_count'])
                    assert snapshot['rng_numpy'] == rng.bit_generator.state
                    if engine == 'full':
                        digest = hashlib.sha256(draws.tobytes()).hexdigest()
                        assert sampling.setdefault((task_id, seed), digest) == digest
                    b_losses = {}
                    for step in result['checkpoints']:
                        b_losses[step] = mse(errors(snap(directory, step)['model'], action, 'B'))
                        assert b_losses[step] == result['selection_losses'][str(step)]
                        if engine == 'short':
                            error = errors(snap(directory, step)['model'], action, 'C')
                            check_saved(error, directory / f'step_{step}_C_errors.npz')
                            assert result['short_curves'][str(step)] == dict(B=b_losses[step], C=mse(error))
                    selected = min(result['checkpoints'], key=b_losses.get)
                    assert selected == result['selected_step']
                    state = snap(directory, selected)['model']
                    for split in ('B', 'C'):
                        error = errors(state, action, split)
                        check_saved(error, directory / f'selected_{split}_errors.npz')
                        assert mse(error) == result['selected_losses'][split]
            for action in ACTIONS:
                directory = path_for(out, task_id, 'full', action, seed)
                child_result = read(directory / 'result.json')
                true_step = child_result['selected_step']
                selected_state = snap(directory, true_step)['model']
                true_C = child_result['selected_losses']['C']
                for split in ('C', 'D'):
                    loss = true_C if split == 'C' else mse(errors(selected_state, action, split))
                    task_truth[split].setdefault(action, []).append(100. * (parent_loss[split] - loss) / parent_loss[split])
                actual_gain = task_truth['C'][action][-1]
                record = records[(task_id, seed, action)]
                scalar_records.append(dict(task=task_id, seed=seed, action=action, actual_gain=actual_gain,
                    predicted={method: record['methods'][method]['gain_percent'] for method in METHODS}))
                true_curves, full_C_errors = {}, {}
                for step in GRID:
                    true_state = snap(directory, step)['model']
                    full_C_errors[step] = errors(true_state, action, 'C')
                    true_curves[step] = dict(B=child_result['selection_losses'][str(step)], C=mse(full_C_errors[step]))
                    with (destination / f'{task_id}_{action}_{seed}_true_{step}_C.npz').open('xb') as stream:
                        np.savez_compressed(stream, error=full_C_errors[step])
                for method in ('M1', 'RAW', 'FIXED'):
                    prediction_dir = out / 'forecasts' / f'{task_id}_{action}_{seed}_{method}'
                    local = read(prediction_dir / 'result.json')
                    for key in ('method', 'task', 'action', 'seed', 'selected_step', 'gain_percent', 'curves', 'assets_sha256'):
                        assert local[key] == record['methods'][method][key]
                    assert record['methods'][method]['category'] == category(local['gain_percent'])
                    for name, digest in local['assets_sha256'].items():
                        assert sha(prediction_dir / name) == digest
                    steps = [parent_result['selected_step']] if method == 'FIXED' else list(GRID)
                    curves = {}
                    for step in steps:
                        if method == 'FIXED':
                            state = parent_state
                        else:
                            state = snap(path_for(out, task_id, 'proxy', action, seed), step)['model']
                            if method == 'M1':
                                real = snap(parent_dir, step)['model']
                                reference = snap(path_for(out, task_id, 'proxy', 'PARENT', seed), step)['model']
                                state = {key: real[key] + (state[key] - reference[key]) for key in state}
                        curves[step] = {}
                        for split in ('B', 'C'):
                            error = errors(state, action, split)
                            check_saved(error, prediction_dir / f'step_{step}_{split}_errors.npz')
                            curves[step][split] = mse(error)
                            assert curves[step][split] == local['curves'][str(step)][split]
                    selected = min(steps, key=lambda k: curves[k]['B'])
                    assert selected == local['selected_step']
                    gain = 100. * (parent_loss['C'] - curves[selected]['C']) / parent_loss['C']
                    assert gain == local['gain_percent']
                    if method in ('M1', 'RAW'):
                        checkpoints.append(dict(task=task_id, action=action, seed=seed, method=method,
                            selected_step=selected, true_step=true_step, hit=selected == true_step,
                            B_selection_regret=true_curves[selected]['B'] - true_curves[true_step]['B'],
                            C_prediction_term=curves[selected]['C'] - true_curves[selected]['C'],
                            C_selection_term=true_curves[selected]['C'] - true_curves[true_step]['C']))
                short = read(path_for(out, task_id, 'short', action, seed) / 'result.json')
                assert record['methods']['SHORT']['gain_percent'] == 100. * (parent_loss['C'] - short['selected_losses']['C']) / parent_loss['C']
                assert record['methods']['SHORT']['steps'] == short['steps']
                assert record['methods']['SHORT']['selected_step'] == short['selected_step']
                assert record['methods']['SHORT']['curves'] == short['short_curves']
                assert record['methods']['SHORT']['category'] == category(record['methods']['SHORT']['gain_percent'])
                for step in GRID:
                    with np.load(out / 'forecasts' / f'{task_id}_{action}_{seed}_RAW' / f'step_{step}_C_errors.npz') as z:
                        raw = z['error']
                    with np.load(out / 'forecasts' / f'{task_id}_{action}_{seed}_M1' / f'step_{step}_C_errors.npz') as z:
                        paired = z['error']
                    eps_T = raw.astype(np.float64) - full_C_errors[step]
                    parent_raw = errors(snap(path_for(out, task_id, 'proxy', 'PARENT', seed), step)['model'], action, 'C')
                    parent_real = errors(snap(parent_dir, step)['model'], action, 'C')
                    eps_0 = parent_raw.astype(np.float64) - parent_real
                    corrected = paired.astype(np.float64) - full_C_errors[step]
                    # Original pointwise tolerance still fails. The separately frozen diagnosis
                    # explained all gaps with parameter/forward rounding and analytical bounds.
                    numerical = numeric_cells[(task_id, seed, action, step)]
                    assert all(numerical[k] for k in ('high_identity_bound_pass', 'accounting_bound_pass',
                        'forward_bound_pass', 'saved_residuals_exact', 'parameter_coordinates_exact', 'manual_forward_exact'))
                    rhs = raw.astype(np.float64) - eps_0
                    gap = paired.astype(np.float64) - rhs
                    violations = int(np.sum(np.abs(gap) > 2e-6 + 2e-5 * np.abs(rhs)))
                    assert violations == numerical['original_violations']
                    assert float(np.max(np.abs(gap))) == numerical['gap_max']
                    eT, e0 = float(np.mean(eps_T ** 2)), float(np.mean(eps_0 ** 2))
                    cross = float(np.mean(eps_T * eps_0))
                    energies.append(dict(task=task_id, action=action, seed=seed, step=step,
                        raw_energy=eT, parent_energy=e0, cross=cross, paired_energy=float(np.mean(corrected ** 2)),
                        identity_rhs=eT + e0 - 2 * cross,
                        alignment=cross / np.sqrt(eT * e0) if eT > 0 and e0 > 0 else None,
                        original_tolerance_violations=violations, numerical_gap_max=numerical['gap_max']))
        tasks[task_id] = {split: {action: float(np.mean(vals)) for action, vals in row.items()}
                          for split, row in task_truth.items()}
    assert len(all_row_ids) == 1543 * 60 and len(run_costs) == 72
    assert proxy_count == 27 and short_count == 18 and full_count == 27 + predictions['short_completed_3000']
    cells, mae_task = [], {method: {} for method in METHODS}
    for task in plan['tasks']:
        task_id = task['id']
        for action in ACTIONS:
            selected = [r for r in scalar_records if r['task'] == task_id and r['action'] == action]
            truth = float(np.mean([r['actual_gain'] for r in selected]))
            means = {m: float(np.mean([r['predicted'][m] for r in selected])) for m in METHODS}
            cells.append(dict(task=task_id, action=action, truth=truth, predictions=means,
                actual_category=category(truth), predicted_categories={m: category(v) for m, v in means.items()},
                unstable=min(r['actual_gain'] for r in selected) < -.25 and max(r['actual_gain'] for r in selected) > .25))
        for method in METHODS:
            mae_task[method][task_id] = float(np.mean([abs(c['predictions'][method] - c['truth']) for c in cells if c['task'] == task_id]))
    mae = {m: float(np.mean(list(mae_task[m].values()))) for m in METHODS}
    best = min(BASELINES, key=lambda m: mae[m])
    task_noninferior = sum(mae_task['M1'][t] <= mae_task[best][t] for t in mae_task['M1'])
    error_gate = mae['M1'] <= .8 * mae[best] and mae[best] - mae['M1'] >= .05 and task_noninferior >= 2
    recommended, realized = {}, {s: {m: [] for m in METHODS} for s in ('C', 'D')}
    for task in plan['tasks']:
        task_id = task['id']
        recommended[task_id] = {}
        for method in METHODS:
            means = {c['action']: c['predictions'][method] for c in cells if c['task'] == task_id}
            choice = recommendation(means)
            assert predictions['recommendations'][task_id][method] == dict(action=choice, means=means, categories={a: category(v) for a, v in means.items()})
            recommended[task_id][method] = choice
            for split in ('C', 'D'):
                realized[split][method].append(0. if choice == 'PARENT' else tasks[task_id][split][choice])
    mean_gain = {s: {m: float(np.mean(v)) for m, v in row.items()} for s, row in realized.items()}
    selection_gate = mean_gain['C']['M1'] >= max(mean_gain['C'][b] for b in BASELINES) + .10 and min(realized['C']['M1']) >= -.25
    D_gate = mean_gain['D']['M1'] >= 0. and min(realized['D']['M1']) >= -.25
    # Reconstruct standalone method costs independently from saved timing receipts.
    costs = []
    for task in plan['tasks']:
        for seed in plan['seeds']:
            task_id = task['id']
            row = dict(task=task_id, seed=seed, real_parent_seconds=read(path_for(out, task_id, 'full', 'PARENT', seed) / 'timing.json')['seconds'],
                M1=read(path_for(out, task_id, 'proxy', 'PARENT', seed) / 'timing.json')['seconds'], RAW=0., FIXED=0., SHORT=0.)
            for action in ACTIONS:
                trajectory = read(path_for(out, task_id, 'proxy', action, seed) / 'timing.json')['seconds']
                row['M1'] += trajectory
                row['RAW'] += trajectory
                for method in ('M1', 'RAW', 'FIXED'):
                    row[method] += read(out / 'forecasts' / f'{task_id}_{action}_{seed}_{method}' / 'timing.json')['seconds']
                row['SHORT'] += read(path_for(out, task_id, 'short', action, seed) / 'timing.json')['seconds']
            registered = next(c for c in predictions['costs'] if c['task'] == task_id and c['seed'] == seed)
            for method in METHODS:
                assert abs(row[method] - registered[method]) < 1e-9
            costs.append(row)
    budget_gate = all(c['M1'] <= 30. and c['SHORT'] <= 30. for c in costs)
    budget_gate = budget_gate and all(r['seconds'] <= 15. for r in run_costs if r['engine'] == 'short')
    assert budget_gate
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    assert sha(out / 'predictions.json') == lock['predictions_sha256']
    assert sha(out / 'audit.log') == original_log_sha and not (out / 'final_audit.json').exists()
    for name, digest in diagnosis['checked_input_hashes'].items():
        assert sha(out / name) == digest
    result = dict(status='PASS_supplementary_verification', original_audit_status='FAILED_preserved',
        original_stage_accepted=False, diagnosis_sha256=sha(diagnosis_path), original_audit_log_sha256=original_log_sha,
        assets_verified=True, original_inputs_unchanged=True, states_verified=checked_states,
        residual_arrays_verified=checked_residuals, full_trainings=full_count, proxy_rollouts=proxy_count,
        short_trainings=short_count, cumulative_full_trainings=304 + full_count,
        prediction_MAE_pp=mae, per_task_MAE_pp=mae_task, best_baseline=best, noninferior_tasks=task_noninferior,
        cells=cells, scalar_records=scalar_records, recommendations=recommended, task_truth=tasks,
        recommendation_gains=realized, mean_recommendation_gain=mean_gain,
        checkpoint_diagnostics=checkpoints, error_energy=energies, costs=costs, run_costs=run_costs,
        budget_gate=budget_gate, error_gate=bool(error_gate), selection_gate=bool(selection_gate), D_gate=bool(D_gate),
        advance=False, descriptive_gate_conjunction=bool(budget_gate and error_gate and selection_gate and D_gate),
        audit_seconds=time.perf_counter() - started, independent_source_claim=False, test_access=False)
    write(supplement / 'supplementary_audit.json', result)
    archive_started = time.perf_counter()
    allowed = {'.py', '.json', '.md', '.log', '.npz', '.pt', '.lock'}
    paths = sorted(p for p in supplement.rglob('*') if p.is_file() and p.suffix in allowed)
    assert all(not p.is_symlink() for p in paths)
    hashes = {p.relative_to(supplement).as_posix(): sha(p) for p in paths}
    archive = supplement / 'response_proxy_supplement_20261013.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in hashes:
            tar.add(supplement / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(hashes)
        for name, digest in hashes.items():
            assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == digest
    write(supplement / 'archive_receipt.json', dict(file=archive.name, bytes=archive.stat().st_size,
          sha256=sha(archive), members_sha256=hashes, status='PASS',
          archive_seconds=time.perf_counter() - archive_started))
    print(json.dumps({k: result[k] for k in ('status', 'prediction_MAE_pp', 'error_gate', 'selection_gate', 'D_gate', 'advance')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnosis', required=True)
    args = parser.parse_args()
    audit(Path(args.source), Path(args.output), Path(args.diagnosis))
