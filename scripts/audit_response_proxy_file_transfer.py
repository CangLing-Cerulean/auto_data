"""Create-only preflight, calibration, prediction and final audits for FTR1."""
import argparse
import ast
import copy
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import audit_file_transfer_core as audit

read, sha, write = audit.read, audit.sha, audit.write
GRID, METHODS, POLICIES = audit.GRID, audit.METHODS, audit.POLICIES


def counters(ctx):
    return dict(states=ctx.states_checked, residuals=ctx.residuals_checked, moments=ctx.moments_checked,
                independent_replay_steps=ctx.replay_steps, independent_replay_trajectories=ctx.replay_trajectories)


def preflight(out):
    import response_proxy_file_transfer as runner
    started = runner.tick()
    plan, prep = audit.verify(out)
    # Extract precisely the established original per-step algorithm; no old
    # runner entry point is executed and no historical asset is changed.
    tree = ast.parse((out / 'response_proxy_selection.py').read_text(encoding='utf-8'))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    loop = next(n for n in ast.walk(train) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step')
    begin = next(i for i, n in enumerate(loop.body) if ast.unparse(n) == 'model.train()')
    end = next(i for i, n in enumerate(loop.body) if ast.unparse(n) == 'opt.step()')
    expected_body = loop.body[begin:end + 1]
    new_tree = ast.parse((out / 'response_proxy_file_transfer.py').read_text(encoding='utf-8'))
    current = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'update_step')
    assert [ast.dump(n) for n in expected_body] == [ast.dump(n) for n in current.body]
    reference = copy.deepcopy(current); reference.name = 'original_step'; reference.body = expected_body
    unit = ast.fix_missing_locations(ast.Module(body=[reference], type_ignores=[]))
    scope = dict(np=np, torch=torch, rp=runner.rp)
    exec(compile(unit, 'frozen_original_step', 'exec'), scope)
    results = []
    for task in plan['tasks']:
        independent = audit.Context(out, plan, task)
        ctx = runner.Context(out, plan, task['id'])
        seed = plan['seeds'][0]
        for action in ('PARENT', task['actions'][0]['id'], task['actions'][-1]['id']):
            for engine in ('full', 'proxy'):
                left, right = ctx.model(seed), ctx.model(seed)
                left_opt, right_opt = ctx.optimizer(left), ctx.optimizer(right)
                left_rng, right_rng = np.random.default_rng(seed + 1000), np.random.default_rng(seed + 1000)
                left_counts, right_counts = np.zeros(ctx.n, np.int64), np.zeros(ctx.n, np.int64)
                moment = ctx.moments(left, action) if engine == 'proxy' else None
                if moment is not None:
                    for name, value in zip(('gram', 'cross', 'target_square'), independent.moments(action)):
                        np.testing.assert_array_equal(getattr(moment, name).cpu().numpy(), value)
                for _ in range(25):
                    runner.update_step(ctx, left, left_opt, left_rng, left_counts, engine, action, moment)
                    scope['original_step'](ctx, right, right_opt, right_rng, right_counts, engine, action, moment)
                audit.equal(left.state_dict(), right.state_dict())
                audit.equal(left_opt.state_dict(), right_opt.state_dict())
                audit.equal(left_rng.bit_generator.state, right_rng.bit_generator.state)
                np.testing.assert_array_equal(left_counts, right_counts)
                assert not left_counts[ctx.probabilities[action] == 0].any()
                results.append(dict(task=ctx.task, action=action, engine=engine, seed=seed,
                                    steps_each=25, paired_model_Adam_RNG_exact=True))
        del independent, ctx
    audit.verify(out)
    write(out / 'preflight_audit.json', dict(status='PASS', plan_sha256=sha(out / 'plan.json'),
        preparation_sha256=sha(out / 'preparation.json'), original_update_AST_equal=True,
        diagnostics=results, diagnostic_trajectories=len(results) * 2, new_full_trainings=0,
        all_rows_windows_probabilities_reconstructed=True, test_access=False, D_access=False,
        seconds=runner.tick() - started))


def check_embedded(directory, record):
    expected = dict(record); seconds = expected.pop('seconds')
    assert expected == read(directory / 'result.json') and seconds == read(directory / 'timing.json')['seconds']


def parents(out):
    started = time.perf_counter(); plan, prep = audit.verify(out)
    preflight_receipt = read(out / 'preflight_audit.json')
    assert preflight_receipt['status'] == 'PASS' and preflight_receipt['preparation_sha256'] == sha(out / 'preparation.json')
    cal = read(out / 'calibration.json'); audit.assets(out, cal['assets_sha256'])
    assert cal['plan_sha256'] == sha(out / 'plan.json')
    assert len(cal['runs']) == 18 and len(cal['measures']) == 9
    assert {p.name for p in (out / 'runs').iterdir()} == {j['id'] for j in plan['jobs'] if j['phase'] == 'parents'}
    expected_budgets, counts = {}, []
    for task in plan['tasks']:
        ctx = audit.Context(out, plan, task); full, proxy, scores = [], [], []
        for seed in plan['seeds']:
            for engine, items in (('full', full), ('proxy', proxy)):
                result = ctx.run(engine, 'PARENT', seed)
                directory = ctx.directory(engine, 'PARENT', seed)
                embedded = next(r for r in cal['runs'] if r['id'] == result['id'])
                check_embedded(directory, embedded); items.append(embedded)
            score = ctx.forecast('PARENT', seed, 'M1')
            assert score['gain_percent'] == 0 and score['curves'] == full[-1]['curves']
            seconds = read(out / 'forecasts' / f'{ctx.task}_PARENT_{seed}_M1' / 'timing.json')['seconds']
            measure = next(m for m in cal['measures'] if m['task'] == ctx.task and m['seed'] == seed)
            assert measure['identity_score_seconds'] == seconds; scores.append(seconds)
        p, s, k = max(r['seconds'] for r in proxy), max(scores), len(task['actions'])
        c = (1 + 1 / k) * p + s
        prefixes = {str(j): min(r['prefix_seconds'][str(j)] for r in full) for j in GRID[:-1]}
        feasible = [j for j in GRID[:-1] if prefixes[str(j)] >= 1.25 * c]
        expected_budgets[ctx.task] = dict(actions=k, max_proxy_parent_seconds=p, max_identity_score_seconds=s,
            estimated_m1_seconds_per_candidate=c, calibration_margin=1.25,
            minimum_parent_prefix_seconds=prefixes, short_steps=feasible[0] if feasible else None)
        counts.append(dict(task=ctx.task, **counters(ctx))); del ctx
    assert cal['tasks'] == expected_budgets
    feasible = all(t['short_steps'] is not None for t in expected_budgets.values())
    assert cal['status'] == ('PASS' if feasible else 'STOP_NO_FEASIBLE_SHORT_PREFIX')
    assert cal['new_full_trainings'] == 9 and cal['cumulative_full_trainings'] == 418
    write(out / 'parents_audit.json', dict(status='PASS', calibration_passed=feasible,
        plan_sha256=sha(out / 'plan.json'), calibration_sha256=sha(out / 'calibration.json'),
        counts=counts, new_full_trainings=9, cumulative_full_trainings=418, seconds=time.perf_counter() - started))


def calibrated(out):
    cal, receipt, sync = (read(out / name) for name in ('calibration.json', 'parents_audit.json', 'calibration_sync_receipt.json'))
    assert cal['status'] == receipt['status'] == 'PASS' and receipt['calibration_passed']
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['plan_sha256'] == sha(out / 'plan.json')
    assert receipt['calibration_sha256'] == sync['calibration_sha256'] == sha(out / 'calibration.json')
    assert sync['parents_audit_sha256'] == sha(out / 'parents_audit.json')
    audit.assets(out, cal['assets_sha256'])
    return cal


def file_metrics(ctx, error, split):
    names = ctx.arrays[f'{split}_files'][ctx.arrays[f'{split}_starts']]
    return {name: dict(windows=int(np.count_nonzero(names == name)), nmse=audit.mse(error[names == name]),
        nmae=float(np.mean(np.abs(error[names == name].astype(np.float64))))) for name in ctx.spec['files'][split]}


def forecast(out):
    started = time.perf_counter(); plan, prep = audit.verify(out); cal = calibrated(out)
    prediction = read(out / 'predictions.json'); audit.assets(out, prediction['assets_sha256'])
    assert prediction['plan_sha256'] == sha(out / 'plan.json') and prediction['calibration_sha256'] == sha(out / 'calibration.json')
    assert prediction['status'] == 'completed' and not prediction['full_candidate_responses_observed']
    assert len(prediction['records']) == 69 and len(prediction['costs']) == len(prediction['cheap_records']) == 9
    records = {(r['task'], r['seed'], r['action']): r for r in prediction['records']}; assert len(records) == 69
    assert {p.name for p in (out / 'runs').iterdir()} == {j['id'] for j in plan['jobs'] if j['phase'] in ('parents', 'forecast')}
    counts, recommendations, costs, per_file = [], {}, [], []
    for task in plan['tasks']:
        ctx = audit.Context(out, plan, task); tid = task['id']; actions = [a['id'] for a in task['actions']]
        cheap_losses = []
        for seed in plan['seeds']:
            source, source_seconds = ctx.source(seed)
            warm = dict(M1=read(ctx.directory('proxy', 'PARENT', seed) / 'timing.json')['seconds'], RAW=0., SOURCE=source_seconds, SHORT=0.)
            parent = read(ctx.directory('full', 'PARENT', seed) / 'result.json')
            parent_state = audit.snap(ctx.directory('full', 'PARENT', seed), parent['selected_step'])['model']
            for split in ('B', 'C'):
                per_file.append(dict(task=tid, seed=seed, method='PARENT', action='PARENT', split=split,
                                     files=file_metrics(ctx, ctx.errors(parent_state, split), split)))
            for action in actions:
                row = records[(tid, seed, action)]
                proxy = ctx.run('proxy', action, seed); short = ctx.run('short', action, seed)
                proxy_seconds = read(ctx.directory('proxy', action, seed) / 'timing.json')['seconds']
                warm['SHORT'] += read(ctx.directory('short', action, seed) / 'timing.json')['seconds']
                for method in ('M1', 'RAW'):
                    local = ctx.forecast(action, seed, method)
                    seconds = read(out / 'forecasts' / f'{tid}_{action}_{seed}_{method}' / 'timing.json')['seconds']
                    assert row['predictions'][method] == dict(local, seconds=seconds)
                    warm[method] += proxy_seconds + seconds
                assert row['predictions']['SOURCE'] == source[action]
                expected_short = dict(method='SHORT', task=tid, action=action, seed=seed, selected_step=short['selected_step'],
                    curves=short['curves'], gain_percent=100. * (parent['selected_losses']['C'] - short['selected_losses']['C']) / parent['selected_losses']['C'])
                assert row['predictions']['SHORT'] == expected_short
                for method in METHODS:
                    chosen = row['predictions'][method]['selected_step']
                    if method == 'SOURCE':
                        with np.load(out / 'source' / f'{tid}_s{seed}' / action / f'step_{chosen}.npz', allow_pickle=False) as z:
                            state = ctx.state(torch.from_numpy(z['coordinates']))
                    else:
                        state = audit.snap(ctx.directory('short' if method == 'SHORT' else 'proxy', action, seed), chosen)['model']
                        if method == 'M1':
                            real = audit.snap(ctx.directory('full', 'PARENT', seed), chosen)['model']
                            pp = audit.snap(ctx.directory('proxy', 'PARENT', seed), chosen)['model']
                            state = {k: real[k] + (state[k] - pp[k]) for k in state}
                    for split in ('B', 'C'):
                        per_file.append(dict(task=tid, seed=seed, method=method, action=action, split=split,
                            files=file_metrics(ctx, ctx.errors(state, split), split)))
            error = ctx.errors(parent_state, 'A')
            cheap_dir = out / 'cheap' / f'{tid}_s{seed}'
            ctx.check_error(error, cheap_dir / 'A_errors.npz')
            cheap = read(cheap_dir / 'result.json')
            expected_losses = {name: values['nmse'] for name, values in file_metrics(ctx, error, 'A').items()}
            assert cheap['file_losses'] == expected_losses and cheap['selected_parent_step'] == parent['selected_step']
            assert cheap['error_summary_sha256'] == sha(cheap_dir / 'A_errors.npz')
            check_embedded(cheap_dir, next(r for r in prediction['cheap_records'] if (r['task'], r['seed']) == (tid, seed)))
            cheap_losses.append(expected_losses)
            warm.update(PARENT=0., LOSS_DROP=read(cheap_dir / 'timing.json')['seconds'])
            cost = next(c for c in prediction['costs'] if (c['task'], c['seed']) == (tid, seed))
            for key, value in warm.items():
                if key != 'LOSS_DROP':
                    assert abs(cost['warm'][key] - value) < 1e-9
            policies = prediction['policies'][tid]
            assert abs(cost['warm']['SIZE_DROP'] - policies['size_seconds'] / 3) < 1e-9
            assert abs(cost['warm']['LOSS_DROP'] - warm['LOSS_DROP'] - policies['loss_reduction_seconds'] / 3) < 1e-9
            common = prep['seconds'] / 9 + (read(out / f'{tid}_forecast_setup.json')['seconds'] + read(out / f'{tid}_parent_setup.json')['seconds']) / 3
            assert cost['common_preparation_allocation_seconds'] == common
            assert cost['real_parent_seconds'] == read(ctx.directory('full', 'PARENT', seed) / 'timing.json')['seconds']
            assert cost['proxy_parent_seconds'] == read(ctx.directory('proxy', 'PARENT', seed) / 'timing.json')['seconds']
            assert cost['identity_calibration_seconds'] == next(m['identity_score_seconds'] for m in cal['measures'] if (m['task'], m['seed']) == (tid, seed))
            assert set(cost['warm']) == set(METHODS + POLICIES)
            for method, value in cost['warm'].items():
                required = common + cost['real_parent_seconds'] + value
                registered = required + cost['identity_calibration_seconds'] + (0. if method == 'M1' else cost['proxy_parent_seconds'])
                assert abs(cost['cold_required_computation'][method] - required) < 1e-9
                assert abs(cost['cold_registered_execution'][method] - registered) < 1e-9
            assert cost['m1_not_more_expensive_than_short'] == (warm['M1'] <= warm['SHORT'])
            costs.append(cost)
            print(json.dumps(dict(audit='forecast', task=tid, seed=seed)), flush=True)
        policies = prediction['policies'][tid]
        names = ctx.arrays['A_files'][ctx.arrays['A_starts']]
        sizes = {a['id']: int(np.count_nonzero(names == a['removed_source_file'])) for a in task['actions']}
        losses = {a['id']: float(np.mean([r[a['removed_source_file']] for r in cheap_losses])) for a in task['actions']}
        assert policies['window_counts'] == sizes and policies['mean_file_losses'] == losses
        assert policies['SIZE_DROP'] == max(actions, key=lambda a: sizes[a])
        assert policies['LOSS_DROP'] == max(actions, key=lambda a: losses[a]) and policies['PARENT'] == 'PARENT'
        recommendations[tid] = {}
        for method in METHODS:
            means = {a: float(np.mean([records[(tid, s, a)]['predictions'][method]['gain_percent'] for s in plan['seeds']])) for a in actions}
            best = max(actions, key=lambda a: means[a]); choice = best if means[best] >= 1. else 'PARENT'
            recommendations[tid][method] = dict(action=choice, means=means)
        for method in POLICIES:
            recommendations[tid][method] = dict(action=policies[method])
        counts.append(dict(task=tid, **counters(ctx))); del ctx
    assert prediction['recommendations'] == recommendations
    agreement = [m for m in ('RAW', 'SOURCE', 'SHORT') + POLICIES if all(recommendations[t['id']]['M1']['action'] == recommendations[t['id']][m]['action'] for t in plan['tasks'])]
    cost_gate = all(c['m1_not_more_expensive_than_short'] for c in costs)
    assert prediction['cost_gate'] == cost_gate and prediction['identical_recommendation_baselines'] == agreement
    assert prediction['eligible_to_label'] == (cost_gate and not agreement)
    assert prediction['new_full_trainings'] == 0 and prediction['cumulative_full_trainings'] == 418
    write(out / 'forecast_audit.json', dict(status='PASS', plan_sha256=sha(out / 'plan.json'),
        predictions_sha256=sha(out / 'predictions.json'), counts=counts, cost_gate=cost_gate,
        eligible_to_label=cost_gate and not agreement, identical_recommendation_baselines=agreement,
        selected_per_file_metrics=per_file, seconds=time.perf_counter() - started,
        new_full_trainings=0, cumulative_full_trainings=418, test_access=False, D_access=False))


def final(out):
    started = time.perf_counter(); plan, prep = audit.verify(out); calibrated(out)
    predictions, labels, sync, forecast_audit = (read(out / n) for n in
        ('predictions.json', 'labels.json', 'prediction_sync_receipt.json', 'forecast_audit.json'))
    assert sync['status'] == 'exact_commit_and_tree_verified' and forecast_audit['status'] == 'PASS'
    assert sync['plan_sha256'] == sha(out / 'plan.json') and labels['plan_sha256'] == sha(out / 'plan.json')
    assert sync['predictions_sha256'] == labels['predictions_sha256'] == forecast_audit['predictions_sha256'] == sha(out / 'predictions.json')
    assert sync['forecast_audit_sha256'] == sha(out / 'forecast_audit.json')
    assert predictions['eligible_to_label'] and forecast_audit['eligible_to_label'] and labels['status'] == 'completed'
    assert read(out / 'labels.lock')['predictions_sha256'] == sha(out / 'predictions.json')
    audit.assets(out, predictions['assets_sha256'])
    assert {p.name for p in (out / 'runs').iterdir()} == {j['id'] for j in plan['jobs']}
    assert len(labels['runs']) == len({r['id'] for r in labels['runs']}) == 69
    records = {(r['task'], r['seed'], r['action']): r for r in predictions['records']}
    rows, energies, costs, counts, per_file = [], [], [], [], []
    for task in plan['tasks']:
        ctx = audit.Context(out, plan, task); tid = task['id']
        for seed in plan['seeds']:
            parent = read(ctx.directory('full', 'PARENT', seed) / 'result.json'); baseline = parent['selected_losses']['C']
            full_seconds = 0.
            for action in task['actions']:
                name = action['id']; result = ctx.run('full', name, seed)
                check_embedded(ctx.directory('full', name, seed), next(r for r in labels['runs'] if r['id'] == result['id']))
                full_seconds += read(ctx.directory('full', name, seed) / 'timing.json')['seconds']
                actual = 100. * (baseline - result['selected_losses']['C']) / baseline
                predicted = {m: records[(tid, seed, name)]['predictions'][m]['gain_percent'] for m in METHODS}
                rows.append(dict(task=tid, seed=seed, action=name, actual_gain=actual, predicted=predicted))
                energies.extend(ctx.energy(seed, name, k) for k in GRID)
                state = audit.snap(ctx.directory('full', name, seed), result['selected_step'])['model']
                for split in ('B', 'C'):
                    per_file.append(dict(task=tid, seed=seed, action=name, split=split,
                                         files=file_metrics(ctx, ctx.errors(state, split), split)))
            warm = next(c['warm']['M1'] for c in predictions['costs'] if (c['task'], c['seed']) == (tid, seed))
            costs.append(dict(task=tid, seed=seed, actual_all_full_children_seconds=full_seconds,
                              M1_seconds=warm, at_most_half_full_cost=warm <= .5 * full_seconds))
            print(json.dumps(dict(audit='final', task=tid, seed=seed)), flush=True)
        counts.append(dict(task=tid, **counters(ctx))); del ctx
    cells, decisions = [], []
    task_mae = {m: {} for m in METHODS}; realized = {m: [] for m in METHODS + POLICIES}
    for task in plan['tasks']:
        tid = task['id']
        for action in task['actions']:
            subset = [r for r in rows if (r['task'], r['action']) == (tid, action['id'])]
            assert len(subset) == 3
            true = float(np.mean([r['actual_gain'] for r in subset]))
            predicted = {m: float(np.mean([r['predicted'][m] for r in subset])) for m in METHODS}
            cells.append(dict(task=tid, action=action['id'], actual_gain=true, predicted=predicted,
                actual_category=audit.previous.category(true), predicted_categories={m: audit.previous.category(v) for m, v in predicted.items()},
                unstable=min(r['actual_gain'] for r in subset) < -.25 and max(r['actual_gain'] for r in subset) > .25))
        subset = [c for c in cells if c['task'] == tid]
        oracle = max([0.] + [c['actual_gain'] for c in subset])
        threshold_oracle = max([0.] + [c['actual_gain'] for c in subset if c['actual_gain'] >= 1.])
        for method in METHODS:
            task_mae[method][tid] = float(np.mean([abs(c['actual_gain'] - c['predicted'][method]) for c in subset]))
        for method in METHODS + POLICIES:
            choice = predictions['recommendations'][tid][method]['action']
            gain = 0. if choice == 'PARENT' else next(c['actual_gain'] for c in subset if c['action'] == choice)
            realized[method].append(gain)
            decisions.append(dict(task=tid, method=method, action=choice, actual_gain=gain,
                raw_oracle=oracle, threshold_oracle=threshold_oracle, raw_regret=oracle - gain,
                threshold_regret=threshold_oracle - gain, accepted_harm=gain < -.25,
                effective_actions=sum(c['actual_gain'] >= 1. for c in subset),
                covered_effective_actions=sum(c['actual_gain'] >= 1. and c['predicted'][method] >= 1. for c in subset) if method in METHODS else None))
    mae = {m: float(np.mean(list(v.values()))) for m, v in task_mae.items()}
    micro_mae = {m: float(np.mean([abs(c['actual_gain'] - c['predicted'][m]) for c in cells])) for m in METHODS}
    seed_mae = {m: float(np.mean([abs(r['actual_gain'] - r['predicted'][m]) for r in rows])) for m in METHODS}
    best = min(('RAW', 'SOURCE', 'SHORT'), key=lambda m: mae[m])
    noninferior = sum(task_mae['M1'][t] <= task_mae[best][t] for t in task_mae['M1'])
    precision_gate = mae['M1'] <= .8 * mae[best] and mae[best] - mae['M1'] >= .05 and noninferior >= 2
    mean_gain = {m: float(np.mean(v)) for m, v in realized.items()}
    decision_gate = mean_gain['M1'] >= max(mean_gain[m] for m in ('RAW', 'SOURCE', 'SHORT') + POLICIES) + .10 and min(realized['M1']) >= -.25
    cost_gate = predictions['cost_gate'] and all(c['at_most_half_full_cost'] for c in costs)
    assert len(rows) == 69 and len(cells) == 23 and len(energies) == 414
    assert labels['new_full_trainings'] == 69 and labels['cumulative_full_trainings'] == 487
    audit.verify(out); audit.assets(out, predictions['assets_sha256'])
    write(out / 'final_audit.json', dict(status='PASS', plan_sha256=sha(out / 'plan.json'),
        predictions_sha256=sha(out / 'predictions.json'), labels_sha256=sha(out / 'labels.json'), counts=counts,
        rows=rows, cells=cells, decisions=decisions, error_energy=energies, actual_costs=costs,
        selected_per_file_metrics=per_file, task_MAE_pp=task_mae, task_macro_MAE_pp=mae,
        cell_micro_MAE_pp=micro_mae, seed_cell_micro_MAE_pp=seed_mae, best_dynamic_baseline=best,
        noninferior_tasks=noninferior, mean_recommendation_gain=mean_gain, recommendation_gains=realized,
        precision_gate=precision_gate, decision_gate=decision_gate, cost_gate=cost_gate,
        advance=precision_gate and decision_gate and cost_gate, new_full_trainings=78,
        cumulative_full_trainings=487, seconds=time.perf_counter() - started, test_access=False,
        D_access=False, known_source_development=True, independent_task_claim=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('preflight', 'parents', 'forecast', 'final'))
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.set_num_threads(4)
    try:
        globals()[args.phase](args.output)
    except Exception:
        if args.output == Path('/root/auto_data_experiments/20261014_file_transfer_r1') and args.output.resolve() == args.output and args.output.is_dir():
            write(args.output / f'{args.phase}_audit_failure.json', dict(status='failed', error=traceback.format_exc()))
        raise
