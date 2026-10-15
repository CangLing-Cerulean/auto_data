"""Frozen four-group M1 phases; candidate labels require independent locks."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import fourgroup_policy as policy
import fourgroup_runtime as run
import response_proxy as rp
import source_dlinear_adapter as source


def prepare(out):
    started = time.perf_counter()
    plan = run.verify(out)
    destination = out / 'prepared'
    destination.mkdir()
    records = []
    for task in plan['tasks']:
        tid = task['id']
        with np.load(out / 'inputs/prepared' / f'{tid}_ABC.npz', allow_pickle=False) as old:
            starts = old['A_starts']
            cycles = old['A_cycles'][starts]
            rows = old['A_base_rows'][starts]
            groups = policy.assign_groups(tid, cycles, task['partitions']['A'])
        for action in run.ACTIONS:
            weights = policy.action_weights(groups, action)
            run.arrays(destination / f'{tid}_{action}.npz', weights=weights.window_weights,
                group_weights=weights.group_weights, groups=groups,
                window_cycles=cycles, window_base_rows=rows)
            records.append(dict(task=tid, action=action, mean_weight=weights.mean_weight,
                mass_deviation=weights.mass_deviation, effective_radius=weights.effective_radius,
                ess_fraction=weights.ess_fraction, windows=6468, cycles=308,
                sampling='fixed_uniform', all_windows_retained=True))
    assets = {p.relative_to(out).as_posix(): run.sha(p) for p in sorted(destination.iterdir())}
    run.write(out / 'preparation.json', dict(status='PASS', tasks=records,
        assets_sha256=assets, plan_sha256=run.sha(out / 'plan.json'),
        seconds=time.perf_counter() - started, new_full_trainings=0, D_access=False, test_access=False))


def preflight_pass(out):
    audit = run.read(out / 'preflight_audit.json')
    assert audit['status'] == 'PASS' and audit['plan_sha256'] == run.sha(out / 'plan.json')
    return audit


def calibration_pass(out):
    preflight_pass(out)
    result, audit, sync = (run.read(out / name) for name in
        ('calibration.json', 'calibration_audit.json', 'calibration_sync_receipt.json'))
    assert result['status'] == 'completed' and result['eligible_to_forecast']
    assert audit['status'] == 'PASS' and audit['eligible_to_forecast']
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['plan_sha256'] == run.sha(out / 'plan.json')
    assert sync['calibration_sha256'] == audit['calibration_sha256'] == run.sha(out / 'calibration.json')
    assert sync['calibration_audit_sha256'] == run.sha(out / 'calibration_audit.json')
    return result


def calibrate(out):
    started = time.perf_counter()
    plan = run.verify(out)
    preflight_pass(out)
    run.write(out / 'calibration.lock', dict(plan_sha256=run.sha(out / 'plan.json')))
    records, identities, setup = [], [], []
    for task in plan['tasks']:
        begin = run.tick()
        ctx = run.Context(out, plan, task['id'])
        setup.append(dict(task=ctx.task, seconds=run.tick() - begin))
        for seed in run.SEEDS:
            for engine in ('full', 'proxy'):
                record = run.train(ctx, engine, 'PARENT', seed, steps=3000, replay=True)
                records.append(record)
                run.write(out / 'calibration_status.json', dict(status='running',
                    replays_completed=len(records), new_full_trainings=0,
                    last_id=record['id']), update=True)
                print(dict(phase='calibration', replay=len(records), id=record['id']), flush=True)
            identities.append(run.score_proxy(ctx, 'PARENT', seed, 'M1', identity=True))
        del ctx
    tasks, failures = {}, []
    for tid in run.TASKS:
        proxy = {seed: next(r['timing']['seconds'] for r in records if
            (r['task'], r['seed'], r['engine']) == (tid, seed, 'proxy')) for seed in run.SEEDS}
        identity = {seed: next(r['seconds'] for r in identities if
            (r['task'], r['seed']) == (tid, seed)) for seed in run.SEEDS}
        prefixes = {seed: {step: next(r['timing']['prefix_seconds'][step] for r in records if
            (r['task'], r['seed'], r['engine']) == (tid, seed, 'full'))
            for step in policy.SHORT_STEPS} for seed in run.SEEDS}
        try:
            tasks[tid] = asdict(policy.calibrate_short(proxy, identity, prefixes))
        except policy.BudgetCalibrationError as error:
            failures.append(dict(task=tid, reason=str(error)))
            tasks[tid] = dict(short_steps=None, proxy_seconds=proxy,
                identity_seconds=identity, prefix_seconds=prefixes, failed=True)
    result = dict(status='calibration_budget_failed' if failures else 'completed',
        eligible_to_forecast=not failures, failures=failures, records=records,
        identity_scores=identities, tasks=tasks, setup=setup,
        original_parent_replays=9, original_proxy_replays=9, replay_update_steps=54000,
        new_full_trainings=0, cumulative_full_trainings=418,
        plan_sha256=run.sha(out / 'plan.json'), phase_seconds=time.perf_counter() - started,
        D_access=False, test_access=False)
    run.write(out / 'calibration.json', result)
    run.write(out / 'calibration_status.json', dict(status=result['status'],
        replays_completed=18, eligible_to_forecast=not failures,
        calibration_sha256=run.sha(out / 'calibration.json')), update=True)


def source_forecast(ctx, action, seed):
    directory = ctx.out / 'forecasts' / f'{ctx.task}_{action}_{seed}_SOURCE'
    directory.mkdir(parents=True)
    started = run.tick()
    model = ctx.model(seed)
    parent_moment, child_moment = (ctx.moments(model, a) for a in ('PARENT', action))
    packed = []
    for name, moment in (('parent', parent_moment), ('action', child_moment)):
        values = {k: getattr(moment, k).cpu().numpy() for k in ('gram', 'cross', 'target_square')}
        run.arrays(directory / f'{name}_moments.npz', **values)
        packed.append(values)
    cpu_model = run.core.DLinear(dict(ctx.plan['training'], channels=8))
    parent = ctx.parent('full', seed)
    baseline = run.read(parent / 'result.json')['selected_losses']['C']
    delta = np.zeros((8, 66, 8), dtype=np.float64)
    curves, diagnostics = {}, []
    for step in run.GRID:
        state = source.parent_coordinates(cpu_model, run.load_snapshot(parent, step))
        delta, checks = source.quadratic_segment(packed[0]['gram'], packed[0]['cross'],
            packed[1]['gram'], packed[1]['cross'], state, delta)
        coordinates = (state['theta'] + delta).astype(np.float32)
        run.arrays(directory / f'step_{step}.npz', delta=delta, coordinates=coordinates)
        rp.load_parameters(model, torch.tensor(coordinates, device='cuda'))
        curves[step] = {s: ctx.evaluate(model, s) for s in ('B', 'C')}
        diagnostics.append(dict(step=step,
            clipped_negative_eigenvalues=sum(c['clipped_negative_eigenvalues'] for c in checks),
            max_roundoff_tolerance=max(c['roundoff_tolerance'] for c in checks)))
    selected = rp.select_checkpoint({k: curves[k]['B']['nmse'] for k in run.GRID}, list(run.GRID))
    result = dict(task=ctx.task, seed=seed, action=action, method='SOURCE', curves=curves,
        selected_step=selected, selected_metrics=curves[selected], baseline_C=baseline,
        gain_percent=100. * (1. - curves[selected]['C']['nmse'] / baseline),
        numerical_diagnostics=diagnostics,
        assets_sha256={p.name: run.sha(p) for p in sorted(directory.iterdir())})
    run.write(directory / 'result.json', result)
    digest = run.sha(directory / 'result.json')
    seconds = run.tick() - started
    run.write(directory / 'timing.json', dict(seconds=seconds, result_sha256=digest,
        boundary='own parent/action moment construction through response, metrics, result and hashing'))
    return dict(result, seconds=seconds, directory=directory.relative_to(ctx.out).as_posix())


def forecast(out):
    started = time.perf_counter()
    plan = run.verify(out)
    calibration = calibration_pass(out)
    run.write(out / 'forecast.lock', dict(plan_sha256=run.sha(out / 'plan.json'),
        calibration_sha256=run.sha(out / 'calibration.json')))
    records, costs, setup = [], [], []
    for task in plan['tasks']:
        begin = run.tick()
        ctx = run.Context(out, plan, task['id'])
        setup.append(dict(task=ctx.task, seconds=run.tick() - begin))
        for seed in run.SEEDS:
            proxy_parent = next(r for r in calibration['records'] if
                (r['task'], r['engine'], r['seed']) == (ctx.task, 'proxy', seed))
            real_parent = next(r for r in calibration['records'] if
                (r['task'], r['engine'], r['seed']) == (ctx.task, 'full', seed))
            cost = dict(task=ctx.task, seed=seed, M1=proxy_parent['timing']['seconds'],
                RAW=0., SOURCE=0., SHORT=0., PARENT=0.,
                real_parent_seconds=real_parent['timing']['seconds'])
            for action in run.ACTIONS:
                proxy = run.train(ctx, 'proxy', action, seed, steps=3000)
                methods = {m: run.score_proxy(ctx, action, seed, m) for m in ('M1', 'RAW')}
                methods['SOURCE'] = source_forecast(ctx, action, seed)
                short = run.train(ctx, 'short', action, seed, steps=calibration['tasks'][ctx.task]['short_steps'])
                baseline = run.read(ctx.parent('full', seed) / 'result.json')['selected_losses']['C']
                methods['SHORT'] = dict(method='SHORT', task=ctx.task, action=action, seed=seed,
                    selected_step=short['selected_step'], selected_metrics=short['selected_metrics'],
                    curves=short['curves'], steps=short['steps'], directory=short['directory'],
                    gain_percent=100. * (1. - short['selected_metrics']['C']['nmse'] / baseline),
                    seconds=short['timing']['seconds'])
                methods['PARENT'] = dict(method='PARENT', gain_percent=0., seconds=0.)
                for method in ('M1', 'RAW'):
                    cost[method] += proxy['timing']['seconds'] + methods[method]['seconds']
                cost['SOURCE'] += methods['SOURCE']['seconds']
                cost['SHORT'] += short['timing']['seconds']
                records.append(dict(task=ctx.task, seed=seed, action=action,
                    proxy_directory=proxy['directory'], methods=methods))
                run.write(out / 'forecast_status.json', dict(status='running', records=len(records),
                    registered_records=72, proxy_rollouts=len(records), short_runs=len(records),
                    new_full_trainings=0, last_id=short['id']), update=True)
                print(dict(phase='forecast', records=len(records), id=short['id']), flush=True)
            cost['budget_pass'] = cost['M1'] <= cost['SHORT']
            costs.append(cost)
        del ctx
    recommendations, choices = {}, {m: {} for m in run.METHODS}
    for tid in run.TASKS:
        recommendations[tid] = {}
        for method in run.METHODS:
            gains = {seed: {action: next(r['methods'][method]['gain_percent'] for r in records if
                (r['task'], r['seed'], r['action']) == (tid, seed, action)) for action in run.ACTIONS}
                for seed in run.SEEDS}
            recommendations[tid][method] = asdict(policy.recommend(gains))
            choices[method][tid] = recommendations[tid][method]['action']
    stop = asdict(policy.recommendation_stop(choices))
    budget_pass = all(c['budget_pass'] for c in costs)
    common_preparation = run.read(out / 'preparation.json')['seconds'] + plan['historical_preparation_seconds']
    for cost in costs:
        common = common_preparation / 9 + next(s['seconds'] for s in setup if s['task'] == cost['task']) / 3
        cost['allocated_common_preparation_seconds'] = common
        cost['cold_allocated_seconds'] = {m: cost[m] + cost['real_parent_seconds'] + common for m in run.METHODS}
    assets = {p.relative_to(out).as_posix(): run.sha(p) for directory in ('runs', 'forecasts')
              for p in sorted((out / directory).rglob('*')) if p.is_file()}
    run.verify(out)
    result = dict(status='completed', records=records, recommendations=recommendations,
        costs=costs, setup=setup, budget_gate=budget_pass, identical_recommendation_stop=stop,
        eligible_to_label=budget_pass and not stop['stop'], assets_sha256=assets,
        plan_sha256=run.sha(out / 'plan.json'), calibration_sha256=run.sha(out / 'calibration.json'),
        proxy_rollouts=72, short_runs=72, new_full_trainings=0, cumulative_full_trainings=418,
        scientific_accuracy=None, actual_gain=None, full_children_started=False,
        phase_seconds=time.perf_counter() - started, D_access=False, test_access=False)
    run.write(out / 'predictions.json', result)
    run.write(out / 'forecast_status.json', dict(status='completed', records=72,
        eligible_to_label=result['eligible_to_label'], predictions_sha256=run.sha(out / 'predictions.json')), update=True)


def labels_pass(out):
    calibration_pass(out)
    prediction, audit, sync, lock = (run.read(out / name) for name in
        ('predictions.json', 'forecast_audit.json', 'prediction_sync_receipt.json', 'prediction_lock.json'))
    assert prediction['status'] == 'completed' and prediction['eligible_to_label']
    assert audit['status'] == 'PASS' and audit['eligible_to_label']
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['plan_sha256'] == lock['plan_sha256'] == run.sha(out / 'plan.json')
    assert sync['predictions_sha256'] == lock['predictions_sha256'] == audit['predictions_sha256'] == run.sha(out / 'predictions.json')
    assert sync['forecast_audit_sha256'] == run.sha(out / 'forecast_audit.json')
    assert sync['prediction_lock_sha256'] == run.sha(out / 'prediction_lock.json')
    for name, digest in prediction['assets_sha256'].items():
        assert run.sha(run.safe_file(out, name)) == digest, name
    return prediction


def labels(out):
    started = time.perf_counter()
    plan = run.verify(out)
    labels_pass(out)
    run.write(out / 'labels.lock', dict(plan_sha256=run.sha(out / 'plan.json'),
        predictions_sha256=run.sha(out / 'predictions.json')))
    records = []
    for task in plan['tasks']:
        ctx = run.Context(out, plan, task['id'])
        for seed in run.SEEDS:
            for action in run.ACTIONS:
                result = run.train(ctx, 'full', action, seed, steps=3000)
                records.append(result)
                run.write(out / 'labels_status.json', dict(status='running', completed=len(records),
                    cumulative_full_trainings=418 + len(records), last_id=result['id']), update=True)
                print(dict(phase='labels', completed=len(records), id=result['id']), flush=True)
        del ctx
    run.write(out / 'labels.json', dict(status='completed', records=records,
        new_full_trainings=72, cumulative_full_trainings=490,
        predictions_sha256=run.sha(out / 'predictions.json'),
        phase_seconds=time.perf_counter() - started, D_access=False, test_access=False))
    run.write(out / 'labels_status.json', dict(status='completed', completed=72,
        cumulative_full_trainings=490), update=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'calibrate', 'forecast', 'labels'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run.configure()
    try:
        globals()[args.phase](args.output)
    except Exception:
        if args.output == run.ROOT and args.output.resolve() == run.ROOT:
            failure = args.output / f'{args.phase}_failure.json'
            if not failure.exists():
                run.write(failure, dict(status='failed', phase=args.phase, error=traceback.format_exc(),
                    D_access=False, test_access=False))
        raise
