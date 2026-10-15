"""Independent NumPy policy checks, saved-state inference and fixed replays."""
import copy
import itertools
import time

import numpy as np
import torch

import adam_risk_sampling as algebra
import adam_sampling_pilot as study
import fourgroup_runtime as rt


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def reference_allocation(a):
    # Enumerate every possible positive free set, independently of active-set code.
    a = np.asarray(a, dtype=float)
    lower = np.full(4, .125)
    if np.all(a == 0):
        return np.full(4, .25)
    candidates = []
    positive = np.flatnonzero(a > 0)
    if len(positive):
        for length in range(1, len(positive)+1):
            for subset in itertools.combinations(positive, length):
                p = lower.copy()
                idx = np.asarray(subset)
                roots = np.sqrt(a[idx]/np.max(a[idx]))
                p[idx] = (1-.125*(4-length))*roots/roots.sum()
                if np.all(p >= lower-1e-15):
                    candidates.append(p)
    else:
        for i in range(4):
            p = lower.copy()
            p[i] += .5
            candidates.append(p)
    return min(candidates, key=lambda p: float(np.sum(a/p)))


def restore(ctx, directory, seed, step):
    saved = rt.load_snapshot(directory, step)
    model = ctx.model(seed)
    model.load_state_dict(saved['model'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
    rt.equal(rt.rp.parameters(model).cpu(), saved['original_coordinates'])
    assert saved['step'] == step
    for parameter in model.parameters():
        assert int(optimizer.state[parameter].get('step', 0)) == step
    return model, optimizer, saved


def numpy_reference(ctx, model, optimizer, step, method):
    theta = study.pack(model).cpu().numpy()
    x, y = ctx.phi['A'].cpu().numpy(), ctx.targets['A'].cpu().numpy()
    residual = np.einsum('ncd,cdh->nhc', x, theta, optimize=True)-y
    if method != 'gradient_norm':
        m = study.pack(model, optimizer, 'exp_avg').cpu().numpy()
        v = study.pack(model, optimizer, 'exp_avg_sq').cpu().numpy()
        mu = np.einsum('ncd,nhc->cdh', x, residual, optimize=True)/(6468*32)
        try:
            center, first, second = algebra.adam_derivatives(theta, m, v, mu, next_step=step+1)
        except algebra.NonsmoothAdam:
            return dict(regular=False, coefficients=np.zeros(4), probabilities=np.full(4, .25))
        xc, yc = ctx.phi['C'].cpu().numpy(), ctx.targets['C'].cpu().numpy()
        rc = np.einsum('ncd,cdh->nhc', xc, center, optimize=True)-yc
        risk = np.einsum('ncd,nhc->cdh', xc, rc, optimize=True)/(1617*32)
        hessian = np.einsum('ncd,nce->cde', xc, xc, optimize=True)/(1617*32)
    scores = []
    for lo in range(0, 6468, 256):
        g = np.einsum('ncd,nhc->ncdh', x[lo:lo+256], residual[lo:lo+256])/32
        if method == 'gradient_norm':
            score = np.sum(g*g, axis=(1,2,3))
        else:
            transformed = g*first
            score = np.einsum('ncdh,cde,nceh->n', transformed, hessian, transformed, optimize=True)
            if method == 'adam_risk':
                score += np.sum(g*g*risk*second, axis=(1,2,3))
        scores.append(score)
    scores = np.concatenate(scores)
    coefficients = np.array([scores[ctx.groups_np == g].mean()/16 for g in range(4)])
    result = dict(regular=True, coefficients=coefficients, probabilities=reference_allocation(coefficients))
    if method != 'gradient_norm':
        result['derivatives'] = [center, first, second]
    return result


def compare_policy(actual, reference):
    assert actual['regular'] == reference['regular']
    close(actual['coefficients'], reference['coefficients'])
    close(actual['probabilities'], reference['probabilities'])
    for a, b in zip(actual.get('derivatives', []), reference.get('derivatives', [])):
        close(a, b)
    return float(np.max(np.abs(np.asarray(actual['coefficients'])-reference['coefficients'])))


def preflight(out, plan):
    started = rt.tick()
    rows = []
    for task in rt.TASKS:
        ctx = study.Context(out, plan, task, 'adam_risk')
        for seed in rt.SEEDS:
            model, optimizer, _ = restore(ctx, ctx.parent(seed), seed, 500)
            reference = numpy_reference(ctx, model, optimizer, 500, 'adam_risk')
            actual = study.proposal(ctx, model, optimizer, 500, 'adam_risk', witness=True)
            error = compare_policy(actual, reference)
            rows.append(dict(task=task, seed=seed, step=500, regular=reference['regular'], max_coefficient_error=error))
            print(dict(phase='preflight', task=task, seed=seed, status='PASS'), flush=True)
    result = dict(status='PASS', rows=rows, parent_states=9, rtol=1e-9, atol=1e-11,
        seconds=rt.tick()-started, plan_sha256=rt.sha(out/'plan.json'), new_full_trainings=0)
    rt.write(out/'preflight.json', result)
    return result


def replay(ctx, directory, seed, method):
    started = rt.tick()
    model = ctx.model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rng = np.random.default_rng(seed+1000)
    rt.equal(rt.snapshot(model, optimizer, rng, 0), rt.load_snapshot(directory, 0))
    cdf = np.cumsum(np.full(6468, 1/6468)); cdf[-1] = 1.
    weights = torch.ones(6468, device='cuda')
    counts = np.zeros(6468, dtype=np.int64)
    for step in range(1, 3001):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ids = np.searchsorted(cdf, rng.random(256), side='right')
        np.add.at(counts, ids, 1)
        index = torch.tensor(ids, device='cuda')
        batch = ctx.batch('A', index)
        error = model(batch[:, :32])-batch[:, 32:]
        loss = error.square().mean() if step <= 500 else (error.square()*weights[index, None, None]).mean()
        loss.backward(); optimizer.step()
        if step in rt.GRID:
            rt.equal(rt.snapshot(model, optimizer, rng, step), rt.load_snapshot(directory, step))
        if step in study.REFRESH:
            p = np.array(rt.read(directory/f'proposal_{step}.json')['probabilities'])
            # Independent construction of inverse CDF and inverse-probability loss.
            window_mass = p[ctx.groups_np]/1617
            cdf = np.add.accumulate(window_mass); cdf[-1] = 1.
            weights = torch.tensor(.25/p[ctx.groups_np], dtype=torch.float32, device='cuda')
    with np.load(directory/'sampling.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['counts'], counts)
    return dict(run=directory.name, states=7, bitexact=True, seconds=rt.tick()-started)


def decision(rows):
    by = {(x['task'], x['seed'], x['method']): x for x in rows}
    tasks = []
    for task in rt.TASKS:
        gains, nmae = {}, {}
        for method in ('uniform',)+study.METHODS:
            gains[method] = [100*(1-by[task,s,method]['best_C']['nmse']/by[task,s,'uniform']['best_C']['nmse']) for s in rt.SEEDS]
            nmae[method] = [100*(1-by[task,s,method]['best_C']['nmae']/by[task,s,'uniform']['best_C']['nmae']) for s in rt.SEEDS]
        increments = {m: (np.array(gains['adam_risk'])-gains[m]).tolist() for m in ('gradient_norm', 'geometry')}
        ratio = np.mean([by[task,s,'adam_risk']['method_seconds'] for s in rt.SEEDS])/np.mean([by[task,s,'uniform']['method_seconds'] for s in rt.SEEDS])
        qualifies = (np.mean(gains['adam_risk']) >= 1 and min(gains['adam_risk']) > 0
            and all(np.mean(x) >= .10 and min(x) > 0 for x in increments.values()) and np.mean(nmae['adam_risk']) >= 0)
        harm_ok = np.mean(gains['adam_risk']) >= -.25 and np.mean(nmae['adam_risk']) >= -.25
        tasks.append(dict(task=task, gain_percent=gains, nmae_gain_percent=nmae,
            increments_pp=increments, risk_to_uniform_cost_ratio=float(ratio),
            benefit_gate=bool(qualifies), harm_gate=bool(harm_ok), cost_gate=bool(ratio <= 1.10)))
    advance = sum(x['benefit_gate'] for x in tasks) >= 2 and all(x['harm_gate'] and x['cost_gate'] for x in tasks)
    return dict(tasks=tasks, advance=advance, conclusion='prepare_confirmation' if advance else 'end_current_method')


def audit(out, plan):
    started = rt.tick()
    rows, replays, policy_checks = [], [], []
    for task_index, task in enumerate(rt.TASKS):
        for seed_index, seed in enumerate(rt.SEEDS):
            for method in ('uniform',)+study.METHODS:
                directory = out/'runs'/study.run_id(task, method, seed)
                result, status = rt.read(directory/'result.json'), rt.read(directory/'status.json')
                assert result['status'] == status['status'] == 'completed'
                assert status['steps_completed'] == result['steps'] == 3000
                for name, digest in result['assets_sha256'].items():
                    assert rt.sha(rt.safe_file(directory, name)) == digest
                ctx = study.Context(out, plan, task, method, for_audit=True)
                for step in rt.GRID:
                    model, optimizer, saved = restore(ctx, directory, seed, step)
                    rt.equal(ctx.evaluate(model, 'B'), result['B_curve'][str(step)])
                    if method != 'uniform' and step in study.REFRESH:
                        record = rt.read(directory/f'proposal_{step}.json')
                        assert record == result['proposals'][study.REFRESH.index(step)]
                        assert record['parent_checkpoint_sha256'] == rt.sha(directory/f'step_{step}.pt')
                        reference = numpy_reference(ctx, model, optimizer, step, method)
                        error = compare_policy(record, reference)
                        p = np.array(record['probabilities'])
                        assert np.all(p >= .125-1e-15) and abs(p.sum()-1) < 1e-12
                        policy_checks.append(dict(run=directory.name, step=step, max_coefficient_error=error, regular=reference['regular']))
                best_step = min(rt.GRID, key=lambda k: result['B_curve'][str(k)]['nmse'])
                assert best_step == result['best_step']
                scores = {}
                for label, step in (('best', best_step), ('last', 3000)):
                    model, _, _ = restore(ctx, directory, seed, step)
                    scores[label] = ctx.evaluate(model, 'C')
                timing = rt.read(directory/'timing.json')
                assert 0 < timing['method_seconds'] <= timing['actual_elapsed_seconds']
                row = dict(task=task, seed=seed, method=method, best_step=best_step,
                    best_C=scores['best'], last_C=scores['last'], method_seconds=timing['method_seconds'])
                rows.append(row)
                if method != 'uniform' and seed_index == task_index:
                    replays.append(replay(ctx, directory, seed, method))
                print(dict(phase='audit', run=directory.name, status='PASS'), flush=True)
    assert len(rows) == 36 and len(replays) == 9 and len(policy_checks) == 135
    result = dict(status='PASS', rows=rows, policy_checks=policy_checks, independent_replays=replays,
        decision=decision(rows), seconds=rt.tick()-started, new_full_trainings=27,
        parent_replays=9, audit_replays=9, cumulative_full_trainings=461,
        plan_sha256=rt.sha(out/'plan.json'), limitations=['C-adaptive development only', 'single source', 'fixed steps, not equal time'])
    rt.write(out/'audit.json', result)
    return result
