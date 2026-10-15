"""Independent ABC, real-batch AD, and complete NumPy path reconstruction."""
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import audit_fourgroup_m1 as prior
import fourgroup_runtime as rt
import n1_path_runtime as run
import response_tangent as reference
import response_tangent_torch as candidate

RTOL, ATOL = 1e-9, 1e-11


def close(left, right):
    def cpu(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    left, right = cpu(left), cpu(right)
    np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL)
    return float(np.max(np.abs(left-right)))


def packed(model, opt, key):
    values = {}
    for name, parameter in model.named_parameters():
        if key == 'model':
            value = parameter.detach()
        elif key == 'gradient':
            value = parameter.grad.detach()
        else:
            value = opt.state[parameter].get(key, torch.zeros_like(parameter))
        values[name] = value
    return prior.coordinates(values).double()


def weighted_gradient(phi, target, theta, a, groups):
    # Independent loss and automatic differentiation; no candidate batch_terms.
    w = torch.tensor(reference.direction_basis(), device=theta.device) @ a
    prediction = torch.einsum('bcd,cdh->bhc', phi, theta)
    loss = ((prediction-target).square() * (1+w[groups])[:, None, None]).mean()
    return torch.autograd.grad(loss, theta, create_graph=True)[0]


def adjusted_adam(theta, m, v, step, g, dm, dv):
    mm, vv = .9*m + .1*g + dm, .999*v + .001*g.square() + dv
    zero = vv.detach() == 0
    safe_v = torch.where(zero, torch.ones_like(vv), vv)
    change = (mm/(1-.9**step)) / (torch.sqrt(safe_v/(1-.999**step)) + 1e-8)
    change = torch.where(zero, mm/((1-.9**step)*1e-8), change)
    return theta-.001*change, mm, vv


def real_jvp(ctx, seed, directory):
    model = prior.new_model(ctx.plan, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rng = np.random.default_rng(seed+1000)
    rt.equal(rt.snapshot(model, opt, rng, 0), rt.load_snapshot(ctx.parent('full', seed), 0))
    state = candidate.TangentState.zeros(run.PARAMETERS, torch.device('cuda'))
    history = torch.ones(run.PARAMETERS, dtype=torch.bool, device='cuda')
    initial = packed(model, opt, 'model')
    records, maximum = [], 0.
    for step in range(1, 4):
        theta = packed(model, opt, 'model')
        m, v = packed(model, opt, 'exp_avg'), packed(model, opt, 'exp_avg_sq')
        ids = np.searchsorted(ctx.cdf, rng.random(256), side='right')
        index = torch.tensor(ids, device='cuda')
        batch = ctx.batch('A', index)
        phi, target, groups = prior.features(batch[:, :32]).double(), batch[:, 32:].double(), ctx.groups[index]
        rt.equal(phi, ctx.phi['A'][index])
        model.train(); opt.zero_grad(set_to_none=True)
        (model(batch[:, :32])-batch[:, 32:]).square().mean().backward()
        g = packed(model, opt, 'gradient')
        opt.step()
        new_theta, mm, vv = packed(model, opt, 'model'), packed(model, opt, 'exp_avg'), packed(model, opt, 'exp_avg_sq')
        _, hs, forcing = candidate.batch_terms(phi, target, theta, groups, state.S)
        state = candidate.tangent_step(theta.flatten(), m.flatten(), v.flatten(), step-1,
            g.flatten(), hs, forcing, state, history_all_zero=history,
            updated_parent_m=mm.flatten(), updated_parent_v=vv.flatten())
        history &= g.flatten() == 0
        zero = torch.zeros(4, dtype=torch.float64, device='cuda')
        analytic = weighted_gradient(phi, target, theta.clone().requires_grad_(True), zero, groups).detach()
        dg = g-analytic
        dm, dv = mm-(.9*m+.1*g), vv-(.999*v+.001*g.square())
        smooth, _, _ = adjusted_adam(theta, m, v, step, g, dm, dv)
        dt = new_theta-smooth
        records.append((phi, target, groups, dg, dm, dv, dt))
        def trajectory(a):
            t = initial.clone().requires_grad_(True)
            mm0, vv0 = torch.zeros_like(t), torch.zeros_like(t)
            for k, (x, y, gr, gd, md, vd, td) in enumerate(records, 1):
                gg = weighted_gradient(x, y, t, a, gr) + gd
                t, mm0, vv0 = adjusted_adam(t, mm0, vv0, k, gg, md, vd)
                t = t+td
            return t, mm0, vv0
        primal = trajectory(zero)
        for actual, expected in zip(primal, (new_theta, mm, vv)):
            maximum = max(maximum, close(actual, expected))
        for j in range(4):
            direction = torch.eye(4, dtype=torch.float64, device='cuda')[j]
            _, derivatives = torch.autograd.functional.jvp(trajectory, zero, direction, strict=True)
            for name, expected in zip(('S', 'M', 'V'), derivatives):
                maximum = max(maximum, close(getattr(state, name)[:, j], expected.flatten()))
        run.save_state(directory, step, model, opt, rng, state, history)
    return dict(task=ctx.task, seed=seed, steps=3, direction_checks=12,
        tangent_array_checks=36, maximum_absolute_error=maximum, rtol=RTOL, atol=ATOL)


def preflight(out, plan):
    directory = out / 'preflight'
    directory.mkdir()
    started = rt.tick()
    rebuilt = prior.rebuild(out, plan)
    rows, parent_states = [], 0
    for task in rt.TASKS:
        ctx = run.Context(out, plan, task)
        for seed in rt.SEEDS:
            parent_states += prior.state_audit(ctx.parent('full', seed), plan, seed, 'full')
            target = directory / f'{task}_{seed}'
            target.mkdir()
            rows.append(real_jvp(ctx, seed, target))
        del ctx
    result = dict(status='PASS', ABC_arrays=rebuilt, original_parent_states=parent_states,
        real_batch_JVP=rows, diagnostic_parent_updates=27, new_full_trainings=0,
        cumulative_full_trainings=418, D_access=False, test_access=False,
        seconds=rt.tick()-started, plan_sha256=rt.sha(out / 'plan.json'))
    rt.write(out / 'preflight_audit.json', result)
    return result


def check_response(out, ctx, directory, model, state, step):
    checked = 0
    for split in ('B', 'C'):
        residual, features = [], []
        model.eval()
        with torch.no_grad():
            for lo in range(0, len(ctx.starts[split]), 2048):
                batch = ctx.batch(split, torch.arange(lo, min(lo+2048, len(ctx.starts[split])), device='cuda'))
                residual.append((model(batch[:, :32])-batch[:, 32:]).cpu().numpy())
                features.append(prior.features(batch[:, :32]).double().cpu().numpy())
        error = np.concatenate(residual)
        phi = np.concatenate(features)
        jac = np.einsum('bcd,cdhj->bhcj', phi, state.S.reshape(*run.SHAPE, 4))
        meta = rt.read(directory / f'step_{step}_{split}.json')
        rt.equal(meta['residual'], prior.metric(error))
        assert meta['jacobian_shape'] == list(jac.shape) and meta['jacobian_dtype'] == 'float64'
        # Cross-implementation floating-point reduction is tolerance checked.
        # Byte binding is checked separately using exactly the persisted S and GPU contraction.
        saved = prior.arrays(directory / f'step_{step}_tangent.npz')['S']
        exact = torch.einsum('bcd,cdhj->bhcj', torch.tensor(phi, device='cuda'),
            torch.tensor(saved.reshape(*run.SHAPE, 4), device='cuda')).cpu().numpy()
        assert hashlib.sha256(exact.tobytes()).hexdigest() == meta['jacobian_byte_sha256']
        close(jac, exact)
        r = error.astype(np.float64)
        expected = dict(cross=np.einsum('bhc,bhcj->j', r, jac)/r.size,
            gram=np.einsum('bhcj,bhck->jk', jac, jac)/r.size,
            channel_cross=np.einsum('bhc,bhcj->cj', r, jac)/(r.shape[0]*r.shape[1]),
            channel_gram=np.einsum('bhcj,bhck->cjk', jac, jac)/(r.shape[0]*r.shape[1]))
        actual = prior.arrays(directory / f'step_{step}_{split}.npz')
        assert actual.keys() == expected.keys()
        for key in expected:
            close(actual[key], expected[key])
        checked += 1
    return checked


def audit_path(out, plan, task, seed):
    directory = out / 'paths' / f'N1_{task}_s{seed}'
    result = rt.read(directory / 'result.json')
    assert result['status'] == 'completed' and result['steps_completed'] == 3000
    assert result['saved_steps'] == list(run.STEPS) and result['plan_sha256'] == rt.sha(out/'plan.json')
    for name, digest in result['assets_sha256'].items():
        assert rt.sha(rt.safe_file(directory, name)) == digest, name
    ctx = rt.Context(out, plan, task)
    abc = prior.load_abc(out, task)
    bounds = next(t['partitions'] for t in plan['tasks'] if t['id'] == task)
    groups = (abc['A_cycles'][abc['A_starts']]-bounds['A'][0])//77
    np.testing.assert_array_equal(np.bincount(groups), np.full(4,1617))
    # Precompute independently using the audit feature implementation.
    phi, target = [], []
    for lo in range(0, ctx.n, 2048):
        batch = ctx.batch('A', torch.arange(lo, min(lo+2048,ctx.n), device='cuda'))
        phi.append(prior.features(batch[:, :32]).double().cpu().numpy())
        target.append(batch[:, 32:].double().cpu().numpy())
    phi, target = np.concatenate(phi), np.concatenate(target)
    model = prior.new_model(plan, seed)
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
    rng = np.random.default_rng(seed+1000)
    state = reference.TangentState.zeros(run.PARAMETERS)
    history = np.ones(run.PARAMETERS, dtype=bool)
    draws = np.zeros(ctx.n, dtype=np.int64)
    rt.equal(rt.snapshot(model,opt,rng,0),rt.load_snapshot(directory,0))
    initial = prior.arrays(directory/'initial_tangent.npz')
    for name in ('S','M','V'):
        np.testing.assert_array_equal(initial[name], getattr(state,name))
    np.testing.assert_array_equal(initial['history_all_zero'],history)
    saved_states, responses, maximum = 1, 0, 0.
    for step in range(1, 3001):
        theta = packed(model,opt,'model').cpu().numpy()
        m, v = packed(model,opt,'exp_avg').cpu().numpy(), packed(model,opt,'exp_avg_sq').cpu().numpy()
        ids = np.searchsorted(ctx.cdf,rng.random(256),side='right')
        np.add.at(draws,ids,1)
        batch = ctx.batch('A',torch.tensor(ids,device='cuda'))
        model.train();opt.zero_grad(set_to_none=True)
        (model(batch[:,:32])-batch[:,32:]).square().mean().backward()
        gradient = packed(model,opt,'gradient').cpu().numpy()
        opt.step()
        mm,vv=packed(model,opt,'exp_avg').cpu().numpy(),packed(model,opt,'exp_avg_sq').cpu().numpy()
        _,hs,forcing=reference.batch_terms(phi[ids],target[ids],theta,groups[ids],state.S)
        state=reference.tangent_step(theta.ravel(),m.ravel(),v.ravel(),step-1,gradient.ravel(),hs,forcing,state,
            history_all_zero=history,updated_parent_m=mm.ravel(),updated_parent_v=vv.ravel())
        history &= gradient.ravel()==0
        if step in run.STEPS:
            rt.equal(rt.snapshot(model,opt,rng,step),rt.load_snapshot(directory,step))
            actual=prior.arrays(directory/f'step_{step}_tangent.npz')
            for name in ('S','M','V'):
                maximum=max(maximum,close(actual[name],getattr(state,name)))
            np.testing.assert_array_equal(history,actual['history_all_zero'])
            saved_states += 1
        if step in (1,2,3):
            old=out/'preflight'/f'{task}_{seed}'
            rt.equal(rt.load_snapshot(directory,step),rt.load_snapshot(old,step))
            ad=prior.arrays(old/f'step_{step}_tangent.npz')
            for name in ('S','M','V'):
                close(ad[name],getattr(state,name))
        if step in rt.GRID:
            rt.equal(rt.snapshot(model,opt,rng,step),rt.load_snapshot(ctx.parent('full',seed),step))
            responses += check_response(out,ctx,directory,model,state,step)
            print(dict(audit_task=task,seed=seed,step=step,states='PASS',tangents='PASS'),flush=True)
    np.testing.assert_array_equal(draws,prior.arrays(directory/'sampling.npz')['draw_count'])
    np.testing.assert_array_equal(draws,prior.arrays(ctx.parent('full',seed)/'sampling.npz')['draw_count'])
    timing=rt.read(directory/'timing.json')
    assert timing['method_seconds']==timing['actual_elapsed_seconds']-timing['comparison_seconds']-timing['diagnostic_write_seconds']
    assert all(timing[key]>=0 for key in timing if key.endswith('seconds'))
    return dict(task=task,seed=seed,parent_states=saved_states,tangent_arrays=saved_states*3,
        residual_and_response_sets=responses,maximum_absolute_error=maximum,method_seconds=timing['method_seconds'])


def paths(out,plan):
    started=rt.tick()
    original=rt.read(out/'paths.json')
    assert original['status']=='completed_pending_independent_audit'
    assert original['plan_sha256']==rt.sha(out/'plan.json')
    rows=[audit_path(out,plan,task,seed) for task in rt.TASKS for seed in rt.SEEDS]
    costs=[]
    for task in rt.TASKS:
        lower=original['input_verification_seconds']/3+original['preparation_seconds'][task]+sum(r['method_seconds'] for r in rows if r['task']==task)
        budget=plan['M1_task_budget_seconds'][task]
        costs.append(dict(task=task,necessary_method_seconds=lower,M1_seconds=budget,lower_bound_pass=lower<=budget))
    rt.equal(costs,original['costs'])
    result=dict(status='PASS',paths=rows,costs=costs,all_cost_lower_bounds_pass=all(r['lower_bound_pass'] for r in costs),
        FD_eligible_pending_published_receipt=all(r['lower_bound_pass'] for r in costs),
        new_full_trainings=0,cumulative_full_trainings=418,audit_parent_replays=9,
        seconds=rt.tick()-started,plan_sha256=rt.sha(out/'plan.json'),paths_sha256=rt.sha(out/'paths.json'),
        rtol=RTOL,atol=ATOL,D_access=False,test_access=False)
    rt.write(out/'paths_audit.json',result)
    return result
