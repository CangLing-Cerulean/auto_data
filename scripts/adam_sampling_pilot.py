"""A bounded online sampling decision: 27 new models and nine parent replays."""
import argparse
from pathlib import Path
import random
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import adam_risk_sampling as solver
import fourgroup_runtime as rt

ROOT = Path('/root/auto_data_experiments/20261015_adam_sampling_pilot_r1')
METHODS = ('adam_risk', 'gradient_norm', 'geometry')
REFRESH = (500, 1000, 1500, 2000, 2500)


def verify(out):
    assert out == ROOT and out.resolve() == out and __debug__
    plan = rt.read(out/'plan.json')
    source, activation = rt.read(out/'source_sync_receipt.json'), rt.read(out/'execution_activation.json')
    assert plan['stage'] == 'adam_sampling_pilot_r1' and plan['output_root'] == str(ROOT)
    assert plan['cumulative_before'] == 434 and plan['maximum_new_models'] == 27
    assert plan['methods'] == list(METHODS) and plan['refresh_steps'] == list(REFRESH)
    assert source['status'] == 'exact_commit_and_tree_verified'
    assert source['plan_sha256'] == rt.sha(out/'plan.json')
    assert source['input_lock_sha256'] == plan['input_lock_sha256'] == rt.sha(out/'input_lock.json')
    assert activation['status'] == 'inputs_environment_storage_verified'
    assert activation['source_commit'] == source['source_commit']
    for key, name in (('plan_sha256', 'plan.json'), ('input_lock_sha256', 'input_lock.json'),
                      ('storage_preflight_sha256', 'storage_preflight.json'),
                      ('environment_sha256', 'runtime_environment.json')):
        assert activation[key] == rt.sha(out/name)
    assert rt.read(out/'runtime_environment.json')['environment'] == plan['expected_environment'] == rt.environment()
    assert rt.sha(out/'storage_estimate.json') == plan['storage_estimate_sha256']
    storage = rt.read(out/'storage_preflight.json')
    assert storage['status'] == 'PASS' and storage['probe_removed']
    assert storage['write_sha256'] == storage['read_sha256']
    assert storage['free_before_bytes'] >= plan['preflight_min_free_bytes']
    for name, digest in plan['code_sha256'].items():
        assert rt.sha(rt.safe_file(out, name)) == digest, name
    for item in rt.read(out/'input_lock.json')['files']:
        file = rt.safe_file(out, item['path'])
        assert file.stat().st_size == item['bytes'] and rt.sha(file) == item['sha256']
    assert shutil.disk_usage(out).free >= plan['runtime_min_free_bytes']
    return plan


def pack(model, optimizer=None, field=None):
    values = {}
    for name, parameter in model.named_parameters():
        value = parameter if field is None else optimizer.state[parameter].get(field, torch.zeros_like(parameter))
        values[name] = value.detach()
    return torch.stack([torch.cat((values[f'seasonal.{c}.weight'].T,
        values[f'seasonal.{c}.bias'][None], values[f'trend.{c}.weight'].T,
        values[f'trend.{c}.bias'][None])) for c in range(8)]).double()


class Context:
    def __init__(self, out, plan, task, method, for_audit=False):
        self.out, self.plan, self.task, self.method = out, plan, task, method
        self.values, self.starts, self.phi, self.targets = {}, {}, {}, {}
        self.offset = torch.arange(40, device='cuda')
        splits = ['A', 'B']
        if for_audit or method in ('geometry', 'adam_risk'):
            splits.append('C')
        with np.load(out/'inputs/prepared'/f'{task}_ABC.npz', allow_pickle=False) as data:
            for split in splits:
                self.values[split] = torch.tensor(data[f'{split}_values'], device='cuda')
                self.starts[split] = torch.tensor(data[f'{split}_starts'], device='cuda')
            lo = next(x['partitions']['A'][0] for x in plan['tasks'] if x['id'] == task)
            self.groups_np = (data['A_cycles'][data['A_starts']]-lo)//77
        np.testing.assert_array_equal(np.bincount(self.groups_np), np.full(4, 1617))
        self.groups = torch.tensor(self.groups_np, dtype=torch.long, device='cuda')
        self.q = np.full(4, .25)
        assert len(self.starts['A']) == 6468 and len(self.starts['B']) == 1617
        feature_splits = [] if method == 'uniform' else ['A']
        if method in ('geometry', 'adam_risk'):
            feature_splits.append('C')
        for split in feature_splits:
            xx, yy = [], []
            for lo in range(0, len(self.starts[split]), 2048):
                batch = self.batch(split, torch.arange(lo, min(lo+2048, len(self.starts[split])), device='cuda'))
                xx.append(rt.rp.features(SimpleNamespace(kernel=25), batch[:, :32]).double())
                yy.append(batch[:, 32:].double())
            self.phi[split], self.targets[split] = torch.cat(xx), torch.cat(yy)
        if 'C' in self.phi:
            self.hessian = 2*torch.einsum('ncd,nce->cde', self.phi['C'], self.phi['C'])/(1617*64)

    def batch(self, split, ids):
        assert split in ('A', 'B', 'C')
        return self.values[split][self.starts[split][ids, None]+self.offset]

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return rt.core.DLinear(dict(self.plan['training'], channels=8)).cuda()

    def parent(self, seed):
        return self.out/'inputs/runs'/f'DEV_{self.task}_full_PARENT_s{seed}'

    @torch.no_grad()
    def evaluate(self, model, split):
        assert split == 'B' or (split == 'C' and 'C' in self.values)
        model.eval()
        residuals = []
        for lo in range(0, len(self.starts[split]), 2048):
            batch = self.batch(split, torch.arange(lo, min(lo+2048, len(self.starts[split])), device='cuda'))
            residuals.append((model(batch[:, :32])-batch[:, 32:]).cpu().numpy())
        return rt.metric(np.concatenate(residuals))


@torch.no_grad()
def proposal(ctx, model, optimizer, step, method, witness=False):
    theta = pack(model)
    residual = torch.einsum('ncd,cdh->nhc', ctx.phi['A'], theta)-ctx.targets['A']
    regular = True
    if method == 'gradient_norm':
        # Exact norm of the channel-wise outer-product gradients; no C access.
        scores = (2/64)**2*(ctx.phi['A'].square().sum(2)*residual.square().sum(1)).sum(1)
    else:
        m, v = pack(model, optimizer, 'exp_avg'), pack(model, optimizer, 'exp_avg_sq')
        mu = 2*torch.einsum('ncd,nhc->cdh', ctx.phi['A'], residual)/(6468*64)
        k = step+1
        a, b = .9*m/(1-.9**k), .1/(1-.9**k)
        c, d = .999*v/(1-.999**k), .001/(1-.999**k)
        n, s = a+b*mu, torch.sqrt(c+d*mu.square())
        regular = not bool((s == 0).any())
        if not regular:
            return dict(probabilities=ctx.q.copy(), coefficients=np.zeros(4), regular=False)
        denominator = s+1e-8
        sp, spp = d*mu/s, d*c/s.pow(3)
        first = -.001*(b/denominator-n*sp/denominator.square())
        second = .001*(2*b*sp/denominator.square()+n*spp/denominator.square()
                      -2*n*sp.square()/denominator.pow(3))
        center = theta-.001*n/denominator
        rc = torch.einsum('ncd,cdh->nhc', ctx.phi['C'], center)-ctx.targets['C']
        risk_gradient = 2*torch.einsum('ncd,nhc->cdh', ctx.phi['C'], rc)/(1617*64)
        chunks = []
        for lo in range(0, 6468, 256):
            g = 2/64*torch.einsum('ncd,nhc->ncdh', ctx.phi['A'][lo:lo+256], residual[lo:lo+256])
            transformed = g*first
            score = torch.einsum('ncdh,cde,nceh->n', transformed, ctx.hessian, transformed)
            if method == 'adam_risk':
                score += (g.square()*risk_gradient*second).sum((1,2,3))
            chunks.append(score)
        scores = torch.cat(chunks)
    assert bool(torch.isfinite(scores).all())
    coefficients = torch.stack([scores[ctx.groups == i].mean()/16 for i in range(4)]).cpu().numpy()
    probability = solver.allocate_signed(coefficients, ctx.q, .5)
    result = dict(probabilities=probability, coefficients=coefficients, regular=regular)
    if witness and method != 'gradient_norm':
        result['derivatives'] = [x.cpu().numpy() for x in (center, first, second)]
    return result


def distribution(ctx, probability):
    per_window = probability[ctx.groups_np]/1617
    cdf = np.cumsum(per_window)
    cdf[-1] = 1.
    weights = torch.tensor(ctx.q[ctx.groups_np]/probability[ctx.groups_np], dtype=torch.float32, device='cuda')
    return cdf, weights


def run_id(task, method, seed):
    return f'ARS_{task}_{method}_s{seed}'


def run_one(out, plan, task, method, seed):
    directory = out/'runs'/run_id(task, method, seed)
    directory.mkdir(parents=True)
    started = rt.tick()
    status = dict(status='running', task=task, method=method, seed=seed, steps_completed=0,
                  new_full_trainings=int(method != 'uniform'))
    rt.write(directory/'status.json', status)
    try:
        ctx = Context(out, plan, task, method)
        model = ctx.model(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        rng = np.random.default_rng(seed+1000)
        clock = rt.tick()
        rt.equal(rt.snapshot(model, optimizer, rng, 0), rt.load_snapshot(ctx.parent(seed), 0))
        verification_seconds = rt.tick()-clock
        rt.save_snapshot(directory/'initial.pt', model, optimizer, rng, 0)
        # Match the original full-window inverse-CDF sampler exactly in warmup.
        cdf = np.cumsum(np.full(6468, 1/6468))
        cdf[-1] = 1.
        weights = torch.ones(6468, dtype=torch.float32, device='cuda')
        draws = np.zeros(6468, dtype=np.int64)
        b_curve, proposals = {}, []
        preparation_seconds = rt.tick()-started
        proposal_seconds, evaluation_seconds = 0., 0.
        for step in range(1, 3001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ids = np.searchsorted(cdf, rng.random(256), side='right')
            np.add.at(draws, ids, 1)
            index = torch.tensor(ids, device='cuda')
            batch = ctx.batch('A', index)
            squared = (model(batch[:, :32])-batch[:, 32:]).square()
            loss = squared.mean() if method == 'uniform' or step <= 500 else (squared*weights[index, None, None]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                clock = rt.tick()
                b_curve[str(step)] = ctx.evaluate(model, 'B')
                rt.save_snapshot(directory/f'step_{step}.pt', model, optimizer, rng, step)
                if method == 'uniform' or step == 500:
                    check_start = rt.tick()
                    rt.equal(rt.snapshot(model, optimizer, rng, step), rt.load_snapshot(ctx.parent(seed), step))
                    verification_seconds += rt.tick()-check_start
                evaluation_seconds += rt.tick()-clock
                if method != 'uniform' and step in REFRESH:
                    clock = rt.tick()
                    chosen = proposal(ctx, model, optimizer, step, method)
                    cdf, weights = distribution(ctx, chosen['probabilities'])
                    elapsed = rt.tick()-clock
                    proposal_seconds += elapsed
                    record = dict(step=step, regular=chosen['regular'],
                        probabilities=chosen['probabilities'].tolist(), coefficients=chosen['coefficients'].tolist(),
                        parent_checkpoint_sha256=rt.sha(directory/f'step_{step}.pt'), seconds=elapsed)
                    rt.write(directory/f'proposal_{step}.json', record)
                    proposals.append(record)
                status['steps_completed'] = step
                rt.write(directory/'status.json', status, update=True)
                print(dict(run=directory.name, step=step), flush=True)
        best_step = min(rt.GRID, key=lambda s: b_curve[str(s)]['nmse'])
        rt.arrays(directory/'sampling.npz', counts=draws)
        assets = {p.name: rt.sha(p) for p in sorted(directory.iterdir())
                  if p.name != 'status.json' and p.is_file()}
        result = dict(status='completed', task=task, method=method, seed=seed, steps=3000,
            best_step=best_step, B_curve=b_curve, proposals=proposals, assets_sha256=assets,
            preparation_seconds=preparation_seconds, proposal_seconds=proposal_seconds,
            evaluation_write_compare_seconds=evaluation_seconds,
            new_full_trainings=int(method != 'uniform'),
            parent_replays=int(method == 'uniform'), C_final_scores_not_read=True)
        rt.write(directory/'result.json', result)
        status.update(status='completed', steps_completed=3000)
        rt.write(directory/'status.json', status, update=True)
        elapsed = rt.tick()-started
        timing = dict(method_seconds=elapsed-verification_seconds, actual_elapsed_seconds=elapsed,
            verification_seconds=verification_seconds,
            boundary='data preparation through completed result/status and hashes; timing receipt excluded')
        rt.write(directory/'timing.json', timing)
        return dict(result, timing=timing)
    except Exception:
        status.update(status='failed', error=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
        rt.write(directory/'status.json', status, update=True)
        raise


def uniform(out, plan):
    records = [run_one(out, plan, task, 'uniform', seed) for task in rt.TASKS for seed in rt.SEEDS]
    result = dict(status='completed', records=records, parent_replays=9, new_full_trainings=0,
                  plan_sha256=rt.sha(out/'plan.json'))
    rt.write(out/'uniform.json', result)
    return result


def methods(out, plan):
    receipt = rt.read(out/'uniform_sync_receipt.json')
    assert receipt['status'] == 'exact_commit_and_tree_verified'
    assert receipt['uniform_sha256'] == rt.sha(out/'uniform.json')
    assert rt.read(out/'uniform.json')['status'] == 'completed'
    records = []
    for task in rt.TASKS:
        for index, seed in enumerate(rt.SEEDS):
            order = METHODS[index:]+METHODS[:index]
            for method in order:
                records.append(run_one(out, plan, task, method, seed))
    result = dict(status='completed', records=records, new_full_trainings=27,
        cumulative_full_trainings=461, plan_sha256=rt.sha(out/'plan.json'))
    rt.write(out/'methods.json', result)
    return result


PHASES = ('preflight', 'uniform', 'methods', 'audit')


def validate_phase(out, phase):
    plan = verify(out)
    assert phase in PHASES and not (out/'stage_failure.json').exists()
    assert not (out/f'{phase}.json').exists() and not (out/f'{phase}_execution.lock').exists()
    if phase != 'preflight':
        assert rt.read(out/'preflight.json')['status'] == 'PASS'
    if phase in ('methods', 'audit'):
        assert rt.read(out/'uniform.json')['status'] == 'completed'
    if phase == 'audit':
        assert rt.read(out/'methods.json')['status'] == 'completed'
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=PHASES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rt.configure()
    started = rt.tick()
    plan = validate_phase(args.output, args.phase)
    rt.write(args.output/f'{args.phase}_execution.lock', dict(started_unix=time.time(), phase=args.phase))
    try:
        if args.phase in ('uniform', 'methods'):
            result = globals()[args.phase](args.output, plan)
        else:
            import audit_adam_sampling_pilot as audit
            result = getattr(audit, args.phase)(args.output, plan)
        print(dict(phase=args.phase, status=result['status'], seconds=rt.tick()-started), flush=True)
    except Exception:
        rt.write(args.output/'stage_failure.json', dict(phase=args.phase, error=traceback.format_exc(),
            seconds=time.perf_counter()-started, status='failed_closed'))
        raise


if __name__ == '__main__':
    main()
