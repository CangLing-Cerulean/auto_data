"""Create-only engineering tests/benchmarks on already disclosed Hydraulic assets."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
import response_proxy as rp


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if replace:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


class Context:
    def __init__(self, out):
        self.out = out
        self.plan = read(out / 'plan.json')
        self.old = Path(self.plan['input_root'])
        assert self.old.resolve() != out.resolve()
        for name, digest in self.plan['code_sha256'].items():
            assert sha(out / name) == digest, name
        assert read(out / 'source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
        self.check_inputs()
        h = load_module('old_hydraulic', self.old / 'hydraulic_signal_validation.py')
        original = h.verify(self.old)
        self.config = dict(original['training'], channels=8)
        self.core = load_module('frozen_core', original['frozen_core'])
        self.proposal = load_module('frozen_proposal', self.old / 'proposal_median3.py')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        arrays, self.params, _ = h.prepared(self.old, original)
        self.values = {s: torch.tensor(arrays[s], device='cuda') for s in ('train', 'validation')}
        self.starts = {s: torch.tensor(arrays[s + '_starts'], device='cuda') for s in self.values}
        self.offset = torch.arange(40, device='cuda')
        self.cdf = np.cumsum(arrays['probability'])
        self.cdf[-1] = 1.
        assert np.all(arrays['probability'] == 1. / len(self.starts['train']))
        assert np.all(arrays['loss_weight'] == 1.)

    def check_inputs(self):
        for name, digest in self.plan['inputs_sha256'].items():
            assert sha(self.old / name) == digest, name

    def model(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return self.core.DLinear(self.config).cuda()

    def optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'],
                                weight_decay=self.config['weight_decay'])

    def transform(self, x, action):
        return x if action == 'PARENT' else self.proposal.transform(x, self.params, 'P1', action == 'CP1')

    def batch(self, split, ids):
        return self.values[split][self.starts[split][ids, None] + self.offset]

    @torch.no_grad()
    def moments(self, model, action):
        n = len(self.starts['train'])
        result = None
        for lo in range(0, n, 2048):
            batch = self.batch('train', torch.arange(lo, min(lo + 2048, n), device='cuda'))
            phi = rp.features(model, self.transform(batch[:, :32], action))
            m = rp.batch_moments(phi, batch[:, 32:], torch.full((len(batch),), 1. / n, device='cuda', dtype=torch.float64))
            if result is None:
                result = m
            else:
                result.add_(m)
        return result

    @torch.no_grad()
    def evaluate(self, model, action):
        model.eval()
        errors = []
        n = len(self.starts['validation'])
        for lo in range(0, n, 2048):
            batch = self.batch('validation', torch.arange(lo, min(lo + 2048, n), device='cuda'))
            errors.append((model(self.transform(batch[:, :32], action)) - batch[:, 32:]).cpu().numpy())
        errors = np.concatenate(errors)
        assert np.isfinite(errors).all()
        return float(np.mean(errors.astype(np.float64) ** 2)), errors


def numerical(ctx):
    out = ctx.out / 'numerical'
    out.mkdir()
    started = tick()
    result = dict(status='running', tests=[], diagnostic_optimizer_steps=0, test_access=False)
    write(out / 'status.json', result)
    try:
        ids = torch.linspace(0, len(ctx.starts['train']) - 1, 256, device='cuda').long()
        batch = ctx.batch('train', ids)
        tol = ctx.plan['numeric_tolerances']
        for action in ctx.plan['actions']:
            for weighted in (False, True):
                direct = ctx.model(ctx.plan['numeric_seed'])
                proxy = copy.deepcopy(direct)
                x, y = ctx.transform(batch[:, :32], action), batch[:, 32:]
                # Synthetic weights, not synthetic sensor records; exercise nonuniform semantics.
                w = torch.linspace(.5, 1.5, len(batch), device='cuda') if weighted else torch.ones(len(batch), device='cuda')
                phi = rp.features(direct, x)
                moment = rp.batch_moments(phi, y, w.double() / len(batch))
                reconstructed = torch.einsum('ncp,cph->nhc', phi.double(), rp.parameters(direct).double())
                torch.testing.assert_close(reconstructed.float(), direct(x), rtol=tol['forward_rtol'], atol=tol['forward_atol'])
                direct_opt, proxy_opt = ctx.optimizer(direct), ctx.optimizer(proxy)
                max_gradient_error = 0.
                for step in range(1, 33):
                    direct_opt.zero_grad(set_to_none=True)
                    loss = ((direct(x) - y).square() * w[:, None, None]).mean()
                    if step == 1:
                        torch.testing.assert_close(moment.loss(rp.parameters(direct)).float(), loss,
                                                   rtol=tol['forward_rtol'], atol=tol['forward_atol'])
                    loss.backward()
                    # Check moment gradients at the same parameters, independent of trajectory drift.
                    probe = copy.deepcopy(direct)
                    rp.assign_gradient(probe, moment)
                    for a, b in zip(direct.parameters(), probe.parameters()):
                        max_gradient_error = max(max_gradient_error, float((a.grad - b.grad).abs().max()))
                        torch.testing.assert_close(a.grad, b.grad, rtol=tol['gradient_rtol'], atol=tol['gradient_atol'])
                    proxy_opt.zero_grad(set_to_none=True)
                    rp.assign_gradient(proxy, moment)
                    direct_opt.step()
                    result['diagnostic_optimizer_steps'] += 1
                    proxy_opt.step()
                    result['diagnostic_optimizer_steps'] += 1
                    torch.testing.assert_close(rp.parameters(direct), rp.parameters(proxy), rtol=0., atol=tol['parameter_atol'])
                for a, b in zip(direct.parameters(), proxy.parameters()):
                    for key in ('exp_avg', 'exp_avg_sq'):
                        torch.testing.assert_close(direct_opt.state[a][key], proxy_opt.state[b][key],
                                                   rtol=tol['state_rtol'], atol=tol['state_atol'])
                    assert direct_opt.state[a]['step'].item() == proxy_opt.state[b]['step'].item() == 32
                base = rp.parameters(direct).detach()
                assert torch.equal(rp.paired_parameters(base, rp.parameters(proxy), rp.parameters(proxy)), base)
                # Same effective mass via probability and weights gives identical moments.
                probability = w.double() / w.double().sum()
                weight = w.double().sum() / len(w)
                other = rp.batch_moments(phi, y, probability * weight)
                for a, b in zip((moment.gram, moment.cross), (other.gram, other.cross)):
                    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
                result['tests'].append(dict(action=action, weighted=weighted, max_gradient_absolute_error=max_gradient_error,
                    max_parameter_absolute_error=float((rp.parameters(direct) - rp.parameters(proxy)).abs().max()), steps=32))
        grid = ctx.plan['grid']
        b = dict(zip(grid, (3., 1., 1., 2., 4., 5.)))
        c = dict(zip(grid, (0., 10., 0., 0., 0., 0.)))
        selected = rp.select_checkpoint(b, grid)
        assert selected == 1000 and c[selected] == 10.
        for invalid in ({500: 1.}, dict(zip(grid, (3., float('nan'), 1., 2., 4., 5.)))):
            try:
                rp.select_checkpoint(invalid, grid)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid selector input accepted')
        result['selection_tests'] = dict(earliest_tie=True, C_not_used=True, rejects_incomplete=True, rejects_nonfinite=True)
        ctx.check_inputs()
        result.update(status='PASS', process_seconds=tick() - started, full_training_runs=0,
                      plan_sha256=sha(ctx.out / 'plan.json'), prototype_sha256=sha(ctx.out / 'response_proxy.py'))
        write(out / 'result.json', result)
    except Exception:
        result.update(status='failed', error=traceback.format_exc(), process_seconds=tick() - started)
        raise
    finally:
        write(out / 'status.json', result, replace=True)


def benchmark_one(ctx, job):
    out = ctx.out / 'runs' / job['id']
    out.mkdir(parents=True)
    status = dict(status='running', **job, steps_completed=0)
    write(out / 'status.json', status)
    torch.cuda.reset_peak_memory_stats()
    started = tick()
    timing = dict(moment_build=0., updates=0., evaluation=0., serialization=0., parent_anchor_load=0.)
    try:
        model = ctx.model(job['seed'])
        opt = ctx.optimizer(model)
        initial = rp.parameters(model).detach().cpu().numpy()
        initial_sha = hashlib.sha256(initial.tobytes()).hexdigest()
        action, engine = job['action'], job['engine']
        generator = np.random.default_rng(job['seed'] + 1000)
        t0 = tick()
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=0,
            rng_numpy=generator.bit_generator.state, rng_torch=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(), original_coordinates=torch.from_numpy(initial)), out / 'initial.pt')
        timing['serialization'] += tick() - t0
        moment = None
        if engine == 'proxy':
            t0 = tick()
            moment = ctx.moments(model, action)
            timing['moment_build'] = tick() - t0
            t0 = tick()
            with (out / 'moments.npz').open('xb') as stream:
                np.savez_compressed(stream, gram=moment.gram.cpu().numpy(), cross=moment.cross.cpu().numpy(), target_square=moment.target_square.cpu().numpy())
            timing['serialization'] += tick() - t0
        predictor = copy.deepcopy(model) if engine == 'proxy' and action != 'PARENT' else model
        draw_count = np.zeros(len(ctx.starts['train']), dtype=np.int64)
        scores, history, snapshots = {}, [], {}
        segment_start = tick()
        for step in range(1, 3001):
            model.train()
            opt.zero_grad(set_to_none=True)
            if engine == 'direct':
                chosen = np.searchsorted(ctx.cdf, generator.random(256), side='right')
                np.add.at(draw_count, chosen, 1)
                batch = ctx.batch('train', torch.tensor(chosen, device='cuda'))
                loss = (model(ctx.transform(batch[:, :32], action)) - batch[:, 32:]).square().mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite direct loss')
                loss.backward()
            else:
                rp.assign_gradient(model, moment)
            opt.step()
            status['steps_completed'] = step
            if step in ctx.plan['grid']:
                timing['updates'] += tick() - segment_start
                parameter = rp.parameters(model).detach()
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError('Nonfinite parameter')
                t0 = tick()
                snapshot = dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
                    rng_numpy=generator.bit_generator.state, rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state(), original_coordinates=parameter.cpu())
                torch.save(snapshot, out / f'step_{step}.pt')
                timing['serialization'] += tick() - t0
                snapshots[step] = f'step_{step}.pt'
                # Proxy-parent is shared setup and needs no checkpoint selection.
                if not (engine == 'proxy' and action == 'PARENT'):
                    if engine == 'proxy':
                        t0 = tick()
                        anchor_dir = ctx.out / 'runs'
                        real = torch.load(anchor_dir / f'ENG_direct_PARENT_s{job["seed"]}' / f'step_{step}.pt', weights_only=True, map_location='cuda')['original_coordinates']
                        reference = torch.load(anchor_dir / f'ENG_proxy_PARENT_s{job["seed"]}' / f'step_{step}.pt', weights_only=True, map_location='cuda')['original_coordinates']
                        rp.load_parameters(predictor, rp.paired_parameters(real, parameter, reference))
                        timing['parent_anchor_load'] += tick() - t0
                    t0 = tick()
                    score, _ = ctx.evaluate(predictor, action)
                    scores[step] = score
                    timing['evaluation'] += tick() - t0
                history.append(dict(step=step, elapsed_seconds=tick() - started))
                write(out / 'status.json', status, replace=True)
                segment_start = tick()
        selected = None
        if scores:
            selected = rp.select_checkpoint(scores, ctx.plan['grid'])
            t0 = tick()
            selected_model = torch.load(out / snapshots[selected], weights_only=True, map_location='cuda')
            model.load_state_dict(selected_model['model'])
            if engine == 'proxy':
                anchor_dir = ctx.out / 'runs'
                real = torch.load(anchor_dir / f'ENG_direct_PARENT_s{job["seed"]}' / f'step_{selected}.pt', weights_only=True, map_location='cuda')['original_coordinates']
                reference = torch.load(anchor_dir / f'ENG_proxy_PARENT_s{job["seed"]}' / f'step_{selected}.pt', weights_only=True, map_location='cuda')['original_coordinates']
                rp.load_parameters(predictor, rp.paired_parameters(real, rp.parameters(model), reference))
            timing['parent_anchor_load'] += tick() - t0
            t0 = tick()
            repeated_score, errors = ctx.evaluate(predictor, action)
            assert repeated_score == scores[selected]
            timing['evaluation'] += tick() - t0
            t0 = tick()
            torch.save(predictor.state_dict(), out / 'selected_prediction_model.pt')
            with (out / 'selected_errors.npz').open('xb') as stream:
                np.savez_compressed(stream, error=errors)
            timing['serialization'] += tick() - t0
        if engine == 'direct':
            t0 = tick()
            with (out / 'sampling.npz').open('xb') as stream:
                np.savez_compressed(stream, draw_count=draw_count)
            timing['serialization'] += tick() - t0
        elapsed = tick() - started
        assets = {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.npz', '.pt')}
        result = dict(**job, status='completed', steps=3000, initial_sha256=initial_sha,
            process_seconds=elapsed, timings=timing, peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
            selection_losses=scores, selected_step=selected, checkpoints=history, assets_sha256=assets,
            purpose='engineering cost only; disclosed task; not independent response evidence',
            direct_training_completed=int(engine == 'direct'), proxy_rollout_completed=int(engine == 'proxy'),
            plan_sha256=sha(ctx.out / 'plan.json'), test_access=False)
        write(out / 'result.json', result)
        status.update(status='completed', process_seconds=elapsed)
        return result
    except Exception:
        status.update(status='failed', error=traceback.format_exc(), process_seconds=tick() - started)
        raise
    finally:
        write(out / 'status.json', status, replace=True)


def benchmark(ctx):
    numerical_result = read(ctx.out / 'numerical' / 'result.json')
    assert numerical_result['status'] == 'PASS'
    assert numerical_result['plan_sha256'] == sha(ctx.out / 'plan.json')
    assert not (ctx.out / 'runs').exists(), 'No benchmark IDs may be rerun'
    write(ctx.out / 'benchmark.lock', dict(plan_sha256=sha(ctx.out / 'plan.json')))
    results = []
    try:
        for job in ctx.plan['jobs']:
            result = benchmark_one(ctx, job)
            results.append(result)
            progress = dict(status='running', completed=len(results), registered=len(ctx.plan['jobs']),
                direct_completed=sum(r['direct_training_completed'] for r in results),
                proxy_completed=sum(r['proxy_rollout_completed'] for r in results), last_id=job['id'])
            write(ctx.out / 'benchmark_status.json', progress, replace=True)
            print(json.dumps(dict(id=job['id'],seconds=result['process_seconds'])), flush=True)
        ctx.check_inputs()
        write(ctx.out / 'benchmark_result.json', dict(status='completed', results=results, inputs_unchanged=True,
              environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), threads=torch.get_num_threads()),
              full_direct_trainings=9, proxy_rollouts=9, test_access=False))
        progress['status'] = 'completed'
    except Exception:
        progress = dict(status='failed', completed=len(results), registered=len(ctx.plan['jobs']),
            direct_completed=sum(r['direct_training_completed'] for r in results),
            proxy_completed=sum(r['proxy_rollout_completed'] for r in results), error=traceback.format_exc())
        raise
    finally:
        write(ctx.out / 'benchmark_status.json', progress, replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('numerical', 'benchmark'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    ctx = Context(Path(args.output))
    optimizer_defaults = ctx.optimizer(ctx.model(ctx.plan['numeric_seed'])).defaults
    torch.cuda.synchronize()
    write(ctx.out / (args.mode + '_setup.json'), dict(seconds=time.perf_counter() - started,
          torch=torch.__version__, gpu=torch.cuda.get_device_name(), mode=args.mode,
          optimizer=optimizer_defaults))
    if args.mode == 'numerical':
        numerical(ctx)
    else:
        benchmark(ctx)


if __name__ == '__main__':
    main()
