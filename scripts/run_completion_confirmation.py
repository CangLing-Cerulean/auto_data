"""New, separately authorized confirmations; historical runs remain immutable."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, digest, write_json, evaluate, predict
from reducible_sampling import load_data
from importance_utils import gradient_norm, distribution

FOLDER = ROOT / 'research_runs/completion_study'


class FixedRepresentation(torch.nn.Module):
    def __init__(self, model, policy):
        super().__init__()
        self.model, self.policy = model, policy

    def forward(self, x):
        return predict(self.model, x, self.policy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--policy', required=True, choices=['uniform', 'drop_earliest', 'reshuffle', 'adaptive_is', 'sources', 'selective_center'])
    parser.add_argument('--origin', type=float, default=1., choices=[.6, .7, .8, 1.])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--steps', type=int, default=3000, choices=[3000, 6000])
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id
    assert not run.exists(), 'A new immutable run ID is required'
    run.mkdir(); begun = time.monotonic()

    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})

    status('preparing')
    try:
        protocol = ROOT / 'research_runs/selection/protocol.json'
        p = json.loads(protocol.read_text()); assert p['steps'] == 3000
        snapshots = {'code_snapshot.py': Path(__file__), 'core_snapshot.py': ROOT / 'scripts/run_research.py',
            'data_helper.py': ROOT / 'scripts/reducible_sampling.py', 'importance_helper.py': ROOT / 'scripts/importance_utils.py',
            'source_helper.py': ROOT / 'scripts/leading_sources.py', 'protocol.json': protocol, 'study_protocol.md': FOLDER / 'PROTOCOL.md'}
        identity = {'run_type': 'main_model', **vars(args), 'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'sources_sha256': {name: digest(path) for name, path in snapshots.items()},
            'budget_authorization': 'User explicitly removed compute budget cap on 2026-10-03; prior elapsed time retained'}
        if args.policy in ('sources', 'selective_center'):
            authorization = FOLDER / 'REPRESENTATION_PROTOCOL.md'
            assert authorization.exists(), 'Representation scope must be registered before execution'
            snapshots['representation_protocol.md'] = authorization
            identity['representation_protocol_sha256'] = digest(authorization)
        if args.steps == 6000:
            timing_protocol = FOLDER / 'LONG_TRAINING_PROTOCOL.md'
            assert timing_protocol.exists(), 'Long-training comparisons require their registered protocol'
            snapshots['long_training_protocol.md'] = timing_protocol
            identity['long_training_protocol_sha256'] = digest(timing_protocol)
        write_json(run / 'identity.json', identity)
        for name, path in snapshots.items(): (run / name).write_bytes(path.read_bytes())
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda()
        eval_data = torch.from_numpy(eval_values).cuda() if args.origin == 1 else data
        eval_starts = torch.from_numpy(future).cuda(); offsets = torch.arange(40, device='cuda')
        model = DLinear(p).cuda()
        if args.policy == 'sources':
            from leading_sources import SourceDLinear, load_mapping
            mapping, mapping_hash = load_mapping(); model = SourceDLinear(model, mapping).cuda()
            write_json(run / 'source_mapping.json', {'mapping': list(map(int, mapping)), 'sha256': mapping_hash})
        elif args.policy == 'selective_center':
            model = FixedRepresentation(model, 'selective_center').cuda()
        assert sum(v.numel() for v in model.parameters()) == 7920
        torch.save(model.state_dict(), run / 'initial.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        rng = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        n = len(starts); total_draws = args.steps * 256
        retained = np.ones(n, dtype=bool)
        if args.policy == 'drop_earliest': retained[np.array_split(np.arange(n), 4)[0]] = False
        static_q = retained.astype(np.float64) / retained.sum()
        plan = None
        if args.policy == 'reshuffle':
            planner = np.random.default_rng(args.seed + 1000)
            plan = np.concatenate([planner.permutation(n) for _ in range(int(np.ceil(total_draws / n)))])[:total_draws].astype(np.int32).reshape(args.steps, 256)
            np.savez_compressed(run / 'sampling_plan.npz', candidate_indices=plan)
            plan_gpu = torch.from_numpy(plan.astype(np.int64)).cuda()
        counts = torch.zeros(n, device='cuda', dtype=torch.int64)
        history = []; phases = []; best = float('inf'); best_metrics = None; best_step = None
        milestone_3000 = None
        for phase in range(0, args.steps, 500):
            scoring_started = time.monotonic()
            norm = None
            if args.policy == 'adaptive_is' and phase > 0:
                model.eval(); norm = torch.zeros(n, device='cuda', dtype=torch.float64)
                torch.save(model.state_dict(), run / f'scoring_{phase:04d}.pt')
                for lo in range(0, n, p['evaluation_batch_size']):
                    batch = data[positions[lo:lo + p['evaluation_batch_size'], None] + offsets]
                    norm[lo:lo + len(batch)] = gradient_norm(model, batch[:, :32], batch[:, 32:])
                q = distribution(norm).cpu().numpy()
            else:
                q = static_q
            assert np.isfinite(q).all() and np.all(q >= 0) and abs(q.sum() - 1) < 1e-12
            weight_gpu = (1 / (n * torch.from_numpy(q).cuda())).float() if args.policy == 'adaptive_is' else torch.ones(n, device='cuda')
            weights = weight_gpu.cpu().numpy()
            cdf = np.cumsum(q); cdf[-1] = 1.; cdf = torch.from_numpy(cdf).cuda()
            scoring_seconds = time.monotonic() - scoring_started
            phase_counts = torch.zeros_like(counts); losses = []; model.train()
            for step in range(phase + 1, phase + 501):
                chosen = plan_gpu[step - 1] if plan is not None else torch.searchsorted(cdf,
                    torch.rand(256, device='cuda', dtype=torch.float64, generator=rng), right=True)
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                squared = (model(batch[:, :32]) - batch[:, 32:]).square()
                # Formal S-series uses per-window reduction; internal T-series uses a direct mean.
                loss = (squared.mean((1, 2)) * weight_gpu[chosen]).mean() if args.policy == 'adaptive_is' or args.origin == 1 else squared.mean()
                assert torch.isfinite(loss); loss.backward(); optimizer.step()
                losses.append(float(loss.detach())); phase_counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            counts += phase_counts
            if args.policy == 'adaptive_is':
                np.savez_compressed(run / f'phase_{phase:04d}.npz', probability=q, loss_weight=weights.astype(np.float32),
                    gradient_norm=np.zeros(n) if norm is None else norm.cpu().numpy(), draw_count=phase_counts.cpu().numpy())
            phases.append({'start_step': phase, 'scoring_seconds': scoring_seconds,
                'scoring_forward_presentations': n if args.policy == 'adaptive_is' and phase > 0 else 0})
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses)), 'elapsed_seconds': time.monotonic() - begun}
            if args.origin == 1:
                metrics = evaluate(model, eval_data, eval_starts, groups, p, 'baseline')
                row['validation'] = metrics['groups']['all']
                if metrics['groups']['all']['NMSE'] < best:
                    best = metrics['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = metrics
                    torch.save(model.state_dict(), run / 'best.pt')
                if args.steps == 6000:
                    torch.save(model.state_dict(), run / f'checkpoint_{phase + 500:04d}.pt')
                    write_json(run / f'validation_{phase + 500:04d}.json', metrics)
                    row['metric_available_elapsed_seconds'] = time.monotonic() - begun
            history.append(row)
            if phase + 500 == 3000:
                milestone_3000 = {'best_step': best_step, 'metrics': best_metrics}
                torch.save(model.state_dict(), run / 'step3000.pt')
                if args.steps == 6000:
                    np.savez_compressed(run / 'draws3000.npz', draw_count=counts.cpu().numpy())
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': rng.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history, 'best': best, 'best_step': best_step}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draws = counts.cpu().numpy(); assert draws.sum() == total_draws and not np.any(draws[~retained])
        if plan is not None: assert np.array_equal(draws, np.bincount(plan.ravel(), minlength=n))
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=future, cell=cells, months=months,
            retained=retained, probability=static_q if args.policy != 'adaptive_is' else q, draw_count=draws)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'steps_completed': args.steps, 'samples_presented': total_draws,
            'history': history, 'phases': phases, 'sensor_columns': manifest['sensor_columns'], 'milestone_3000': milestone_3000,
            'windows_sha256': digest(run / 'windows.npz')}
        if args.origin < 1: result['future'] = evaluate(model, data, eval_starts, groups, p, 'baseline')
        else: result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', args.steps)
        print(json.dumps({'completed': args.run_id, 'metrics': result.get('future', result)['groups']['all']}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
