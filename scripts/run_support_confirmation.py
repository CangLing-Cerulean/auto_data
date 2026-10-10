"""Second-origin confirmation of two frozen full-support sampling recipes."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
import coverage_sampling
from run_research import ROOT, DLinear, digest, write_json, evaluate
from reducible_sampling import load_data, used_seconds
from covariate_sampling import ROUND_LIMIT

FOLDER = ROOT / 'research_runs/support_confirmation'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.8, 1.], required=True)
    parser.add_argument('--policy', choices=['balanced', 'reshuffle'], required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(); assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id
    assert not run.exists(), 'A new ID is required; completed runs cannot be overwritten'
    allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 0
    run.mkdir(); begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})
    def check(step):
        if time.monotonic() - begun >= allowance: raise TimeoutError(f'Shared study budget exhausted at {step}')
    status('preparing')
    try:
        protocol = ROOT / 'research_runs/selection/protocol.json'; p = json.loads(protocol.read_text())
        assert p['steps'] == 3000 and p['batch_size'] == 256 and p['input_length'] == 32 and p['prediction_length'] == 8
        identity = {'run_type': 'main_model', 'policy': args.policy, 'origin': args.origin, 'seed': args.seed,
            'parent_prefix60': 'T023_prefix60_balanced' if args.policy == 'balanced' else 'T042_prefix60_reshuffle',
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'), 'protocol_sha256': digest(protocol),
            'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'), 'code_sha256': digest(Path(__file__)),
            'data_helper_sha256': digest(ROOT / 'scripts/reducible_sampling.py'), 'core_sha256': digest(ROOT / 'scripts/run_research.py'),
            'coverage_helper_sha256': digest(ROOT / 'scripts/coverage_sampling.py'),
            'group_labels_sha256': digest(ROOT / 'research_runs/cluster_selection/window_groups.npz'),
            'budget_helper_sha256': digest(ROOT / 'scripts/covariate_sampling.py'), 'round_limit_seconds': ROUND_LIMIT}
        write_json(run / 'identity.json', identity)
        for source, name in [(Path(__file__), 'code_snapshot.py'), (protocol, 'protocol.json'),
            (FOLDER / 'PROTOCOL.md', 'study_protocol.md'), (ROOT / 'scripts/reducible_sampling.py', 'data_helper_snapshot.py'),
            (ROOT / 'scripts/run_research.py', 'core_snapshot.py'), (ROOT / 'scripts/coverage_sampling.py', 'coverage_helper.py'),
            (ROOT / 'scripts/covariate_sampling.py', 'budget_helper.py')]:
            (run / name).write_bytes(source.read_bytes())
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda(); offsets = torch.arange(40, device='cuda')
        model = DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters()) == 7920
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        n = len(starts); plan = None; q = None
        generator = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        if args.policy == 'balanced':
            coverage_sampling.identity('balanced')
            q = coverage_sampling.build(starts, 'balanced', run)
            cdf = np.cumsum(q); cdf[-1] = 1; cdf = torch.from_numpy(cdf).cuda()
        else:
            rng = np.random.default_rng(args.seed + 1000)
            plan = np.concatenate([rng.permutation(n) for _ in range(int(np.ceil(768000 / n)))])[:768000].reshape(3000, 256).astype(np.int32)
            for start in range(0, plan.size, n):
                piece = plan.ravel()[start:min(start + n, plan.size)]
                assert len(np.unique(piece)) == len(piece)
            np.savez_compressed(run / 'sampling_plan.npz', candidate_indices=plan)
            write_json(run / 'sampling_plan.json', {'sha256': digest(run / 'sampling_plan.npz'), 'seed': args.seed + 1000,
                'shape': list(plan.shape), 'eligible_windows': n, 'source_mapping': 'candidate_indices indexes train_start in windows.npz'})
            plan_gpu = torch.from_numpy(plan.astype(np.int64)).cuda()
        counts = torch.zeros(n, device='cuda', dtype=torch.int64)
        history = []; best = float('inf'); best_step = None; best_metrics = None
        eval_data = torch.from_numpy(eval_values).cuda() if args.origin == 1 else data
        future_gpu = torch.from_numpy(future).cuda()
        for phase in range(0, 3000, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                check(step)
                chosen = plan_gpu[step - 1] if plan is not None else torch.searchsorted(cdf,
                    torch.rand(256, device='cuda', dtype=torch.float64, generator=generator), right=True)
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses))}
            if args.origin == 1:
                check(phase + 500); metrics = evaluate(model, eval_data, future_gpu, groups, p, 'baseline')
                row['validation'] = metrics['groups']['all']
                if metrics['groups']['all']['NMSE'] < best:
                    best = metrics['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = metrics
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': generator.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history, 'best': best, 'best_step': best_step,
                'best_metrics': best_metrics}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draw_count = counts.cpu().numpy(); assert draw_count.sum() == 768000
        if plan is not None: assert np.array_equal(draw_count, np.bincount(plan.ravel(), minlength=n))
        content = {'train_start': starts, 'probe_start': future, 'cell': cells, 'months': months, 'draw_count': draw_count}
        if q is not None: content['q'] = q
        np.savez_compressed(run / 'windows.npz', **content)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'],
            'history': history, 'steps_completed': 3000, 'samples_presented': 768000,
            'sensor_columns': manifest['sensor_columns'], 'windows_sha256': digest(run / 'windows.npz'),
            'unique_windows_presented': int((draw_count > 0).sum()), 'unique_fraction': float((draw_count > 0).mean())}
        if args.origin < 1:
            check(3000); result['future'] = evaluate(model, data, future_gpu, groups, p, 'baseline')
        else:
            result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', 3000)
        print(json.dumps({'completed': args.run_id, 'future': result.get('future'), 'validation': result.get('groups')}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
