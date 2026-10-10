"""Uniform-objective reshuffling and equal-CDF temporal stratification."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, DLinear, digest, write_json, evaluate
from reducible_sampling import load_data, used_seconds, ROUND_LIMIT, STEPS, BATCH_SIZE

FOLDER = ROOT / 'research_runs/batch_sampling'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    parser.add_argument('--policy', choices=['reshuffle', 'time_stratified'], required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(); assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id
    assert not (run / 'result.json').exists(), 'Cannot overwrite completed run'
    run.mkdir(exist_ok=True)
    protocol = ROOT / 'research_runs/selection/protocol.json'; p = json.loads(protocol.read_text())
    assert p['steps'] == STEPS and p['batch_size'] == BATCH_SIZE and p['input_length'] == 32 and p['prediction_length'] == 8
    identity = {'policy': args.policy, 'origin': args.origin, 'seed': args.seed, 'run_type': 'main_model',
        'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'), 'protocol_sha256': digest(protocol),
        'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'), 'code_sha256': digest(Path(__file__)),
        'data_helper_sha256': digest(ROOT / 'scripts/reducible_sampling.py'), 'core_sha256': digest(ROOT / 'scripts/run_research.py'),
        'round_limit_seconds': ROUND_LIMIT}
    if (run / 'identity.json').exists(): assert json.loads((run / 'identity.json').read_text()) == identity
    else:
        write_json(run / 'identity.json', identity)
        for file, name in [(Path(__file__), 'code_snapshot.py'), (ROOT / 'scripts/reducible_sampling.py', 'helper_snapshot.py'),
                           (ROOT / 'scripts/run_research.py', 'core_snapshot.py'), (protocol, 'protocol.json'),
                           (FOLDER / 'PROTOCOL.md', 'study_protocol.md')]: (run / name).write_bytes(file.read_bytes())
    allowance = ROUND_LIMIT - used_seconds() - 20
    assert allowance > 0
    old = json.loads((run / 'status.json').read_text()).get('elapsed_seconds', 0) if (run / 'status.json').exists() else 0
    begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': old + time.monotonic() - begun})
    def check(step):
        if time.monotonic() - begun >= allowance: raise TimeoutError(f'Original study budget exhausted at step {step}')
    status('preparing')
    try:
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        write_json(run / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda(); offsets = torch.arange(40, device='cuda')
        model = DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters()) == 7920
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        n = len(starts); plan_path = run / 'sampling_plan.npz'
        if plan_path.exists():
            assert digest(plan_path) == json.loads((run / 'sampling_plan.json').read_text())['sha256']
            with np.load(plan_path) as z: draw_plan = z['candidate_indices']
        else:
            if args.policy == 'reshuffle':
                rng = np.random.default_rng(args.seed + 1000)
                cycles = int(np.ceil(STEPS * BATCH_SIZE / n))
                draw_plan = np.concatenate([rng.permutation(n) for _ in range(cycles)])[:STEPS * BATCH_SIZE].reshape(STEPS, BATCH_SIZE).astype(np.int32)
                for start in range(0, draw_plan.size, n):
                    piece = draw_plan.ravel()[start:min(start + n, draw_plan.size)]
                    assert len(np.unique(piece)) == len(piece)
            else:
                generator = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
                cdf = torch.from_numpy(np.cumsum(np.full(n, 1 / n, dtype=np.float64))).cuda(); cdf[-1] = 1
                strata = torch.arange(BATCH_SIZE, device='cuda', dtype=torch.float64)
                planned = []
                for step in range(STEPS):
                    check(step)
                    u = (strata + torch.rand(BATCH_SIZE, device='cuda', dtype=torch.float64, generator=generator)) / BATCH_SIZE
                    planned.append(torch.searchsorted(cdf, u, right=True))
                draw_plan = torch.stack(planned).cpu().numpy().astype(np.int32)
            np.savez_compressed(plan_path, candidate_indices=draw_plan)
            write_json(run / 'sampling_plan.json', {'policy': args.policy, 'seed': args.seed + 1000,
                'shape': list(draw_plan.shape), 'sha256': digest(plan_path), 'eligible_windows': n,
                'source_mapping': 'candidate_indices index train_start in windows.npz; source unchanged'})
        assert draw_plan.shape == (STEPS, BATCH_SIZE) and draw_plan.min() >= 0 and draw_plan.max() < n
        draw_gpu = torch.from_numpy(draw_plan.astype(np.int64)).cuda()
        counts = torch.zeros(n, device='cuda', dtype=torch.int64)
        step0 = 0; history = []; best = float('inf'); best_step = None; best_metrics = None
        if (run / 'latest.pt').exists():
            saved = torch.load(run / 'latest.pt', map_location='cuda', weights_only=False)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            counts = saved['counts']; step0 = saved['step']; history = saved['history']
            best = saved['best']; best_step = saved['best_step']; best_metrics = saved['best_metrics']
            assert saved['sampling_plan_sha256'] == digest(plan_path)
        for phase in range(step0, STEPS, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                check(step); chosen = draw_gpu[step - 1]
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses))}
            if args.origin == 1:
                check(phase + 500)
                measured = evaluate(model, torch.from_numpy(eval_values).cuda(), torch.from_numpy(future).cuda(), groups, p, 'baseline')
                row['validation'] = measured['groups']['all']
                if measured['groups']['all']['NMSE'] < best:
                    best = measured['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = measured
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'counts': counts,
                'step': phase + 500, 'history': history, 'best': best, 'best_step': best_step, 'best_metrics': best_metrics,
                'sampling_plan_sha256': digest(plan_path)}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draw_counts = counts.cpu().numpy()
        assert int(draw_counts.sum()) == STEPS * BATCH_SIZE
        assert np.array_equal(draw_counts, np.bincount(draw_plan.ravel(), minlength=n))
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=future, cell=cells, months=months, draw_count=draw_counts)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'],
            'history': history, 'steps_completed': STEPS, 'samples_presented': STEPS * BATCH_SIZE,
            'sampling_plan_sha256': digest(plan_path), 'windows_sha256': digest(run / 'windows.npz'),
            'sensor_columns': manifest['sensor_columns'], 'unique_windows_presented': int((draw_counts > 0).sum()),
            'unique_fraction': float((draw_counts > 0).mean()), 'minimum_presentations': int(draw_counts.min()),
            'maximum_presentations': int(draw_counts.max()), 'additional_model_forward_presentations': 0}
        if args.origin < 1:
            check(STEPS); result['future'] = evaluate(model, data, torch.from_numpy(future).cuda(), groups, p, 'baseline')
        else: result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = old + time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', STEPS)
        print(json.dumps({'completed': args.run_id, 'future': result.get('future'), 'validation': result.get('groups')}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
