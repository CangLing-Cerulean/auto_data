"""Fixed DLinear with static input-leverage sampling and optional correction."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, DLinear, digest, write_json, evaluate
from reducible_sampling import load_data, used_seconds
from leverage_sampling import FOLDER, ROUND_LIMIT, probability_for


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    parser.add_argument('--policy', choices=['uncorrected', 'corrected', 'shuffle'], required=True)
    parser.add_argument('--selected-policy', choices=['uncorrected', 'corrected'])
    parser.add_argument('--score-run', required=True); parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and Path(args.score_run).name == args.score_run
    run = ROOT / 'research_runs/results' / args.run_id; assert not run.exists()
    allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 100
    run.mkdir(); begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})
    def check():
        if time.monotonic() - begun >= allowance: raise TimeoutError('Shared round budget exhausted')
    status('preparing')
    try:
        protocol = ROOT / 'research_runs/selection/protocol.json'; p = json.loads(protocol.read_text())
        assert (p['steps'], p['batch_size'], p['input_length'], p['prediction_length']) == (3000, 256, 32, 8)
        score_folder = ROOT / 'research_runs/results' / args.score_run
        assert json.loads((score_folder / 'status.json').read_text())['phase'] == 'completed'
        assert json.loads((score_folder / 'identity.json').read_text())['origin'] == args.origin
        score_starts, q, weights, score_hash = probability_for(args.score_run, args.policy, args.selected_policy)
        sources = {'code_snapshot.py': Path(__file__), 'policy_snapshot.py': ROOT / 'scripts/leverage_sampling.py',
            'data_helper.py': ROOT / 'scripts/reducible_sampling.py', 'core_snapshot.py': ROOT / 'scripts/run_research.py',
            'budget_helper.py': ROOT / 'scripts/cadence_selection.py', 'normalization_helper.py': ROOT / 'scripts/covariate_sampling.py',
            'protocol.json': protocol, 'study_protocol.md': FOLDER / 'PROTOCOL.md'}
        identity = {'run_type': 'main_model', 'policy': args.policy, 'selected_policy': args.selected_policy,
            'origin': args.origin, 'seed': args.seed, 'score_run': args.score_run, 'score_weights_sha256': score_hash,
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'sources_sha256': {name: digest(path) for name, path in sources.items()}, 'round_limit_seconds': ROUND_LIMIT}
        write_json(run / 'identity.json', identity)
        for name, source in sources.items(): (run / name).write_bytes(source.read_bytes())
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        assert np.array_equal(starts, score_starts); write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        write_json(run / 'environment.json', {'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda()
        eval_data = torch.from_numpy(eval_values).cuda() if args.origin == 1 else data
        future_gpu = torch.from_numpy(future).cuda(); offsets = torch.arange(40, device='cuda')
        cdf = np.cumsum(q); cdf[-1] = 1.; cdf = torch.from_numpy(cdf).cuda()
        weights_gpu = torch.from_numpy(weights.astype(np.float32)).cuda()
        model = DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters()) == 7920
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        rng = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        counts = torch.zeros(len(starts), device='cuda', dtype=torch.int64)
        history = []; best = float('inf'); best_step = None; best_metrics = None
        corrected = (args.selected_policy if args.policy == 'shuffle' else args.policy) == 'corrected'
        for phase in range(0, 3000, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                check()
                chosen = torch.searchsorted(cdf, torch.rand(256, device='cuda', dtype=torch.float64, generator=rng), right=True)
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                error = (model(batch[:, :32]) - batch[:, 32:]).square()
                loss = (error.mean((1, 2)) * weights_gpu[chosen]).mean() if corrected else error.mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses))}
            if args.origin == 1:
                check(); metrics = evaluate(model, eval_data, future_gpu, groups, p, 'baseline')
                row['validation'] = metrics['groups']['all']
                if metrics['groups']['all']['NMSE'] < best:
                    best = metrics['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = metrics
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': rng.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draws = counts.cpu().numpy(); assert draws.sum() == 768000
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=future, cell=cells, months=months,
            probability=q, loss_weight=weights, draw_count=draws)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'], 'history': history,
            'steps_completed': 3000, 'samples_presented': 768000, 'sensor_columns': manifest['sensor_columns'],
            'corrected': corrected, 'expected_loss_weight': float(q @ weights),
            'min_relative_probability': float(q.min() * len(q)), 'max_relative_probability': float(q.max() * len(q)),
            'windows_sha256': digest(run / 'windows.npz')}
        if args.origin < 1:
            check(); result['future'] = evaluate(model, data, future_gpu, groups, p, 'baseline')
        else:
            result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', 3000)
        print(json.dumps({'completed': args.run_id, 'metrics': result.get('future', result)['groups']['all']}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
