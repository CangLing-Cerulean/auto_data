"""Fixed forecasting model with an immutable train-only data policy."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, digest, write_json, evaluate
from reducible_sampling import load_data

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--score-run', required=True); parser.add_argument('--policy', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .7, .8, 1.], required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--model', choices=['dlinear', 'residual_mlp'], default='dlinear')
    args = parser.parse_args()
    assert all(Path(value).name == value for value in (args.run_id, args.score_run, args.policy))
    run = ROOT / 'research_runs/results' / args.run_id; assert not run.exists(); run.mkdir()
    begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})
    status('preparing')
    try:
        source = ROOT / 'research_runs/results' / args.score_run
        score_identity = json.loads((source / 'identity.json').read_text())
        assert score_identity['origin'] == args.origin
        assert json.loads((source / 'status.json').read_text())['phase'] == 'completed'
        policy = source / ('policy_' + args.policy + '.npz')
        policy_entry = next(row for row in json.loads((source / 'result.json').read_text())['policies'] if row['policy'] == args.policy)
        assert digest(policy) == policy_entry['sha256']
        protocol = ROOT / 'research_runs/selection/protocol.json'; p = json.loads(protocol.read_text())
        snapshots = {'code_snapshot.py': Path(__file__), 'core_snapshot.py': ROOT / 'scripts/run_research.py',
            'data_helper.py': ROOT / 'scripts/reducible_sampling.py', 'protocol.json': protocol,
            'study_protocol.md': FOLDER / 'PROTOCOL.md', 'policies.md': FOLDER / 'POLICIES.md'}
        if args.model == 'residual_mlp':
            assert (FOLDER / 'MODEL_TRANSFER_PROTOCOL.md').exists()
            snapshots['model_transfer_protocol.md'] = FOLDER / 'MODEL_TRANSFER_PROTOCOL.md'
            snapshots['model_snapshot.py'] = ROOT / 'scripts/forecast_models.py'
        identity = {'run_type': 'main_model', **vars(args), 'policy_sha256': digest(policy),
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'sources_sha256': {name: digest(path) for name, path in snapshots.items()}}
        write_json(run / 'identity.json', identity)
        for name, path in snapshots.items(): (run / name).write_bytes(path.read_bytes())
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        write_json(run / 'data_audit.json', audit)
        with np.load(policy) as saved:
            assert np.array_equal(starts, saved['train_start']) and np.array_equal(cells, saved['cell'])
            q = saved['probability'].copy(); weight = saved['loss_weight'].copy(); expectation = float(saved['expected_loss_weight'])
        assert weight.shape == (len(starts),) or weight.shape == (len(starts), 8, 15)
        assert np.isfinite(q).all() and np.all(q >= 0) and abs(q.sum() - 1) < 1e-12
        assert np.isfinite(weight).all() and weight.min() > 0 and expectation > 0
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda(); offsets = torch.arange(40, device='cuda')
        eval_data = torch.from_numpy(eval_values).cuda() if args.origin == 1 else data
        future_gpu = torch.from_numpy(future).cuda()
        cdf = np.cumsum(q); cdf[-1] = 1.; cdf = torch.from_numpy(cdf).cuda()
        weight_gpu = torch.from_numpy(weight.astype(np.float32)).cuda()
        if args.model == 'dlinear':
            model = DLinear(p).cuda(); expected_parameters = 7920
        else:
            from forecast_models import CrossChannelForecast
            model = CrossChannelForecast('residual_mlp').cuda(); expected_parameters = 77048
        assert sum(v.numel() for v in model.parameters()) == expected_parameters
        torch.save(model.state_dict(), run / 'initial.pt')
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        rng = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        counts = torch.zeros(len(starts), device='cuda', dtype=torch.int64)
        history = []; best = float('inf'); best_step = None; best_metrics = None
        plain_internal = args.origin < 1 and weight.ndim == 1 and np.array_equal(weight, np.ones(len(starts), dtype=np.float32))
        for phase in range(0, 3000, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                chosen = torch.searchsorted(cdf, torch.rand(256, device='cuda', dtype=torch.float64, generator=rng), right=True)
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                squared = (model(batch[:, :32]) - batch[:, 32:]).square()
                if plain_internal:
                    loss = squared.mean()
                elif weight.ndim == 1:
                    loss = (squared.mean((1, 2)) * weight_gpu[chosen]).mean() / expectation
                else:
                    loss = (squared * weight_gpu[chosen]).mean() / expectation
                assert torch.isfinite(loss); loss.backward(); optimizer.step()
                losses.append(float(loss.detach())); counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses)), 'elapsed_seconds': time.monotonic() - begun}
            if args.origin == 1:
                metrics = evaluate(model, eval_data, future_gpu, groups, p, 'baseline')
                row['validation'] = metrics['groups']['all']
                if metrics['groups']['all']['NMSE'] < best:
                    best = metrics['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = metrics
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': rng.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history, 'best': best, 'best_step': best_step}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draws = counts.cpu().numpy(); assert draws.sum() == 768000 and not np.any(draws[q == 0])
        if np.array_equal(q, np.full(len(q), 1 / len(q))) and args.seed == 2026 and args.origin in (.6, .8, 1):
            baseline = { .6: 'T000_prefix60_full', .8: 'T005_prefix80_full', 1.: 'S000_uniform_s2026'}[args.origin]
            filename = 'sampling_weights.npz' if args.origin == 1 else 'windows.npz'
            with np.load(ROOT / 'research_runs/results' / baseline / filename) as reference:
                assert np.array_equal(draws, reference['draw_count'])
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=future, cell=cells, months=months,
            probability=q, draw_count=draws)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'steps_completed': 3000, 'samples_presented': 768000,
            'history': history, 'sensor_columns': manifest['sensor_columns'], 'expected_loss_weight': expectation,
            'weights_shape': list(weight.shape), 'weights_min': float(weight.min()), 'weights_max': float(weight.max()),
            'retained_windows': int((q > 0).sum()), 'windows_sha256': digest(run / 'windows.npz')}
        if args.origin < 1: result['future'] = evaluate(model, data, future_gpu, groups, p, 'baseline')
        else: result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', 3000)
        print(json.dumps({'completed': args.run_id, 'metrics': result.get('future', result)['groups']['all']}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
