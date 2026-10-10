"""Original DLinear trained with frozen covariate-based window probabilities."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, DLinear, digest, write_json, evaluate
from covariate_sampling import FOLDER, ROUND_LIMIT, load_data, used_seconds, probability_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    parser.add_argument('--policy', choices=['global', 'state_matched', 'shuffle'], required=True)
    parser.add_argument('--selected-policy', choices=['global', 'state_matched'])
    parser.add_argument('--score-run', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and Path(args.score_run).name == args.score_run
    run = ROOT / 'research_runs/results' / args.run_id
    assert not (run / 'result.json').exists(), 'Cannot overwrite completed run'
    protocol = ROOT / 'research_runs/selection/protocol.json'; p = json.loads(protocol.read_text())
    assert p['steps'] == 3000 and p['batch_size'] == 256 and p['input_length'] == 32 and p['prediction_length'] == 8
    q, score_starts, score_digest = probability_for(args.score_run, args.policy, args.selected_policy)
    score_identity = json.loads((ROOT / 'research_runs/results' / args.score_run / 'identity.json').read_text())
    assert score_identity['origin'] == args.origin
    identity = {'policy': args.policy, 'selected_policy': args.selected_policy, 'origin': args.origin, 'seed': args.seed,
        'run_type': 'main_model', 'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
        'protocol_sha256': digest(protocol), 'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'),
        'code_sha256': digest(Path(__file__)), 'score_run': args.score_run, 'score_weights_sha256': score_digest,
        'scoring_code_sha256': digest(ROOT / 'scripts/covariate_sampling.py'),
        'data_helper_sha256': digest(ROOT / 'scripts/reducible_sampling.py'),
        'core_sha256': digest(ROOT / 'scripts/run_research.py'), 'round_limit_seconds': ROUND_LIMIT}
    run.mkdir(exist_ok=True)
    if (run / 'identity.json').exists(): assert json.loads((run / 'identity.json').read_text()) == identity
    else:
        write_json(run / 'identity.json', identity)
        for source, name in [(Path(__file__), 'code_snapshot.py'), (ROOT / 'scripts/covariate_sampling.py', 'scoring_snapshot.py'),
            (ROOT / 'scripts/reducible_sampling.py', 'data_helper_snapshot.py'), (ROOT / 'scripts/run_research.py', 'core_snapshot.py'),
            (protocol, 'protocol.json'), (FOLDER / 'PROTOCOL.md', 'study_protocol.md')]:
            (run / name).write_bytes(source.read_bytes())
    allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 0
    old = json.loads((run / 'status.json').read_text()).get('elapsed_seconds', 0) if (run / 'status.json').exists() else 0
    begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': old + time.monotonic() - begun})
    def check(step):
        if time.monotonic() - begun >= allowance: raise TimeoutError(f'Study budget exhausted at step {step}')
    status('preparing')
    try:
        manifest, values, starts, _, cells, months, eval_values, future, groups, audit = load_data(args.origin)
        assert np.array_equal(starts, score_starts)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda(); offsets = torch.arange(40, device='cuda')
        model = DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters()) == 7920
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        cumulative = np.cumsum(q); cumulative[-1] = 1
        cdf = torch.from_numpy(cumulative).cuda()
        counts = torch.zeros(len(starts), dtype=torch.int64, device='cuda')
        step0 = 0; history = []; best = float('inf'); best_step = None; best_metrics = None
        if (run / 'latest.pt').exists():
            saved = torch.load(run / 'latest.pt', map_location='cuda', weights_only=False)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['sampling_rng'].cpu()); counts = saved['counts']; step0 = saved['step']
            history = saved['history']; best = saved['best']; best_step = saved['best_step']; best_metrics = saved['best_metrics']
        evaluation_data = torch.from_numpy(eval_values).cuda() if args.origin == 1 else data
        future_gpu = torch.from_numpy(future).cuda()
        for phase in range(step0, 3000, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                check(step)
                chosen = torch.searchsorted(cdf, torch.rand(256, device='cuda', dtype=torch.float64, generator=generator), right=True)
                batch = data[positions[chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses))}
            if args.origin == 1:
                check(phase + 500)
                measured = evaluate(model, evaluation_data, future_gpu, groups, p, 'baseline')
                row['validation'] = measured['groups']['all']
                if measured['groups']['all']['NMSE'] < best:
                    best = measured['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = measured
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': generator.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history, 'best': best, 'best_step': best_step,
                'best_metrics': best_metrics}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draw_counts = counts.cpu().numpy(); assert draw_counts.sum() == 768000
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=future, cell=cells, months=months, q=q, draw_count=draw_counts)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'], 'history': history,
            'steps_completed': 3000, 'samples_presented': 768000, 'sensor_columns': manifest['sensor_columns'],
            'unique_windows_presented': int((draw_counts > 0).sum()), 'unique_fraction': float((draw_counts > 0).mean()),
            'min_relative_probability': float(q.min() * len(q)), 'max_relative_probability': float(q.max() * len(q)),
            'windows_sha256': digest(run / 'windows.npz')}
        if args.origin < 1:
            check(3000); result['future'] = evaluate(model, data, future_gpu, groups, p, 'baseline')
        else:
            result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = old + time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', 3000)
        print(json.dumps({'completed': args.run_id, 'future': result.get('future'), 'validation': result.get('groups')}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
