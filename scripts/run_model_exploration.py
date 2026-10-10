"""Fixed model experiments; development reads train and validation only."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from forecast_models import CrossChannelForecast
from run_research import ROOT, digest, write_json, prepare, evaluate
from reducible_sampling import used_seconds

FOLDER = ROOT / 'research_runs/model_exploration'
ROUND_START = 5945.514959004708
ROUND_LIMIT = ROUND_START + 1000
BASELINES = {2026: 'S000_uniform_s2026', 2027: 'S005_uniform_s2027', 2028: 'S010_uniform_s2028'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--kind', choices=['cross_linear', 'residual_mlp'], required=True)
    parser.add_argument('--seed', type=int, choices=list(BASELINES), required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id
    assert not run.exists(), 'Cannot overwrite a run'
    allowance = ROUND_LIMIT - used_seconds() - 20
    assert allowance > 100
    run.mkdir(); begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': time.monotonic() - begun})
    def check():
        if time.monotonic() - begun >= allowance:
            raise TimeoutError('Shared development budget exhausted')
    status('preparing')
    try:
        protocol = ROOT / 'research_runs/selection/protocol.json'
        p = json.loads(protocol.read_text())
        assert (p['steps'], p['batch_size'], p['input_length'], p['prediction_length'], p['channels']) == (3000, 256, 32, 8, 15)
        manifest_path = ROOT / 'data/metropt3_v1/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        sources = { 'code_snapshot.py': Path(__file__), 'model_snapshot.py': ROOT / 'scripts/forecast_models.py',
            'core_snapshot.py': ROOT / 'scripts/run_research.py', 'budget_helper.py': ROOT / 'scripts/reducible_sampling.py',
            'protocol.json': protocol, 'study_protocol.md': FOLDER / 'PROTOCOL.md'}
        identity = {'run_type': 'main_model', 'model': args.kind, 'seed': args.seed,
            'dataset_id': manifest['dataset_id'], 'manifest_sha256': digest(manifest_path),
            'baseline': BASELINES[args.seed], 'round_limit_seconds': ROUND_LIMIT,
            'sources_sha256': {name: digest(path) for name, path in sources.items()}}
        write_json(run / 'identity.json', identity)
        for name, path in sources.items(): (run / name).write_bytes(path.read_bytes())
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        arrays, starts, groups, audit = prepare(p, manifest)
        write_json(run / 'data_audit.json', audit)
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = {s: torch.from_numpy(v).cuda() for s, v in arrays.items()}
        indices = {s: torch.from_numpy(v).cuda() for s, v in starts.items()}
        n = len(starts['train'])
        q = np.full(n, 1. / n); cdf = np.cumsum(q); cdf[-1] = 1.
        cdf = torch.from_numpy(cdf).cuda()
        model = CrossChannelForecast(args.kind).cuda()
        params = sum(v.numel() for v in model.parameters())
        assert params == {'cross_linear': 57720, 'residual_mlp': 77048}[args.kind]
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        generator = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        counts = torch.zeros(n, device='cuda', dtype=torch.int64)
        offsets = torch.arange(40, device='cuda')
        best = float('inf'); best_step = None; best_metrics = None; history = []
        for phase in range(0, 3000, 500):
            losses = []
            for step in range(phase + 1, phase + 501):
                check()
                chosen = torch.searchsorted(cdf, torch.rand(256, device='cuda', dtype=torch.float64, generator=generator), right=True)
                batch = data['train'][indices['train'][chosen, None] + offsets]
                optimizer.zero_grad(set_to_none=True)
                loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean((1, 2)).mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                counts.scatter_add_(0, chosen, torch.ones_like(chosen))
            check()
            metrics = evaluate(model, data['validation'], indices['validation'], groups, p, 'baseline')
            row = {'step': phase + 500, 'train_loss': float(np.mean(losses)), 'validation': metrics['groups']['all']}
            history.append(row)
            if metrics['groups']['all']['NMSE'] < best:
                best = metrics['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = metrics
                torch.save(model.state_dict(), run / 'best.pt')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'sampling_rng': generator.get_state(),
                'counts': counts, 'step': phase + 500, 'history': history}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        draw = counts.cpu().numpy(); assert draw.sum() == 768000
        with np.load(ROOT / 'research_runs/results' / BASELINES[args.seed] / 'sampling_weights.npz') as ref:
            assert np.array_equal(draw, ref['draw_count']), 'Paired sampling mismatch'
        np.savez_compressed(run / 'windows.npz', train_start=starts['train'], validation_start=starts['validation'], draw_count=draw)
        torch.save(model.state_dict(), run / 'final.pt')
        result = {**best_metrics, 'run_id': args.run_id, 'identity': identity, 'parameter_count': params,
            'best_step': best_step, 'history': history, 'steps_completed': 3000, 'samples_presented': 768000,
            'sensor_columns': manifest['sensor_columns'], 'paired_sampling_exact': True,
            'raw_unit_MSE': float(np.mean(np.array(best_metrics['per_channel_NMSE']) * np.array(audit['scale']) ** 2)),
            'elapsed_seconds': time.monotonic() - begun}
        write_json(run / 'result.json', result); status('completed', 3000)
        print(json.dumps({'completed': args.run_id, 'metrics': result['groups']['all']}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
