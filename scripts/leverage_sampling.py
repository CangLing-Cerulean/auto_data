"""Fixed train-input ridge leverage scores for the original DLinear task."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, digest, write_json
from reducible_sampling import load_data, used_seconds
from covariate_sampling import bounded_weights
from cadence_selection import ROUND_START, ROUND_LIMIT

FOLDER = ROOT / 'research_runs/leverage_sampling'
STAGE_START = 6219.7936278916895
RIDGE_FRACTION = .001
SCORE_BATCH = 8192
SHUFFLE_SEED = 20261004


def probability_for(score_run, policy, selected):
    path = ROOT / 'research_runs/results' / score_run / 'leverage_weights.npz'
    with np.load(path) as z:
        starts = z['train_start'].copy(); q = z['probability'].copy(); cells = z['cell'].copy()
    if policy == 'shuffle':
        assert selected in ('uncorrected', 'corrected')
        rng = np.random.default_rng(SHUFFLE_SEED)
        for cell in np.unique(cells):
            indices = np.flatnonzero(cells == cell)
            q[indices] = q[rng.permutation(indices)]
    else:
        assert selected is None
    q /= q.sum(); n = len(q)
    assert np.isfinite(q).all() and q.min() * n >= .5 - 1e-10 and q.max() * n <= 2 + 1e-10
    corrected = (selected if policy == 'shuffle' else policy) == 'corrected'
    weight = 1. / (n * q) if corrected else np.ones(n)
    assert abs(float(q @ weight) - 1) < 1e-10
    return starts, q, weight, digest(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    args = parser.parse_args(); assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id; assert not run.exists()
    allowance = ROUND_LIMIT - used_seconds() - 20; assert allowance > 100
    run.mkdir(); begun = time.monotonic()
    def status(phase):
        write_json(run / 'status.json', {'phase': phase, 'elapsed_seconds': time.monotonic() - begun})
    def check():
        if time.monotonic() - begun >= allowance: raise TimeoutError('Shared round budget exhausted')
    status('fitting_input_geometry')
    try:
        sources = {'code_snapshot.py': Path(__file__), 'normalization_helper.py': ROOT / 'scripts/covariate_sampling.py',
            'data_helper.py': ROOT / 'scripts/reducible_sampling.py', 'core_snapshot.py': ROOT / 'scripts/run_research.py',
            'budget_helper.py': ROOT / 'scripts/cadence_selection.py', 'study_protocol.md': FOLDER / 'PROTOCOL.md'}
        identity = {'run_type': 'input_geometry_diagnostic', 'origin': args.origin, 'ridge_fraction': RIDGE_FRACTION,
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'sources_sha256': {name: digest(path) for name, path in sources.items()}, 'round_limit_seconds': ROUND_LIMIT}
        write_json(run / 'identity.json', identity)
        for name, source in sources.items(): (run / name).write_bytes(source.read_bytes())
        manifest, values, starts, _, cells, months, _, future, _, audit = load_data(args.origin, reference_only=True)
        assert future is None and len(values) == audit['train_raw_rows'][1]
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        data = torch.from_numpy(values).cuda(); indices = torch.from_numpy(starts).cuda()
        offsets = torch.arange(32, device='cuda'); n = len(starts)
        def features(lo):
            x = data[indices[lo:lo + SCORE_BATCH, None] + offsets].double().permute(2, 0, 1)
            return torch.cat((x, torch.ones((15, x.shape[1], 1), device='cuda', dtype=torch.float64)), dim=2)
        gram = torch.zeros((15, 33, 33), dtype=torch.float64, device='cuda')
        for lo in range(0, n, SCORE_BATCH):
            check(); z = features(lo); gram += z.transpose(1, 2) @ z
        gram /= n; gram = (gram + gram.transpose(1, 2)) / 2
        ridge = RIDGE_FRACTION * gram.diagonal(dim1=1, dim2=2).mean(1)
        inverse = torch.linalg.inv(gram + ridge[:, None, None] * torch.eye(33, device='cuda', dtype=torch.float64))
        effective_df = (gram * inverse.transpose(1, 2)).sum((1, 2)); assert (effective_df > 0).all()
        score = np.empty(n, dtype=np.float64)
        for lo in range(0, n, SCORE_BATCH):
            check(); z = features(lo)
            batch_score = (((z @ inverse) * z).sum(2) / effective_df[:, None]).mean(0)
            score[lo:lo + len(batch_score)] = batch_score.cpu().numpy()
        assert np.isfinite(score).all() and score.min() > 0 and abs(score.mean() - 1) < 1e-8
        q = bounded_weights(np.log(score), np.zeros(n, dtype=np.int8)) / n
        np.savez_compressed(run / 'geometry.npz', gram=gram.cpu().numpy(), inverse=inverse.cpu().numpy(),
            ridge=ridge.cpu().numpy(), effective_df=effective_df.cpu().numpy())
        np.savez_compressed(run / 'leverage_weights.npz', train_start=starts, cell=cells, months=months,
            score=score, probability=q)
        result = {'identity': identity, 'dataset_id': manifest['dataset_id'], 'windows': n,
            'score_mean': float(score.mean()), 'score_quantiles': np.quantile(score, [0, .25, .5, .75, .9, .99, 1]).tolist(),
            'min_relative_probability': float(q.min() * n), 'max_relative_probability': float(q.max() * n),
            'effective_df': effective_df.cpu().tolist(), 'test_accessed': False,
            'future_target_values_used_in_score': False, 'model_fitted': False,
            'weights_sha256': digest(run / 'leverage_weights.npz'),
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name()}}
        write_json(run / 'result.json', result); status('completed')
        print(json.dumps({'completed': args.run_id, 'windows': n, 'score_mean': result['score_mean']}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
