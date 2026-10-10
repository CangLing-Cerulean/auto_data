"""Train-only input density-ratio scoring; no predictive-model changes."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, digest, write_json
from reducible_sampling import load_data, used_seconds

FOLDER = ROOT / 'research_runs/covariate_sampling'
ROUND_START = 5391.239370419644
ROUND_LIMIT = ROUND_START + 1200
RECENT_DAYS = 30
DOMAIN_SEED = 20261012
SHUFFLE_SEED = 20261013
DOMAIN_SAMPLE_SIZE = 32768
LOW_WEIGHT = .5
HIGH_WEIGHT = 2.
REGULARIZATION = .001


def bounded_weights(log_ratio, groups):
    ratio = np.exp(np.clip(log_ratio, -20, 20))
    weights = np.empty_like(ratio)
    for group in np.unique(groups):
        mask = groups == group; values = ratio[mask]
        lo, hi = 0., max(1., HIGH_WEIGHT / values.min())
        for _ in range(80):
            middle = (lo + hi) / 2
            if np.clip(middle * values, LOW_WEIGHT, HIGH_WEIGHT).mean() < 1:
                lo = middle
            else:
                hi = middle
        weights[mask] = np.clip((lo + hi) / 2 * values, LOW_WEIGHT, HIGH_WEIGHT)
        assert abs(weights[mask].mean() - 1) < 1e-10
    assert weights.min() >= LOW_WEIGHT and weights.max() <= HIGH_WEIGHT
    return weights


def probability_for(score_run, policy, selected_policy=None):
    path = ROOT / 'research_runs/results' / score_run / 'density_weights.npz'
    with np.load(path) as z:
        original = selected_policy if policy == 'shuffle' else policy
        q = z['q_' + original].copy(); cells = z['cell'].copy(); starts = z['train_start'].copy()
        if policy == 'shuffle':
            rng = np.random.default_rng(SHUFFLE_SEED)
            for cell in np.unique(cells):
                members = np.flatnonzero(cells == cell)
                q[members] = q[rng.permutation(members)]
    q /= q.sum()
    assert np.isfinite(q).all() and q.min() * len(q) >= LOW_WEIGHT - 1e-9
    assert q.max() * len(q) <= HIGH_WEIGHT + 1e-9
    return q, starts, digest(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    args = parser.parse_args(); assert Path(args.run_id).name == args.run_id
    run = ROOT / 'research_runs/results' / args.run_id
    assert not run.exists(), 'Auxiliary fit requires a new run ID'
    remaining = ROUND_LIMIT - used_seconds() - 20; assert remaining > 0
    run.mkdir(); begun = time.monotonic()
    def status(phase):
        write_json(run / 'status.json', {'phase': phase, 'elapsed_seconds': time.monotonic() - begun})
    def check():
        if time.monotonic() - begun >= remaining: raise TimeoutError('Study budget exhausted')
    status('preparing')
    try:
        identity = {'run_type': 'auxiliary_domain_classifier', 'origin': args.origin,
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'code_sha256': digest(Path(__file__)), 'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'),
            'data_helper_sha256': digest(ROOT / 'scripts/reducible_sampling.py'),
            'core_sha256': digest(ROOT / 'scripts/run_research.py'), 'round_limit_seconds': ROUND_LIMIT}
        write_json(run / 'identity.json', identity)
        for src, dst in [(Path(__file__), 'code_snapshot.py'), (FOLDER / 'PROTOCOL.md', 'study_protocol.md'),
            (ROOT / 'scripts/reducible_sampling.py', 'data_helper_snapshot.py'),
            (ROOT / 'scripts/run_research.py', 'core_snapshot.py')]:
            (run / dst).write_bytes(src.read_bytes())
        manifest, values, starts, _, cells, months, _, _, _, audit = load_data(args.origin, reference_only=True)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        write_json(run / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda()
        offsets = torch.arange(32, device='cuda'); parts = []
        with torch.no_grad():
            for start in range(0, len(starts), 8192):
                check(); batch = data[positions[start:start + 8192, None] + offsets]
                parts.append(torch.cat([batch.mean(1), batch.std(1, unbiased=False), batch[:, -1] - batch[:, 0]], dim=1).double())
            features = torch.cat(parts); del parts
            mean = features.mean(0); std = features.std(0, unbiased=False)
            scale = torch.where(std == 0, torch.ones_like(std), std)
            features = (features - mean) / scale
            assert torch.isfinite(features).all()
        with (ROOT / 'data/metropt3_v1/train.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.reader(handle); next(reader)
            timestamps = np.array([row[1] for row in reader], dtype='datetime64[s]')
        input_end = timestamps[starts + 31]
        recent = input_end >= input_end.max() - np.timedelta64(RECENT_DAYS, 'D')
        m = min(DOMAIN_SAMPLE_SIZE, int(recent.sum()), len(starts)); assert m > 0
        rng = np.random.default_rng(DOMAIN_SEED)
        source_indices = rng.choice(len(starts), m, replace=False)
        target_indices = rng.choice(np.flatnonzero(recent), m, replace=False)
        train_indices = torch.from_numpy(np.r_[source_indices, target_indices]).cuda()
        inputs = features[train_indices]
        targets = torch.cat([torch.zeros(m, device='cuda', dtype=torch.float64), torch.ones(m, device='cuda', dtype=torch.float64)])
        coefficients = torch.nn.Parameter(torch.zeros(45, dtype=torch.float64, device='cuda'))
        intercept = torch.nn.Parameter(torch.zeros((), dtype=torch.float64, device='cuda'))
        optimizer = torch.optim.LBFGS([coefficients, intercept], lr=1, max_iter=100, history_size=50,
                                     tolerance_grad=1e-7, line_search_fn='strong_wolfe')
        losses = []
        def closure():
            check(); optimizer.zero_grad(set_to_none=True)
            logits = inputs @ coefficients + intercept
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets) + REGULARIZATION / 2 * coefficients.square().sum()
            assert torch.isfinite(loss); loss.backward(); losses.append(float(loss.detach())); return loss
        status('fitting'); optimizer.step(closure); check()
        with torch.no_grad():
            log_ratio = (features @ coefficients + intercept).cpu().numpy()
            fitted_logits = inputs @ coefficients + intercept
            accuracy = float(((fitted_logits >= 0) == (targets == 1)).double().mean())
            final_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(fitted_logits, targets))
        states = cells // len(np.unique(months))
        global_weights = bounded_weights(log_ratio, np.zeros(len(starts), dtype=np.int8))
        matched_weights = bounded_weights(log_ratio, states)
        q_global = global_weights / global_weights.sum(); q_matched = matched_weights / matched_weights.sum()
        profiles = {}
        for name, q in [('global', q_global), ('state_matched', q_matched)]:
            profiles[name] = {'minimum_relative_probability': float(q.min() * len(q)),
                'maximum_relative_probability': float(q.max() * len(q)),
                'effective_sample_size': float(1 / np.square(q).sum()),
                'recent_probability_mass': float(q[recent].sum()),
                'state_mass': {str(int(g)): float(q[states == g].sum()) for g in np.unique(states)}}
        np.savez_compressed(run / 'density_weights.npz', train_start=starts, cell=cells, state=states, months=months,
            input_end=input_end.astype(np.int64), recent=recent, log_density_ratio=log_ratio,
            q_global=q_global, q_state_matched=q_matched, feature_mean=mean.cpu().numpy(), feature_scale=scale.cpu().numpy(),
            coefficients=coefficients.detach().cpu().numpy(), intercept=intercept.detach().cpu().numpy(),
            source_fit_indices=source_indices, recent_fit_indices=target_indices)
        torch.save({'coefficients': coefficients.detach(), 'intercept': intercept.detach(), 'optimizer': optimizer.state_dict(),
            'feature_mean': mean, 'feature_scale': scale}, run / 'classifier.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'],
            'eligible_windows': len(starts), 'recent_windows': int(recent.sum()), 'recent_population_share': float(recent.mean()),
            'latest_input_end': str(input_end.max()), 'recent_start': str(input_end.max() - np.timedelta64(RECENT_DAYS, 'D')),
            'fitted_rows_per_domain': m, 'source_recent_fit_overlap': int(len(np.intersect1d(source_indices, target_indices))),
            'in_sample_domain_accuracy': accuracy, 'in_sample_domain_BCE': final_loss,
            'optimizer_closure_calls': len(losses), 'optimizer_loss_history': losses,
            'profiles': profiles, 'weights_sha256': digest(run / 'density_weights.npz'),
            'elapsed_seconds': time.monotonic() - begun, 'test_accessed': False}
        write_json(run / 'result.json', result); status('completed')
        print(json.dumps({k: result[k] for k in ['run_id', 'eligible_windows', 'recent_population_share', 'in_sample_domain_accuracy', 'profiles', 'elapsed_seconds']}))
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
