"""Train-only scoring, grouped analytic meta-weighting, and coverage coresets."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, digest, write_json
from reducible_sampling import load_data
from completion_policies import (MEMBER_SEEDS, oof_weights, protected_windows, input_signatures,
    quad_groups, coreset_probability, shuffled_weights, quality_profile, verify_arithmetic)

FOLDER = ROOT / 'research_runs/completion_study'
BATCH = 8192
RIDGE_FRACTION = .001
SKETCH_DIMENSIONS = 16


def moments(raw_gpu, starts, mean, scale, labels=None, count_groups=1):
    mean = torch.as_tensor(mean, device='cuda', dtype=torch.float64)
    scale = torch.as_tensor(scale, device='cuda', dtype=torch.float64)
    gram = torch.zeros((count_groups, 15, 33, 33), device='cuda', dtype=torch.float64)
    cross = torch.zeros((count_groups, 15, 33, 8), device='cuda', dtype=torch.float64)
    target_sq = torch.zeros((count_groups, 15, 8), device='cuda', dtype=torch.float64)
    counts = np.zeros(count_groups, dtype=np.int64)
    offsets = torch.arange(40, device='cuda')
    for lo in range(0, len(starts), BATCH):
        ids = torch.from_numpy(starts[lo:lo + BATCH]).cuda()
        batch = (raw_gpu[ids[:, None] + offsets] - mean) / scale
        x = batch[:, :32].permute(2, 0, 1)
        z = torch.cat([x, torch.ones((15, len(batch), 1), device='cuda', dtype=torch.float64)], dim=2)
        y = batch[:, 32:].permute(2, 0, 1)
        local = np.zeros(len(batch), dtype=np.int64) if labels is None else labels[lo:lo + len(batch)]
        for group in np.unique(local):
            selected = torch.from_numpy(local == group).cuda()
            zz, yy = z[:, selected], y[:, selected]
            gram[group] += zz.transpose(1, 2) @ zz
            cross[group] += zz.transpose(1, 2) @ yy
            target_sq[group] += yy.square().sum(1)
            counts[group] += int((local == group).sum())
    assert counts.sum() == len(starts)
    return gram, cross, target_sq, counts


def fit_affine(stats):
    gram, cross, _, counts = stats
    n = int(counts.sum()); g = gram.sum(0) / n; b = cross.sum(0) / n
    ridge = RIDGE_FRACTION * g.diagonal(dim1=1, dim2=2).mean(1)
    regularizer = torch.eye(33, device='cuda', dtype=torch.float64); regularizer[-1, -1] = 0
    coefficient = torch.linalg.solve(g + ridge[:, None, None] * regularizer, b)
    assert torch.isfinite(coefficient).all()
    return coefficient, ridge


def mse_from_stats(coefficient, stats, scale_ratio):
    gram, cross, target_sq, counts = stats
    n = int(counts.sum()); g = gram.sum(0) / n; b = cross.sum(0) / n; yy = target_sq.sum(0) / n
    error = (coefficient * (g @ coefficient)).sum(1) - 2 * (coefficient * b).sum(1) + yy
    return (error * scale_ratio[:, None].square()).mean()


def fit_oof(raw, raw_gpu, starts, cells, full_scale, run, status):
    n = len(starts); errors = np.zeros((n, 8, 15), dtype=np.float32); scored = np.zeros(n, dtype=bool)
    fold_id = np.full(n, -1, dtype=np.int8); folds = []
    offsets = torch.arange(40, device='cuda')
    for fold, fraction in enumerate((.25, .5, .75)):
        status('oof_teacher_' + str(fold))
        fit_end = int(len(raw) * fraction); score_end = int(len(raw) * (fraction + .25))
        fitted = starts[starts + 40 <= fit_end]
        members = np.flatnonzero((starts >= fit_end) & (starts + 40 <= score_end))
        assert fitted.max() + 40 <= starts[members].min()
        mean = raw[:fit_end].mean(0); std = raw[:fit_end].std(0); scale = np.where(std == 0, 1, std)
        stats = moments(raw_gpu, fitted, mean, scale); coefficient, ridge = fit_affine(stats)
        mu = torch.from_numpy(mean).cuda(); sigma = torch.from_numpy(scale).cuda()
        ratio = torch.from_numpy(scale / full_scale).cuda()
        for lo in range(0, len(members), BATCH):
            selected = members[lo:lo + BATCH]; ids = torch.from_numpy(starts[selected]).cuda()
            batch = (raw_gpu[ids[:, None] + offsets] - mu) / sigma
            x = batch[:, :32].permute(2, 0, 1)
            z = torch.cat([x, torch.ones((15, len(batch), 1), device='cuda', dtype=torch.float64)], dim=2)
            prediction = (z @ coefficient).permute(1, 2, 0)
            error = ((prediction - batch[:, 32:]) * ratio).square()
            errors[selected] = error.cpu().numpy().astype(np.float32)
        scored[members] = True; fold_id[members] = fold
        np.savez_compressed(run / f'oof_teacher_{fold}.npz', mean=mean, scale=scale,
            coefficient=coefficient.cpu().numpy(), ridge=ridge.cpu().numpy(), gram=stats[0].cpu().numpy(),
            cross=stats[1].cpu().numpy(), fit_start=fitted, score_start=starts[members])
        folds.append({'fold': fold, 'fit_raw_rows': [0, fit_end], 'score_raw_rows': [fit_end, score_end],
            'fit_windows': len(fitted), 'scored_windows': len(members), 'last_fit_end_exclusive': int(fitted.max() + 40),
            'first_score_start': int(starts[members].min()), 'mean_scored_NMSE': float(errors[members].mean(dtype=np.float64))})
    assert np.isfinite(errors).all()
    status('ranking_oof_losses')
    window, target, permutation = oof_weights(errors, scored, cells)
    np.savez_compressed(run / 'oof_scores.npz', train_start=starts, squared_error=errors, scored=scored,
        fold_id=fold_id, control_permutation=permutation)
    write_json(run / 'oof_audit.json', {'folds': folds, 'scored_windows': int(scored.sum()),
        'unscored_default_windows': int((~scored).sum()), 'window_weight_mean': float(window.mean(dtype=np.float64)),
        'target_mean_max_deviation': float(np.abs(target.mean(0, dtype=np.float64) - 1).max()),
        'future_validation_or_test_used': False, 'teacher_solver': 'past-only standardized affine ridge; not Adam main model'})
    return {'oof_window': window, 'oof_window_shuffle': window[permutation],
        'oof_target': target, 'oof_target_shuffle': target[permutation]}


def gradient_signatures(values, starts, teacher, run):
    p = json.loads((ROOT / 'research_runs/selection/protocol.json').read_text())
    source = ROOT / 'research_runs/results' / teacher / 'final.pt'
    assert source.exists(), 'The specified uniform final checkpoint must exist'
    model = DLinear(p).cuda(); model.load_state_dict(torch.load(source, map_location='cuda', weights_only=True)); model.eval()
    rng = np.random.default_rng(20261021)
    projectors = {}
    for kind in ('seasonal', 'trend'):
        projectors[kind] = torch.from_numpy(rng.choice([-1., 1.], size=(15, 32, 8, SKETCH_DIMENSIONS)) / np.sqrt(SKETCH_DIMENSIONS)).cuda()
        projectors[kind + '_bias'] = torch.from_numpy(rng.choice([-1., 1.], size=(15, 8, SKETCH_DIMENSIONS)) / np.sqrt(SKETCH_DIMENSIONS)).cuda()
    data = torch.from_numpy(values).cuda(); offsets = torch.arange(40, device='cuda')

    @torch.no_grad()
    def sketch(batch):
        x, y = batch[:, :32], batch[:, 32:]
        channels = x.transpose(1, 2)
        trend = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(channels, (12, 12), mode='replicate'), 25, stride=1)
        seasonal = channels - trend
        projected = torch.einsum('bcl,clhk->bchk', seasonal.double(), projectors['seasonal'])
        projected += torch.einsum('bcl,clhk->bchk', trend.double(), projectors['trend'])
        projected += projectors['seasonal_bias'][None] + projectors['trend_bias'][None]
        error = (model(x) - y).double().permute(0, 2, 1)
        return (projected * error[..., None]).sum((1, 2)) * (2 / 120)

    probe_indices = np.linspace(0, len(starts) - 1, 5, dtype=int)
    probe = data[torch.from_numpy(starts[probe_indices]).cuda()[:, None] + offsets]
    measured = sketch(probe); actual = []
    for batch in probe:
        loss = (model(batch[None, :32]) - batch[None, 32:]).square().mean()
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        projected = torch.zeros(SKETCH_DIMENSIONS, device='cuda', dtype=torch.float64)
        for (name, _), gradient in zip(model.named_parameters(), gradients):
            kind, channel, parameter = name.split('.')
            matrix = projectors[kind][int(channel)].permute(1, 0, 2) if parameter == 'weight' else projectors[kind + '_bias'][int(channel)]
            projected += (gradient.double()[..., None] * matrix).reshape(-1, SKETCH_DIMENSIONS).sum(0)
        actual.append(projected)
    actual = torch.stack(actual)
    assert torch.allclose(actual, measured, atol=1e-6, rtol=1e-5)
    signatures = np.empty((len(starts), SKETCH_DIMENSIONS), dtype=np.float32)
    for lo in range(0, len(starts), 2048):
        positions = torch.from_numpy(starts[lo:lo + 2048]).cuda()
        signatures[lo:lo + len(positions)] = sketch(data[positions[:, None] + offsets]).cpu().numpy().astype(np.float32)
    np.savez_compressed(run / 'gradient_projection.npz', signature=signatures,
        **{key: value.cpu().numpy() for key, value in projectors.items()})
    write_json(run / 'gradient_projection_audit.json', {'teacher': teacher, 'checkpoint_sha256': digest(source),
        'probe_source_starts': starts[probe_indices].tolist(), 'max_absolute_error': float((actual - measured).abs().max()),
        'atol': 1e-6, 'rtol': 1e-5, 'dimensions': SKETCH_DIMENSIONS})
    return signatures


def fit_bilevel(raw, raw_gpu, values, starts, cells, months, full_scale, run):
    fit_end = int(len(raw) * .6); meta_end = int(len(raw) * .8)
    fitted = starts + 40 <= fit_end
    meta = (starts >= fit_end) & (starts + 40 <= meta_end)
    held = starts >= meta_end
    assert starts[fitted].max() + 40 <= starts[meta].min()
    assert starts[meta].max() + 40 <= starts[held].min()
    changes = np.abs(np.diff(values.astype(np.float64), axis=0)).mean(1)
    prefix = np.r_[0, np.cumsum(changes)]
    variation = (prefix[starts + 31] - prefix[starts]) / 31
    thresholds = np.quantile(variation[fitted], [1 / 3, 2 / 3])
    clusters = cells // len(np.unique(months))
    labels = clusters * 3 + np.searchsorted(thresholds, variation, side='right')
    count_groups = 12
    mean = raw[:fit_end].mean(0); std = raw[:fit_end].std(0); scale = np.where(std == 0, 1, std)
    stats = moments(raw_gpu, starts[fitted], mean, scale, labels[fitted], count_groups)
    meta_stats = moments(raw_gpu, starts[meta], mean, scale)
    held_stats = moments(raw_gpu, starts[held], mean, scale)
    baseline, ridge = fit_affine(stats)
    g, b, _, counts = stats
    count_gpu = torch.from_numpy(counts.astype(np.float64)).cuda()
    proportions = torch.from_numpy(np.bincount(labels, minlength=count_groups).astype(np.float64) / len(labels)).cuda()
    ratio = torch.from_numpy(scale / full_scale).cuda()
    regularizer = torch.eye(33, device='cuda', dtype=torch.float64); regularizer[-1, -1] = 0

    def objective(theta):
        value = torch.tanh(theta)
        weights = 1 + .25 * (value - (proportions * value).sum())
        count = (count_gpu * weights).sum()
        gram = (g * weights[:, None, None, None]).sum(0) / count
        cross = (b * weights[:, None, None, None]).sum(0) / count
        coefficient = torch.linalg.solve(gram + ridge[:, None, None] * regularizer, cross)
        meta_mse = mse_from_stats(coefficient, meta_stats, ratio)
        loss = meta_mse + .01 * weights.log().square().mean()
        return loss, meta_mse, coefficient, weights

    theta = torch.zeros(count_groups, device='cuda', dtype=torch.float64, requires_grad=True)
    initial = objective(theta)[0]
    analytic = torch.autograd.grad(initial, theta)[0]
    finite_differences = []
    for index in (0, 5, 11):
        delta = torch.zeros_like(theta); delta[index] = 1e-4
        numerical = (objective(theta.detach() + delta)[0] - objective(theta.detach() - delta)[0]) / 2e-4
        difference = float((numerical - analytic[index]).abs())
        assert difference < 1e-6
        finite_differences.append({'parameter': index, 'analytic': float(analytic[index]), 'finite_difference': float(numerical), 'absolute_difference': difference})
    optimizer = torch.optim.Adam([theta], lr=.05)
    history = []
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss, meta_mse, _, weights = objective(theta)
        assert torch.isfinite(loss); loss.backward(); optimizer.step()
        if step == 1 or step % 20 == 0:
            history.append({'step': step, 'objective_before_update': float(loss.detach()), 'meta_NMSE_before_update': float(meta_mse.detach())})
    with torch.no_grad():
        loss, meta_mse, coefficient, group_weights = objective(theta)
        held_mse = mse_from_stats(coefficient, held_stats, ratio)
        baseline_meta = mse_from_stats(baseline, meta_stats, ratio)
        baseline_held = mse_from_stats(baseline, held_stats, ratio)
    weight = group_weights.detach().cpu().numpy()[labels].astype(np.float32)
    assert weight.min() >= .5 and weight.max() <= 1.5 and abs(weight.mean(dtype=np.float64) - 1) < 1e-7
    control, permutation = shuffled_weights(weight, cells)
    np.savez_compressed(run / 'bilevel_fit.npz', gram=g.cpu().numpy(), cross=b.cpu().numpy(), counts=counts,
        mean=mean, scale=scale, thresholds=thresholds, labels=labels, theta=theta.detach().cpu().numpy(),
        coefficient=coefficient.cpu().numpy(), group_weight=group_weights.cpu().numpy(), permutation=permutation,
        meta_gram=meta_stats[0].cpu().numpy(), meta_cross=meta_stats[1].cpu().numpy(), meta_target_sq=meta_stats[2].cpu().numpy(),
        held_gram=held_stats[0].cpu().numpy(), held_cross=held_stats[1].cpu().numpy(), held_target_sq=held_stats[2].cpu().numpy())
    write_json(run / 'bilevel_audit.json', {'fit_raw_rows': [0, fit_end], 'meta_raw_rows': [fit_end, meta_end],
        'held_diagnostic_raw_rows': [meta_end, len(raw)], 'fit_windows': int(fitted.sum()), 'meta_windows': int(meta.sum()),
        'held_windows': int(held.sum()), 'gradient_checks': finite_differences, 'history': history,
        'baseline_meta_NMSE': float(baseline_meta), 'weighted_meta_NMSE': float(meta_mse),
        'baseline_held_NMSE': float(baseline_held), 'weighted_held_NMSE': float(held_mse),
        'group_weights': group_weights.cpu().tolist(), 'final_objective': float(loss), 'outer_evaluation_used': False})
    return {'bilevel': weight, 'bilevel_shuffle': control}


def previous_overlap(starts, probability):
    rows = []
    for name in ('T007_prefix60_disjoint', 'T017_prefix60_cluster1', 'T018_prefix60_cluster2', 'T019_prefix60_cluster3', 'T052_prefix60_cadence_drop'):
        path = ROOT / 'research_runs/results' / name / 'windows.npz'
        with np.load(path) as old:
            old_starts = old['train_start']; old_removed = old['probability'] == 0
        common, left, right = np.intersect1d(starts, old_starts, return_indices=True)
        a, b = probability[left] == 0, old_removed[right]
        union = int((a | b).sum()); intersection = int((a & b).sum())
        rows.append({'run': name, 'common_windows': len(common), 'intersection': intersection, 'union': union,
            'deletion_jaccard_on_common_rows': intersection / union if union else 0})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .7, .8, 1.], required=True)
    parser.add_argument('--teacher', required=True); args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and Path(args.teacher).name == args.teacher
    run = ROOT / 'research_runs/results' / args.run_id; assert not run.exists(); run.mkdir()
    begun = time.monotonic()
    def status(phase):
        write_json(run / 'status.json', {'phase': phase, 'elapsed_seconds': time.monotonic() - begun})
    status('preparing')
    try:
        assert verify_arithmetic()
        snapshots = {'code_snapshot.py': Path(__file__), 'policy_helper.py': ROOT / 'scripts/completion_policies.py',
            'data_helper.py': ROOT / 'scripts/reducible_sampling.py', 'core_snapshot.py': ROOT / 'scripts/run_research.py',
            'study_protocol.md': FOLDER / 'PROTOCOL.md', 'policies.md': FOLDER / 'POLICIES.md'}
        identity = {'run_type': 'train_only_policy_fitting_diagnostic', **vars(args),
            'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
            'sources_sha256': {name: digest(path) for name, path in snapshots.items()}}
        write_json(run / 'identity.json', identity)
        for name, path in snapshots.items(): (run / name).write_bytes(path.read_bytes())
        manifest, values, starts, _, cells, months, _, future, _, audit = load_data(args.origin, reference_only=True)
        assert future is None
        write_json(run / 'data_audit.json', audit)
        raw = np.loadtxt(ROOT / 'data/metropt3_v1/train.csv', delimiter=',', skiprows=1, max_rows=len(values),
            usecols=range(2, 17), dtype=np.float64)
        write_json(run / 'measurement_quality.json', quality_profile(raw, manifest['sensor_columns']))
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        assert torch.cuda.is_available()
        raw_gpu = torch.from_numpy(raw).cuda(); n = len(starts); uniform = np.full(n, 1 / n)
        policy_rows = []
        def save_policy(name, probability, weight=None):
            weight = np.ones(n, dtype=np.float32) if weight is None else weight
            per_window = weight if weight.ndim == 1 else weight.mean((1, 2), dtype=np.float64)
            expectation = float(probability @ per_window)
            assert np.isfinite(probability).all() and abs(probability.sum() - 1) < 1e-12 and expectation > 0
            assert np.isfinite(weight).all() and weight.min() > 0
            path = run / ('policy_' + name + '.npz')
            np.savez_compressed(path, train_start=starts, cell=cells, probability=probability,
                loss_weight=weight, expected_loss_weight=expectation)
            row = {'policy': name, 'path': path.name, 'sha256': digest(path), 'retained_windows': int((probability > 0).sum()),
                'weight_shape': list(weight.shape), 'weight_min': float(weight.min()), 'weight_max': float(weight.max()),
                'expected_loss_weight': expectation}
            policy_rows.append(row)
            print(json.dumps({'policy_saved': name, 'retained': row['retained_windows']}), flush=True)
        oof = fit_oof(raw, raw_gpu, starts, cells, np.array(audit['scale']), run, status)
        for name, weight in oof.items(): save_policy(name, uniform, weight)
        del oof
        status('bilevel_fitting')
        for name, weight in fit_bilevel(raw, raw_gpu, values, starts, cells, months, np.array(audit['scale']), run).items():
            save_policy(name, uniform, weight)
        status('coverage_and_gradient_representatives')
        protected = protected_windows(raw, starts)
        features, feature_mean, feature_scale = input_signatures(values, starts)
        gradients = gradient_signatures(values, starts, args.teacher, run)
        np.savez_compressed(run / 'coverage_features.npz', train_start=starts, protected=protected,
            input_features=features, feature_mean=feature_mean, feature_scale=feature_scale)
        coverage_rows = []
        for seed in MEMBER_SEEDS:
            quads = quad_groups(cells, protected, seed)
            np.savez_compressed(run / f'quads_{seed}.npz', candidate_indices=quads)
            for family, signature in (('coverage_core', features), ('supervised_core', gradients), ('coreset_random', features)):
                q, dropped = coreset_probability(signature, quads, n, seed if family == 'coreset_random' else None)
                assert not np.any(protected[dropped])
                for cell in np.unique(cells):
                    assert abs(q[cells == cell].sum() - float((cells == cell).mean())) < 1e-12
                name = family + '_' + str(seed); save_policy(name, q)
                coverage_rows.append({'policy': name, 'seed': seed, 'protected_windows': int(protected.sum()),
                    'dropped_windows': len(dropped), 'deletion_overlap': previous_overlap(starts, q),
                    'cells': [{'cell': int(c), 'windows': int((cells == c).sum()),
                        'protected': int(((cells == c) & protected).sum()), 'retained': int(((cells == c) & (q > 0)).sum())} for c in np.unique(cells)]})
        write_json(run / 'coverage_audit.json', coverage_rows)
        write_json(run / 'result.json', {'run_id': args.run_id, 'identity': identity, 'policies': policy_rows,
            'analytic_teacher_fits': 3, 'bilevel_weight_fit': 1, 'measurement_quality_diagnostic_completed': True,
            'confirmed_measurement_faults': [], 'test_accessed': False, 'elapsed_seconds': time.monotonic() - begun,
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'device': torch.cuda.get_device_name()}})
        status('completed')
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
