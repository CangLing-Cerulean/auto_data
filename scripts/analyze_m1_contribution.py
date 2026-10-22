"""Read-only post-hoc attribution of the completed prefix study; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mean(values):
    return float(np.mean(values))


def output_prediction(value, values, starts):
    value = value.cuda()
    outputs, targets = [], []
    for lo in range(0, len(starts), 2048):
        batch = values[starts[lo:lo + 2048, None] + torch.arange(40, device='cuda')]
        x = batch[:, :32].transpose(1, 2)
        trend = F.avg_pool1d(F.pad(x, (12, 12), mode='replicate'), 25, stride=1)
        seasonal = x - trend
        prediction = torch.stack([
            F.linear(seasonal[:, c], value[c, :32].T.contiguous(), value[c, 32])
            + F.linear(trend[:, c], value[c, 33:65].T.contiguous(), value[c, 65])
            for c in range(8)], dim=-1)
        outputs.append(prediction.cpu().numpy())
        targets.append(batch[:, 32:].cpu().numpy())
    return np.concatenate(outputs), np.concatenate(targets)


def loss(yhat, y):
    # Match the original float32 residual, float64 squared-error accumulation.
    return mean(np.square((yhat - y).astype(np.float64)))


def select(curves):
    return min(sorted(curves, key=int), key=lambda t: curves[t]['B'])


@torch.no_grad()
def main(root, output):
    started = time.perf_counter()
    assert not output.exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec, pred = read(root / 'spec.json'), read(root / 'predictions.json')
    parents, labels = read(root / 'parents.json')['records'], read(root / 'labels.json')['records']
    audit = read(root / 'audit_revision4.json')
    assert audit['status'] == 'PASS' and len(labels) == 36 and len(parents) == 9
    assert sha(root / 'predictions.json') == read(root / 'prediction_lock.json')['sha256']
    inputs = {}
    for filename in ('spec.json', 'parents.json', 'predictions.json', 'labels.json',
                     'audit_revision4.json', 'prediction_lock.json', 'source_lock.json'):
        inputs[filename] = sha(root / filename)

    def checked(path, expected):
        relative = path.relative_to(root).as_posix()
        if relative not in inputs:
            inputs[relative] = sha(path)
        assert inputs[relative] == expected, relative
        return path

    def coords(record, step):
        path = checked(root / record['directory'] / f'step_{step}.pt', record['assets_sha256'][f'step_{step}.pt'])
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step
        return saved['original_coordinates']

    def forecast(record, step):
        path = checked(root / record['directory'] / f'step_{step}.npz', record['assets_sha256'][f'step_{step}.npz'])
        with np.load(path, allow_pickle=False) as saved:
            return torch.from_numpy(saved['coordinates'].copy())

    points, candidates = [], []
    metric_checks = formula_checks = 0
    for task in spec['tasks']:
        path = checked(root / 'inputs' / f'{task}_ABC.npz', spec['input_sha256'][task])
        with np.load(path, allow_pickle=False) as source:
            arrays = {s: (torch.from_numpy(source[f'{s}_values']).cuda(),
                          torch.from_numpy(source[f'{s}_starts']).cuda()) for s in ('B', 'C')}
        for seed in spec['seeds']:
            parent = next(p for p in parents if (p['task'], p['seed']) == (task, seed))
            records = [r for r in pred['records'] if (r['task'], r['seed']) == (task, seed)]
            assert {r['action'] for r in records} == set(spec['actions'])
            base = parent['selected_losses']['C']
            parent_outputs = {}
            for step in spec['grid']:
                h0, l0 = coords(parent, step), coords(records[0]['parent_mean'], step)
                h0_out, y = output_prediction(h0, *arrays['C'])
                np.testing.assert_allclose(loss(h0_out, y), parent['curves'][str(step)]['C'], rtol=2e-13, atol=1e-15)
                metric_checks += 1
                l0_out, _ = output_prediction(l0, *arrays['C'])
                parent_outputs[step] = h0, l0, h0_out, l0_out
            for r in records:
                action = r['action']
                full = next(p for p in labels if (p['task'], p['seed'], p['action']) == (task, seed, action))
                raw_curves, cell_points = {}, []
                for step in spec['grid']:
                    key = str(step)
                    ha, la = coords(full, step), coords(r['mean'], step)
                    m1 = forecast(r['methods']['M1'], step)
                    h0, l0, h0_out, l0_out = parent_outputs[step]
                    assert torch.equal(m1, h0 + (la - l0))
                    formula_checks += 1
                    raw_curves[key] = {}
                    for split in ('B', 'C'):
                        raw_out, target = output_prediction(la, *arrays[split])
                        raw_curves[key][split] = loss(raw_out, target)
                        true_out, _ = output_prediction(ha, *arrays[split])
                        corrected_out, _ = output_prediction(m1, *arrays[split])
                        for actual, expected in ((loss(true_out, target), full['curves'][key][split]),
                                                 (loss(corrected_out, target), r['methods']['M1']['curves'][key][split])):
                            np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-15)
                            metric_checks += 1
                        if split != 'C':
                            continue
                        ea = raw_out.astype(np.float64) - true_out
                        e0 = l0_out.astype(np.float64) - h0_out
                        d = corrected_out.astype(np.float64) - true_out
                        truth_residual = true_out.astype(np.float64) - target
                        raw_distance, corrected_distance = mean(ea * ea), mean(d * d)
                        parent_distance, dot = mean(e0 * e0), mean(ea * e0)
                        theoretical = mean((ea - e0) ** 2)
                        np.testing.assert_allclose(theoretical, raw_distance + parent_distance - 2 * dot, rtol=1e-10, atol=1e-10)
                        crossing, square = 2 * mean(truth_residual * d), corrected_distance
                        double_loss_delta = mean((corrected_out.astype(np.float64) - target) ** 2) - mean(truth_residual ** 2)
                        np.testing.assert_allclose(crossing + square, double_loss_delta, rtol=1e-10, atol=1e-10)
                        p = dict(task=task, seed=seed, action=action, step=step,
                                 raw_prediction_distance=raw_distance, corrected_prediction_distance=corrected_distance,
                                 parent_prediction_distance=parent_distance, child_parent_error_dot=dot,
                                 error_cosine=dot / np.sqrt(raw_distance * parent_distance) if raw_distance * parent_distance > 0 else None,
                                 corrected_to_raw_distance_ratio=corrected_distance / raw_distance if raw_distance else None,
                                 cancellation_distance_identity_residual=theoretical - (raw_distance + parent_distance - 2 * dot),
                                 float32_additive_prediction_rounding_max=float(np.max(np.abs(d - (ea - e0)))),
                                 loss_cross_term_pp=100 * crossing / base, loss_square_term_pp=100 * square / base,
                                 float32_loss_rounding_pp=100 * ((loss(corrected_out, target) - loss(true_out, target)) - double_loss_delta) / base,
                                 raw_fixed_step_response_error_pp=100 * (full['curves'][key]['C'] - raw_curves[key]['C']) / base,
                                 m1_fixed_step_response_error_pp=100 * (full['curves'][key]['C'] - r['methods']['M1']['curves'][key]['C']) / base)
                        points.append(p)
                        cell_points.append(p)
                ts, rs = str(full['selected_step']), select(raw_curves)
                true_gain = 100 * (base - full['curves'][ts]['C']) / base
                row = dict(task=task, seed=seed, action=action, true_gain_percent=true_gain, true_selected_step=int(ts),
                           raw_curves=raw_curves, methods={})
                for method in ('RAW', 'M1', 'PREFIX'):
                    curves = raw_curves if method == 'RAW' else r['methods'][method]['curves']
                    ms = select(curves)
                    if method != 'RAW':
                        assert int(ms) == r['methods'][method]['selected_step']
                    gain = 100 * (base - curves[ms]['C']) / base
                    approximation = 100 * (full['curves'][ms]['C'] - curves[ms]['C']) / base
                    selection = 100 * (full['curves'][ts]['C'] - full['curves'][ms]['C']) / base
                    np.testing.assert_allclose(approximation + selection, gain - true_gain, rtol=1e-10, atol=1e-10)
                    row['methods'][method] = dict(selected_step=int(ms), gain_percent=gain,
                        response_error_pp=gain - true_gain, approximation_component_pp=approximation,
                        checkpoint_component_pp=selection)
                candidates.append(row)
            print(json.dumps(dict(completed_task=task, seed=seed, candidate_count=len(candidates))), flush=True)
    assert len(points) == 216 and len(candidates) == 36
    summary = {}
    for task in ('ALL', *spec['tasks']):
        rows = [r for r in candidates if task == 'ALL' or r['task'] == task]
        ps = [r for r in points if task == 'ALL' or r['task'] == task]
        result = dict(candidate_count=len(rows), point_count=len(ps), methods={})
        for method in ('RAW', 'M1', 'PREFIX'):
            values = [r['methods'][method] for r in rows]
            result['methods'][method] = dict(mae_pp=mean([abs(v['response_error_pp']) for v in values]),
                signed_error_pp=mean([v['response_error_pp'] for v in values]),
                approximation_mean_absolute_pp=mean([abs(v['approximation_component_pp']) for v in values]),
                checkpoint_mean_absolute_pp=mean([abs(v['checkpoint_component_pp']) for v in values]),
                checkpoint_mismatch_count=sum(v['selected_step'] != r['true_selected_step'] for v, r in zip(values, rows)))
        result['cancellation'] = dict(points_with_distance_reduction=sum(p['corrected_prediction_distance'] < p['raw_prediction_distance'] for p in ps),
            ratio_of_total_prediction_distances=sum(p['corrected_prediction_distance'] for p in ps) / sum(p['raw_prediction_distance'] for p in ps),
            median_error_cosine=float(np.median([p['error_cosine'] for p in ps if p['error_cosine'] is not None])))
        result['fixed_step_mae_pp'] = {str(step): {m: mean([abs(p[f'{m.lower()}_fixed_step_response_error_pp']) for p in ps if p['step'] == step]) for m in ('RAW', 'M1')} for step in spec['grid']}
        summary[task] = result
    recommendations = {}
    for task in spec['tasks']:
        recommendations[task] = {}
        rows = [r for r in candidates if r['task'] == task]
        for method in ('RAW', 'M1', 'PREFIX'):
            gains = {a: mean([r['methods'][method]['gain_percent'] for r in rows if r['action'] == a]) for a in spec['actions']}
            action = max(spec['actions'], key=lambda a: gains[a])
            if gains[action] < 1.:
                action = 'PARENT'
            achieved = 0. if action == 'PARENT' else mean([r['true_gain_percent'] for r in rows if r['action'] == action])
            recommendations[task][method] = dict(action=action, predicted_means=gains, achieved_gain_percent=achieved)
            if method != 'RAW':
                assert action == audit['recommendations'][task]['methods'][method]['action']
    for m in ('M1', 'PREFIX'):
        np.testing.assert_allclose(summary['ALL']['methods'][m]['mae_pp'], audit['mae_pp'][m], rtol=1e-12, atol=1e-12)
    result = dict(status='PASS', scope='posthoc_known_source_attribution_not_new_blind_evaluation',
        new_training_steps=0, new_full_trainings=0, historical_full_completed=593,
        source_stage=root.name, input_sha256=inputs, metric_checks=metric_checks,
        formula_checks=formula_checks, summary=summary, recommendations=recommendations,
        mean_selected_gain_percent={m: mean([recommendations[t][m]['achieved_gain_percent'] for t in spec['tasks']]) for m in ('RAW', 'M1', 'PREFIX')},
        points=points, candidates=candidates, analysis_seconds=time.perf_counter() - started)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='PASS', checks=metric_checks, summary=summary, recommendations=recommendations)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.output)
