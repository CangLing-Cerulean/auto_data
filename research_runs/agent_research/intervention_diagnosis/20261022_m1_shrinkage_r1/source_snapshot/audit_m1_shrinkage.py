"""Independently reconstruct prefix fits, matrices and scores before label evaluation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from m1_shrinkage_inference import read, select, sha

METHODS = ('RAW_PREFIX', 'M1_PREFIX', 'SCALAR_SHRINK', 'CHANNEL_SHRINK')


@torch.no_grad()
def prefix_outputs(value, arrays):
    layers = []
    for c in range(8):
        pair = []
        for start, bias in ((0, 32), (33, 65)):
            layer = torch.nn.Linear(32, 8)
            layer.weight.copy_(value[c, start:start + 32].T)
            layer.bias.copy_(value[c, bias])
            pair.append(layer)
        layers.append(pair)
    values, starts = arrays
    outputs = []
    for offset in range(0, len(starts), 2048):
        batch = values[starts[offset:offset + 2048, None] + torch.arange(40)]
        x = batch[:, :32].transpose(1, 2)
        trend = torch.nn.AvgPool1d(25, stride=1)(torch.nn.ReplicationPad1d(12)(x))
        outputs.append(torch.stack([layers[c][0]((x - trend)[:, c]) + layers[c][1](trend[:, c])
                                    for c in range(8)], -1).cpu().numpy())
    return np.concatenate(outputs)


def coordinates(saved):
    model = saved['model']
    value = torch.stack([torch.cat((model[f'seasonal.{c}.weight'].T, model[f'seasonal.{c}.bias'][None],
                                   model[f'trend.{c}.weight'].T, model[f'trend.{c}.bias'][None])) for c in range(8)])
    assert torch.equal(value, saved['original_coordinates'])
    return value


def errors(value, arrays):
    result = {}
    for split, (values, starts) in arrays.items():
        prediction = prefix_outputs(value, (values, starts))
        target = values[starts[:, None] + torch.arange(32, 40)].numpy()
        residual = (prediction - target).astype(np.float64)
        result[split] = float(np.square(residual).mean())
    return result


@torch.no_grad()
def main(source, stage):
    started = time.perf_counter()
    assert not (stage / 'audit.json').exists()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pred = read(stage / 'predictions.json')
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256']
    assert sha(stage / 'source_lock.json') == pred['source_lock_sha256']
    for name, digest in read(stage / 'source_lock.json')['sha256'].items():
        assert sha(stage / name) == digest, name
    for name, digest in pred['source_input_sha256'].items():
        assert sha(source / name) == digest, name
        assert name != 'labels.json' and '_child_' not in name, name
    spec, old = read(source / 'spec.json'), read(source / 'predictions.json')['records']
    parents = read(source / 'parents.json')['records']
    expected = {(t, s, a) for t in spec['tasks'] for s in spec['seeds'] for a in spec['actions']}
    assert len(pred['records']) == 36 and {(r['task'], r['seed'], r['action']) for r in pred['records']} == expected
    counts = dict(coefficients=0, matrices=0, scores=0, original_raw_scores=0)
    device_differences = []

    def load(record, step):
        name = f'step_{step}.pt'
        path = source / record['directory'] / name
        assert sha(path) == record['assets_sha256'][name]
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['step'] == step
        return coordinates(saved)

    raw_records = []
    for task in spec['tasks']:
        with np.load(source / 'inputs' / f'{task}_ABC.npz', allow_pickle=False) as z:
            arrays = {s: (torch.from_numpy(z[f'{s}_values']),
                          torch.from_numpy(z[f'{s}_starts'])) for s in ('B', 'C')}
        for row in (r for r in pred['records'] if r['task'] == task):
            key = tuple(row[x] for x in ('task', 'seed', 'action'))
            r = next(r for r in old if tuple(r[x] for x in ('task', 'seed', 'action')) == key)
            parent = next(p for p in parents if (p['task'], p['seed']) == key[:2])
            h0, l0, ha, la = (load(q, 500) for q in (parent, r['parent_mean'], r['prefix'], r['mean']))
            u = prefix_outputs(h0, arrays['B']).astype(np.float64) - prefix_outputs(l0, arrays['B'])
            v = prefix_outputs(ha, arrays['B']).astype(np.float64) - prefix_outputs(la, arrays['B'])
            for method in ('SCALAR_SHRINK', 'CHANNEL_SHRINK'):
                groups = [(u.ravel(), v.ravel())] if method == 'SCALAR_SHRINK' else [(u[..., c].ravel(), v[..., c].ravel()) for c in range(8)]
                fitted = []
                for x, y in groups:
                    denominator, numerator = float(x @ x), float(x @ y)
                    fitted.append(min(1., max(0., numerator / denominator)) if denominator else 0.)
                if len(fitted) == 1:
                    fitted *= 8
                np.testing.assert_allclose(row['coefficients'][method]['alpha'], fitted, rtol=5e-13, atol=5e-15)
                counts['coefficients'] += 1
            raw_curves = {}
            for step in spec['grid']:
                raw = load(r['mean'], step).numpy()
                filename = f'step_{step}.npz'
                original = r['methods']['M1']
                path = source / original['directory'] / filename
                assert sha(path) == original['assets_sha256'][filename]
                with np.load(path, allow_pickle=False) as z:
                    m1 = z['coordinates']
                raw_curves[str(step)] = errors(torch.from_numpy(raw), arrays)
                counts['original_raw_scores'] += 2
                for method in METHODS:
                    result = row['methods'][method]
                    path = stage / result['directory'] / filename
                    assert sha(path) == result['assets_sha256'][filename]
                    with np.load(path, allow_pickle=False) as z:
                        actual = z['coordinates']
                    if step == 500:
                        expected_value = ha.numpy()
                    elif method == 'RAW_PREFIX':
                        expected_value = raw
                    elif method == 'M1_PREFIX':
                        expected_value = m1
                    else:
                        expected_value = np.empty_like(raw)
                        for c, a in enumerate(row['coefficients'][method]['alpha']):
                            expected_value[c] = raw[c] if a == 0 else m1[c] if a == 1 else (
                                raw[c].astype(np.float64) + a * (m1[c].astype(np.float64) - raw[c])).astype(raw.dtype)
                    np.testing.assert_array_equal(actual, expected_value)
                    counts['matrices'] += 1
                    scores = errors(torch.from_numpy(actual), arrays)
                    for split in ('B', 'C'):
                        np.testing.assert_allclose(scores[split], result['curves'][str(step)][split], rtol=2e-13, atol=1e-15)
                        counts['scores'] += 1
                if step == 500:
                    for split in ('B', 'C'):
                        local = row['methods']['RAW_PREFIX']['curves']['500'][split]
                        original_score = r['prefix']['curves']['500'][split]
                        np.testing.assert_allclose(local, original_score, rtol=2e-5, atol=2e-6)
                        device_differences.append(dict(task=task, seed=key[1], action=key[2], split=split,
                            absolute_loss_difference=abs(local - original_score),
                            relative_loss_difference=abs(local - original_score) / original_score))
            for method in METHODS:
                result = row['methods'][method]
                selected = select(result['curves'])
                assert int(selected) == result['selected_step']
                base = parent['selected_losses']['C']
                np.testing.assert_allclose(result['gain_percent'], 100. * (base - result['curves'][selected]['C']) / base, rtol=1e-13, atol=1e-13)
            raw_records.append(dict(task=task, seed=key[1], action=key[2], curves=raw_curves))
        print(json.dumps(dict(audited_task=task, counts=counts)), flush=True)
    assert counts == dict(coefficients=72, matrices=864, scores=1728, original_raw_scores=432)
    result = dict(status='PASS', counts=counts, predictions_sha256=sha(stage / 'predictions.json'),
                  source_lock_sha256=sha(stage / 'source_lock.json'), raw_records=raw_records,
                  cpu_gpu_prefix_score_differences=device_differences,
                  seconds=time.perf_counter() - started)
    with (stage / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status='PASS', counts=counts, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.stage)
