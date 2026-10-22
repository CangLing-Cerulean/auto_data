"""Independent NumPy reconstruction of fitted models and exact fold boundaries."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from m1_shrinkage_inference import read, sha


def gains(rows, alpha):
    values, steps = [], []
    for r, a in zip(rows, alpha):
        p = np.asarray(r['polynomial'])
        curves = np.empty((6, 2))
        for t in range(6):
            for s in range(2):
                curves[t, s] = sum(p[t, s, c, 0] + p[t, s, c, 1] * a[c] + p[t, s, c, 2] * a[c] ** 2 for c in range(8)) / 8
        if 'prefix_p' in r:
            curves[0] = r['prefix_p']
        selected = min(range(6), key=lambda t: curves[t, 0])
        values.append(100. * (r['parent_c'] - curves[selected, 1]) / r['parent_c'])
        steps.append((selected + 1) * 500)
    return np.asarray(values), steps


def ridge_x(rows):
    return np.column_stack((np.asarray([r['x'] for r in rows]).reshape(len(rows), 24),
        *(gains(rows, np.full((len(rows), 8), a))[0] for a in (0., .5, 1.))))


def main(stage):
    started = time.perf_counter()
    assert not (stage / 'audit.json').exists()
    pred, manifest = read(stage / 'predictions.json'), read(stage / 'fold_manifest.json')
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256']
    assert sha(stage / 'fold_manifest.json') == pred['fold_manifest_sha256']
    for name, digest in read(stage / 'source_lock.json')['sha256'].items():
        assert sha(stage / name) == digest
    features = read(stage / 'features.json')
    feedback = read(stage / 'prefix_feedback.json')
    truth = read(stage / 'evaluation_truth.json')
    for name, key in (('features.json', 'features_sha256'), ('prefix_feedback.json', 'feedback_sha256'), ('evaluation_truth.json', 'truth_sha256')):
        assert sha(stage / name) == manifest[key]
    feature_map = {r['id']: r for r in features}
    assert len(feature_map) == 36
    counts = dict(folds=0, models=0, predictions=0, ridge_normal_equations=0)
    for result in pred['records']:
        entry = next(e for e in manifest['folds'] if e['id'] == result['id'])
        assert sha(stage / entry['path']) == entry['sha256']
        fold = read(stage / entry['path'])
        assert len(fold['history']) == 24 and len(fold['probes']) == 6 and len(fold['targets']) == 12
        for r in fold['history']:
            expected = dict(feature_map[r['id']], **feedback[r['id']], y=truth[r['id']]['y'])
            assert r == expected and r['task'] != fold['task']
        for r in fold['probes']:
            assert r == dict(feature_map[r['id']], **feedback[r['id']])
            assert r['task'] == fold['task'] and r['action'] in fold['probe_actions']
        for r in fold['targets']:
            expected = dict(feature_map[r['id']], probed=r['action'] in fold['probe_actions'])
            if expected['probed']:
                expected['prefix_p'] = feedback[r['id']]['prefix_p']
            assert r == expected and r['task'] == fold['task']
        assert len({r['id'] for r in fold['history']}) == 24
        assert len({r['id'] for r in fold['targets']}) == 12
        counts['folds'] += 1
        rows = fold['targets']
        for method, registered in result['methods'].items():
            assert sha(stage / registered['model']) == registered['model_sha256']
            model = read(stage / registered['model'])
            if method == 'RIDGE_Y':
                x = ridge_x(rows)
                design = np.column_stack((np.ones(len(rows)), (x - model['center']) / model['scale']))
                predicted = np.asarray([r['m1_gain'] for r in rows]) + model['y_scale'] * (design @ model['weights'])
                hx = ridge_x(fold['history'])
                hd = np.column_stack((np.ones(len(hx)), (hx - model['center']) / model['scale']))
                target = np.asarray([r['y'] - r['m1_gain'] for r in fold['history']]) / model['y_scale']
                w = np.asarray(model['weights'])
                penalty = w.copy()
                penalty[0] = 0.
                normal = hd.T @ (hd @ w - target) / len(hx) + penalty
                np.testing.assert_allclose(normal, np.zeros_like(normal), rtol=0, atol=1e-10)
                counts['ridge_normal_equations'] += 1
            else:
                x = (np.asarray([r['x'] for r in rows]) - model['center']) / model['scale']
                design = np.concatenate((np.ones((*x.shape[:2], 1)), x), -1)
                logits = np.sum(design * np.asarray(model['weights']), axis=-1)
                alpha = np.exp(-np.logaddexp(0., -logits))
                raw, selected = gains(rows, alpha)
                predicted = raw + model['beta'][0] * model['y_scale'] + model['beta'][1] * raw
                assert model['updates'] == 1000
                for i, saved in enumerate(registered['predictions']):
                    np.testing.assert_allclose(saved['alpha'], alpha[i], rtol=1e-12, atol=1e-12)
                    assert saved['selected_step'] == selected[i]
            assert [p['id'] for p in registered['predictions']] == [r['id'] for r in rows]
            np.testing.assert_allclose([p['gain_percent'] for p in registered['predictions']], predicted, rtol=1e-10, atol=1e-8)
            counts['models'] += 1
            counts['predictions'] += len(rows)
    assert counts == dict(folds=18, models=72, predictions=864, ridge_normal_equations=18)
    result = dict(status='PASS', counts=counts, predictions_sha256=sha(stage / 'predictions.json'),
                  seconds=time.perf_counter() - started)
    with (stage / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
