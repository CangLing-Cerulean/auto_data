"""Create fixed replay folds. This preparer fits no learner or selects no result."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from m1_shrinkage_inference import output_prediction, read, select, sha


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def identifier(r):
    return f'{r["task"]}_{r["action"]}_{r["seed"]}'


def ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


@torch.no_grad()
def main(source, shrinkage, stage):
    started = time.perf_counter()
    assert not (stage / 'fold_manifest.json').exists()
    for name, digest in read(stage / 'source_lock.json')['sha256'].items():
        assert sha(stage / name) == digest
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    spec = read(source / 'spec.json')
    parents = read(source / 'parents.json')['records']
    old = read(source / 'predictions.json')['records']
    assert sha(source / 'predictions.json') == read(source / 'prediction_lock.json')['sha256']
    previous = read(shrinkage / 'predictions.json')
    assert sha(shrinkage / 'predictions.json') == read(shrinkage / 'prediction_lock.json')['sha256']
    inputs = {name: sha(source / name) for name in ('spec.json', 'parents.json', 'predictions.json', 'prediction_lock.json')}

    def checked(path, digest):
        key = path.relative_to(source).as_posix()
        if key not in inputs:
            inputs[key] = sha(path)
        assert inputs[key] == digest, key
        return path

    def load(record, step):
        name = f'step_{step}.pt'
        path = checked(source / record['directory'] / name, record['assets_sha256'][name])
        saved = torch.load(path, weights_only=True, map_location='cpu')
        assert saved['step'] == step
        return saved['original_coordinates']

    features, feedback, endpoint_differences = [], {}, []
    for task in spec['tasks']:
        path = checked(source / 'inputs' / f'{task}_ABC.npz', spec['input_sha256'][task])
        with np.load(path, allow_pickle=False) as z:
            arrays = {s: (torch.from_numpy(z[f'{s}_values']), torch.from_numpy(z[f'{s}_starts'])) for s in ('B', 'C')}
        for r in (r for r in old if r['task'] == task):
            rid = identifier(r)
            parent = next(p for p in parents if (p['task'], p['seed']) == (task, r['seed']))
            h0, _ = output_prediction(load(parent, 500), *arrays['B'])
            l0, target = output_prediction(load(r['parent_mean'], 500), *arrays['B'])
            la, _ = output_prediction(load(r['mean'], 500), *arrays['B'])
            u = h0.astype(np.float64) - l0
            d = la.astype(np.float64) - l0
            e0 = l0.astype(np.float64) - target
            energy_u, energy_d, energy_e0 = [np.mean(v * v, axis=(0, 1)) for v in (u, d, e0)]
            nu, nd, ne = [np.sqrt(v) for v in (energy_u, energy_d, energy_e0)]
            x = np.stack((ratio(np.mean(u * d, axis=(0, 1)), nu * nd),
                          np.log1p(ratio(nd, nu)), np.log1p(ratio(nu, ne))), -1)
            ha, _ = output_prediction(load(r['prefix'], 500), *arrays['B'])
            v = ha.astype(np.float64) - la
            alpha = np.clip(ratio(np.mean(u * v, axis=(0, 1)), energy_u), 0., 1.)
            previous_row = next(p for p in previous['records'] if identifier(p) == rid)
            np.testing.assert_allclose(alpha, previous_row['coefficients']['CHANNEL_SHRINK']['alpha'], rtol=2e-5, atol=2e-6)
            feedback[rid] = dict(prefix_p=[r['prefix']['curves']['500'][s] for s in ('B', 'C')],
                                  z=alpha.tolist(), z_energy=energy_u.tolist())
            polynomial = []
            for step in spec['grid']:
                raw = load(r['mean'], step)
                name = f'step_{step}.npz'
                original = r['methods']['M1']
                path = checked(source / original['directory'] / name, original['assets_sha256'][name])
                with np.load(path, allow_pickle=False) as z:
                    m1 = torch.from_numpy(z['coordinates'])
                splits = []
                for split in ('B', 'C'):
                    raw_out, y = output_prediction(raw, *arrays[split])
                    m1_out, _ = output_prediction(m1, *arrays[split])
                    low_loss = np.mean((raw_out - y).astype(np.float64) ** 2, axis=(0, 1))
                    high_loss = np.mean((m1_out - y).astype(np.float64) ** 2, axis=(0, 1))
                    square = np.mean((m1_out.astype(np.float64) - raw_out) ** 2, axis=(0, 1))
                    splits.append(np.stack((low_loss, high_loss - low_loss - square, square), -1).tolist())
                    old_score = r['methods']['M1']['curves'][str(step)][split]
                    np.testing.assert_allclose(high_loss.mean(), old_score, rtol=2e-5, atol=2e-6)
                    endpoint_differences.append(abs(float(high_loss.mean()) - old_score))
                polynomial.append(splits)
            raw_curves = {str(step): {s: float(np.asarray(polynomial[j])[k, :, 0].mean())
                                     for k, s in enumerate(('B', 'C'))} for j, step in enumerate(spec['grid'])}
            chosen = select(raw_curves)
            base = parent['selected_losses']['C']
            row = dict(id=rid, task=task, seed=r['seed'], action=r['action'], x=x.tolist(),
                polynomial=polynomial, parent_c=base,
                parent_prefix_p=[parent['curves']['500'][s] for s in ('B', 'C')],
                m1_gain=r['methods']['M1']['gain_percent'],
                raw_gain=100. * (base - raw_curves[chosen]['C']) / base)
            assert np.isfinite(np.asarray(x)).all() and min(row['parent_prefix_p']) > 0 and base > 0
            features.append(row)
        print(json.dumps(dict(prepared_task=task, candidates=len(features))), flush=True)
    assert len(features) == len(feedback) == 36
    # Only the preparation/evaluation programs see the full old label table.
    labels = read(source / 'labels.json')['records']
    inputs['labels.json'] = sha(source / 'labels.json')
    truth = {}
    for label in labels:
        rid = identifier(label)
        row = next(r for r in features if r['id'] == rid)
        y = 100. * (row['parent_c'] - label['selected_losses']['C']) / row['parent_c']
        reference = next(r for r in previous['records'] if identifier(r) == rid)
        truth[rid] = dict(y=y, channel_reference=reference['methods']['CHANNEL_SHRINK']['gain_percent'])
    folds = []
    for task in spec['tasks']:
        for pair in itertools.combinations(spec['actions'], 2):
            fold_id = f'{task}_{pair[0]}__{pair[1]}'
            history = []
            for r in features:
                if r['task'] != task:
                    history.append(dict(copy.deepcopy(r), **feedback[r['id']], y=truth[r['id']]['y']))
            probes, targets = [], []
            for r in features:
                if r['task'] != task:
                    continue
                target = copy.deepcopy(r)
                target['probed'] = r['action'] in pair
                if target['probed']:
                    target['prefix_p'] = feedback[r['id']]['prefix_p']
                    probes.append(dict(copy.deepcopy(r), **feedback[r['id']]))
                targets.append(target)
            assert len(history) == 24 and len(probes) == 6 and len(targets) == 12
            historical_labels = [p for p in labels if p['task'] != task]
            historical_parents = [p for p in parents if p['task'] != task]
            target_parents = [p for p in parents if p['task'] == task]
            def proxies(task_is_target):
                records = [r for r in old if (r['task'] == task) == task_is_target]
                unique_parents = {r['parent_mean']['directory']: r['parent_mean']['seconds'] for r in records}
                return sum(r['mean']['seconds'] + r['methods']['M1']['seconds'] for r in records) + sum(unique_parents.values())
            cost = dict(historical_full_seconds=sum(p['charged_full_seconds'] for p in historical_labels) + sum(p['seconds'] for p in historical_parents),
                historical_proxy_seconds=proxies(False), target_parent_seconds=sum(p['seconds'] for p in target_parents),
                target_proxy_seconds=proxies(True), target_prefix_seconds=sum(r['prefix']['seconds'] for r in old if r['task'] == task and r['action'] in pair),
                reference_all_prefix_seconds=sum(r['prefix']['seconds'] for r in old if r['task'] == task),
                scope='Historical GPU recorded work; reused across folds, not new execution or cold-start timing')
            fold = dict(id=fold_id, task=task, actions=spec['actions'], seeds=spec['seeds'], probe_actions=list(pair),
                        history=history, probes=probes, targets=targets, cost=cost)
            path = stage / 'folds' / f'{fold_id}.json'
            write(path, fold)
            folds.append(dict(id=fold_id, path=path.relative_to(stage).as_posix(), sha256=sha(path)))
    write(stage / 'features.json', features)
    write(stage / 'prefix_feedback.json', feedback)
    write(stage / 'evaluation_truth.json', truth)
    write(stage / 'fold_manifest.json', dict(folds=folds, source_inputs_sha256=inputs,
        shrinkage_predictions_sha256=sha(shrinkage / 'predictions.json'),
        source_lock_sha256=sha(stage / 'source_lock.json'), features_sha256=sha(stage / 'features.json'),
        feedback_sha256=sha(stage / 'prefix_feedback.json'), truth_sha256=sha(stage / 'evaluation_truth.json'),
        max_original_m1_loss_difference=max(endpoint_differences), preparation_seconds=time.perf_counter() - started,
        scope='known_source_replay_preparation_no_model_fits', environment=dict(torch=torch.__version__, device='CPU', threads=4)))
    print(json.dumps(dict(status='PREPARED', folds=len(folds), seconds=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--shrinkage', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.shrinkage, args.stage)
