"""Frozen known-source development forecasts; never reads completed child labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from m1_shrinkage_inference import loss, output_prediction, read, select, sha
from m1_prefix_shrinkage import blend_parameters, fit_coefficients

METHODS = ('RAW_PREFIX', 'M1_PREFIX', 'SCALAR_SHRINK', 'CHANNEL_SHRINK')


def write(path, obj):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


@torch.no_grad()
def main(source, stage):
    started = time.perf_counter()
    assert not (stage / 'predictions.json').exists()
    lock = read(stage / 'source_lock.json')
    for name, digest in lock['sha256'].items():
        assert sha(stage / name) == digest, name
    for name, digest in lock['source_metadata_sha256'].items():
        assert sha(source / name) == digest, name
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = read(source / 'spec.json')
    previous = read(source / 'predictions.json')['records']
    parents = read(source / 'parents.json')['records']
    assert spec['prefix_steps'] == 500 and len(previous) == 36 and len(parents) == 9
    assert sha(source / 'predictions.json') == read(source / 'prediction_lock.json')['sha256']
    inputs = dict(lock['source_metadata_sha256'])

    def checked(path, digest):
        relative = path.relative_to(source).as_posix()
        if relative not in inputs:
            inputs[relative] = sha(path)
        assert inputs[relative] == digest, relative
        return path

    def coords(record, step):
        assert record['kind'] in ('parent', 'mean', 'prefix'), record['kind']
        if record['kind'] == 'prefix':
            assert step == 500
        filename = f'step_{step}.pt'
        path = checked(source / record['directory'] / filename, record['assets_sha256'][filename])
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value['step'] == step
        return value['original_coordinates'].numpy().copy()

    def paired(record, step):
        filename = f'step_{step}.npz'
        path = checked(source / record['directory'] / filename, record['assets_sha256'][filename])
        with np.load(path, allow_pickle=False) as data:
            return data['coordinates'].copy()

    records = []
    for task in spec['tasks']:
        path = checked(source / 'inputs' / f'{task}_ABC.npz', spec['input_sha256'][task])
        with np.load(path, allow_pickle=False) as z:
            arrays = {s: (torch.from_numpy(z[f'{s}_values']),
                          torch.from_numpy(z[f'{s}_starts'])) for s in ('B', 'C')}
        for seed in spec['seeds']:
            parent = next(p for p in parents if (p['task'], p['seed']) == (task, seed))
            children = [r for r in previous if (r['task'], r['seed']) == (task, seed)]
            assert len(children) == 4 and {r['action'] for r in children} == set(spec['actions'])
            calibration_started = time.perf_counter()
            h0, l0 = coords(parent, 500), coords(children[0]['parent_mean'], 500)
            high, _ = output_prediction(torch.from_numpy(h0), *arrays['B'])
            low, _ = output_prediction(torch.from_numpy(l0), *arrays['B'])
            u = high.astype(np.float64) - low
            parent_calibration_seconds = time.perf_counter() - calibration_started
            for r in children:
                calibration_started = time.perf_counter()
                prefix = coords(r['prefix'], 500)
                raw_prefix = coords(r['mean'], 500)
                high, _ = output_prediction(torch.from_numpy(prefix), *arrays['B'])
                low, _ = output_prediction(torch.from_numpy(raw_prefix), *arrays['B'])
                v = high.astype(np.float64) - low
                child_calibration_seconds = time.perf_counter() - calibration_started
                coefficients = {}
                for method, by_channel in (('SCALAR_SHRINK', False), ('CHANNEL_SHRINK', True)):
                    tick = time.perf_counter()
                    alpha, moments = fit_coefficients(u, v, by_channel)
                    coefficients[method] = dict(alpha=alpha.tolist(), **moments,
                        fit_seconds=time.perf_counter() - tick)
                models = {m: {} for m in METHODS}
                for step in spec['grid']:
                    if step == 500:
                        for method in METHODS:
                            models[method][step] = prefix
                    else:
                        raw = coords(r['mean'], step)
                        m1 = paired(r['methods']['M1'], step)
                        models['RAW_PREFIX'][step], models['M1_PREFIX'][step] = raw, m1
                        for method in coefficients:
                            models[method][step] = blend_parameters(raw, m1, np.array(coefficients[method]['alpha']))
                result = dict(task=task, seed=seed, action=r['action'], coefficients=coefficients,
                    calibration_seconds=dict(parent_shared_amortized=parent_calibration_seconds / 4.,
                                             child=child_calibration_seconds), methods={})
                for method in METHODS:
                    tick = time.perf_counter()
                    directory = Path('forecasts') / f'{task}_{r["action"]}_{seed}_{method}'
                    (stage / directory).mkdir(parents=True, exist_ok=False)
                    curves, assets = {}, {}
                    for step, value in models[method].items():
                        curves[str(step)] = {}
                        for split in ('B', 'C'):
                            prediction, target = output_prediction(torch.from_numpy(value), *arrays[split])
                            curves[str(step)][split] = loss(prediction, target)
                        filename = f'step_{step}.npz'
                        np.savez(stage / directory / filename, coordinates=value)
                        assets[filename] = sha(stage / directory / filename)
                    selected = select(curves)
                    base = parent['selected_losses']['C']
                    scoring_seconds = time.perf_counter() - tick
                    reused_seconds = r['prefix']['seconds'] + r['mean']['seconds']
                    if method != 'RAW_PREFIX':
                        reused_seconds += r['parent_mean']['seconds'] / 4. + r['methods']['M1']['seconds']
                    calibration_seconds = (parent_calibration_seconds / 4. + child_calibration_seconds +
                        coefficients[method]['fit_seconds']) if method in coefficients else 0.
                    result['methods'][method] = dict(curves=curves, selected_step=int(selected),
                        gain_percent=100. * (base - curves[selected]['C']) / base,
                        directory=directory.as_posix(), assets_sha256=assets,
                        reused_origin_seconds=reused_seconds, calibration_seconds=calibration_seconds,
                        new_score_save_hash_seconds=scoring_seconds,
                        conditional_reuse_ledger_seconds=reused_seconds + calibration_seconds + scoring_seconds)
                records.append(result)
            print(json.dumps(dict(task=task, seed=seed, candidates=len(records))), flush=True)
    expected = {(t, s, a) for t in spec['tasks'] for s in spec['seeds'] for a in spec['actions']}
    keys = [(r['task'], r['seed'], r['action']) for r in records]
    assert len(keys) == 36 and set(keys) == expected
    result = dict(status='FORECAST_COMPLETE', scope='posthoc_known_source_development', records=records,
        source_stage=source.name, source_lock_sha256=sha(stage / 'source_lock.json'),
        source_input_sha256=inputs, new_training_steps=0, new_full_trainings=0,
        historical_full_completed=593, actual_new_forecast_seconds=time.perf_counter() - started,
        environment=dict(device='CPU', torch_version=torch.__version__, threads=4),
        cost_scope='Mixed-device bookkeeping only: reused original GPU work and new CPU inference. Old M1 formation/scoring is conservatively charged. Excludes new common loading/blending and process/transfer cost. Shared prefix calibration charged fully to each shrinkage version. Not a timing or budget comparison.')
    write(stage / 'predictions.json', result)
    write(stage / 'prediction_lock.json', dict(sha256=sha(stage / 'predictions.json'),
        source_lock_sha256=sha(stage / 'source_lock.json'), locked_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='FORECAST_COMPLETE', seconds=result['actual_new_forecast_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.stage)
