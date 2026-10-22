"""Run all fixed learners from folds only; no access to evaluation labels."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from m1_shrinkage_inference import read, sha
from transfer_diagnosis_core import METHODS, fit, predict, ridge_fit, ridge_predict


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def validate_fold(fold):
    assert len(fold['history']) == 24 and len(fold['probes']) == 6 and len(fold['targets']) == 12
    assert all(r['task'] != fold['task'] and 'y' in r for r in fold['history'])
    assert all(r['task'] == fold['task'] and r['action'] in fold['probe_actions'] and 'y' not in r for r in fold['probes'])
    assert len({r['id'] for r in fold['targets']}) == 12
    for r in fold['targets']:
        assert r['task'] == fold['task'] and not {'y', 'z', 'z_energy'}.intersection(r)
        assert ('prefix_p' in r) == (r['action'] in fold['probe_actions']) == r['probed']
    assert {r['id'] for r in fold['probes']} == {r['id'] for r in fold['targets'] if r['probed']}
    assert {r['id'] for r in fold['history']}.isdisjoint({r['id'] for r in fold['targets']})


def main(stage):
    started = time.perf_counter()
    assert not (stage / 'predictions.json').exists()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read(stage / 'fold_manifest.json')
    assert sha(stage / 'source_lock.json') == manifest['source_lock_sha256']
    for name, digest in read(stage / 'source_lock.json')['sha256'].items():
        assert sha(stage / name) == digest
    assert len(manifest['folds']) == 18
    results = []
    for entry in manifest['folds']:
        path = stage / entry['path']
        assert sha(path) == entry['sha256']
        fold = read(path)
        validate_fold(fold)
        methods = {}
        for method in (*METHODS, 'RIDGE_Y'):
            tick = time.perf_counter()
            model = ridge_fit(fold) if method == 'RIDGE_Y' else fit(fold, method)
            predictions = ridge_predict(model, fold['targets']) if method == 'RIDGE_Y' else predict(model, fold['targets'])
            assert len(predictions) == 12 and all(np.isfinite(p['gain_percent']) for p in predictions)
            state_path = stage / 'models' / f'{fold["id"]}_{method}.json'
            write(state_path, model)
            methods[method] = dict(predictions=predictions, seconds=time.perf_counter() - tick,
                                  model=state_path.relative_to(stage).as_posix(), model_sha256=sha(state_path))
        results.append(dict(id=fold['id'], task=fold['task'], probe_actions=fold['probe_actions'], methods=methods))
        print(json.dumps(dict(completed=len(results), total=18, fold=fold['id'])), flush=True)
    write(stage / 'predictions.json', dict(records=results, source_lock_sha256=sha(stage / 'source_lock.json'),
        fold_manifest_sha256=sha(stage / 'fold_manifest.json'), actual_run_seconds=time.perf_counter() - started,
        auxiliary_models=54, auxiliary_updates=54000, ridge_fits=18, new_backbone_trainings=0, historical_full_completed=593))
    write(stage / 'prediction_lock.json', dict(sha256=sha(stage / 'predictions.json'),
        source_lock_sha256=sha(stage / 'source_lock.json'), locked_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(status='PREDICTIONS_LOCKED', seconds=time.perf_counter() - started)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
