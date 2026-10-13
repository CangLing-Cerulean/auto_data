"""Failure closure only: JSON/hash inventory and archive; no torch, data load or inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
import time


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    started = time.perf_counter()
    assert not (out / 'final_audit.json').exists()
    assert not (out / 'failure_closure.json').exists()
    log = (out / 'audit.log').read_text(encoding='utf-8')
    assert 'Not equal to tolerance rtol=2e-05, atol=2e-06' in log
    plan = read(out / 'plan.json')
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    for name, digest in plan['inputs_sha256'].items():
        assert sha(Path(plan['input_root']) / name) == digest
    assert sha(Path(plan['core'])) == plan['core_sha256']
    pred, lock, sync = (read(out / name) for name in ('predictions.json', 'prediction_lock.json', 'prediction_sync_receipt.json'))
    assert lock['predictions_sha256'] == sync['predictions_sha256'] == sha(out / 'predictions.json')
    assert sync['prediction_lock_sha256'] == sha(out / 'prediction_lock.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    for name, digest in pred['assets_sha256'].items():
        assert sha(out / name) == digest
    labels = read(out / 'labels.json')
    assert labels['status'] == 'completed'
    assert sha(out / 'labels.json') == read(out / 'labels_lock.json')['labels_sha256']
    for saved in labels['results']:
        recorded = dict(saved)
        seconds = recorded.pop('seconds')
        directory = out / 'runs' / saved['id']
        assert recorded == read(directory / 'result.json')
        assert seconds == read(directory / 'timing.json')['seconds']
    assert {p.name for p in (out / 'runs').iterdir()} == {j['id'] for j in plan['jobs']}
    rows, results = [], {}
    for job in plan['jobs']:
        directory = out / 'runs' / job['id']
        result = read(directory / 'result.json')
        status = read(directory / 'status.json')
        assert result['status'] == status['status'] == 'completed'
        assert result['steps'] == status['steps_completed']
        for key in ('id', 'task', 'engine', 'action', 'seed'):
            assert result[key] == job[key]
        for name, digest in result['assets_sha256'].items():
            assert sha(directory / name) == digest
        row = dict(job, steps=result['steps'], seconds=read(directory / 'timing.json')['seconds'])
        rows.append(row)
        results[(job['task'], job['seed'], job['engine'], job['action'])] = result
    full = sum(r['engine'] != 'proxy' and r['steps'] == 3000 for r in rows)
    assert full == 27 and sum(r['engine'] == 'proxy' for r in rows) == 27
    assert sum(r['engine'] == 'short' for r in rows) == 18
    methods = ('M1', 'RAW', 'FIXED', 'SHORT')
    records = {(r['task'], r['seed'], r['action']): r for r in pred['records']}
    cells = []
    for task in plan['tasks']:
        for action in ('P1', 'CP1'):
            truth = []
            for seed in plan['seeds']:
                parent = results[(task['id'], seed, 'full', 'PARENT')]['selected_losses']['C']
                child = results[(task['id'], seed, 'full', action)]['selected_losses']['C']
                truth.append(100. * (parent - child) / parent)
            means = {m: statistics.mean(records[(task['id'], s, action)]['methods'][m]['gain_percent']
                                       for s in plan['seeds']) for m in methods}
            cells.append(dict(task=task['id'], action=action, recorded_truth_mean=statistics.mean(truth),
                              recorded_truth_seeds=truth, predictions=means))
    mae = {m: statistics.mean(abs(c['recorded_truth_mean'] - c['predictions'][m]) for c in cells) for m in methods}
    for task in plan['tasks']:
        assert pred['recommendations'][task['id']]['M1']['action'] == pred['recommendations'][task['id']]['FIXED']['action']
    partial = sorted(p.name for p in (out / 'audit').glob('*.npz'))
    result = dict(status='closed_after_failed_audit', stage_accepted=False, scientific_advance=False,
        selection_gate=False, selection_gate_reason='Frozen M1 and FIXED policies are identical; strict positive increment impossible',
        official_error_gate=None, official_D_gate=None, independent_audit='FAILED_original_tolerance',
        audit_log_sha256=sha(out / 'audit.log'), code_unchanged=True, original_inputs_unchanged=True,
        forecasts_unchanged=True, locked_labels_and_all_run_asset_hashes_verified=True,
        new_model_execution=False, new_sensor_array_decoding=False, tolerance_changed=False,
        completed_full_trainings=full, cumulative_completed_full_trainings=304 + full,
        last_independently_accepted_full_count=304, proxy_rollouts=27, short_trainings=18,
        short_optimizer_steps=sum(r['steps'] for r in rows if r['engine'] == 'short'),
        training_failures=0, audit_failures=1, test_access=False, model_queue=False,
        D_access='Partial original audit reached G3/seed 2026101701; no accepted D result; closure never reads arrays',
        failure_location='G3/P1/2026101701 mechanism loop; exact grid point absent from original log',
        provisional_recorded_metrics=dict(status='DESCRIPTIVE_ONLY_NOT_FULLY_INDEPENDENTLY_ACCEPTED', MAE_pp=mae, cells=cells),
        registered_costs=pred['costs'], runs=rows, totals_seconds={e: sum(r['seconds'] for r in rows if r['engine'] == e) for e in ('full', 'proxy', 'short')},
        forecast_phase_seconds=pred['phase_seconds'], labels_phase_seconds=labels['phase_seconds'], failed_audit_seconds=None,
        cost_unknown_is_not_zero=True, partial_audit_residual_files=partial,
        closed_utc=datetime.now(timezone.utc).isoformat(), closure_seconds=time.perf_counter() - started)
    write(out / 'failure_closure.json', result)
    archive_started = time.perf_counter()
    allowed = {'.py', '.json', '.md', '.log', '.npz', '.pt', '.lock'}
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in allowed and p.name != 'closure.log')
    assert all(not p.is_symlink() for p in paths)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    archive = out / 'response_proxy_development_failed_audit_20261013.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in hashes:
            tar.add(out / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(hashes)
        for name, digest in hashes.items():
            assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == digest
    write(out / 'failure_archive_receipt.json', dict(status='PASS_archive_integrity_only',
        file=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), members_sha256=hashes,
        archive_seconds=time.perf_counter() - archive_started))
    print(json.dumps(dict(status=result['status'], full=full, archive_bytes=archive.stat().st_size,
                         provisional_MAE_pp=mae)), flush=True)


if __name__ == '__main__':
    main()
