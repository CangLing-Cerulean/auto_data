"""Checkpoint continuation in a separate, hash-bound aggregate; originals stay immutable."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch


PARTIAL = 'SELR1_G1_full_FAVOR_SECOND_s2026101703'
CHECK = 'SELR1_G1_full_KEEP_FIRST_s2026101701'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value, replace=False):
    path = Path(path)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if replace:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(text, encoding='utf-8')
        tmp.replace(path)
    else:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)


def utc():
    return datetime.now(timezone.utc).isoformat()


def runner(root):
    sys.path.insert(0, str(root))
    import response_proxy_selection as rs
    rs.configure()
    return rs


def assert_quiescent(original):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        # Match exact original script arguments, avoiding our parent shell command.
        assert str(original / 'response_proxy_selection.py').encode() not in args, 'Original runner still exists'


def verify_original(original, hashes):
    paths = sorted(p for p in original.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.is_symlink() for p in paths)
    assert {p.relative_to(original).as_posix() for p in paths} == set(hashes)
    for name, digest in hashes.items():
        assert sha(original / name) == digest, name


def inspect(original, out):
    assert_quiescent(original)
    rs = runner(original)
    plan = rs.verify(original)
    prediction = read(original / 'predictions.json')
    lock, sync = (read(original / n) for n in ('prediction_lock.json', 'prediction_sync_receipt.json'))
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sha(original / 'predictions.json') == lock['predictions_sha256'] == sync['predictions_sha256']
    assert sha(original / 'prediction_lock.json') == sync['prediction_lock_sha256']
    assert prediction['budget_gate'] and not prediction['D_loaded'] and not prediction['test_access']
    for name, digest in prediction['assets_sha256'].items():
        assert sha(original / name) == digest, name
    complete, unstarted = [], []
    for job in (j for j in plan['jobs'] if j['phase'] == 'labels'):
        directory = original / 'runs' / job['id']
        if job['id'] == PARTIAL:
            assert {p.name for p in directory.iterdir()} == {'status.json', 'initial.pt', 'step_500.pt'}
            assert read(directory / 'status.json')['steps_completed'] == 500
            continue
        if not directory.exists():
            unstarted.append(job['id'])
            continue
        result, status = (read(directory / n) for n in ('result.json', 'status.json'))
        assert result['status'] == status['status'] == 'completed' and result['steps'] == 3000
        assert result['id'] == job['id'] and result['plan_sha256'] == sha(original / 'plan.json')
        assert read(directory / 'timing.json')['seconds'] > 0
        for name, digest in result['assets_sha256'].items():
            assert sha(directory / name) == digest
        complete.append(job['id'])
    assert len(complete) == 11 and len(unstarted) == 24
    snap = torch.load(original / 'runs' / PARTIAL / 'step_500.pt', map_location='cpu', weights_only=True)
    assert snap['step'] == 500 and len(snap['optimizer']['state']) == 32
    assert all(torch.isfinite(v).all() for v in snap['model'].values())
    for state in snap['optimizer']['state'].values():
        assert state['step'].item() == 500
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
    rng = np.random.default_rng(2026101703 + 1000)
    for _ in range(500):
        rng.random(256)
    assert snap['rng_numpy'] == rng.bit_generator.state
    assert snap['rng_torch'].dtype == snap['rng_cuda'].dtype == torch.uint8
    paths = sorted(p for p in original.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.is_symlink() for p in paths)
    hashes = {p.relative_to(original).as_posix(): sha(p) for p in paths}
    write(out / 'inspection.json', dict(status='PASS', inspected_at_utc=utc(), original_root=str(original),
        original_process_absent=True, termination_cause='unknown', complete_ids=complete, unstarted_ids=unstarted,
        partial_id=PARTIAL, checkpoint_step=500, checkpoint_sha256=hashes[f'runs/{PARTIAL}/step_500.pt'],
        original_files_sha256=hashes, original_predictions_sha256=sha(original / 'predictions.json'),
        original_plan_sha256=sha(original / 'plan.json'), pre_interruption_partial_seconds=None,
        torch_version=torch.__version__, numpy_version=np.__version__, cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), new_training_steps=0, D_access=False, test_access=False))
    print(json.dumps(dict(status='PASS', complete=11, partial=1, unstarted=24, checkpoint_step=500)), flush=True)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def restore(ctx, directory, seed, action):
    model = ctx.model(seed)
    opt = ctx.optimizer(model)
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=True)
    equal_tree(model.state_dict(), initial['model'])
    snap = torch.load(directory / 'step_500.pt', map_location='cpu', weights_only=True)
    assert snap['step'] == 500
    model.load_state_dict(snap['model'])
    opt.load_state_dict(snap['optimizer'])
    # Restore after all construction; reconstruction consumes no model updates.
    rng = np.random.default_rng(seed + 1000)
    draws = np.zeros(ctx.n, dtype=np.int64)
    for _ in range(500):
        np.add.at(draws, np.searchsorted(ctx.cdfs[action], rng.random(256), side='right'), 1)
    assert rng.bit_generator.state == snap['rng_numpy']
    rng.bit_generator.state = snap['rng_numpy']
    torch.set_rng_state(snap['rng_torch'].cpu())
    torch.cuda.set_rng_state(snap['rng_cuda'].cpu())
    return model, opt, rng, draws


def step_model(ctx, model, opt, rng, draws, action):
    model.train()
    opt.zero_grad(set_to_none=True)
    ids = np.searchsorted(ctx.cdfs[action], rng.random(256), side='right')
    np.add.at(draws, ids, 1)
    batch = ctx.batch('A', torch.tensor(ids, device='cuda'))
    loss = (model(ctx.transform(batch[:, :32], action)) - batch[:, 32:]).square().mean()
    assert bool(torch.isfinite(loss))
    loss.backward()
    opt.step()


def equivalence(rs, aggregate, plan):
    start = rs.tick()
    ctx = rs.Context(aggregate, plan, 'G1')
    directory = aggregate / 'runs' / CHECK
    model, opt, rng, draws = restore(ctx, directory, 2026101701, 'KEEP_FIRST')
    for _ in range(501, 1001):
        step_model(ctx, model, opt, rng, draws, 'KEEP_FIRST')
    expected = torch.load(directory / 'step_1000.pt', map_location='cpu', weights_only=True)
    equal_tree(model.state_dict(), expected['model'])
    equal_tree(opt.state_dict(), expected['optimizer'])
    equal_tree(rng.bit_generator.state, expected['rng_numpy'])
    equal_tree(torch.get_rng_state(), expected['rng_torch'])
    equal_tree(torch.cuda.get_rng_state(), expected['rng_cuda'])
    receipt = dict(status='PASS', execution_id='SELR1_RECOVERY_CHECK_500_TO_1000',
        reference_id=CHECK, restored_step=500, checked_step=1000, diagnostic_optimizer_steps=500,
        model_optimizer_numpy_torch_cuda_bitexact=True, seconds=rs.tick() - start,
        counts_as_new_full_training=False, original_reference_unchanged=True, D_access=False, test_access=False)
    write(aggregate / 'recovery' / 'checkpoint_equivalence.json', receipt)
    return receipt


def continuation(rs, aggregate, plan):
    start = rs.tick()
    ctx = rs.Context(aggregate, plan, 'G1')
    directory = aggregate / 'runs' / PARTIAL
    model, opt, rng, draws = restore(ctx, directory, 2026101703, 'FAVOR_SECOND')
    scores = {500: ctx.evaluate(model, 'FAVOR_SECOND', 'B')[0]}
    state = read(directory / 'status.json')
    state['recovery_execution_id'] = 'SELR1_CONTINUE_FROM_500_G1_FAVOR_SECOND_s2026101703'
    for step in range(501, 3001):
        step_model(ctx, model, opt, rng, draws, 'FAVOR_SECOND')
        state['steps_completed'] = step
        if step in rs.GRID:
            path = directory / f'step_{step}.pt'
            assert not path.exists()
            rs.snapshot(model, opt, rng, step, path)
            scores[step] = ctx.evaluate(model, 'FAVOR_SECOND', 'B')[0]
            write(directory / 'status.json', state, replace=True)
    selected = rs.rp.select_checkpoint(scores, list(rs.GRID))
    model.load_state_dict(rs.load_state(directory / f'step_{selected}.pt')['model'])
    losses = {}
    for split in ('B', 'C'):
        losses[split], error = ctx.evaluate(model, 'FAVOR_SECOND', split)
        rs.arrays_save(directory / f'selected_{split}_errors.npz', error=error)
    assert losses['B'] == scores[selected]
    rs.arrays_save(directory / 'sampling.npz', draw_count=draws)
    assets = {p.name: sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.pt', '.npz')}
    initial = torch.load(directory / 'initial.pt', map_location='cpu', weights_only=True)['original_coordinates']
    result = dict(state, status='completed', steps=3000, selected_step=selected, selection_losses=scores,
        selected_losses=losses, checkpoints=list(rs.GRID), short_curves={}, assets_sha256=assets,
        initial_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(), plan_sha256=sha(aggregate / 'plan.json'),
        recovery=dict(start_step=501, saved_prefix_steps=500, new_optimizer_steps=2500,
                      original_interrupted_artifacts_preserved=True, same_logical_training=True))
    write(directory / 'result.json', result)
    state['status'] = 'completed'
    write(directory / 'status.json', state, replace=True)
    measured = rs.tick() - start
    write(directory / 'timing.json', dict(seconds=None, measured_continuation_seconds=measured,
        pre_interruption_seconds=None, boundary='total unknown; measured continuation includes load, RNG count reconstruction, B rescore and remaining steps/assets'))
    write(aggregate / 'recovery' / 'continuation.json', dict(status='completed', logical_id=PARTIAL,
        execution_id=state['recovery_execution_id'], measured_continuation_seconds=measured,
        total_logical_run_seconds=None, original_run_status_unchanged=True, new_optimizer_steps=2500))
    result['seconds'] = None
    return result


def run(original, out):
    started = time.perf_counter()
    assert_quiescent(original)
    inspection = read(out / 'inspection.json')
    recovery_plan = read(out / 'recovery_plan.json')
    sync = read(out / 'recovery_source_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['recovery_plan_sha256'] == sha(out / 'recovery_plan.json')
    assert recovery_plan['inspection_sha256'] == sha(out / 'inspection.json')
    assert recovery_plan['original_root'] == str(original) and recovery_plan['recovery_root'] == str(out)
    for name, digest in recovery_plan['code_sha256'].items():
        assert sha(out / name) == digest
    verify_original(original, inspection['original_files_sha256'])
    aggregate = out / 'aggregate'
    aggregate.mkdir()
    for name, digest in inspection['original_files_sha256'].items():
        target = aggregate / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
        assert sha(target) == digest
    recovery = aggregate / 'recovery'
    recovery.mkdir()
    for name in ('inspection.json', 'recovery_plan.json', 'recovery_source_sync_receipt.json',
                 'recover_proxy_selection.py', 'RECOVERY_PROTOCOL.md'):
        shutil.copy2(out / name, recovery / name)
    rs = runner(aggregate)
    plan = rs.verify(aggregate)
    results = []
    try:
        equivalence(rs, aggregate, plan)
        for job in (j for j in plan['jobs'] if j['phase'] == 'labels'):
            ident = job['id']
            if ident in inspection['complete_ids']:
                directory = aggregate / 'runs' / ident
                result = read(directory / 'result.json')
                result['seconds'] = read(directory / 'timing.json')['seconds']
            elif ident == PARTIAL:
                result = continuation(rs, aggregate, plan)
            else:
                assert ident in inspection['unstarted_ids']
                ctx = rs.Context(aggregate, plan, job['task'])
                result = rs.train(ctx, 'full', job['action'], job['seed'])
                del ctx
            results.append(result)
            status = dict(status='running', completed=len(results), registered=36, last_id=ident,
                          original_completed_reused=11, recovery_completed=max(0, len(results) - 11))
            write(out / 'recovery_status.json', status, replace=True)
            print(json.dumps(status), flush=True)
        verify_original(original, inspection['original_files_sha256'])
        rs.verify(aggregate)
        assert sha(aggregate / 'predictions.json') == inspection['original_predictions_sha256']
        assert len(results) == 36
        write(aggregate / 'labels.json', dict(status='completed', results=results,
            predictions_sha256=inspection['original_predictions_sha256'], phase_seconds=None,
            measured_recovery_phase_seconds=time.perf_counter() - started, D_loaded=False, test_access=False,
            original_completed_reused=11, resumed_logical_trainings=1, newly_started_full_trainings=24,
            partial_original_timing_unknown=True))
        write(aggregate / 'labels_status.json', dict(status='completed', completed=36, registered=36,
            recovered=True, original_labels_status_unchanged=True), replace=True)
        write(out / 'recovery_status.json', dict(status='completed', completed=36, registered=36,
            original_completed_reused=11, resumed=1, newly_started=24, cumulative_full_trainings=367,
            full_total_timing_known=False), replace=True)
    except Exception:
        write(out / 'recovery_status.json', dict(status='failed', completed=len(results),
            error=traceback.format_exc(), original_artifacts_preserved=True), replace=True)
        raise


def audit(original, out):
    start = time.perf_counter()
    inspection = read(out / 'inspection.json')
    assert read(out / 'recovery_status.json')['status'] == 'completed'
    verify_original(original, inspection['original_files_sha256'])
    aggregate = out / 'aggregate'
    plan = read(out / 'recovery_plan.json')
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
        assert sha(aggregate / 'recovery' / name) == digest
    changed = {'labels_status.json', f'runs/{PARTIAL}/status.json'}
    for name, digest in inspection['original_files_sha256'].items():
        if name not in changed:
            assert sha(aggregate / name) == digest, name
    check = read(aggregate / 'recovery' / 'checkpoint_equivalence.json')
    assert check['status'] == 'PASS' and check['model_optimizer_numpy_torch_cuda_bitexact']
    assert check['diagnostic_optimizer_steps'] == 500 and not check['counts_as_new_full_training']
    continuation_receipt = read(aggregate / 'recovery' / 'continuation.json')
    timing = read(aggregate / 'runs' / PARTIAL / 'timing.json')
    assert timing['seconds'] is None and timing['pre_interruption_seconds'] is None
    assert continuation_receipt['total_logical_run_seconds'] is None
    assert continuation_receipt['measured_continuation_seconds'] == timing['measured_continuation_seconds'] > 0
    labels = read(aggregate / 'labels.json')
    assert labels['phase_seconds'] is None and labels['partial_original_timing_unknown']
    assert sum(r['seconds'] is None for r in labels['results']) == 1
    assert next(r for r in labels['results'] if r['id'] == PARTIAL)['seconds'] is None
    write(aggregate / 'recovery' / 'provenance_audit.json', dict(status='PASS',
        original_files_verified=len(inspection['original_files_sha256']),
        changed_copy_paths=sorted(changed), originals_unchanged=True,
        logical_matrix_complete=36, original_complete_reused=11, checkpoint_continuation=1,
        newly_started=24, cumulative_full_trainings=367, total_label_timing_known=False))
    write(out / 'audit_status.json', dict(status='running'))
    try:
        with (out / 'original_auditor.log').open('xb') as stream:
            subprocess.run([sys.executable, '-B', str(aggregate / 'audit_response_proxy_selection.py'),
                '--output', str(aggregate)], cwd=aggregate, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'), check=True)
        result = read(aggregate / 'final_audit.json')
        assert result['status'] == 'PASS' and result['cumulative_full_trainings'] == 367
        verify_original(original, inspection['original_files_sha256'])
        known = sum(r['seconds'] for r in labels['results'] if r['seconds'] is not None)
        known += timing['measured_continuation_seconds']
        write(out / 'recovery_audit.json', dict(status='PASS', original_auditor_unchanged=True,
            original_final_audit_sha256=sha(aggregate / 'final_audit.json'),
            archive_receipt_sha256=sha(aggregate / 'archive_receipt.json'),
            original_files_unchanged=True, restored_state_test=check,
            cumulative_full_trainings=367, original_complete_reused=11,
            resumed_logical_trainings=1, newly_started_full_trainings=24,
            known_label_run_seconds=known, unknown_pre_interruption_run_seconds=None,
            full_label_run_seconds=None, measured_recovery_phase_seconds=labels['measured_recovery_phase_seconds'],
            audit_wrapper_seconds=time.perf_counter() - start,
            D_access=False, test_access=False, advance=result['advance']))
        write(out / 'audit_status.json', dict(status='PASS'), replace=True)
    except Exception:
        write(out / 'audit_status.json', dict(status='failed', error=traceback.format_exc()), replace=True)
        raise


def launch(original, out, phase='run'):
    control = out / ('control' if phase == 'run' else 'audit_control')
    assert not control.exists()
    if phase == 'run':
        assert not (out / 'aggregate').exists()
    control.mkdir()
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (control / 'run.log').open('xb') as stream:
        process = subprocess.Popen([sys.executable, '-B', str(out / 'recover_proxy_selection.py'), phase,
            '--original', str(original), '--output', str(out)], cwd=out, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    write(control / 'launch.json', dict(pid=process.pid, started_at_utc=utc(), output_transport='regular_file'))
    print(json.dumps(dict(status='launched', pid=process.pid)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect', 'launch', 'run', 'audit', 'launch-audit'))
    parser.add_argument('--original', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.mode == 'launch-audit':
        launch(Path(args.original), Path(args.output), phase='audit')
    else:
        globals()[args.mode](Path(args.original), Path(args.output))
