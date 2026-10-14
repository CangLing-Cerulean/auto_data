"""Create-only SOURCE forecasts for one registered task and seed.

The caller owns source preparation, protocol verification and the phase ledger.
This module uses the unchanged SOURCE adapter and never trains a model.
"""
import hashlib
from pathlib import Path
import time
import traceback

import numpy as np
import torch

import response_proxy as rp
import response_proxy_existing as existing
import response_proxy_selection as frozen
import source_dlinear_adapter as adapter


GRID = frozen.GRID


def _component(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('Task and action IDs must be single path components')
    return value


def _error_summary(error, loss):
    if not isinstance(error, np.ndarray) or error.dtype != np.float32 or not np.isfinite(error).all():
        raise ValueError('Evaluation must return finite float32 error arrays')
    nmse = float(np.mean(error.astype(np.float64) ** 2))
    if not np.isfinite(nmse) or nmse != float(loss):
        raise ValueError('Evaluation NMSE differs from complete error array')
    return dict(shape=list(error.shape), dtype=str(error.dtype),
                byte_sha256=hashlib.sha256(error.tobytes()).hexdigest(), nmse=nmse,
                nmae=float(np.mean(np.abs(error.astype(np.float64)))))


def run_source(ctx, seed):
    """Return all action forecasts and actual, unamortized task/seed costs.

    A pre-existing task/seed directory is rejected without writing into it.
    B alone selects checkpoints; C measures the selected response against the
    full parent's selected C loss. Error assets contain summaries, not arrays.
    """
    task = _component(ctx.task)
    started = time.perf_counter()
    directory = Path(ctx.out) / 'source' / f'{task}_s{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        frozen.tick()
        actions = tuple(ctx.actions)
        if not actions or len(set(actions)) != len(actions) or 'PARENT' in actions:
            raise ValueError('Expected unique registered candidate actions')
        for action in actions:
            _component(action)
        training = ctx.plan['training']
        if (ctx.channels != 8 or training['input_length'] != 32 or training['prediction_length'] != 8
                or training['moving_average_kernel'] != 25):
            raise ValueError('SOURCE requires the frozen eight-channel 32-to-8 task with kernel 25')
        model = ctx.model(seed)
        initial_sha = hashlib.sha256(rp.parameters(model).detach().cpu().numpy().tobytes()).hexdigest()
        moment_directory = directory / 'moments'
        moment_directory.mkdir()
        begin = frozen.tick()
        moments = {}
        for action in ('PARENT',) + actions:
            moment = ctx.moments(model, action)
            packed = {key: getattr(moment, key).detach().cpu().numpy()
                      for key in ('gram', 'cross', 'target_square')}
            if any(not np.isfinite(value).all() for value in packed.values()):
                raise ValueError('Nonfinite SOURCE moments')
            frozen.arrays_save(moment_directory / f'{action}.npz', **packed)
            moments[action] = packed
            del moment
        moment_seconds = frozen.tick() - begin

        begin = frozen.tick()
        parent = Path(ctx.parent_dir('full', seed))
        parent_result = frozen.read(parent / 'result.json')
        if parent_result['initial_sha256'] != initial_sha:
            raise ValueError('Parent initialization differs from the context model')
        baseline = float(parent_result['selected_losses']['C'])
        if not np.isfinite(baseline) or baseline <= 0:
            raise ValueError('Parent selected C loss must be finite and positive')
        cpu_model = ctx.core.DLinear(dict(training, channels=ctx.channels))
        states, parent_hashes = {}, {'result.json': frozen.sha(parent / 'result.json')}
        for step in GRID:
            path = parent / f'step_{step}.pt'
            snapshot = torch.load(path, map_location='cpu', weights_only=True)
            state = adapter.parent_coordinates(cpu_model, snapshot)
            if state['step'] != step:
                raise ValueError('Parent checkpoint step differs from frozen grid')
            states[step] = state
            parent_hashes[path.name] = frozen.sha(path)
            del snapshot
        parent_state_seconds = frozen.tick() - begin

        action_seconds = 0.
        for action in actions:
            begin = frozen.tick()
            action_directory = directory / action
            action_directory.mkdir()
            delta = np.zeros_like(states[GRID[0]]['theta'])
            curves, scores, diagnostics, summaries = {}, {}, [], {}
            for step in GRID:
                delta, checks = adapter.quadratic_segment(
                    moments['PARENT']['gram'], moments['PARENT']['cross'],
                    moments[action]['gram'], moments[action]['cross'], states[step], delta)
                coordinates = (states[step]['theta'] + delta).astype(np.float32)
                frozen.arrays_save(action_directory / f'step_{step}.npz',
                                   delta=delta, coordinates=coordinates)
                rp.load_parameters(model, torch.tensor(coordinates, device=next(model.parameters()).device))
                curves[step], summaries[step] = {}, {}
                for split in ('B', 'C'):
                    loss, error = ctx.evaluate(model, action, split)
                    summary = _error_summary(error, loss)
                    existing.arrays_save(action_directory / f'step_{step}_{split}_errors.npz', error=error)
                    summaries[step][split] = summary
                    curves[step][split] = summary['nmse']
                    del error
                scores[step] = curves[step]['B']
                diagnostics.append(dict(step=step, blocks=checks,
                    clipped_negative_eigenvalues=sum(c['clipped_negative_eigenvalues'] for c in checks),
                    max_roundoff_tolerance=max(c['roundoff_tolerance'] for c in checks)))
            selected = rp.select_checkpoint(scores, GRID)
            gain = 100. * (baseline - curves[selected]['C']) / baseline
            result = dict(method='SOURCE', task=task, seed=seed, action=action,
                selected_step=selected, baseline_C=baseline, selected_losses=curves[selected],
                gain_percent=gain, category=frozen.category(gain), curves=curves,
                error_summaries=summaries, numerical_diagnostics=diagnostics,
                assets_sha256={p.name: frozen.sha(p) for p in sorted(action_directory.iterdir())},
                new_full_trainings=0)
            frozen.write(action_directory / 'result.json', result)
            seconds = frozen.tick() - begin
            frozen.write(action_directory / 'timing.json', dict(seconds=seconds,
                boundary='action directory creation through propagation/evaluation/result/assets/hash; timing receipt excluded'))
            records.append(dict(result, seconds=seconds))
            action_seconds += seconds

        assets = {p.relative_to(directory).as_posix(): frozen.sha(p)
                  for p in sorted(directory.rglob('*')) if p.is_file()}
        booked = moment_seconds + parent_state_seconds + action_seconds
        result = dict(status='completed', method='SOURCE', task=task, seed=seed,
            records=records, moment_seconds=moment_seconds,
            parent_state_seconds=parent_state_seconds, action_seconds=action_seconds,
            initial_sha256=initial_sha,
            parent_directory=str(parent), parent_assets_sha256=parent_hashes,
            assets_sha256=assets, new_full_trainings=0,
            cost_accounting='Moment building/saving, parent loading/validation/hashing, and action intervals are disjoint. '
                'Total includes directory creation, model initialization, action timing receipts, final asset hashing and result.json write; no seed amortization. '
                'Context preparation and real parent training are caller-owned costs. '
                'Only the final timing.json receipt write is excluded from this invocation; Python startup and network are outside it.')
        frozen.write(directory / 'result.json', result)
        total_seconds = frozen.tick() - started
        other_seconds = total_seconds - booked
        frozen.write(directory / 'timing.json', dict(total_seconds=total_seconds,
            other_seconds=other_seconds,
            boundary='directory creation through final result.json write; only this timing receipt write excluded'))
        return dict(result, total_seconds=total_seconds, other_seconds=other_seconds)
    except Exception:
        frozen.write(directory / 'failure.json', dict(status='failed', task=task, seed=seed,
            completed_actions=len(records), seconds=time.perf_counter() - started,
            failure_clock='CPU perf_counter; no CUDA synchronization attempted',
            error=traceback.format_exc(), new_full_trainings=0))
        raise
