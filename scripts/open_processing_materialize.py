"""Trusted in-memory parent adapter for the isolated numeric-program worker."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import open_processing_program as program
import open_processing_sandbox as sandbox


WALL_SECONDS = 150
MAX_STDERR_CHARACTERS = 8000
MIN_EFFECTIVE_WINDOWS = 256
MAX_WEIGHT_MULTIPLIER = 20


class ProgramExecutionError(RuntimeError):
    """An unsuccessful attempt, with the original process evidence attached."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _array_hash(value):
    array = np.asarray(value)
    header = f'{array.dtype.str}:{array.shape}:'.encode('ascii')
    return _sha256(header + array.tobytes(order='C'))


def _fail(receipt, message):
    receipt.update(status='FAIL', error=message)
    raise ProgramExecutionError(message, receipt)


def _capture(receipt, stdout, stderr):
    stdout = stdout or b''
    stderr = stderr or b''
    if isinstance(stderr, bytes):
        stderr = stderr.decode('utf-8', errors='replace')
    receipt.update(stdout_bytes=len(stdout), stdout_sha256=_sha256(stdout),
                   stderr=stderr[:MAX_STDERR_CHARACTERS],
                   stderr_characters=len(stderr),
                   stderr_truncated=len(stderr) > MAX_STDERR_CHARACTERS)


def run_worker(source, kind, action, rows, state=None,
               include_weights=False, include_target=False):
    """Return (metadata, arrays, receipt); never retry or bypass isolation."""
    receipt = dict(status='FAIL', kind=kind, action=action, returncode=None,
                   timed_out=False, seconds=0., source_sha256=None,
                   rows_sha256=None, state_sha256=None, request_sha256=None,
                   wall_limit_seconds=WALL_SECONDS)
    _capture(receipt, b'', b'')
    try:
        if not isinstance(source, str):
            raise ValueError('source must be text')
        receipt['source_sha256'] = _sha256(source.encode('utf-8'))
        if kind not in {'measurement', 'pipeline'} or action not in {'fit', 'apply'}:
            raise ValueError('unknown kind or action')
        rows = np.asarray(rows)
        if (rows.ndim != 2 or rows.shape[1] != sandbox.CHANNELS
                or rows.dtype.kind not in 'biuf' or not np.isfinite(rows).all()):
            raise ValueError('rows must be a finite real array with 370 channels')
        receipt['rows_sha256'] = _array_hash(rows)
        metadata = dict(source=source, kind=kind, action=action,
                        include_weights=bool(include_weights),
                        include_target=bool(include_target))
        if action == 'apply':
            state = program.validate_state(state, max_bytes=sandbox.MAX_STATE_BYTES)
        tree, arrays = sandbox.encode_state(state)
        receipt['state_sha256'] = _sha256(sandbox.pack(dict(state=tree), arrays))
        if action == 'apply':
            metadata['state'] = tree
        arrays['rows'] = rows
        payload = sandbox.pack(metadata, arrays)
        receipt['request_sha256'] = _sha256(payload)
    except Exception as exc:
        _fail(receipt, f'Invalid worker request: {type(exc).__name__}: {exc}')
    environment = os.environ.copy()
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS'):
        environment[variable] = '1'
    command = [sys.executable, '-B', str(Path(__file__).with_name('open_processing_sandbox.py')), 'worker']
    start = time.perf_counter()
    try:
        result = subprocess.run(command, input=payload, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=WALL_SECONDS,
                                close_fds=True, env=environment, check=False)
    except subprocess.TimeoutExpired as exc:
        receipt.update(seconds=time.perf_counter() - start, timed_out=True)
        _capture(receipt, exc.stdout, exc.stderr)
        _fail(receipt, 'Worker exceeded the 150 second wall limit')
    except Exception as exc:
        receipt['seconds'] = time.perf_counter() - start
        _fail(receipt, f'Worker launch failed: {type(exc).__name__}: {exc}')
    receipt.update(seconds=time.perf_counter() - start, returncode=result.returncode)
    _capture(receipt, result.stdout, result.stderr)
    if result.returncode != 0:
        _fail(receipt, f'Worker exited with code {result.returncode}')
    try:
        metadata, arrays = sandbox.unpack(result.stdout)
        if not isinstance(metadata, dict):
            raise ValueError('worker metadata must be an object')
        receipt['worker_metadata'] = metadata
        if metadata.get('status') != 'PASS':
            raise ValueError(f'worker status is {metadata.get("status")!r}')
    except Exception as exc:
        _fail(receipt, f'Invalid worker result: {type(exc).__name__}: {exc}')
    receipt['status'] = 'PASS'
    return metadata, arrays, receipt


def fit_program(source, kind, a):
    """Fit only the supplied A rows and validate the returned numeric state."""
    metadata, arrays, receipt = run_worker(source, kind, 'fit', a)
    try:
        tree = metadata['state']
        state = sandbox.decode_state(tree, arrays)
        state = program.validate_state(state, max_bytes=sandbox.MAX_STATE_BYTES)
        _, referenced = sandbox.encode_state(state)
        # decode_state can otherwise silently ignore unexpected archive fields.
        def keys(node):
            if node['type'] == 'array':
                return [node['key']]
            if node['type'] == 'dict':
                return [key for child in node['items'].values() for key in keys(child)]
            if node['type'] == 'sequence':
                return [key for child in node['items'] for key in keys(child)]
            return []
        if set(keys(tree)) != set(arrays) or len(referenced) != len(arrays):
            raise ValueError('unexpected or duplicated state array fields')
        receipt['fitted_state_sha256'] = _sha256(sandbox.pack(dict(state=tree), arrays))
    except Exception as exc:
        _fail(receipt, f'Invalid fitted state: {type(exc).__name__}: {exc}')
    return state, receipt


def apply_program(source, kind, rows, state, include_weights=False, include_target=False):
    """Materialize and strictly validate arrays without coercion or repair."""
    metadata, arrays, receipt = run_worker(source, kind, 'apply', rows, state,
                                          include_weights, include_target)
    try:
        n = len(rows) - sandbox.LENGTH - sandbox.HORIZON + 1
        if n <= 0 or metadata.get('windows') != n:
            raise ValueError('invalid output window count')
        if kind == 'measurement':
            expected = dict(measurement=((n,), np.dtype('float64')))
        else:
            expected = dict(model_input=((n, 32, 370), np.dtype('float32')),
                            offset=((n, 370), np.dtype('float32')),
                            scale=((n, 370), np.dtype('float32')),
                            sampling=((n,), np.dtype('float64')),
                            loss=((n,), np.dtype('float64')))
            if include_target:
                expected['target'] = ((n, 8, 370), np.dtype('bool'))
        if set(arrays) != set(expected):
            raise ValueError('output fields differ from the required fields')
        for name, (shape, dtype) in expected.items():
            value = arrays[name]
            if (not isinstance(value, np.ndarray) or value.shape != shape
                    or value.dtype != dtype or not np.isfinite(value).all()):
                raise ValueError(f'{name} must have shape {shape}, dtype {dtype}, and finite values')
        if kind == 'pipeline':
            if not (arrays['scale'] > 0).all():
                raise ValueError('scale must be positive')
            if (arrays['sampling'] < 0).any() or (arrays['loss'] < 0).any():
                raise ValueError('weights must be nonnegative')
        receipt['array_sha256'] = {name: _array_hash(value) for name, value in arrays.items()}
    except Exception as exc:
        _fail(receipt, f'Invalid materialized arrays: {type(exc).__name__}: {exc}')
    return arrays, receipt


def normalize_weights(sampling, loss):
    """Normalize finite weights and reject insufficient support or concentration."""
    sampling = np.asarray(sampling)
    loss = np.asarray(loss)
    if (sampling.ndim != 1 or sampling.shape != loss.shape or sampling.size == 0
            or sampling.dtype.kind not in 'biuf' or loss.dtype.kind not in 'biuf'
            or not np.isfinite(sampling).all() or not np.isfinite(loss).all()
            or (sampling < 0).any() or (loss < 0).any()):
        raise ValueError('sampling and loss must be same-sized finite nonnegative vectors')
    sampling = sampling.astype(np.float64)
    loss = loss.astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        total = sampling.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError('sampling total must be finite and positive')
        probabilities = sampling / total
        mean_loss = np.sum(probabilities * loss) / probabilities.sum()
        if not np.isfinite(mean_loss) or mean_loss <= 0:
            raise ValueError('expected loss must be finite and positive')
        normalized_loss = loss / mean_loss
        effective = int(np.count_nonzero((sampling > 0) & (loss > 0)))
        maximum_sampling = float(np.max(probabilities) * sampling.size)
        maximum_loss = float(np.max(normalized_loss))
        if effective < MIN_EFFECTIVE_WINDOWS:
            raise ValueError('at least 256 effective windows are required')
        if maximum_sampling > MAX_WEIGHT_MULTIPLIER:
            raise ValueError('sampling multiplier exceeds 20')
        if not np.isfinite(normalized_loss).all() or maximum_loss > MAX_WEIGHT_MULTIPLIER:
            raise ValueError('normalized loss exceeds 20')
        stats = dict(n=int(sampling.size), positive_effective=effective,
                     positive_fraction=effective / sampling.size,
                     ess_sampling=float(1 / np.sum(probabilities ** 2)),
                     max_sampling_multiplier=maximum_sampling, max_loss=maximum_loss,
                     loss_std_under_p=float(np.sqrt(np.sum(probabilities * (normalized_loss - 1) ** 2))))
    return probabilities, normalized_loss, stats
