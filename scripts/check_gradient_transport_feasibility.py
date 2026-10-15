"""Fixed-shape algebra and CPU operator costs; never loads research arrays."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import contextlib
import hashlib
import io
import json
import platform
import time
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def statistics(x, y, parent, groups):
    scale = 2 / (x.shape[1] * y.shape[0] * y.shape[2])
    residual = x @ parent - y
    return np.stack([np.swapaxes(x[:, groups == g], -1, -2) @
                     residual[:, groups == g] * scale for g in range(4)])


def curvatures(x, groups, outputs):
    scale = 2 / (x.shape[1] * x.shape[0] * outputs)
    return np.stack([np.swapaxes(x[:, groups == g], -1, -2) @
                     x[:, groups == g] * scale for g in range(4)])


def combine(weights, values):
    return (weights @ values.reshape(4, -1)).reshape(len(weights), *values.shape[1:])


def direct(x, y, groups, weights, theta):
    residual = x[None] @ theta - y[None]
    residual = residual * weights[:, groups][:, None, :, None]
    return np.swapaxes(x, -1, -2)[None] @ residual * (2 / y.size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage
    lock = json.loads((stage / 'source_lock.json').read_text(encoding='utf-8'))
    for item in lock['files']:
        assert sha(Path(item['path'])) == item['sha256']
    destination = stage / 'kernel_check.json'
    if destination.exists():
        raise FileExistsError('Preserve the first execution result')
    started = time.perf_counter()
    rows = []
    for seed in (92001, 92002, 92003):
        for dtype in (np.float32, np.float64):
            for count in (1, 8, 24):
                rng = np.random.default_rng(seed)
                xfull = rng.normal(size=(8, 1024, 66)).astype(dtype)
                xfull[:, :, 32] = 1
                xfull[:, :, 65] = 1  # Duplicate biases, as in the original coordinates.
                full_groups = np.arange(1024) % 4
                ids = rng.integers(0, 1024, size=256)
                x, groups = xfull[:, ids], full_groups[ids]
                y = rng.normal(size=(8, 256, 8)).astype(dtype)
                parent = (rng.normal(size=(8, 66, 8)) * .1).astype(dtype)
                theta = (parent[None] + rng.normal(size=(count, 8, 66, 8)) * .03).astype(dtype)
                theta[0] = parent
                weights = rng.uniform(.25, 1.75, size=(count, 4)).astype(dtype)
                weights /= weights.mean(axis=1, keepdims=True)
                weights[0] = 1
                if count > 1:
                    weights[-1] = 0
                tic = time.perf_counter()
                hbar = curvatures(xfull, full_groups, 8)
                moment_seconds = time.perf_counter() - tic
                tic = time.perf_counter()
                hw = combine(weights, hbar)
                combine_seconds = time.perf_counter() - tic
                q = statistics(x, y, parent, groups)
                affine = q - hbar @ parent
                reference = combine(weights, q) + hw @ (theta - parent)
                compact = hw @ theta + combine(weights, affine)
                batch_h = curvatures(x, groups, 8)
                exact_transport = combine(weights, q) + combine(weights, batch_h) @ (theta - parent)
                exact_direct = direct(x, y, groups, weights, theta)
                rtol, atol = ((3e-5, 3e-6) if dtype == np.float32 else (1e-11, 1e-12))
                np.testing.assert_allclose(compact, reference, rtol=rtol, atol=atol)
                np.testing.assert_allclose(exact_transport, exact_direct, rtol=rtol, atol=atol)
                np.testing.assert_allclose(compact[0], exact_direct[0], rtol=rtol, atol=atol)
                if count > 1:
                    np.testing.assert_array_equal(compact[-1], 0)

                def prepare():
                    return statistics(x, y, parent, groups) - hbar @ parent

                def pipeline():
                    return hw @ theta + combine(weights, prepare())

                functions = {
                    'direct_batch_gradient': lambda: direct(x, y, groups, weights, theta),
                    'cached_surrogate_gradient': lambda: hw @ theta + combine(weights, affine),
                    'parent_group_affine': prepare,
                    'parent_affine_plus_candidates': pipeline,
                }
                for function in functions.values():
                    for _ in range(3):
                        function()
                samples = {key: [] for key in functions}
                for repeat in range(7):
                    names = list(functions)
                    if repeat % 2:
                        names.reverse()
                    for name in names:
                        tic = time.perf_counter()
                        for _ in range(20):
                            functions[name]()
                        samples[name].append((time.perf_counter() - tic) / 20)
                medians = {key: float(np.median(value)) for key, value in samples.items()}
                rows.append(dict(seed=seed, dtype=np.dtype(dtype).name, candidates=count,
                    compact_error=float(np.max(np.abs(compact-reference))),
                    batch_transport_error=float(np.max(np.abs(exact_transport-exact_direct))),
                    parent_error=float(np.max(np.abs(compact[0]-exact_direct[0]))),
                    full_moment_seconds=moment_seconds, weight_combine_seconds=combine_seconds,
                    samples_seconds=samples, median_seconds=medians,
                    pipeline_to_direct_ratio=medians['parent_affine_plus_candidates']/medians['direct_batch_gradient']))
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    # Exact byte counts, no archive or allocator compression assumptions.
    storage = dict(parameters=8*66*8, steps=3000, groups=4,
        float32_original_q_plus_parent_bytes=3000*(4+1)*8*66*8*4,
        float32_affine_only_bytes=3000*4*8*66*8*4,
        float32_static_group_h_bytes=4*8*66*66*4,
        float32_full_batch_group_h_trace_bytes=3000*4*8*66*66*4,
        float32_stream_affine_bytes=4*8*66*8*4)
    result = dict(status='PASS', source_lock_sha256=sha(stage/'source_lock.json'),
        scope='synthetic_single_step_CPU_operators_not_training_or_end_to_end',
        rows=rows, storage=storage, wall_seconds=time.perf_counter()-started,
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        numpy_config=config.getvalue(), new_full_trainings=0, cumulative_full_trainings=434,
        real_arrays_accessed=False, optimizer_updates=0, real_experiments_enabled=False)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], rows=len(rows), storage=storage,
                         wall_seconds=result['wall_seconds']), indent=2))


if __name__ == '__main__':
    main()
