"""Independent saved-asset audit and owned-file archive for proxy engineering."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import tarfile
import time
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coordinates(state):
    # Independent mapping from original state_dict; do not call prototype packer.
    return torch.stack([torch.cat([state[f'seasonal.{c}.weight'].T,
        state[f'seasonal.{c}.bias'][None], state[f'trend.{c}.weight'].T,
        state[f'trend.{c}.bias'][None]]) for c in range(8)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    started = time.perf_counter()
    plan, complete = read(out / 'plan.json'), read(out / 'benchmark_result.json')
    assert complete['status'] == 'completed' and len(complete['results']) == len(plan['jobs']) == 18
    assert read(out / 'benchmark_status.json')['status'] == 'completed'
    for name, digest in plan['code_sha256'].items():
        assert sha(out / name) == digest
    old = Path(plan['input_root'])
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    h = module('old_hydraulic', old / 'hydraulic_signal_validation.py')
    original = h.verify(old)
    arrays, params, _ = h.prepared(old, original)
    core = module('old_core', original['frozen_core'])
    proposal = module('old_proposal', old / 'proposal_median3.py')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    values = torch.tensor(arrays['validation'], device='cuda')
    starts = torch.tensor(arrays['validation_starts'], device='cuda')
    offset = torch.arange(40, device='cuda')
    rows, initialization, sampling = [], {}, {}
    for expected, recorded in zip(plan['jobs'], complete['results']):
        directory = out / 'runs' / expected['id']
        r = read(directory / 'result.json')
        assert r == recorded
        assert all(r[key] == value for key, value in expected.items())
        assert read(directory / 'status.json')['status'] == 'completed' and r['steps'] == 3000
        assert r['plan_sha256'] == sha(out / 'plan.json')
        assert r['process_seconds'] > 0 and r['peak_gpu_memory_bytes'] > 0
        assert sum(r['timings'].values()) <= r['process_seconds'] + 1e-6
        assert [x['step'] for x in r['checkpoints']] == plan['grid']
        assert all(a['elapsed_seconds'] < b['elapsed_seconds'] for a, b in zip(r['checkpoints'], r['checkpoints'][1:]))
        for name, digest in r['assets_sha256'].items():
            assert sha(directory / name) == digest
        for step in [0] + plan['grid']:
            snap = torch.load(directory / ('initial.pt' if step == 0 else f'step_{step}.pt'), weights_only=True, map_location='cpu')
            packed = coordinates(snap['model'])
            assert torch.equal(packed, snap['original_coordinates']) and bool(torch.isfinite(packed).all())
            assert snap['step'] == step
            if step:
                assert len(snap['optimizer']['state']) == 32
                for state in snap['optimizer']['state'].values():
                    assert state['step'].item() == step
                    assert bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all())
            else:
                assert not snap['optimizer']['state']
                digest = hashlib.sha256(packed.numpy().tobytes()).hexdigest()
                assert digest == r['initial_sha256']
                assert initialization.setdefault(r['seed'], digest) == digest
        if r['engine'] == 'direct':
            generator = np.random.default_rng(r['seed'] + 1000)
            cdf = np.cumsum(arrays['probability'])
            cdf[-1] = 1.
            expected_draws = np.zeros(len(cdf), dtype=np.int64)
            for _ in range(3000):
                np.add.at(expected_draws, np.searchsorted(cdf, generator.random(256), side='right'), 1)
            with np.load(directory / 'sampling.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['draw_count'], expected_draws)
            assert generator.bit_generator.state == snap['rng_numpy']
            digest = hashlib.sha256(expected_draws.tobytes()).hexdigest()
            assert sampling.setdefault(r['seed'], digest) == digest
        if r['engine'] == 'proxy' and r['action'] == 'PARENT':
            assert r['selection_losses'] == {} and r['selected_step'] is None
        else:
            scores = r['selection_losses']
            selected = min(plan['grid'], key=lambda step: scores[str(step)])
            assert selected == r['selected_step']
            state = torch.load(directory / f'step_{selected}.pt', weights_only=True, map_location='cpu')['model']
            if r['engine'] == 'proxy':
                parent = out / 'runs' / f'ENG_direct_PARENT_s{r["seed"]}'
                proxy = out / 'runs' / f'ENG_proxy_PARENT_s{r["seed"]}'
                a = torch.load(parent / f'step_{selected}.pt', weights_only=True, map_location='cpu')['model']
                b = torch.load(proxy / f'step_{selected}.pt', weights_only=True, map_location='cpu')['model']
                state = {key: a[key] + (state[key] - b[key]) for key in state}
            saved_state = torch.load(directory / 'selected_prediction_model.pt', weights_only=True, map_location='cpu')
            assert all(torch.equal(value, saved_state[key]) for key, value in state.items())
            model = core.DLinear(dict(original['training'], channels=8)).cuda().eval()
            model.load_state_dict(state)
            errors = []
            with torch.no_grad():
                for lo in range(0, len(starts), 2048):
                    batch = values[starts[lo:lo + 2048, None] + offset]
                    x = batch[:, :32]
                    if r['action'] != 'PARENT':
                        x = proposal.transform(x, params, 'P1', r['action'] == 'CP1')
                    errors.append((model(x) - batch[:, 32:]).cpu().numpy())
            errors = np.concatenate(errors)
            with np.load(directory / 'selected_errors.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(errors, saved['error'])
            assert float(np.mean(errors.astype(np.float64) ** 2)) == scores[str(selected)]
        rows.append(dict(id=r['id'], seconds=r['process_seconds'], engine=r['engine'], action=r['action']))
    for name, digest in plan['inputs_sha256'].items():
        assert sha(old / name) == digest
    medians = {f'{engine}_{action}': statistics.median(row['seconds'] for row in rows if row['engine'] == engine and row['action'] == action)
               for engine in ('direct', 'proxy') for action in plan['actions']}
    shared, parent = medians['proxy_PARENT'], medians['direct_PARENT']
    proxy_cost = shared + medians['proxy_P1'] + medians['proxy_CP1']
    direct_cost = medians['direct_P1'] + medians['direct_CP1']
    audit_seconds = time.perf_counter() - started
    result = dict(status='PASS', runs_verified=18, selected_residual_arrays_verified=15,
        full_direct_trainings=9, proxy_rollouts=9, cumulative_full_trainings=304,
        cost_medians_seconds=medians, two_candidate_proxy_seconds=proxy_cost,
        two_candidate_direct_seconds=direct_cost, cold_proxy_seconds=parent + proxy_cost,
        cold_direct_seconds=parent + direct_cost, cost_gate=proxy_cost < direct_cost,
        cost_gate_scope='two registered candidates; compute feasibility only; no accuracy or decision claim',
        audit_seconds=audit_seconds, original_inputs_unchanged=True, test_access=False)
    write(out / 'final_audit.json', result)
    # Only this new, owned output tree: never follow links or include original data.
    allowed = {'.py', '.json', '.md', '.log', '.pt', '.npz'}
    paths = sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in allowed and p.name != 'audit.log')
    assert all(not p.is_symlink() and 'data' not in p.relative_to(out).parts for p in paths)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in paths}
    archive = out / 'response_proxy_engineering_20261013.tar.gz'
    with tarfile.open(archive, 'x:gz') as tar:
        for name in hashes:
            tar.add(out / name, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(hashes)
        for name, digest in hashes.items():
            assert hashlib.file_digest(tar.extractfile(name), 'sha256').hexdigest() == digest
    write(out / 'archive_receipt.json', dict(status='PASS', file=archive.name,
        bytes=archive.stat().st_size, sha256=sha(archive), members_sha256=hashes))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
