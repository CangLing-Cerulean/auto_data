"""Independent source-file reconstruction and tensor checks for FTR1.

Only generic forward/moment/residual helpers are reused from the audited r2
auditor. Its old partition, job IDs, budget rules and phase entry points are
never called here.
"""
import ast
import csv
import hashlib
import math
from pathlib import Path
import random

import numpy as np
import torch

import audit_response_proxy_existing_r2 as previous

read, sha, write = previous.read, previous.sha, previous.write
coordinates, snap, mse, gamma = previous.coordinates, previous.snap, previous.mse, previous.gamma
GRID = (500, 1000, 1500, 2000, 2500, 3000)
METHODS = ('M1', 'RAW', 'SOURCE', 'SHORT')
POLICIES = ('PARENT', 'SIZE_DROP', 'LOSS_DROP')


def ident(task, engine, action, seed):
    return f'FTR1_{task}_{engine}_{action}_s{seed}'


def verify(out):
    plan = read(out / 'plan.json')
    assert str(out) == plan['output_root'] == '/root/auto_data_experiments/20261014_file_transfer_r1'
    sync = read(out / 'source_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    assert sync['plan_sha256'] == sha(out / 'plan.json')
    assert sync['registration_sha256'] == sha(out / 'task_registration.json')
    for name, digest in plan['code_sha256'].items():
        assert Path(name).name == name and sha(out / name) == digest
    assert sha(plan['core']) == plan['core_sha256'] == '4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'
    data = plan['dataset']
    for name in ('manifest', 'train'):
        assert sha(data[name]) == data[name + '_sha256']
    manifest = read(data['manifest'])
    train = next(s for s in manifest['splits'] if s['name'] == 'train')
    registration = read(out / 'task_registration.json')
    assert data['manifest_sha256'] == registration['manifest_sha256'] == '372d99fd9188151d3429171c1978c3ea13a34b735bf3ec7d031d8b6845c5ecc2'
    assert data['train_sha256'] == registration['train_sha256'] == train['sha256']
    names = sorted((s['series_id'] for s in train['series_coverage']),
                   key=lambda n: (hashlib.sha256(('file-transfer-r1|' + n).encode('utf-8')).hexdigest(), n))
    assert len(set(names)) == 23 and train['rows'] == 32802
    expected = []
    for group in range(3):
        files = {role: sorted(n for i, n in enumerate(names) if i % 3 == (group + offset) % 3)
                 for offset, role in enumerate(('A', 'B', 'C'))}
        expected.append(dict(id=f'F{group}', files=files, actions=[dict(id=f'DROP_{i:02d}', removed_source_file=n)
                        for i, n in enumerate(files['A'])]))
    assert plan['tasks'] == expected == read(out / 'task_registration.json')['tasks']
    assert plan['seeds'] == [2026102001, 2026102002, 2026102003]
    assert plan['training'] == dict(input_length=32, prediction_length=8, moving_average_kernel=25)
    assert plan['grid'] == list(GRID) and data['sensors'] == manifest['sensor_columns']
    assert plan['cumulative_before'] == 409 and plan['expected_cumulative_after'] == 487
    jobs = set()
    for task in expected:
        for seed in plan['seeds']:
            for engine in ('full', 'proxy'):
                jobs.add((ident(task['id'], engine, 'PARENT', seed), task['id'], engine, 'PARENT', seed, 'parents'))
            for action in task['actions']:
                for engine, phase in (('proxy', 'forecast'), ('short', 'forecast'), ('full', 'labels')):
                    jobs.add((ident(task['id'], engine, action['id'], seed), task['id'], engine, action['id'], seed, phase))
    assert len(plan['jobs']) == len(jobs) == 225
    assert jobs == {(j['id'], j['task'], j['engine'], j['action'], j['seed'], j['phase']) for j in plan['jobs']}
    prep = read(out / 'preparation.json')
    assert prep['status'] == 'PASS' and prep['plan_sha256'] == sha(out / 'plan.json')
    assert prep['registration_sha256'] == sha(out / 'task_registration.json')
    assets(out, prep['assets_sha256'])
    return plan, prep


def assets(directory, hashes):
    for name, digest in hashes.items():
        path = directory / name
        assert path.resolve().is_relative_to(directory.resolve()) and sha(path) == digest, name


def reconstruct(out, plan, task):
    data = plan['dataset']
    manifest = read(data['manifest'])
    with Path(data['train']).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == manifest['columns_raw']
        rows = list(reader)
    assert len(rows) == 32802
    raw = np.array([[float(r[c]) for c in data['sensors']] for r in rows], dtype=np.float64)
    assert np.isfinite(raw).all()
    files = np.array([r['source_file'] for r in rows])
    source = np.array([int(r['source_row']) for r in rows], dtype=np.int64)
    times = np.array([r['datetime'] for r in rows], dtype='datetime64[s]').astype(np.int64)
    coverage = next(s for s in manifest['splits'] if s['name'] == 'train')['series_coverage']
    assert set(files) == {s['series_id'] for s in coverage}
    for item in coverage:
        ids = sorted(np.flatnonzero(files == item['series_id']), key=lambda i: source[i])
        np.testing.assert_array_equal(source[ids], np.arange(item['source_row_start_inclusive'], item['source_row_end_exclusive']))
        assert len(ids) == item['rows'] and np.all(np.diff(times[ids]) > 0)
        assert times[ids[0]] == np.datetime64(item['first_time'], 's').astype(np.int64)
        assert times[ids[-1]] == np.datetime64(item['last_time'], 's').astype(np.int64)
    ids = {role: np.array(sorted((i for i, name in enumerate(files) if name in names),
                                key=lambda i: (files[i], source[i])), dtype=np.int64)
           for role, names in task['files'].items()}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(ids.values()))), np.arange(len(raw)))
    assert all(set(files[ids[a]]).isdisjoint(files[ids[b]]) for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')))
    mean, scale = raw[ids['A']].mean(0), raw[ids['A']].std(0)
    scale[scale == 0] = 1.
    with np.load(out / 'prepared' / f'{task["id"]}_ABC.npz', allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    np.testing.assert_array_equal(arrays['mean'], mean)
    np.testing.assert_array_equal(arrays['scale'], scale)
    summary = next(s for s in read(out / 'preparation.json')['tasks'] if s['task'] == task['id'])
    for role, subset in ids.items():
        for key, value in [('base_rows', subset), ('files', files[subset]), ('source_rows', source[subset]),
                           ('times', times[subset]), ('values', ((raw[subset] - mean) / scale).astype(np.float32))]:
            np.testing.assert_array_equal(arrays[f'{role}_{key}'], value)
        starts = np.arange(len(subset) - 39, dtype=np.int64)
        valid = np.ones(len(starts), dtype=bool)
        for offset in range(39):
            left, right = subset[starts + offset], subset[starts + offset + 1]
            valid &= (files[left] == files[right]) & (source[right] - source[left] == 1)
            valid &= (times[right] > times[left]) & (times[right] - times[left] <= 30)
        starts = starts[valid]
        np.testing.assert_array_equal(arrays[f'{role}_starts'], starts)
        assert len(starts) >= 256
        counts = {n: int(np.count_nonzero(files[subset[starts]] == n)) for n in task['files'][role]}
        assert all(c > 0 for c in counts.values()) and summary['source_window_counts'][role] == counts
        assert summary['rows'][role] == len(subset) and summary['windows'][role] == len(starts)
        elapsed = times[subset[starts + 39]] - times[subset[starts + 31]]
        assert summary['forecast_elapsed'][role] == dict(min_seconds=int(elapsed.min()), max_seconds=int(elapsed.max()))
    assert summary['mean'] == mean.tolist() and summary['scale'] == scale.tolist()
    window_files = arrays['A_files'][arrays['A_starts']]
    probabilities = {'PARENT': np.full(len(window_files), 1. / len(window_files))}
    for action in task['actions']:
        removed = window_files == action['removed_source_file']
        assert removed.any() and np.count_nonzero(~removed) >= 256
        p = np.zeros(len(removed), dtype=np.float64)
        p[~removed] = 1. / np.count_nonzero(~removed)
        probabilities[action['id']] = p
        with np.load(out / 'prepared' / f'{task["id"]}_{action["id"]}.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['removed'], removed)
    for action, p in probabilities.items():
        with np.load(out / 'prepared' / f'{task["id"]}_{action}.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['probability'], p)
    return arrays, probabilities


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


class Context(previous.AuditContext):
    def __init__(self, out, plan, task):
        self.out, self.plan, self.task, self.spec = out, plan, task['id'], task
        self.arrays, self.probabilities = reconstruct(out, plan, task)
        self.values = {s: torch.tensor(self.arrays[f'{s}_values'], device='cuda') for s in ('A', 'B', 'C')}
        self.starts = {s: torch.tensor(self.arrays[f'{s}_starts'], device='cuda') for s in self.values}
        self.core = previous.module('file_transfer_audit_core', plan['core'])
        self.channels = 8
        self.model = self.core.DLinear(dict(plan['training'], channels=8)).cuda().eval()
        self.states_checked = self.residuals_checked = self.moments_checked = 0
        self.replay_steps = self.replay_trajectories = 0
        self.moment_reference = {}

    def directory(self, engine, action, seed):
        return self.out / 'runs' / ident(self.task, engine, action, seed)

    def state(self, packed):
        result = {}
        for c in range(8):
            for kind, weight, bias in (('seasonal', packed[c, :32].T, packed[c, 32]),
                                       ('trend', packed[c, 33:65].T, packed[c, 65])):
                result[f'{kind}.{c}.weight'] = weight
                result[f'{kind}.{c}.bias'] = bias
        return result

    def run(self, engine, action, seed):
        directory = self.directory(engine, action, seed)
        result, status = read(directory / 'result.json'), read(directory / 'status.json')
        assert result['status'] == status['status'] == 'completed'
        assert result['id'] == status['id'] == directory.name and result['plan_sha256'] == sha(self.out / 'plan.json')
        assert (result['task'], result['engine'], result['action'], result['seed']) == (self.task, engine, action, seed)
        assets(directory, result['assets_sha256'])
        maximum = read(self.out / 'calibration.json')['tasks'][self.task]['short_steps'] if engine == 'short' else 3000
        assert maximum in (GRID[:-1] if engine == 'short' else (3000,))
        steps = [k for k in GRID if k <= maximum]
        assert result['steps'] == result['steps_completed'] == status['steps_completed'] == maximum
        assert result['checkpoints'] == steps and set(result['prefix_seconds']) == set(map(str, steps))
        prefixes = [result['prefix_seconds'][str(k)] for k in steps]
        assert 0 < prefixes[0] and all(a < b for a, b in zip(prefixes, prefixes[1:]))
        assert prefixes[-1] < read(directory / 'timing.json')['seconds']
        initial = snap(self.directory('full', 'PARENT', seed), 0)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        registered = self.core.DLinear(dict(self.plan['training'], channels=8))
        equal(initial['model'], registered.state_dict())
        equal(initial['rng_torch'], torch.get_rng_state())
        equal(initial['rng_cuda'], torch.cuda.get_rng_state())
        replay = registered.cuda()
        replay_opt = torch.optim.Adam(replay.parameters(), lr=.001, weight_decay=0.)
        replay_moments = [torch.from_numpy(v).cuda() for v in self.moments(action)] if engine == 'proxy' else None
        rng = np.random.default_rng(seed + 1000)
        counts = np.zeros(len(self.probabilities[action]), dtype=np.int64)
        cdf = np.cumsum(self.probabilities[action]); cdf /= cdf[-1]
        cdf[np.flatnonzero(self.probabilities[action])[-1]:] = 1.
        last = 0
        for step in [0] + steps:
            for _ in range(last, step):
                replay.train(); replay_opt.zero_grad(set_to_none=True)
                if engine == 'proxy':
                    packed = coordinates(replay.state_dict()).double()
                    gradient = (2. / 64) * (replay_moments[0] @ packed - replay_moments[1])
                    for channel in range(8):
                        for parameter, value in zip((replay.seasonal[channel].weight, replay.seasonal[channel].bias,
                            replay.trend[channel].weight, replay.trend[channel].bias),
                            (gradient[channel, :32].T, gradient[channel, 32], gradient[channel, 33:65].T, gradient[channel, 65])):
                            parameter.grad = value.to(parameter.dtype).contiguous()
                else:
                    sampled = np.searchsorted(cdf, rng.random(256), side='right')
                    np.add.at(counts, sampled, 1)
                    starts = self.starts['A'][torch.tensor(sampled, device='cuda')]
                    batch = self.values['A'][starts[:, None] + torch.arange(40, device='cuda')]
                    loss = (replay(batch[:, :32]) - batch[:, 32:]).square().mean()
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                replay_opt.step()
                self.replay_steps += 1
            saved = snap(directory, step)
            equal(saved['model'], replay.state_dict())
            equal(saved['optimizer'], replay_opt.state_dict())
            packed = coordinates(saved['model'])
            assert saved['step'] == step and torch.equal(saved['original_coordinates'], packed) and bool(torch.isfinite(packed).all())
            equal(saved['rng_numpy'], rng.bit_generator.state)
            equal(saved['rng_torch'], initial['rng_torch']); equal(saved['rng_cuda'], initial['rng_cuda'])
            group = saved['optimizer']['param_groups']
            assert len(group) == 1 and group[0]['params'] == list(range(32))
            assert group[0]['lr'] == .001 and tuple(group[0]['betas']) == (.9, .999)
            assert group[0]['eps'] == 1e-8 and group[0]['weight_decay'] == 0.
            assert not group[0]['amsgrad'] and not group[0]['maximize']
            if step == 0:
                equal(saved['model'], initial['model'])
                assert not saved['optimizer']['state']
                assert hashlib.sha256(packed.numpy().tobytes()).hexdigest() == result['initial_sha256']
            else:
                assert set(saved['optimizer']['state']) == set(range(32))
                for index, param in enumerate(saved['model'].values()):
                    state = saved['optimizer']['state'][index]
                    assert state['step'].item() == step
                    for key in ('exp_avg', 'exp_avg_sq'):
                        assert state[key].shape == param.shape and bool(torch.isfinite(state[key]).all())
                    assert bool((state['exp_avg_sq'] >= 0).all())
            last = step
            self.states_checked += 1
        self.replay_trajectories += 1
        if engine == 'proxy':
            with np.load(directory / 'moments.npz', allow_pickle=False) as z:
                for key, reference in zip(('gram', 'cross', 'target_square'), self.moments(action)):
                    np.testing.assert_array_equal(z[key], reference)
            self.moments_checked += 1
            assert result['selected_step'] is None and result['selected_losses'] is None
            assert result['selection_losses'] == result['curves'] == {}
            return result
        with np.load(directory / 'sampling.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['draw_count'], counts)
        assert counts.sum() == maximum * 256 and not counts[self.probabilities[action] == 0].any()
        for step in steps:
            state = snap(directory, step)['model']
            for split in ('B', 'C'):
                error = self.errors(state, split)
                self.check_error(error, directory / f'step_{step}_{split}_errors.npz')
                assert mse(error) == result['curves'][str(step)][split]
            assert result['selection_losses'][str(step)] == result['curves'][str(step)]['B']
        chosen = min(steps, key=lambda k: result['curves'][str(k)]['B'])
        assert result['selected_step'] == chosen and result['selected_losses'] == result['curves'][str(chosen)]
        for split in ('B', 'C'):
            self.check_error(self.errors(snap(directory, chosen)['model'], split), directory / f'selected_{split}_errors.npz')
        return result

    def forecast(self, action, seed, method):
        directory = self.out / 'forecasts' / f'{self.task}_{action}_{seed}_{method}'
        local = read(directory / 'result.json')
        assert (local['task'], local['action'], local['seed'], local['method']) == (self.task, action, seed, method)
        assets(directory, local['assets_sha256'])
        for step in GRID:
            state = snap(self.directory('proxy', action, seed), step)['model']
            if method == 'M1':
                real = snap(self.directory('full', 'PARENT', seed), step)['model']
                parent = snap(self.directory('proxy', 'PARENT', seed), step)['model']
                state = {k: real[k] + (state[k] - parent[k]) for k in state}
            for split in ('B', 'C'):
                error = self.errors(state, split)
                self.check_error(error, directory / f'step_{step}_{split}_errors.npz')
                assert mse(error) == local['curves'][str(step)][split]
        self.selection(local, seed)
        return local

    def selection(self, record, seed):
        assert list(record['curves']) == [str(k) for k in GRID]
        assert all(set(v) == {'B', 'C'} and all(math.isfinite(x) and x >= 0 for x in v.values()) for v in record['curves'].values())
        chosen = min(GRID, key=lambda k: record['curves'][str(k)]['B'])
        baseline = read(self.directory('full', 'PARENT', seed) / 'result.json')['selected_losses']['C']
        assert baseline > 0 and record['selected_step'] == chosen
        assert record['gain_percent'] == 100. * (baseline - record['curves'][str(chosen)]['C']) / baseline

    def source(self, seed):
        import source_dlinear_adapter as adapter
        directory = self.out / 'source' / f'{self.task}_s{seed}'
        result = read(directory / 'result.json')
        assert result['status'] == 'completed' and result['task'] == self.task and result['seed'] == seed
        assets(directory, result['assets_sha256'])
        assets(self.directory('full', 'PARENT', seed), result['parent_assets_sha256'])
        parent = read(self.directory('full', 'PARENT', seed) / 'result.json')
        assert result['initial_sha256'] == parent['initial_sha256']
        for action in self.probabilities:
            with np.load(directory / 'moments' / f'{action}.npz', allow_pickle=False) as z:
                for key, reference in zip(('gram', 'cross', 'target_square'), self.moments(action)):
                    np.testing.assert_array_equal(z[key], reference)
            self.moments_checked += 1
        states = {}
        for step in GRID:
            saved = snap(self.directory('full', 'PARENT', seed), step)
            state = adapter.parent_coordinates(self.core.DLinear(dict(self.plan['training'], channels=8)), saved)
            np.testing.assert_array_equal(state['theta'], coordinates(saved['model']).numpy().astype(np.float64))
            for key in ('exp_avg', 'exp_avg_sq'):
                named = {name: saved['optimizer']['state'][i][key] for i, name in enumerate(saved['model'])}
                expected = coordinates(named).numpy().astype(np.float64)
                np.testing.assert_array_equal(state[key], expected)
            expected_p = 1. / (np.sqrt(state['exp_avg_sq'] / (-math.expm1(step * math.log(.999)))) + 1e-8)
            np.testing.assert_array_equal(state['P'], expected_p)
            states[step] = state
        records = {}
        for action_index, action in enumerate(self.spec['actions']):
            name = action['id']; target = directory / name
            local = read(target / 'result.json'); assets(target, local['assets_sha256'])
            delta = np.zeros((8, 66, 8), dtype=np.float64)
            gp, cp, _ = self.moments('PARENT'); ga, ca, _ = self.moments(name)
            for step in GRID:
                previous_delta = delta.copy()
                delta, checks = adapter.quadratic_segment(gp, cp, ga, ca, states[step], delta)
                with np.load(target / f'step_{step}.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['delta'], delta)
                    np.testing.assert_array_equal(z['coordinates'], (states[step]['theta'] + delta).astype(np.float32))
                    state = self.state(torch.from_numpy(z['coordinates']))
                # Fixed, predeclared independent block-exponential probes. All
                # other blocks reuse the unchanged previously verified operator.
                if action_index == 0 and step in (500, 3000):
                    p = states[step]['P'][0, :, 0]; root = np.sqrt(p)
                    h = 2. / 64 * gp[0]
                    forcing = (2. / 64 * ((ga - gp) @ states[step]['theta'] - (ca - cp)))[0, :, 0]
                    generator = np.zeros((67, 67))
                    generator[:66, :66] = -root[:, None] * ((h + h.T) / 2) * root[None, :]
                    generator[:66, 66] = -root * forcing
                    exponential = torch.matrix_exp(torch.from_numpy(.5 * generator)).numpy()
                    probe = root * (exponential @ np.r_[previous_delta[0, :, 0] / root, 1.])[:66]
                    np.testing.assert_allclose(delta[0, :, 0], probe, rtol=1e-7, atol=1e-7)
                diagnostics = next(d for d in local['numerical_diagnostics'] if d['step'] == step)
                assert diagnostics['blocks'] == checks
                assert diagnostics['clipped_negative_eigenvalues'] == sum(c['clipped_negative_eigenvalues'] for c in checks)
                assert diagnostics['max_roundoff_tolerance'] == max(c['roundoff_tolerance'] for c in checks)
                for split in ('B', 'C'):
                    error = self.errors(state, split)
                    self.check_error(error, target / f'step_{step}_{split}_errors.npz')
                    assert mse(error) == local['curves'][str(step)][split]
                    summary = dict(shape=list(error.shape), dtype=str(error.dtype),
                        byte_sha256=hashlib.sha256(error.tobytes()).hexdigest(), nmse=mse(error),
                        nmae=float(np.mean(np.abs(error.astype(np.float64)))))
                    assert local['error_summaries'][str(step)][split] == summary
            self.selection(local, seed)
            assert local['baseline_C'] == parent['selected_losses']['C']
            assert local['selected_losses'] == local['curves'][str(local['selected_step'])]
            assert local['category'] == previous.category(local['gain_percent'])
            record = dict(local, seconds=read(target / 'timing.json')['seconds'])
            records[name] = record
        assert result['records'] == list(records.values())
        timing = read(directory / 'timing.json')
        booked = result['moment_seconds'] + result['parent_state_seconds'] + result['action_seconds']
        assert result['action_seconds'] == sum(r['seconds'] for r in records.values())
        assert timing['total_seconds'] > 0 and timing['other_seconds'] >= 0
        assert abs(timing['total_seconds'] - booked - timing['other_seconds']) < 1e-9
        return records, timing['total_seconds']

    @torch.no_grad()
    def energy(self, seed, action, step):
        real = snap(self.directory('full', 'PARENT', seed), step)['model']
        reference = snap(self.directory('proxy', 'PARENT', seed), step)['model']
        child = snap(self.directory('proxy', action, seed), step)['model']
        truth = snap(self.directory('full', action, seed), step)['model']
        paired = {k: real[k] + (child[k] - reference[k]) for k in child}
        pc, pr, pp, combo = [coordinates(v).double().cuda() for v in (child, real, reference, paired)]
        total = np.zeros(4); count = 0; gap_max = 0.
        for _, batch in self.batches('C'):
            errors = []
            for state in (child, real, reference, paired, truth):
                self.model.load_state_dict(state)
                errors.append((self.model(batch[:, :32]) - batch[:, 32:]).double())
            raw, e_real, e_ref, corr, true = errors
            e0, et, corrected = e_ref - e_real, raw - true, corr - true
            phi, target = self.features(batch), batch[:, 32:].double()
            pbound = torch.einsum('nci,cij->njc', phi.abs(), (combo - (pr + pc - pp)).abs())
            magnitude = sum(torch.einsum('nci,cij->njc', phi.abs(), v.abs()) + target.abs() for v in (pc, pr, pp, combo))
            bound = pbound + gamma(70, np.float32) * magnitude + gamma(256, np.float64) * (magnitude + pbound)
            gap = (corr - (raw - e0)).abs()
            assert bool((gap <= bound).all())
            gap_max = max(gap_max, float(gap.max()))
            total += [float((et ** 2).sum()), float((e0 ** 2).sum()), float((et * e0).sum()), float((corrected ** 2).sum())]
            count += et.numel()
        return dict(task=self.task, seed=seed, action=action, step=step, raw_energy=total[0] / count,
                    parent_energy=total[1] / count, cross=total[2] / count, paired_energy=total[3] / count,
                    rounding_bound_pass=True, identity_gap_max=gap_max)
