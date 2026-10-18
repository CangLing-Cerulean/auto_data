"""Independent local A/B-only audit. Never reads candidate/C/D/test arrays."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
BASE = STAGE.parent
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('audited_parent_tables', ROOT / 'scripts/parent_measurement_tables.py')
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)
checks = []
findings = []
metrics = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check(name, condition, **details):
    checks.append(dict(name=name, passed=bool(condition), **details))


def compare(name, actual, expected, atol=2e-12):
    check(name, np.shape(actual) == np.shape(expected) and np.allclose(actual, expected, rtol=2e-12, atol=atol), max_abs_difference=float(np.max(np.abs(actual - expected))))


def independent_inputs(values, starts, length):
    """One window at a time, full FFT reference; no subject helper calls."""
    rows, parseval = [], []
    cutoff = (length // 2 + 2) // 2
    for start in starts:
        window = values[int(start):int(start) + length].astype(np.float64)
        average = np.average(window, axis=0)
        centered = window - average
        full_power = np.square(np.abs(np.fft.fft(centered, axis=0)))
        positive = full_power[1:length // 2 + 1]
        denominator = positive.sum()
        fraction = positive[cutoff - 1:].sum() / denominator if denominator else 0.
        energy = full_power[1:].sum()
        ids = np.arange(length)
        high_mask = np.minimum(ids, length - ids) >= cutoff
        parseval.append(full_power[high_mask].sum() / energy if energy else 0.)
        rows.append([window.sum()/window.size, np.dot(average, average)/window.shape[1],
                     np.square(centered).sum()/window.size,
                     np.square(window[1:] - window[:-1]).sum()/((length-1)*window.shape[1]),
                     np.square(window[-1] - window[0]).sum()/window.shape[1],
                     np.sqrt(np.square(np.square(window).mean(0) - np.square(window).mean(0).mean()).mean()),
                     fraction, np.abs(window).sum()/window.size])
    return np.array(rows), np.array(parseval)


def independent_errors(error):
    error = error.astype(np.float64)
    squared = np.square(error)
    denominator = squared.sum()/squared.size
    n, horizon, channels = error.shape
    half = horizon//2
    return np.stack([squared.sum((1, 2))/(horizon*channels*denominator),
                     squared[:, half:].sum((1, 2))/(half*channels*denominator),
                     squared[:, :half].sum((1, 2))/(half*channels*denominator),
                     error.sum((1, 2))/(horizon*channels*np.sqrt(denominator)),
                     np.abs(error).sum((1, 2))/(horizon*channels*np.sqrt(denominator))], axis=1)


def main():
    hydraulic = read(STAGE/'hydraulic_measurements.json')
    ett = read(STAGE/'ett_measurements.json')
    receipts = {**hydraulic['assets'], **ett['assets']}
    initial = {name:sha(STAGE/'tables'/name) for name in receipts}
    lock = read(STAGE/'execution_lock.json')
    deployment = read(STAGE/'deployment_lock.json')
    for name, entry in lock['files'].items():
        path = STAGE/name if name == 'PROTOCOL.md' else ROOT/'scripts'/name
        check('execution_lock:'+name, sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'])
    bundle = STAGE/'private/reader.tar.gz'
    check('deployment_bundle', sha(bundle) == deployment['sha256'] and bundle.stat().st_size == deployment['bytes'])
    with tarfile.open(bundle) as archive:
        check('deployment_members', {m.name for m in archive.getmembers()} == set(deployment['files']))
        for name, digest in deployment['files'].items():
            check('deployed_file:'+name, hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest)
    check('deployment_receipt', read(STAGE/'deployment_receipt.json')['bundle_sha256'] == deployment['sha256'])
    check('extraction_exit', read(STAGE/'exit.json')['exit_code'] == 0)
    input_rows, error_rows = 0, 0
    for name, receipt in receipts.items():
        path = STAGE/'tables'/name
        check('table_hash:'+name, initial[name] == receipt['sha256'] and path.stat().st_size == receipt['bytes'])
        is_ett = name.startswith('ETT_')
        na, nb = (8449, 1249) if is_ett else (6468, 1617)
        seeds = (2026101901, 2026101902, 2026101903) if is_ett else (2026101701, 2026101702, 2026101703)
        with np.load(path, allow_pickle=False) as saved:
            expected = {'A_inputs', 'B_inputs', 'A_first', 'A_starts', 'B_starts'} | {f'B_errors_s{s}' for s in seeds}
            check('table_keys:'+name, set(saved.files) == expected)
            for split, size in [('A', na), ('B', nb)]:
                values, starts = saved[split+'_inputs'], saved[split+'_starts']
                check('shape_finite:'+name+':'+split, values.shape == (size,8) and values.dtype == np.float64 and np.isfinite(values).all())
                check('complete_starts:'+name+':'+split, np.array_equal(starts, np.arange(size) if is_ett else (np.arange(size//21)[:,None]*60+np.arange(21)).ravel()))
                check('input_bounds:'+name+':'+split, (values[:,1:] >= -1e-14).all() and (values[:,6] <= 1+1e-14).all())
                input_rows += size
            expected_first = np.arange(na)+96 < 4320 if is_ett else np.repeat(np.arange(308)<154,21)
            check('A_first:'+name, saved['A_first'].dtype == bool and np.array_equal(saved['A_first'], expected_first))
            for seed in seeds:
                error = saved[f'B_errors_s{seed}']
                check('errors_shape_finite:'+name+':'+str(seed), error.shape == (nb,5) and error.dtype == np.float64 and np.isfinite(error).all())
                check('error_identities:'+name+':'+str(seed), abs(error[:,0].mean()-1)<1e-12 and np.allclose(error[:,0], (error[:,1]+error[:,2])/2, rtol=1e-12, atol=1e-12))
                check('error_bounds:'+name+':'+str(seed), (error[:,:3]>=0).all() and (error[:,4]>=0).all() and (error[:,3]**2 <= error[:,0]+1e-12).all() and (error[:,4]**2 <= error[:,0]+1e-12).all() and (abs(error[:,3])<=error[:,4]+1e-12).all())
                error_rows += nb
    plan = read(BASE/'20261014_weighted_response_r1/plan.json')
    dev = BASE/'20261013_response_proxy_development_r1/archives/verified'
    for task in plan['tasks']:
        tid = task['id']
        source = dev/'prepared'/f'{tid}_ABC.npz'
        expected_hash = {v for k,v in plan['input_sha256'].items() if k.endswith('/'+tid+'_ABC.npz')}
        check('Hydraulic_source_hash:'+tid, expected_hash == {sha(source)})
        with np.load(source, allow_pickle=False) as src, np.load(STAGE/'tables'/f'Hydraulic_{tid}.npz', allow_pickle=False) as table:
            for split in ('A','B'):
                values, starts = src[split+'_values'], src[split+'_starts']
                cycles, offsets = src[split+'_cycles'], src[split+'_offsets']
                lo, hi = task['partitions'][split]
                check('Hydraulic_source_shape:'+tid+split, values.shape == ((hi-lo)*60,8))
                check('Hydraulic_source_cycles:'+tid+split, np.array_equal(cycles, np.repeat(np.arange(lo,hi),60)) and np.array_equal(offsets, np.tile(np.arange(60),hi-lo)))
                valid = np.arange(len(values)-39)
                valid = valid[(cycles[valid] == cycles[valid+39]) & (offsets[valid+39]-offsets[valid] == 39)]
                check('Hydraulic_complete_windows:'+tid+split, np.array_equal(starts,valid) and np.array_equal(starts,table[split+'_starts']))
                reference, parseval = independent_inputs(values, starts,32)
                compare('Hydraulic_input_recompute:'+tid+split,table[split+'_inputs'],reference)
                delta = table[split+'_inputs'][:,6]-parseval
                metrics[tid+split+'_frequency_parseval'] = dict(affected_windows=int(np.count_nonzero(abs(delta)>1e-12)),maximum_absolute_difference=float(abs(delta).max()),mean_absolute_difference=float(abs(delta).mean()))
            for seed in (2026101701,2026101702,2026101703):
                run = dev/'runs'/f'DEV_{tid}_full_PARENT_s{seed}'
                path = run/'selected_B_errors.npz'
                result = read(run/'result.json')
                entry = next(c for c in hydraulic['checks'] if c['task']==tid and c['seed']==seed)
                check('Hydraulic_error_source_hash:'+tid+str(seed), sha(path)==result['assets_sha256'][path.name]==entry['parent_B_sha256'])
                with np.load(path, allow_pickle=False) as error_source:
                    error = error_source['error']
                check('Hydraulic_error_layout:'+tid+str(seed), error.shape==(1617,8,8))
                check('Hydraulic_parent_loss:'+tid+str(seed), abs(np.square(error.astype(np.float64)).mean()-result['selected_losses']['B'])<1e-12)
                compare('Hydraulic_error_recompute:'+tid+str(seed),table[f'B_errors_s{seed}'],independent_errors(error))
    old = ROOT/'research_runs/ett_confirmation'
    for station in ('ETTh1','ETTh2'):
        check('ETT_frozen_window_metadata:'+station, read(old/station/'preprocessing.json')['windows'] == dict(A=8449,B=1249,C=1249))
        for entry in [c for c in ett['checks'] if c['station']==station]:
            run = old/station/'runs'/f'PARENT_P_s{entry["seed"]}'
            check('ETT_parent_checkpoint:'+station+str(entry['seed']), sha(run/'best.pt') == entry['parent_checkpoint_sha256'])
            curve = read(run/'B_curve.json')
            selected = min(curve,key=lambda r:(r['NMSE'],r['step']))
            check('ETT_parent_B_curve:'+station+str(entry['seed']), len(selected['residual_sha256'])==64 and entry['B_residual_hash_match'] and entry['windows']==1249)
    synthetic = {}
    for length in (32,96):
        time = np.arange(length)
        values = (np.cos(2*np.pi*time/length)+(-1.)**time)[:,None]
        measured = float(subject.inputs(values,np.array([0]),length)[0,6])
        expected = 2/3
        synthetic[str(length)] = dict(actual=measured,parseval_energy_expected=expected,absolute_difference=abs(measured-expected))
        check('frequency_synthetic_reproduction:'+str(length),abs(measured-.8)<1e-12)
        error = np.empty((2,8 if length==32 else 96,3))
        error[:,:error.shape[1]//2]=1
        error[:,error.shape[1]//2:]=3
        compare('target_axis_synthetic:'+str(length), subject.errors(error), independent_errors(error))
    metrics['frequency_synthetic'] = synthetic
    findings.append(dict(classification='reproduced_definition_mismatch',file='scripts/parent_measurement_tables.py',line=21,trigger='An even-length centered window with both Nyquist and non-Nyquist energy',actual='Unweighted rFFT bins give 0.8 for cos(2*pi*t/N)+(-1)^t',expected='A real-signal band energy fraction uses conjugate-pair weights (interior bins twice, Nyquist once), giving 2/3',impact='The high_frequency_fraction differs from Parseval band-energy fraction for real A/B windows; if unweighted bin power is intentional it must be explicitly defined and named accordingly.'))
    check('all_tables_unchanged', all(sha(STAGE/'tables'/name)==digest for name,digest in initial.items()))
    output = dict(status='PASS_WITH_FINDING' if all(c['passed'] for c in checks) else 'FAIL',scope='5 local tables; Hydraulic A/B source arrays and 9 saved parent B residuals; ETT table/reader/receipt and 6 checkpoint hashes only',checks=checks,counts=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),tables=5,input_windows=input_rows,parent_error_windows=error_rows,Hydraulic_independent_input_tables=6,Hydraulic_independent_parent_errors=9,ETT_checkpoint_hashes=6),metrics=metrics,findings=findings,limitations=['ETT raw A/B tensors and residuals were not independently recomputed: local tables, frozen reader, receipts and parent checkpoint hashes only; no server access.', 'No C/D/test numerical arrays, child responses, training, optimizer or server operations were used.'],new_full_trainings=0,cumulative_full_trainings=464)
    with (STAGE/'audit.json').open('x',encoding='utf-8') as stream:
        json.dump(output,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=output['status'],**output['counts'],findings=len(findings))))


if __name__ == '__main__':
    main()
