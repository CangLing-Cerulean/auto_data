"""Narrow supplement audit: local corrected column only; no C/D/test or server."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
CORRECTED = STAGE/'frequency_corrected'
checks, metrics = [], {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check(name, passed, **details):
    checks.append(dict(name=name, passed=bool(passed), **details))


def reference(values, starts, length):
    """Full two-sided DFT, directly selecting both halves of high bands."""
    output = []
    bins = np.arange(length)
    cutoff = (length//2 + 2)//2
    selected = np.minimum(bins, length-bins) >= cutoff
    for start in starts:
        x = values[int(start):int(start)+length].astype(np.float64)
        x = x - x.mean(0)
        power = abs(np.fft.fft(x,axis=0))**2
        denominator = power[1:].sum()
        np.testing.assert_allclose(denominator/length, (x*x).sum(),rtol=1e-12,atol=1e-12)
        output.append(power[selected].sum()/denominator if denominator>0 else 0.)
    return np.asarray(output)


def compare(name, actual, expected):
    difference = float(np.max(abs(actual-expected)))
    check(name, np.allclose(actual,expected,rtol=1e-12,atol=1e-12), maximum_absolute_difference=difference)
    return difference


def main():
    spec = importlib.util.spec_from_file_location('audited_frequency_correction',ROOT/'scripts/correct_parent_frequency.py')
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
    preserved = [STAGE/'audit.json', STAGE/'audit_check.py', STAGE/'execution_lock.json', STAGE/'deployment_lock.json',ROOT/'scripts/parent_measurement_tables.py',ROOT/'scripts/extract_ett_parent_measurements.py']
    baseline = {str(path):sha(path) for path in preserved}
    total_changed, unchanged_arrays, hydraulic_rows, ett_rows = 0,0,0,0
    for family, old_receipt_name in [('Hydraulic','hydraulic_measurements.json'),('ETT','ett_measurements.json')]:
        receipt = read(CORRECTED/f'{family}_correction.json')
        old_receipt = read(STAGE/old_receipt_name)
        check('correction_code_hash:'+family, receipt['code_sha256']==sha(ROOT/'scripts/correct_parent_frequency.py'))
        check('correction_asset_set:'+family, set(receipt['assets'])==set(old_receipt['assets']))
        for name, entry in receipt['assets'].items():
            old_path, new_path = STAGE/'tables'/name, CORRECTED/name
            baseline[str(old_path)] = sha(old_path)
            baseline[str(new_path)] = sha(new_path)
            check('original_hash:'+name, sha(old_path)==old_receipt['assets'][name]['sha256'])
            check('corrected_hash_size:'+name, sha(new_path)==entry['sha256'] and new_path.stat().st_size==entry['bytes'])
            task = name[len(family)+1:-4]
            record = next(c for c in receipt['checks'] if c['task']==task)
            check('receipt_original_hash:'+name, record['old_sha256']==sha(old_path))
            with np.load(old_path,allow_pickle=False) as old, np.load(new_path,allow_pickle=False) as new:
                check('same_keys:'+name,set(old.files)==set(new.files))
                for key in old.files:
                    a,b = old[key],new[key]
                    check('same_shape_dtype:'+name+':'+key,a.shape==b.shape and a.dtype==b.dtype)
                    if key not in ('A_inputs','B_inputs'):
                        check('unchanged_array:'+name+':'+key,a.tobytes()==b.tobytes())
                        unchanged_arrays += 1
                    else:
                        columns = [0,1,2,3,4,5,7]
                        check('other_columns_byte_equal:'+name+':'+key,a[:,columns].tobytes()==b[:,columns].tobytes())
                        check('frequency_valid:'+name+':'+key,np.isfinite(b[:,6]).all() and (b[:,6]>=0).all() and (b[:,6]<=1).all())
                        changed = int(np.count_nonzero(a[:,6]!=b[:,6]))
                        maximum = float(abs(a[:,6]-b[:,6]).max())
                        change = next(c for c in record['changes'] if c['split']==key[0])
                        check('receipt_change:'+name+':'+key,change['windows']==len(b) and change['changed']==changed and change['max_abs_change']==maximum)
                        total_changed += changed
                        metrics[name+':'+key]=dict(changed=changed,windows=len(b),maximum_absolute_change=maximum)
                if family=='Hydraulic':
                    source=STAGE.parent/'20261013_response_proxy_development_r1/archives/verified/prepared'/f'{task}_ABC.npz'
                    check('hydraulic_prepared_hash:'+task,sha(source)==record['prepared_sha256'])
                    with np.load(source,allow_pickle=False) as original:
                        for split in ('A','B'):
                            starts=new[split+'_starts']
                            expected=reference(original[split+'_values'],starts,32)
                            compare('Hydraulic_two_sided_DFT:'+task+split,new[split+'_inputs'][:,6],expected)
                            hydraulic_rows += len(starts)
                else:
                    check('ETT_prepared_receipt:'+task,record['prepared_sha256']==read(ROOT/'research_runs/ett_confirmation'/task/'preprocessing.json')['prepared_sha256'])
                    ett_rows += len(new['A_starts'])+len(new['B_starts'])
    for length in (32,96):
        t=np.arange(length)
        mixed=(np.cos(2*np.pi*t/length)+(-1.)**t)[:,None]
        actual=subject.frequency(mixed,np.array([0]),length)
        compare('mixed_Nyquist_expected_2_over_3:'+str(length),actual,np.array([2/3]))
        check('constant_zero_frequency:'+str(length),subject.frequency(np.full((length,3),7.),np.array([0]),length)[0]==0.)
        # Distinct channels and permuted/duplicate starts spanning chunk boundary.
        rng=np.random.default_rng(length)
        values=rng.normal(size=(length+310,4))
        starts=rng.integers(0,311,size=519)
        compare('frequency_axis_window_chunk:'+str(length),subject.frequency(values,starts,length),reference(values,starts,length))
    check('original_and_corrected_materials_unchanged',all(sha(Path(path))==digest for path,digest in baseline.items()))
    counts=dict(checks=len(checks),passed=sum(c['passed'] for c in checks),failed=sum(not c['passed'] for c in checks),tables=5,unchanged_non_input_arrays=unchanged_arrays,changed_frequency_windows=total_changed,Hydraulic_independently_recomputed_windows=hydraulic_rows,ETT_windows_receipt_and_change_only=ett_rows,synthetic_axis_window_chunk_windows=1038)
    result=dict(status='PASS' if counts['failed']==0 else 'FAIL',counts=counts,checks=checks,metrics=metrics,findings=[],limitations=['ETT independent recomputation from source A/B arrays was not performed; its 19,396 corrected values are covered by hash, receipt, change-scope, bounds and generic N96 algorithm tests only.'],new_full_trainings=0,cumulative_full_trainings=464,C_D_test_numeric_access=False,server_operations=0)
    with (STAGE/'correction_audit.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'],**counts)))


if __name__=='__main__':
    main()
