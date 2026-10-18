"""Only the r8 RMS correction and revised lock; retain first gate failure."""
import hashlib
import json
import math
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
import conditional_interactions as impl

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(value, message):
    if not value: raise AssertionError(message)

checks=[]
def check(name, function):
    try: checks.append(dict(check=name, status='PASS', details=function()))
    except Exception as exc: checks.append(dict(check=name, status='FAIL', error=repr(exc)))

def extremes():
    rows=[]
    for value in (1e200, 1e-200):
        records=[dict(family='synthetic',task='one',features=dict(q=value)),
                 dict(family='synthetic',task='two',features=dict(q=-value))]
        result=impl.normalizers(records, ('q',))['q']
        require(math.isfinite(result) and result>0 and math.isclose(result,abs(value),rel_tol=1e-12), f'RMS mismatch {value}: {result}')
        rows.append(dict(input_magnitude=value, actual=result, expected=abs(value)))
    return rows

def current_RMS():
    lock=read(STAGE/'execution_lock.json'); values=[]
    for source,norms in lock['normalizers'].items():
        records=[r for r in read(impl.OLD/'replay'/f'{source}_augmented_records.json')['records'] if r['family']==source]
        tasks=sorted({(r['family'],r['task']) for r in records})
        actual=impl.normalizers(records,tuple(norms))
        for name,expected in norms.items():
            weighted=[]
            for task in tasks:
                group=[r for r in records if (r['family'],r['task'])==task]
                weighted.extend(float(r['features'][name])/math.sqrt(len(tasks)*len(group)) for r in group)
            independent=math.hypot(*weighted)
            require(actual[name]==expected, 'prepared normalizer differs from new implementation')
            require(math.isclose(actual[name],independent,rel_tol=1e-12,abs_tol=1e-15), 'independent norm differs')
            values.append(dict(source=source,name=name,locked=expected,independent=independent))
    require(len(values)==16,'expected 16 source normalizers')
    return values

def revised_hashes():
    lock=read(STAGE/'execution_lock.json'); old=read(STAGE/'preflight_first_lock.json')
    for name,expected in lock['scripts'].items():
        require(sha(ROOT/'scripts'/name)==expected,'script hash '+name)
    require(lock['protocol_sha256']==sha(STAGE/'PROTOCOL.md'),'protocol hash')
    require(old['scripts']['conditional_interactions.py']!=lock['scripts']['conditional_interactions.py'],'missing source correction')
    require(old['inputs']==lock['inputs'],'frozen inputs changed')
    require(read(STAGE/'gate_audit.json')['status']=='FAIL','first failure not retained')
    return dict(current_source_sha256=lock['scripts']['conditional_interactions.py'],
                old_lock_sha256=sha(STAGE/'preflight_first_lock.json'),
                first_gate_sha256=sha(STAGE/'gate_audit.json'), scripts_verified=len(lock['scripts']))

if __name__=='__main__':
    for name,function in [('finite_extreme_RMS_correction',extremes),('all_16_source_RMS',current_RMS),('revised_code_lock_and_retained_failure',revised_hashes)]: check(name,function)
    receipt=dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL', checks=checks,
                 execution_lock_sha256=sha(STAGE/'execution_lock.json'), checker_sha256=sha(__file__),
                 scope='RMS correction only; prior seven passing gates not repeated',
                 real_policy_trajectories_executed=0,new_full_trainings=0,new_LLM_calls=0)
    with (STAGE/'gate_resolution.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],checks=len(checks),passed=sum(c['status']=='PASS' for c in checks))))
    sys.exit(0 if receipt['status']=='PASS' else 1)
