"""Read-only verification of the stopped forecast batch; no model or labels loaded."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

METHODS=('M1','RAW','FIXED','SHORT')
ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ident(t,e,a,s):return f'EXSR1_{t}_{e}_{a}_s{s}'
def close(a,b):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9,(a,b)

def audit(out):
    started=time.perf_counter();plan=read(out/'plan.json');pred=read(out/'predictions.json')
    cal=read(out/'calibration.json');prep=read(out/'preparation.json')
    assert pred['plan_sha256']==sha(out/'plan.json')
    assert pred['calibration_sha256']==sha(out/'calibration.json')
    assert pred['preparation_sha256']==sha(out/'preparation.json')
    assert not pred['full_children_started'] and not pred['budget_gate']
    assert read(out/'forecast_status.json')['status']=='completed'
    assert read(out/'parents_audit.json')['status']=='PASS'
    assert read(out/'calibration_sync_receipt.json')['calibration_sha256']==sha(out/'calibration.json')
    assert not any((out/n).exists() for n in ('labels.lock','labels.json','labels_status.json','final_audit.json'))
    assert not list(out.glob('*_failure.json'))
    assert sha(Path(plan['core']))==plan['core_sha256']
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for task in plan['tasks']:
        assert sha(Path(task['train']))==task['train_sha256']
        assert sha(Path(task['manifest']))==task['manifest_sha256']
    hashes={}
    for document in (prep,cal,pred):
        for name,digest in document['assets_sha256'].items():
            if name in hashes:assert hashes[name]==digest
            hashes[name]=digest
    for name,digest in hashes.items():assert sha(out/name)==digest,name
    expected={j['id'] for j in plan['jobs'] if j['phase']!='labels'}
    assert len(expected)==60 and {p.name for p in (out/'runs').iterdir()}==expected
    runs={}
    for name in sorted(expected):
        directory=out/'runs'/name;result=read(directory/'result.json');timing=read(directory/'timing.json')
        assert result['status']==read(directory/'status.json')['status']=='completed'
        assert result['id']==name and result['plan_sha256']==sha(out/'plan.json')
        assert math.isfinite(timing['seconds']) and timing['seconds']>0
        if result['engine']=='short':assert result['steps']<3000
        runs[name]=dict(engine=result['engine'],steps=result['steps'],seconds=timing['seconds'])
    records={(r['task'],r['seed'],r['action']):r for r in pred['records']}
    assert len(records)==len(pred['records'])==24
    costs=[];violations=[];shorts=[]
    for task in plan['tasks']:
        tid=task['id'];budget=cal['budgets'][tid]
        for seed in plan['seeds']:
            cost={m:0. for m in METHODS};cost['M1']=runs[ident(tid,'proxy','PARENT',seed)]['seconds']
            for action in ACTIONS:
                proxy=runs[ident(tid,'proxy',action,seed)]['seconds'];short=runs[ident(tid,'short',action,seed)]
                for m in ('M1','RAW','FIXED'):
                    directory=out/'forecasts'/f'{tid}_{action}_{seed}_{m}'
                    seconds=read(directory/'timing.json')['seconds']
                    close(seconds,records[(tid,seed,action)]['methods'][m]['seconds'])
                    cost[m]+=seconds
                cost['M1']+=proxy;cost['RAW']+=proxy;cost['SHORT']+=short['seconds']
                close(short['seconds'],records[(tid,seed,action)]['methods']['SHORT']['seconds'])
                row=dict(id=ident(tid,'short',action,seed),task=tid,seed=seed,action=action,
                    seconds=short['seconds'],steps=short['steps'],cap=budget['short_budget_seconds'])
                shorts.append(row)
                if row['seconds']>row['cap']:violations.append(dict(kind='short_run',**row,excess_seconds=row['seconds']-row['cap']))
            registered=next(c for c in pred['costs'] if (c['task'],c['seed'])==(tid,seed))
            parent=runs[ident(tid,'full','PARENT',seed)]['seconds']
            common=prep['seconds']/6+(read(out/f'{tid}_parent_setup.json')['seconds']+read(out/f'{tid}_setup.json')['seconds'])/3
            close(parent,registered['real_parent_seconds']);close(common,registered['allocated_common_preparation_seconds'])
            for m in METHODS:
                close(cost[m],registered[m]);close(cost[m]+parent+common,registered['cold_total_seconds'][m])
                if m in ('M1','SHORT') and cost[m]>budget['selection_budget_seconds']:
                    violations.append(dict(kind='selection_total',task=tid,seed=seed,method=m,seconds=cost[m],
                        cap=budget['selection_budget_seconds'],excess_seconds=cost[m]-budget['selection_budget_seconds']))
            costs.append(dict(task=tid,seed=seed,**cost))
    for m in METHODS:close(sum(c['cold_total_seconds'][m] for c in pred['costs']),pred['batch_cold_total_seconds'][m])
    assert violations
    choices={t:{m:pred['recommendations'][t][m]['action'] for m in METHODS} for t in ('M','S')}
    for task in ('M','S'):
        for method in METHODS:
            means={a:sum(records[(task,s,a)]['methods'][method]['gain_percent'] for s in plan['seeds'])/3 for a in ACTIONS}
            for a in ACTIONS:close(means[a],pred['recommendations'][task][method]['means'][a])
            choice=max(ACTIONS,key=lambda a:means[a]);choice=choice if means[choice]>=1. else 'PARENT'
            assert choices[task][method]==choice
    same_as_raw=all(c['M1']==c['RAW'] for c in choices.values())
    counts={e:sum(r['engine']==e for r in runs.values()) for e in ('full','proxy','short')}
    assert counts==dict(full=6,proxy=30,short=24)
    result=dict(status='PASS',scope='file integrity, run counts, costs, budget stop and recommendation identity; no numerical model reforward',
        auditor_sha256=sha(Path(__file__)),plan_sha256=sha(out/'plan.json'),predictions_sha256=sha(out/'predictions.json'),
        assets_verified=len(hashes),counts=counts,new_full_trainings=6,cumulative_full_trainings=373,
        costs=costs,short_runs=shorts,budget_violations=violations,budget_gate=False,
        recommendations=choices,M1_same_actions_as_RAW=same_as_raw,
        selection_gate=False if same_as_raw else None,error_gate=None,full_child_labels_started=0,
        model_numerical_audit_complete=False,test_access=False,base_validation_access=False,
        phase_training_seconds={e:sum(r['seconds'] for r in runs.values() if r['engine']==e) for e in counts},
        audit_seconds=time.perf_counter()-started)
    with (out/'budget_stop_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','assets_verified','counts','budget_violations','recommendations','selection_gate')}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
