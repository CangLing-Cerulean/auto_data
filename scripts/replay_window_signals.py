"""Source-isolated retrospective evaluation of executable window measurements."""
import argparse
import copy
import csv
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import LinearGaussian,DiagnosticPredictor,PrequentialLedger,digest
from diagnosis_pool_features import read,sha,write
from replay_conditional_diagnosis import BASE_FEATURES,EXPERTS,feature_vector,view
from window_measurement_dsl import evaluate,summary,INPUT_COLUMNS,ERROR_COLUMNS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT=BASE/'20261018_measurement_generation_r4'
METHODS=('reference','original_quadratic','window_fixed','window_quadratic','manual_uniform','manual_conditional','ordinary_uniform','ordinary_joint','ordinary_conditional','ordinary_combined','mismatch_uniform','mismatch_joint','mismatch_conditional','mismatch_combined')

class DualUnitGaussian(LinearGaussian):
    @staticmethod
    def _fit_mean(x,y,group,ridge):
        names,counts=np.unique(group,return_counts=True);lookup=dict(zip(names,counts))
        w=np.array([1/lookup[g] for g in group]);center=np.average(x,axis=0,weights=w)
        scale=np.sqrt(np.maximum(np.average((x-center)**2,axis=0,weights=w),1.))
        design=(x-center)/scale;ym=np.average(y,weights=w);a=design*np.sqrt(w)[:,None]
        coefficients=a.T@np.linalg.solve(a@a.T+ridge*np.eye(len(y)),np.sqrt(w)*(y-ym))
        return center,scale,np.r_[ym,coefficients]

def fit(records,names,quadratic=False):
    names=tuple(names);expanded=names+tuple(f'{names[i]}*{names[j]}' for i in range(len(names)) for j in range(i,len(names))) if quadratic else names
    x=np.array([feature_vector(r,names,quadratic) for r in records]);y=np.array([r['response']['Y'] for r in records]);z=np.array([r['response']['Z'] for r in records])
    groups=[r['family']+':'+r['task'] for r in records];ids=[r['id'] for r in records]
    return DiagnosticPredictor(DualUnitGaussian.fit(x,y,groups,ids,expanded),DualUnitGaussian.fit(np.column_stack([x,y]),z,groups,ids,[*expanded,'observed_Y_scoring_only']))

def fixed_measurements(table,features):
    output={};s=summary(table);shift=features['exposure_shift_first']
    for row in s['statistics']:
        key=row['input'];delta=(row['A_first_mean']-row['A_second_mean'])/(1+row['B_std'])
        output['window_coverage_'+key]=shift*delta
        for error,corr in row['B_error_correlations'].items():
            output['window_correlation_'+key+'_'+error]=corr
            output['window_aligned_'+key+'_'+error]=shift*delta*corr
    return output

def augment(records,source_family,responses):
    result=copy.deepcopy(records);runtime=[]
    for family,task in sorted({(r['family'],r['task']) for r in records}):
        path=BASE/'20261018_parent_measurements_r3/frequency_corrected'/f'{family}_{task}.npz'
        with np.load(path,allow_pickle=False) as z:table=dict(z)
        for record in result:
            if (record['family'],record['task'])!=(family,task):continue
            start=time.perf_counter();record['features'].update(fixed_measurements(table,record['features']))
            for mode in ('ordinary','mismatch'):
                for j,signal in enumerate(responses[mode]['signals']):record['features'][f'{mode}_measurement_{j}']=evaluate(signal['expression'],table,record['features'])
            runtime.append(dict(id=record['id'],seconds=time.perf_counter()-start))
    return result,runtime

def run(training,held,method,original_names,signal_names):
    extra={};quadratic=method in ('original_quadratic','window_quadratic');names=tuple(BASE_FEATURES)
    if method=='original_quadratic':names=tuple(original_names)
    rule=method.rsplit('_',1)[-1];ledger=PrequentialLedger(rule if rule in ('uniform','joint','conditional') else 'uniform')
    if method.startswith('window_'):names=tuple(sorted(k for k in training[0]['features'] if k in original_names or k.startswith('window_')))
    if method.startswith('manual_'):extra={k:v for k,v in EXPERTS.items()}
    for mode in ('ordinary','mismatch'):
        if method.startswith(mode+'_'):
            if method.endswith('_combined'):names+=tuple(signal_names[mode])
            else:extra={k:k for k in signal_names[mode]}
    visible=list(training);oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
    trace=[];nodes=[];start=time.perf_counter()
    for step in range(1,len(held)+1):
        reference=fit(visible,names,quadratic)
        extra_names={h:(*BASE_FEATURES,feature) for h,feature in extra.items()}
        models={h:fit(visible,n) for h,n in extra_names.items()};predictions={};vectors={}
        for r in candidates:
            xb=feature_vector(r,names,quadratic);xh={h:feature_vector(r,n) for h,n in extra_names.items()}
            predictions[r['id']]=ledger.predict_gain(reference,models,xb,xh);vectors[r['id']]=(xb,xh)
        selected=min(candidates,key=lambda r:(-predictions[r['id']],r['id']));xb,xh=vectors[selected['id']]
        registration=ledger.register(selected['id'],reference,models,xb,xh)
        snapshot=dict(reference=reference.snapshot(),diagnostics={h:m.snapshot() for h,m in models.items()});assert digest(snapshot)==registration['model_hash']
        before=[r['id'] for r in visible];revealed=oracle[selected['id']]
        event=ledger.observe(selected['id'],revealed['response']['Y'],revealed['response']['Z'],revealed['response']['NMAE_gain'],revealed['id'])
        visible.append(revealed);candidates=[r for r in candidates if r['id']!=selected['id']]
        recommendation=ledger.best_observed();chosen=oracle.get(recommendation)
        trace.append(dict(step=step,visible_before=before,selected=selected['id'],predictions_before=predictions,model_snapshots=snapshot,registration=registration,event=event,recommendation=recommendation))
        if step in (2,4,8):
            costs=[oracle[e['event_id']]['original_full_training_seconds'] for e in ledger.events]
            nodes.append(dict(queries=step,recommendation=recommendation,gain=chosen['response']['Y'] if chosen else 0.,NMAE_gain=chosen['response']['NMAE_gain'] if chosen else 0.,negative_seed_count=sum(y<0 for y in chosen['seed_Y']) if chosen else 0,logical_candidate_full_trainings=step*3,shared_parent_full_trainings=3,query_seconds=None if any(c is None for c in costs) else sum(costs),policy_seconds=time.perf_counter()-start))
    return dict(method=method,trace=trace,nodes=nodes,seconds=time.perf_counter()-start)

def main():
    lock=read(OUT/'evaluation_lock.json')
    assert set(lock['script_sha256'])=={'window_measurement_dsl.py','prepare_window_signal_inputs.py','replay_window_signals.py','generate_window_signals.py'}
    for name,value in lock['script_sha256'].items():assert sha(ROOT/'scripts'/name)==value
    assert sha(OUT/'PROTOCOL.md')==lock['protocol_sha256']
    assert set(lock['table_sha256'])=={'ETT_ETTh1.npz','ETT_ETTh2.npz','Hydraulic_G1.npz','Hydraulic_G2.npz','Hydraulic_G3.npz'}
    for name,value in lock['table_sha256'].items():assert sha(BASE/'20261018_parent_measurements_r3/frequency_corrected'/name)==value
    outputs=read(OUT/'generated_outputs_lock.json')
    assert outputs['evaluation_lock_sha256']==sha(OUT/'evaluation_lock.json')
    assert outputs['generation_lock_sha256']==sha(OUT/'generation/generation_lock.json')
    assert set(outputs['calls'])=={f'{family}_{mode}' for family in ('ETT','Hydraulic') for mode in ('ordinary','mismatch')}
    for name,entry in outputs['calls'].items():
        assert set(entry)=={'response.json','receipt.json','prompt.txt','events.jsonl'}
        for filename,value in entry.items():assert sha(OUT/'generation'/name/filename)==value
    old=BASE/'20261018_conditional_diagnosis_r1/development_pool.json';assert sha(old)==lock['pool_sha256']
    records=read(old)['records'];original_names=tuple(sorted(records[0]['features']));results=[];rows=[];timings=[];output=OUT/'replay';output.mkdir()
    for source in ('ETT','Hydraulic'):
        responses={}
        for mode in ('ordinary','mismatch'):
            path=OUT/'generation'/f'{source}_{mode}';receipt=read(path/'receipt.json');assert receipt['status']=='PASS' and sha(path/'response.json')==receipt['response_sha256']
            responses[mode]=read(path/'response.json')
        augmented,runtime=augment(records,source,responses);timings.append(dict(source=source,records=runtime))
        write(output/f'{source}_augmented_records.json',dict(records=augmented,source_family=source,generated_from_target=False))
        signal_names={mode:[f'{mode}_measurement_{j}' for j in range(len(responses[mode]['signals']))] for mode in responses}
        training=[r for r in augmented if r['family']==source]
        for family,task in sorted({(r['family'],r['task']) for r in augmented if r['family']!=source}):
            held=[r for r in augmented if (r['family'],r['task'])==(family,task)]
            oracle=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
            for method in METHODS:
                result=run(training,held,method,original_names,signal_names);result.update(source_family=source,family=family,task=task,oracle_gain=oracle)
                for node in result['nodes']:node['regret']=oracle-node['gain'];rows.append(dict(family=family,task=task,method=method,**node))
                # Per-run outputs bound serialization memory and preserve snapshots.
                target=output/f'{family}_{task}_{method}.json';write(target,result)
                results.append(dict(family=family,task=task,method=method,sha256=sha(target),seconds=result['seconds']))
    summaries=[]
    for method in METHODS:
        for count in (2,4,8):
            subset=[r for r in rows if r['method']==method and r['queries']==count]
            summaries.append(dict(method=method,queries=count,tasks=len(subset),mean_gain=float(np.mean([r['gain'] for r in subset])),mean_regret=float(np.mean([r['regret'] for r in subset]))))
    with (output/'decisions.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'summary.json',dict(status='PASS',runs=results,summaries=summaries,measurement_timings=timings,evaluation_lock_sha256=sha(OUT/'evaluation_lock.json'),new_full_trainings=0,cumulative_full_trainings=464))
    print(dict(status='PASS',runs=len(results),nodes=len(rows)))

if __name__=='__main__':main()
