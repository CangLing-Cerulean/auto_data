"""One-report interventions with independently continued, partially observed policies."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import DiagnosticPredictor,digest,diagnostic_evidence
from diagnosis_pool_features import read,sha,write
from replay_conditional_diagnosis import BASE_FEATURES,feature_vector,view
from replay_window_signals import DualUnitGaussian,fit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
PREVIOUS=BASE/'20261018_measurement_generation_r4'
OUT=BASE/'20261018_feedback_forks_r6'
METHODS=('original_quadratic','mismatch_conditional','mismatch_joint')
BRANCHES=('full','global_only','hidden_current')
TASKS=(('ETT','ETTh1'),('ETT','ETTh2'),('Hydraulic','G1'),('Hydraulic','G2'),('Hydraulic','G3'))
SIGNALS=('mismatch_measurement_0','mismatch_measurement_1')

def records_for(family):
    source='Hydraulic' if family=='ETT' else 'ETT'
    return read(PREVIOUS/'replay'/f'{source}_augmented_records.json')['records']

def original_names(records):
    return tuple(sorted(k for k in records[0]['features'] if not k.startswith(('window_','ordinary_measurement_','mismatch_measurement_'))))

def public_record(record,mode):
    if mode not in BRANCHES:raise ValueError('Unknown report branch')
    result=copy.deepcopy(view(record));keys={'full':('Y','Z','NMAE_gain'),'global_only':('Y','NMAE_gain'),'hidden_current':()}[mode]
    result['response']={k:record['response'][k] for k in keys}
    return result

def fit_partial(records,names,quadratic=False):
    names=tuple(names)
    expanded=names+tuple(f'{names[i]}*{names[j]}' for i in range(len(names)) for j in range(i,len(names))) if quadratic else names
    def component(conditional):
        rows=[r for r in records if 'Y' in r['response'] and (not conditional or 'Z' in r['response'])]
        x=np.array([feature_vector(r,names,quadratic) for r in rows]);y=np.array([r['response']['Y'] for r in rows])
        target=np.array([r['response']['Z'] for r in rows]) if conditional else y
        features=[*expanded,'observed_Y_scoring_only'] if conditional else expanded
        if conditional:x=np.column_stack([x,y])
        return DualUnitGaussian.fit(x,target,[r['family']+':'+r['task'] for r in rows],[r['id'] for r in rows],features)
    return DiagnosticPredictor(component(False),component(True))

def model_set(visible,method,names):
    quadratic=method=='original_quadratic';base_names=names if quadratic else tuple(BASE_FEATURES)
    reference=fit_partial(visible,base_names,quadratic)
    extras={} if quadratic else {h:fit_partial(visible,(*BASE_FEATURES,h)) for h in SIGNALS}
    return reference,extras,base_names,quadratic

def weights(rho):
    if not rho:return {'reference':1.}
    versions=list(rho);logits=np.r_[0.,[rho[h]-math.log(len(rho)) for h in versions]];logits-=logits.max()
    values=np.exp(logits);values/=values.sum()
    return dict(zip(['reference',*versions],map(float,values)))

def prediction_state(visible,candidates,method,names,rho):
    reference,models,base_names,quadratic=model_set(visible,method,names)
    w=weights(rho);predictions={};vectors={}
    for r in candidates:
        xb=feature_vector(r,base_names,quadratic);xh={h:feature_vector(r,(*BASE_FEATURES,h)) for h in models}
        predictions[r['id']]=w['reference']*reference.predict_gain(xb).mean+sum(w[h]*m.predict_gain(xh[h]).mean for h,m in models.items())
        vectors[r['id']]=(xb,xh)
    snapshot=dict(reference=reference.snapshot(),diagnostics={h:m.snapshot() for h,m in models.items()})
    return predictions,vectors,reference,models,snapshot,w

def update_evidence(method,reference,models,xb,xh,response,rho):
    scores={};applied={};before=dict(rho)
    for h,model in models.items():
        if 'Z' in response:
            scores[h]=diagnostic_evidence(reference,model,xb,xh[h],response['Y'],response['Z'])
            applied[h]=scores[h]['conditional' if method.endswith('conditional') else 'joint']
        elif 'Y' in response:
            marginal=model.predict_gain(xh[h]).logpdf(response['Y'])-reference.predict_gain(xb).logpdf(response['Y'])
            scores[h]=dict(marginal=marginal,conditional=None,joint_observed=marginal)
            applied[h]=marginal if method.endswith('joint') else 0.
        else:
            scores[h]=dict(marginal=None,conditional=None,joint_observed=None);applied[h]=0.
        rho[h]+=applied[h]
    return dict(scores=scores,applied=applied,before=before,after=dict(rho))

def choose_observed(observed,continuation_only=False):
    eligible=[r for r in observed if (not continuation_only or r['query_step']>1) and r['response'].get('Y',-math.inf)>=.01 and r['response'].get('NMAE_gain',-math.inf)>=-.01]
    return max(eligible,key=lambda r:(r['response']['Y'],r['id']))['id'] if eligible else 'PARENT'

def make_prefix(records,family,task):
    source=[public_record(r,'full') for r in records if r['family']!=family]
    candidates=[view(r) for r in records if (r['family'],r['task'])==(family,task)]
    model=fit_partial(source,BASE_FEATURES);predictions={r['id']:model.predict_gain(feature_vector(r,BASE_FEATURES)).mean for r in candidates}
    chosen=min(candidates,key=lambda r:(-predictions[r['id']],r['id']))['id']
    return dict(family=family,task=task,selected=chosen,predictions=predictions,reference_snapshot=model.snapshot(),response_revealed=False)

def run_trajectory(records,family,task,method,branch,prefix,folder=None):
    names=original_names(records);visible=[public_record(r,'full') for r in records if r['family']!=family]
    held=[r for r in records if (r['family'],r['task'])==(family,task)]
    oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
    rho={} if method=='original_quadratic' else {h:0. for h in SIGNALS}
    observed=[];trace=[];nodes=[];queried=[];start=time.perf_counter()
    for step in range(1,5):
        predictions,vectors,reference,models,snapshot,w=prediction_state(visible,candidates,method,names,rho)
        selected=prefix['selected'] if step==1 else min(candidates,key=lambda r:(-predictions[r['id']],r['id']))['id']
        registration=dict(step=step,selected=selected,model_hash=digest(snapshot),visible_hash=digest(visible),visible_before=[dict(id=r['id'],response_keys=sorted(r['response'])) for r in visible],predictions_before=predictions,weights_before=w,model_snapshots=snapshot,report_mode=branch if step==1 else 'full')
        if folder:write(folder/f'query_{step}_registration.json',registration)
        # The selected label is read only after the complete prediction registration.
        revealed=public_record(oracle[selected],registration['report_mode']);xb,xh=vectors[selected]
        event=update_evidence(method,reference,models,xb,xh,revealed['response'],rho)
        observed.append(dict(**revealed,query_step=step));visible.append(revealed);queried.append(selected)
        candidates=[r for r in candidates if r['id']!=selected]
        decision=choose_observed(observed);continuation=choose_observed(observed,True)
        row=dict(registration=registration,visible_report=revealed,event=event,recommendation=decision,continuation_recommendation=continuation)
        if folder:write(folder/f'query_{step}_result.json',row)
        trace.append(row)
        if step in (2,4):
            # Evaluation-only lookups never return to visible, models or selection.
            costs=[oracle[key]['original_full_training_seconds'] for key in queried]
            oracle_gain=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
            def outcome(key):
                r=oracle.get(key)
                gain=r['response']['Y'] if r else 0.
                return dict(recommendation=key,gain=gain,regret=oracle_gain-gain,NMAE_gain=r['response']['NMAE_gain'] if r else 0.,negative_seed_count=sum(y<0 for y in r['seed_Y']) if r else 0)
            nodes.append(dict(queries=step,actual=outcome(decision),continuation=outcome(continuation),oracle_gain=oracle_gain,query_seconds=None if any(v is None for v in costs) else sum(costs),logical_candidate_full_trainings=3*step,shared_parent_full_trainings=3))
    return dict(family=family,task=task,method=method,branch=branch,prefix=prefix['selected'],trace=trace,nodes=nodes,seconds=time.perf_counter()-start)

def prepare():
    assert read(BASE/'20261018_strong_agent_r5/sync_receipt.json')['status']=='result_commit_and_release_verified'
    prefixes=[make_prefix(records_for(family),family,task) for family,task in TASKS]
    write(OUT/'prefixes.json',dict(prefixes=prefixes,selection_uses_target_responses=False))
    scripts=('feedback_forks.py','replay_window_signals.py','conditional_diagnosis.py','diagnosis_pool_features.py','replay_conditional_diagnosis.py','window_measurement_dsl.py','parent_measurement_tables.py')
    write(OUT/'execution_lock.json',dict(scripts={name:sha(ROOT/'scripts'/name) for name in scripts},protocol_sha256=sha(OUT/'PROTOCOL.md'),prefixes_sha256=sha(OUT/'prefixes.json'),pools={family:sha(PREVIOUS/'replay'/f'{family}_augmented_records.json') for family in ('ETT','Hydraulic')},definitions={family:sha(PREVIOUS/'generation'/f'{family}_mismatch/response.json') for family in ('ETT','Hydraulic')},methods=METHODS,branches=BRANCHES,tasks=TASKS,queries=4,new_full_trainings=0,new_LLM_calls=0))
    print({'status':'PREPARED','prefixes':len(prefixes)})

def run():
    lock=read(OUT/'execution_lock.json')
    assert read(OUT/'source_publication_receipt.json')['status']=='exact_commit_and_tree_verified'
    for name,value in lock['scripts'].items():assert sha(ROOT/'scripts'/name)==value
    assert sha(OUT/'PROTOCOL.md')==lock['protocol_sha256'] and sha(OUT/'prefixes.json')==lock['prefixes_sha256']
    for family,value in lock['pools'].items():assert sha(PREVIOUS/'replay'/f'{family}_augmented_records.json')==value
    for family,value in lock['definitions'].items():assert sha(PREVIOUS/'generation'/f'{family}_mismatch/response.json')==value
    prefixes=read(OUT/'prefixes.json')['prefixes'];output=OUT/'replay';output.mkdir();rows=[];traces=[];start=time.perf_counter()
    for prefix in prefixes:
        family,task=prefix['family'],prefix['task'];records=records_for(family)
        for method in METHODS:
            for branch in BRANCHES:
                key=f'{family}_{task}_{method}_{branch}';folder=output/key;folder.mkdir()
                result=run_trajectory(records,family,task,method,branch,prefix,folder)
                path=output/f'{key}.json';write(path,result)
                traces.append(dict(family=family,task=task,method=method,branch=branch,path=path.relative_to(OUT).as_posix(),sha256=sha(path),seconds=result['seconds']))
                for node in result['nodes']:rows.append(dict(family=family,task=task,method=method,branch=branch,**node))
    summaries=[];contrasts=[]
    for method in METHODS:
        for q in (2,4):
            for branch in BRANCHES:
                subset=[r for r in rows if r['method']==method and r['branch']==branch and r['queries']==q]
                summaries.append(dict(method=method,branch=branch,queries=q,actual_gain=float(np.mean([r['actual']['gain'] for r in subset])),continuation_gain=float(np.mean([r['continuation']['gain'] for r in subset]))))
            full=next(r for r in summaries if r['method']==method and r['branch']=='full' and r['queries']==q)
            for branch in ('global_only','hidden_current'):
                other=next(r for r in summaries if r['method']==method and r['branch']==branch and r['queries']==q)
                contrasts.append(dict(method=method,queries=q,comparison='full-minus-'+branch,actual_gain_difference=full['actual_gain']-other['actual_gain'],continuation_gain_difference=full['continuation_gain']-other['continuation_gain']))
    write(OUT/'results.json',dict(status='COMPUTED_AUDIT_PENDING',traces=traces,nodes=rows,summaries=summaries,contrasts=contrasts,outer_seconds=time.perf_counter()-start,execution_lock_sha256=sha(OUT/'execution_lock.json'),new_full_trainings=0,cumulative_full_trainings=464,new_LLM_calls=0))
    print(dict(status='COMPUTED_AUDIT_PENDING',trajectories=len(traces),nodes=len(rows),contrasts=contrasts))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));globals()[parser.parse_args().action]()
