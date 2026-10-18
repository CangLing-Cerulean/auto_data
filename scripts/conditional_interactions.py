"""One gain learner with persistent signal features and evidence-gated interactions."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import DiagnosticPredictor, digest
from diagnosis_pool_features import read, sha, write
from replay_conditional_diagnosis import BASE_FEATURES, EXPERTS, feature_vector, view
from replay_window_signals import DualUnitGaussian

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
OLD=BASE/'20261018_measurement_generation_r4'
OUT=BASE/'20261018_conditional_interactions_r8'
METHODS={'A00':('fixed','none'),'A01':('fixed','conditional'),
         'A10':('mismatch','none'),'A11':('mismatch','conditional'),
         'JOINT':('mismatch','joint'),'ORDINARY_FIXED':('ordinary','none'),
         'ORDINARY_CONDITIONAL':('ordinary','conditional')}
CORE=tuple(sorted((*BASE_FEATURES,*EXPERTS.values())))
CORE_NAMES=CORE+tuple(f'{CORE[i]}*{CORE[j]}' for i in range(len(CORE)) for j in range(i,len(CORE)))

def signal_names(mode):
    return tuple(EXPERTS.values())+(() if mode=='fixed' else tuple(f'{mode}_measurement_{j}' for j in range(2)))

def normalizers(records,names):
    if not records:
        raise ValueError('Source records are required')
    tasks=sorted({(r['family'],r['task']) for r in records})
    output={}
    for name in names:
        values=[float(r['features'][name]) for r in records]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Source signals must be finite')
        peak=max(map(abs,values))
        if peak==0:
            output[name]=0.
            continue
        task_moments=[]
        for task in tasks:
            scaled=[float(r['features'][name])/peak for r in records if (r['family'],r['task'])==task]
            task_moments.append(math.fsum(value*value for value in scaled)/len(scaled))
        output[name]=peak*math.sqrt(math.fsum(task_moments)/len(tasks))
    return output

def gain_gate(rho):
    return float(np.exp(-.5*np.logaddexp(0.,-rho)))

def basis(record,names,norm,rho):
    values=list(feature_vector(record,CORE,True));columns=list(CORE_NAMES)
    for name in names:
        u=math.tanh(record['features'][name]/norm[name]) if norm[name]>0 else 0.
        values.append(u);columns.append('signal:'+name)
        for implementation in ('S','W'):
            gate=gain_gate(rho.get(name,{}).get(implementation,0.))
            values.append(u*gate*float(record['implementation']==implementation))
            columns.append('interaction:'+name+':'+implementation)
    return np.array(values),tuple(columns)

def fit(records,names,norm,rho):
    vectors=[basis(r,names,norm,rho) for r in records]
    x=np.array([v for v,_ in vectors]);columns=vectors[0][1]
    y=np.array([r['response']['Y'] for r in records]);z=np.array([r['response']['Z'] for r in records])
    groups=[r['family']+':'+r['task'] for r in records];ids=[r['id'] for r in records]
    return DiagnosticPredictor(DualUnitGaussian.fit(x,y,groups,ids,columns),
                               DualUnitGaussian.fit(np.column_stack([x,y]),z,groups,ids,(*columns,'observed_Y_scoring_only')))

def visible_response(record):
    result=view(record);result['response']={k:record['response'][k] for k in ('Y','Z','NMAE_gain')}
    return result

def policy(visible,candidates,names,norm,rho):
    gain=fit(visible,names,norm,rho)
    ref=fit(visible,(),norm,{})
    experts={name:fit(visible,(name,),norm,{}) for name in names}
    predictions={r['id']:gain.predict_gain(basis(r,names,norm,rho)[0]).mean for r in candidates}
    selected=min(candidates,key=lambda r:(-predictions[r['id']],r['id']))
    return selected,predictions,gain,ref,experts

def update_support(rho,scores,implementation,rule):
    after=copy.deepcopy(rho)
    if rule!='none':
        for name,value in scores.items():after[name][implementation]+=value[rule]
    return after

def choose(queried):
    eligible=[r for r in queried if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]
    return max(eligible,key=lambda r:(r['response']['Y'],r['id'])) if eligible else None

def run_task(training,held,method,norm,folder):
    mode,rule=METHODS[method];names=signal_names(mode)
    rho={name:dict(S=0.,W=0.) for name in names};visible=[visible_response(r) for r in training]
    oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
    trace=[];nodes=[];queried=[];initial=[];start=time.perf_counter();folder.mkdir()
    for step in range(1,len(held)+1):
        selected,predictions,gain,ref,experts=policy(visible,candidates,names,norm,rho)
        snapshots=dict(gain=gain.snapshot(),reference=ref.snapshot(),diagnostics={h:m.snapshot() for h,m in experts.items()})
        registration=dict(step=step,visible_before=[r['id'] for r in visible],candidate_ids=sorted(predictions),
                          selected=selected['id'],predictions_before=predictions,rho_before=copy.deepcopy(rho),
                          normalizers=norm,model_snapshots=snapshots,model_hash=digest(snapshots),
                          gain_std=gain.marginal.std)
        reg_path=folder/f'{step:02d}_registration.json';write(reg_path,registration)
        if step==1:
            # Seal the whole initial forecast before reading any target response.
            initial=[dict(id=r['id'],prediction=predictions[r['id']],std=gain.marginal.std) for r in candidates]
            write(folder/'initial_forecasts.json',dict(rows=initial,registration_sha256=sha(reg_path)))
        revealed=oracle[selected['id']]
        y,z=(revealed['response'][k] for k in ('Y','Z'))
        reference_scores=ref.log_scores(basis(selected,(),norm,{})[0],y,z)
        scores={name:{key:value-reference_scores[key] for key,value in model.log_scores(basis(selected,(name,),norm,{})[0],y,z).items()} for name,model in experts.items()}
        for score in scores.values():
            assert math.isclose(score['joint']-score['marginal'],score['conditional'],rel_tol=1e-9,abs_tol=1e-9)
        after=update_support(rho,scores,selected['implementation'],rule)
        event=dict(step=step,selected=selected['id'],registration_sha256=sha(reg_path),
                   response={k:revealed['response'][k] for k in ('Y','Z','NMAE_gain')},
                   scores=scores,rho_after=after,order='registered_then_scored_then_updated_then_refit_next_step')
        write(folder/f'{step:02d}_event.json',event)
        queried.append(revealed);visible.append(visible_response(revealed));rho=after
        candidates=[r for r in candidates if r['id']!=selected['id']]
        assert len({r['id'] for r in visible})==len(visible)
        chosen=choose(queried)
        trace.append(dict(step=step,selected=selected['id'],registration=reg_path.relative_to(OUT).as_posix(),registration_sha256=sha(reg_path),event=(folder/f'{step:02d}_event.json').relative_to(OUT).as_posix(),event_sha256=sha(folder/f'{step:02d}_event.json')))
        if step in (2,4,8):
            costs=[r['original_full_training_seconds'] for r in queried]
            gains=[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]
            actual=chosen['response']['Y'] if chosen else 0.
            nodes.append(dict(queries=step,recommendation=chosen['id'] if chosen else 'PARENT',gain=actual,
                              NMAE_gain=chosen['response']['NMAE_gain'] if chosen else 0.,
                              regret=max([0.,*gains])-actual,negative_seed_count=sum(y<0 for y in chosen['seed_Y']) if chosen else 0,
                              query_negative_mean_count=sum(r['response']['Y']<0 for r in queried),
                              target_harm_queries=sum(r['response']['d_target']<0 for r in queried),
                              complement_harm_queries=sum(r['response']['d_complement']<0 for r in queried),
                              query_seconds=None if any(c is None for c in costs) else sum(costs),logical_candidate_full_trainings=3*step,shared_parent_full_trainings=3))
    # Post-run forecast evaluation is never an input to policy or support updates.
    initial_evaluation=[]
    for prediction in initial:
        actual=oracle[prediction['id']]['response']['Y'];half=1.959963984540054*prediction['std']
        initial_evaluation.append(dict(**prediction,actual=actual,absolute_error=abs(prediction['prediction']-actual),covered=abs(prediction['prediction']-actual)<=half,interval_width=2*half))
    order_error=[]
    for i,left in enumerate(initial_evaluation):
        for right in initial_evaluation[i+1:]:
            truth=left['actual']-right['actual'];pred=left['prediction']-right['prediction']
            if truth!=0:order_error.append(1. if truth*pred<0 else (.5 if pred==0 else 0.))
    return dict(method=method,names=names,trace=trace,nodes=nodes,initial_evaluation=initial_evaluation,
                initial_pairwise_order_error=float(np.mean(order_error)) if order_error else None,seconds=time.perf_counter()-start)

def prepare():
    assert not (OUT/'execution_lock.json').exists()
    files={}
    for source in ('ETT','Hydraulic'):
        p=OLD/'replay'/f'{source}_augmented_records.json';files[p.relative_to(ROOT).as_posix()]=sha(p)
        for mode in ('ordinary','mismatch'):
            p=OLD/'generation'/f'{source}_{mode}/response.json';files[p.relative_to(ROOT).as_posix()]=sha(p)
    scripts=('conditional_interactions.py','conditional_diagnosis.py','diagnosis_pool_features.py','replay_conditional_diagnosis.py','replay_window_signals.py','window_measurement_dsl.py')
    lock=dict(protocol_sha256=sha(OUT/'PROTOCOL.md'),scripts={name:sha(ROOT/'scripts'/name) for name in scripts},inputs=files,methods=METHODS,normalizers={})
    for source in ('ETT','Hydraulic'):
        records=read(OLD/'replay'/f'{source}_augmented_records.json')['records'];training=[r for r in records if r['family']==source]
        lock['normalizers'][source]=normalizers(training,(*signal_names('mismatch'),*signal_names('ordinary')[4:]))
    write(OUT/'execution_lock.json',lock);print(dict(status='PREPARED',methods=len(METHODS),tasks=5,new_full_trainings=0,new_LLM_calls=0))

def run():
    lock=read(OUT/'execution_lock.json');assert sha(OUT/'PROTOCOL.md')==lock['protocol_sha256']
    for name,value in lock['scripts'].items():assert sha(ROOT/'scripts'/name)==value
    for name,value in lock['inputs'].items():assert sha(ROOT/name)==value
    assert not (OUT/'replay').exists();(OUT/'replay').mkdir()
    results=[];outer=time.perf_counter()
    for source in ('ETT','Hydraulic'):
        records=read(OLD/'replay'/f'{source}_augmented_records.json')['records'];training=[r for r in records if r['family']==source]
        for family,task in sorted({(r['family'],r['task']) for r in records if r['family']!=source}):
            held=[r for r in records if (r['family'],r['task'])==(family,task)]
            for method in METHODS:
                identity=f'{family}_{task}_{method}'
                result=run_task(training,held,method,lock['normalizers'][source],OUT/'replay'/identity)
                result.update(source_family=source,family=family,task=task)
                path=OUT/'replay'/f'{identity}.json';write(path,result)
                results.append(dict(family=family,task=task,method=method,path=path.relative_to(OUT).as_posix(),sha256=sha(path),seconds=result['seconds'],nodes=result['nodes']))
                print(identity, 'completed',flush=True)
    summaries=[]
    for method in METHODS:
        for count in (2,4,8):
            nodes=[node for result in results if result['method']==method for node in result['nodes'] if node['queries']==count]
            summaries.append(dict(method=method,queries=count,tasks=len(nodes),mean_gain=float(np.mean([n['gain'] for n in nodes])),mean_regret=float(np.mean([n['regret'] for n in nodes]))))
    write(OUT/'results.json',dict(status='COMPUTED_AUDIT_PENDING',runs=results,summaries=summaries,outer_seconds=time.perf_counter()-outer,execution_lock_sha256=sha(OUT/'execution_lock.json'),new_full_trainings=0,new_LLM_calls=0,cumulative_full_trainings=464))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'));args=parser.parse_args()
    prepare() if args.mode=='prepare' else run()
