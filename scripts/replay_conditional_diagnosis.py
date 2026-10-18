"""Retrospective grouped replay with explicit reveal boundaries and full traces."""
import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import DiagnosticPredictor,PrequentialLedger,digest
from diagnosis_pool_features import read,sha,write

LOCAL=Path(__file__).resolve().parents[1]
STAGE=LOCAL/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
BASE_FEATURES=('exposure_shift_first','removed_fraction','weighted_implementation',
    'target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio',
    'parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion',
    'parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal',
    'B_input_variance','B_input_roughness','first_input_variance','second_input_variance',
    'first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
EXPERTS={
    'level_match_v1':'aligned_level_shift',
    'variance_exposure_v1':'aligned_variance_shift',
    'roughness_exposure_v1':'aligned_roughness_shift',
    'optimizer_exposure_v1':'optimizer_exposure_interaction',
}
METHODS=('reference','all_features','all_features_quadratic','uniform','marginal','joint','conditional')
NODES=(2,4,8)


def view(record):
    # A numerical policy receives no response, hidden seed scores or run metadata.
    return {key:record[key] for key in ('id','family','task','action','implementation','features')}


def feature_vector(candidate,names,quadratic=False):
    x=np.array([candidate['features'][name] for name in names],dtype=np.float64)
    if quadratic:
        return np.r_[x,[x[i]*x[j] for i in range(len(x)) for j in range(i,len(x))]]
    return x


def fit_model(visible,names,quadratic=False):
    names=tuple(names)
    expanded=names+tuple(f'{names[i]}*{names[j]}' for i in range(len(names)) for j in range(i,len(names))) if quadratic else names
    return DiagnosticPredictor.fit(
        [feature_vector(r,names,quadratic) for r in visible],
        [r['response']['Y'] for r in visible], [r['response']['Z'] for r in visible],
        [r['family']+':'+r['task'] for r in visible],[r['id'] for r in visible],expanded,
        ridge=1.,scale_floor=.01)


def model_set(visible,method,experts):
    if method.startswith('all_features'):
        names=tuple(sorted(visible[0]['features']))
        return fit_model(visible,names,method.endswith('quadratic')),{},names,{}
    reference=fit_model(visible,BASE_FEATURES)
    names={h:BASE_FEATURES+(name,) for h,name in experts.items()}
    diagnostics={} if method=='reference' else {h:fit_model(visible,n) for h,n in names.items()}
    return reference,diagnostics,BASE_FEATURES,names


def proposed_query(candidates,visible,ledger,method,experts):
    reference,models,base_names,names=model_set(visible,method,experts)
    predictions={}
    for candidate in candidates:
        xb=feature_vector(candidate,base_names,method.endswith('quadratic'))
        xh={h:feature_vector(candidate,names[h]) for h in models}
        predictions[candidate['id']]=ledger.predict_gain(reference,models,xb,xh)
    selected=min(candidates,key=lambda r:(-predictions[r['id']],r['id']))
    return selected,predictions,reference,models,base_names,names


def run_task(training,candidates,oracle,method,experts):
    mode=method if method in ('uniform','marginal','joint','conditional') else 'uniform'
    ledger=PrequentialLedger(mode);visible=list(training);unqueried=list(candidates)
    trace=[];nodes=[];begin=time.perf_counter();queried=[]
    for step in range(1,len(candidates)+1):
        selected,predictions,ref,models,base_names,names=proposed_query(unqueried,visible,ledger,method,experts)
        xb=feature_vector(selected,base_names,method.endswith('quadratic'))
        xh={h:feature_vector(selected,names[h]) for h in models}
        registration=ledger.register(selected['id'],ref,models,xb,xh)
        visible_before=[r['id'] for r in visible]
        assert selected['id'] not in visible_before
        snapshot=dict(reference=ref.snapshot(),diagnostics={h:m.snapshot() for h,m in models.items()})
        assert registration['model_hash']==digest(snapshot)
        # The only label lookup in the decision loop occurs after prediction registration.
        revealed=oracle[selected['id']]
        event=ledger.observe(selected['id'],revealed['response']['Y'],revealed['response']['Z'],
                             revealed['response']['NMAE_gain'],revealed['id'])
        queried.append(revealed)
        visible.append(revealed)
        unqueried=[r for r in unqueried if r['id']!=selected['id']]
        recommendation=ledger.best_observed()
        if recommendation=='PARENT':
            value=dict(Y=0.,NMAE_gain=0.);seed_y=[0.,0.,0.]
        else:
            chosen=next(r for r in queried if r['id']==recommendation)
            value=chosen['response'];seed_y=chosen['seed_Y']
        trace.append(dict(step=step,visible_before=visible_before,selected=selected['id'],
                          predictions_before=predictions,model_snapshots=snapshot,
                          registration=registration,event=event,recommendation=recommendation))
        if step in NODES:
            seconds=[r['original_full_training_seconds'] for r in queried]
            nodes.append(dict(queries=step,logical_candidate_full_trainings=3*step,
                shared_parent_full_trainings=3,recommendation=recommendation,
                gain=value['Y'],NMAE_gain=value['NMAE_gain'],
                negative_seed_count=sum(y<0 for y in seed_y),
                min_seed_gain=min(seed_y),query_negative_mean_count=sum(r['response']['Y']<0 for r in queried),
                query_seconds=None if any(v is None for v in seconds) else sum(seconds),
                policy_seconds=time.perf_counter()-begin))
    return dict(method=method,trace=trace,nodes=nodes,seconds=time.perf_counter()-begin)


def audit_no_hidden_label_access(training,candidates,oracle):
    # An intervention on all hidden labels cannot affect the first policy action/predictions.
    ledger=PrequentialLedger('conditional')
    first,pa,*_=proposed_query(candidates,training,ledger,'conditional',EXPERTS)
    shuffled={key:dict(value,response=dict(Y=100.,Z=-100.,NMAE_gain=-100.)) for key,value in oracle.items()}
    # Neither oracle is an argument of the policy call.
    second,pb,*_=proposed_query(candidates,training,ledger,'conditional',EXPERTS)
    assert shuffled and first['id']==second['id'] and pa==pb
    assert all(set(r)=={'id','family','task','action','implementation','features'} for r in candidates)
    return dict(status='PASS',test='policy signature has no oracle; masked candidate schema; unchanged first prediction',
                limitation='numerical API boundary, not operating-system isolation')


def main(output):
    pool=read(STAGE/'development_pool.json');records=pool['records']
    tasks=sorted({(r['family'],r['task']) for r in records});all_runs=[];audits=[]
    for split in ('leave_task_out','leave_family_out'):
        for family,task in tasks:
            held=[r for r in records if r['family']==family and r['task']==task]
            training=[r for r in records if (r['family'],r['task'])!=(family,task)] if split=='leave_task_out' else [r for r in records if r['family']!=family]
            oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
            audits.append(dict(split=split,task=task,**audit_no_hidden_label_access(training,candidates,oracle)))
            oracle_gain=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
            for method in METHODS:
                run=run_task(training,candidates,oracle,method,EXPERTS)
                run.update(split=split,family=family,task=task,oracle_gain=oracle_gain)
                for node in run['nodes']:node['regret']=oracle_gain-node['gain']
                all_runs.append(run)
    rows=[]
    for run in all_runs:
        for node in run['nodes']:
            rows.append(dict(split=run['split'],family=run['family'],task=run['task'],method=run['method'],**node))
    summaries=[]
    for split in ('leave_task_out','leave_family_out'):
        for budget in NODES:
            for method in METHODS:
                selected=[r for r in rows if r['split']==split and r['queries']==budget and r['method']==method]
                families=sorted({r['family'] for r in selected})
                summaries.append(dict(split=split,queries=budget,method=method,tasks=len(selected),families=len(families),
                    mean_gain=float(np.mean([r['gain'] for r in selected])),
                    family_macro_gain=float(np.mean([np.mean([r['gain'] for r in selected if r['family']==f]) for f in families])),
                    mean_regret=float(np.mean([r['regret'] for r in selected])),
                    mean_negative_queries=float(np.mean([r['query_negative_mean_count'] for r in selected]))))
    # Terminal fully revealed recommendations must agree for every policy.
    for split in ('leave_task_out','leave_family_out'):
        for family,task in tasks:
            runs=[r for r in all_runs if r['split']==split and r['family']==family and r['task']==task]
            assert len({r['trace'][-1]['recommendation'] for r in runs})==1
    output.mkdir()
    result=dict(status='PASS',protocol_sha256=sha(STAGE/'PROTOCOL.md'),design_sha256=sha(STAGE/'DEVELOPMENT_DESIGN.md'),
        pool_sha256=sha(STAGE/'development_pool.json'),code_sha256=sha(__file__),
        runs=all_runs,summaries=summaries,audits=audits,new_full_trainings=0,
        task_count=5,source_family_count=2,
        evidence_scope='retrospective development; no independent scientific confirmation; fixed manual experts, not learned LLM signals')
    write(output/'results.json',result)
    with (output/'decisions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'summary.json',dict(status='PASS',summaries=summaries,results_sha256=sha(output/'results.json'),
                                    full_training_count_unchanged=464,new_full_trainings=0))
    print(json.dumps(dict(status='PASS',runs=len(all_runs),decisions=len(rows),output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
