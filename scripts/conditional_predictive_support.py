"""A separately frozen unit-scale ridge control for conditional-diagnosis development."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import LinearGaussian,DiagnosticPredictor,PrequentialLedger,digest
from diagnosis_pool_features import read,sha,write
from replay_conditional_diagnosis import BASE_FEATURES,EXPERTS,METHODS,NODES,feature_vector,view

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_predictive_support_r2'


class UnitScaleGaussian(LinearGaussian):
    @staticmethod
    def _fit_mean(x,y,group,ridge):
        names,counts=np.unique(group,return_counts=True);lookup=dict(zip(names,counts))
        weight=np.array([1/lookup[g] for g in group])
        center=np.average(x,axis=0,weights=weight)
        variance=np.average((x-center)**2,axis=0,weights=weight)
        scale=np.sqrt(np.maximum(variance,1.))
        design=np.column_stack([np.ones(len(x)),(x-center)/scale])
        penalty=np.eye(design.shape[1])*ridge;penalty[0,0]=0.
        beta=np.linalg.solve(design.T@(weight[:,None]*design)+penalty,design.T@(weight*y))
        return center,scale,beta


def fit(visible,names,quadratic=False):
    names=tuple(names)
    expanded=names+tuple(f'{names[i]}*{names[j]}' for i in range(len(names)) for j in range(i,len(names))) if quadratic else names
    x=np.array([feature_vector(r,names,quadratic) for r in visible])
    y=np.array([r['response']['Y'] for r in visible]);z=np.array([r['response']['Z'] for r in visible])
    groups=[r['family']+':'+r['task'] for r in visible];ids=[r['id'] for r in visible]
    return DiagnosticPredictor(UnitScaleGaussian.fit(x,y,groups,ids,expanded),
        UnitScaleGaussian.fit(np.column_stack([x,y]),z,groups,ids,[*expanded,'observed_Y_scoring_only']))


def run(training,held,method):
    ledger=PrequentialLedger(method if method in ('uniform','marginal','joint','conditional') else 'uniform')
    visible=list(training);oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
    names=tuple(sorted(training[0]['features'])) if method.startswith('all_features') else BASE_FEATURES
    extra={} if method=='reference' or method.startswith('all_features') else EXPERTS
    quadratic=method.endswith('quadratic');trace=[];nodes=[];begin=time.perf_counter()
    for step in range(1,len(held)+1):
        ref=fit(visible,names,quadratic)
        models={h:fit(visible,BASE_FEATURES+(name,)) for h,name in extra.items()}
        predictions={};vectors={}
        for candidate in candidates:
            xb=feature_vector(candidate,names,quadratic)
            xh={h:feature_vector(candidate,BASE_FEATURES+(extra[h],)) for h in models}
            predictions[candidate['id']]=ledger.predict_gain(ref,models,xb,xh)
            vectors[candidate['id']]=(xb,xh)
        selected=min(candidates,key=lambda r:(-predictions[r['id']],r['id']))
        xb,xh=vectors[selected['id']]
        registration=ledger.register(selected['id'],ref,models,xb,xh)
        snapshot=dict(reference=ref.snapshot(),diagnostics={h:m.snapshot() for h,m in models.items()})
        assert digest(snapshot)==registration['model_hash']
        before=[r['id'] for r in visible]
        revealed=oracle[selected['id']]
        event=ledger.observe(selected['id'],revealed['response']['Y'],revealed['response']['Z'],revealed['response']['NMAE_gain'],revealed['id'])
        visible.append(revealed);candidates=[r for r in candidates if r['id']!=selected['id']]
        recommendation=ledger.best_observed()
        chosen=oracle.get(recommendation);gain=chosen['response']['Y'] if chosen else 0.
        trace.append(dict(step=step,visible_before=before,selected=selected['id'],predictions_before=predictions,
            model_snapshots=snapshot,registration=registration,event=event,recommendation=recommendation))
        if step in NODES:
            queried=[oracle[e['event_id']] for e in ledger.events];cost=[r['original_full_training_seconds'] for r in queried]
            nodes.append(dict(queries=step,recommendation=recommendation,gain=gain,
                NMAE_gain=chosen['response']['NMAE_gain'] if chosen else 0.,
                negative_seed_count=sum(y<0 for y in chosen['seed_Y']) if chosen else 0,
                min_seed_gain=min(chosen['seed_Y']) if chosen else 0.,
                query_negative_mean_count=sum(r['response']['Y']<0 for r in queried),
                logical_candidate_full_trainings=step*3,shared_parent_full_trainings=3,
                query_seconds=None if any(v is None for v in cost) else sum(cost),policy_seconds=time.perf_counter()-begin))
    return dict(method=method,trace=trace,nodes=nodes,seconds=time.perf_counter()-begin)


def decompose(records):
    old=read(OLD/'manual_expert_replay/results.json');out=[]
    for prior in old['runs']:
        if prior['split']!='leave_family_out' or prior['method']!='reference':continue
        model=prior['trace'][0]['model_snapshots']['reference']['marginal']
        center,scale,beta=map(np.array,(model['center'],model['scale'],model['beta']))
        for r in records:
            if r['task']!=prior['task'] or r['family']!=prior['family']:continue
            x=np.array([r['features'][name] for name in model['feature_names']]);terms=(x-center)/scale*beta[1:]
            assert np.isfinite(terms).all()
            out.append(dict(id=r['id'],intercept=float(beta[0]),prediction=float(beta[0]+terms.sum()),
                components=[dict(name=name,value=float(x[i]),source_center=float(center[i]),source_scale=float(scale[i]),contribution=float(terms[i])) for i,name in enumerate(model['feature_names'])]))
    write(STAGE/'r1_extrapolation_decomposition.json',dict(status='PASS',records=out,source_sha256=sha(OLD/'manual_expert_replay/results.json'),kind='read-only algebra of frozen predictions; no adjusted forecasts'))


def conditional_packets(records):
    output=[]
    for family in ('ETT','Hydraulic'):
        source=[r for r in records if r['family']==family];facts=[]
        tasks=sorted({r['task'] for r in source});anonymous={t:f'T{i+1}' for i,t in enumerate(tasks)}
        for held in tasks:
            train=[r for r in source if r['task']!=held]
            for quadratic in (False,True):
                names=tuple(sorted(source[0]['features'])) if quadratic else BASE_FEATURES
                x=np.array([feature_vector(r,names,quadratic) for r in train]);y=np.array([r['response']['Y'] for r in train]);z=np.array([r['response']['Z'] for r in train]);groups=np.array([r['task'] for r in train])
                cy,sy,by=UnitScaleGaussian._fit_mean(x,y,groups,1.)
                cz,sz,bz=UnitScaleGaussian._fit_mean(np.column_stack([x,y]),z,groups,1.)
                for r in source:
                    if r['task']!=held:continue
                    q=feature_vector(r,names,quadratic);yp=float(np.r_[1.,(q-cy)/sy]@by)
                    zp=[float(np.r_[1.,(np.r_[q,seed_y]-cz)/sz]@bz) for seed_y in r['seed_Y']]
                    residual=[observed-predicted for observed,predicted in zip(r['seed_Z'],zp)]
                    qualified=abs(float(np.mean(residual)))>.01 and (all(e>.0025 for e in residual) or all(e<-.0025 for e in residual))
                    facts.append(dict(record=f'R{source.index(r)+1}',task=anonymous[held],predictor='quadratic' if quadratic else 'linear',features=r['features'],response=r['response'],seed_Y=r['seed_Y'],seed_Z=r['seed_Z'],Y_prediction=yp,Z_given_observed_seed_Y=zp,conditional_seed_residual=residual,qualified_conditional_mismatch=qualified,
                        frozen_mean_functions=dict(Y=dict(center=cy.tolist(),scale=sy.tolist(),beta=by.tolist()),Z=dict(center=cz.tolist(),scale=sz.tolist(),beta=bz.tolist())),train_records=[f'R{source.index(t)+1}' for t in train]))
        path=STAGE/f'{family}_conditional_source_packet.json';write(path,dict(family_role='anonymous source tasks only',facts=facts,held_target_records_included=False,LLM_calls=0))
        output.append(dict(source=family,records=len(source),point_predictions=len(facts),qualified_linear=sum(f['qualified_conditional_mismatch'] for f in facts if f['predictor']=='linear'),qualified_quadratic=sum(f['qualified_conditional_mismatch'] for f in facts if f['predictor']=='quadratic'),sha256=sha(path)))
    return output


def main():
    lock=read(STAGE/'execution_lock.json');assert sha(__file__)==lock['code_sha256'] and sha(STAGE/'PROTOCOL.md')==lock['protocol_sha256']
    records=read(OLD/'development_pool.json')['records'];assert sha(OLD/'development_pool.json')==lock['pool_sha256']
    decompose(records);runs=[];rows=[];static=[]
    for split in ('leave_task_out','leave_family_out'):
        for family,task in sorted({(r['family'],r['task']) for r in records}):
            held=[r for r in records if r['family']==family and r['task']==task]
            training=[r for r in records if (r['family'],r['task'])!=(family,task)] if split=='leave_task_out' else [r for r in records if r['family']!=family]
            best=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
            for method in METHODS:
                result=run(training,held,method);result.update(split=split,family=family,task=task,oracle_gain=best)
                for node in result['nodes']:node['regret']=best-node['gain'];rows.append(dict(split=split,family=family,task=task,method=method,**node))
                runs.append(result)
            if split=='leave_family_out':
                for method,names,quadratic in [('reference',BASE_FEATURES,False),('all_features',tuple(sorted(records[0]['features'])),False),('all_features_quadratic',tuple(sorted(records[0]['features'])),True)]:
                    model=fit(training,names,quadratic)
                    for r in held:
                        pred=model.predict_gain(feature_vector(r,names,quadratic))
                        static.append(dict(id=r['id'],family=family,task=task,method=method,mean=pred.mean,std=pred.std,Y=r['response']['Y'],absolute_error=abs(pred.mean-r['response']['Y']),covered95=abs(pred.mean-r['response']['Y'])<=1.959963984540054*pred.std,model_snapshot=model.snapshot()))
    for split in ('leave_task_out','leave_family_out'):
        for task in {r['task'] for r in records}:assert len({r['trace'][-1]['recommendation'] for r in runs if r['task']==task and r['split']==split})==1
    packets=conditional_packets(records)
    summaries=[]
    for split in ('leave_task_out','leave_family_out'):
        for method in METHODS:
            for node in NODES:
                subset=[r for r in rows if r['split']==split and r['method']==method and r['queries']==node]
                summaries.append(dict(split=split,method=method,queries=node,tasks=len(subset),mean_gain=float(np.mean([r['gain'] for r in subset])),mean_regret=float(np.mean([r['regret'] for r in subset]))))
    write(STAGE/'results.json',dict(status='PASS',runs=runs,static_predictions=static,conditional_packets=packets,lock_sha256=sha(STAGE/'execution_lock.json')))
    write(STAGE/'summary.json',dict(status='PASS',summaries=summaries,conditional_packets=packets,results_sha256=sha(STAGE/'results.json'),new_full_trainings=0,cumulative_full_trainings=464,LLM_calls=0))
    with (STAGE/'decisions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='PASS',runs=len(runs),nodes=len(rows),static_predictions=len(static),conditional_packets=packets)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run'))
    if parser.parse_args().action=='freeze':
        write(STAGE/'execution_lock.json',dict(code_sha256=sha(__file__),protocol_sha256=sha(STAGE/'PROTOCOL.md'),pool_sha256=sha(OLD/'development_pool.json'),new_full_trainings=0,new_LLM_calls=0,variant='unit_feature_scale_floor_1_only'))
    else:main()
