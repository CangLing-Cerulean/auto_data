"""Independent r6 dual-form audit. Execute only on parent 'results ready'.

No production imports, fit calls, policy execution, training, or LLM calls.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[3]
PREVIOUS=STAGE.parent/'20261018_measurement_generation_r4'
TASKS=[('ETT','ETTh1'),('ETT','ETTh2'),('Hydraulic','G1'),('Hydraulic','G2'),('Hydraulic','G3')]
METHODS=['original_quadratic','mismatch_conditional','mismatch_joint']
BRANCHES=['full','global_only','hidden_current']
SIGNALS=['mismatch_measurement_0','mismatch_measurement_1']
BASE_FEATURES=('exposure_shift_first','removed_fraction','weighted_implementation',
 'target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio',
 'parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion',
 'parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal',
 'B_input_variance','B_input_roughness','first_input_variance','second_input_variance',
 'first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
CHECKS=[];ERRORS=[];NUMERIC=[];COMPONENTS=0

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
    if not value: raise AssertionError(message)
def close(a,b,context):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),context+': shape/nonfinite')
    difference=float(np.max(np.abs(a-b))) if a.size else 0.
    NUMERIC.append(dict(context=context,max_absolute_error=difference))
    require(np.allclose(a,b,rtol=2e-9,atol=1e-9),context+f': numerical disagreement {difference}')
def same(a,b,context):
    if isinstance(b,dict):
        require(isinstance(a,dict) and set(a)==set(b),context+': keys')
        for k in b: same(a[k],b[k],context+'/'+str(k))
    elif isinstance(b,(list,tuple)):
        require(isinstance(a,(list,tuple)) and len(a)==len(b),context+': length')
        if b and all(type(y) in (int,float) for y in b): close(a,b,context)
        else:
            for i,(x,y) in enumerate(zip(a,b)): same(x,y,context+'/'+str(i))
    elif type(b) in (float,int) and not isinstance(b,bool): close(a,b,context)
    else: require(a==b,context+': value')
def audited(name,function):
    try:
        value=function();CHECKS.append(dict(check=name,status='PASS',details=value));return value
    except Exception as exc:
        ERRORS.append(dict(check=name,error=repr(exc)));CHECKS.append(dict(check=name,status='FAIL',error=repr(exc)))

def vector(record,names,quadratic=False):
    values=[record['features'][k] for k in names]
    return values+[values[i]*values[j] for i in range(len(values)) for j in range(i,len(values))] if quadratic else values
def public(record,mode):
    value={k:copy.deepcopy(record[k]) for k in ('id','family','task','action','implementation','features')}
    keys={'full':['Y','Z','NMAE_gain'],'global_only':['Y','NMAE_gain'],'hidden_current':[]}[mode]
    value['response']={k:record['response'][k] for k in keys}
    return value
def mean_fit(x,y,groups):
    counts=Counter(groups);weights=np.array([1./counts[g] for g in groups])
    center=np.sum(weights[:,None]*x,axis=0)/weights.sum()
    scale=np.sqrt(np.maximum(np.sum(weights[:,None]*(x-center)**2,axis=0)/weights.sum(),1.))
    design=(x-center)/scale;intercept=float(weights@y/weights.sum())
    a=np.sqrt(weights)[:,None]*design;target=np.sqrt(weights)*(y-intercept)
    beta=a.T@np.linalg.solve(np.eye(len(y))+a@a.T,target)
    require(np.linalg.norm(a.T@(a@beta-target)+beta)<=2e-6*(1+np.linalg.norm(beta)),'ridge stationarity')
    return center,scale,np.r_[intercept,beta]
def model_fit(records,names,quadratic,conditional):
    global COMPONENTS
    rows=[r for r in records if 'Y' in r['response'] and (not conditional or 'Z' in r['response'])]
    x=np.asarray([vector(r,names,quadratic) for r in rows],dtype=float)
    y=np.asarray([r['response']['Y'] for r in rows],dtype=float)
    expanded=list(names)+[f'{names[i]}*{names[j]}' for i in range(len(names)) for j in range(i,len(names))] if quadratic else list(names)
    if conditional: x=np.column_stack([x,y]);y=np.asarray([r['response']['Z'] for r in rows]);expanded+=['observed_Y_scoring_only']
    groups=np.asarray([r['family']+':'+r['task'] for r in rows]);residual=np.empty(len(y))
    require(len(set(groups))>=2,'two training tasks required')
    for g in sorted(set(groups)):
        train=groups!=g;c,s,b=mean_fit(x[train],y[train],groups[train])
        residual[~train]=y[~train]-(b[0]+((x[~train]-c)/s)@b[1:])
    std=max(float(np.sqrt(np.mean([np.mean(residual[groups==g]**2) for g in sorted(set(groups))]))),.01)
    center,scale,beta=mean_fit(x,y,groups);COMPONENTS+=1
    return dict(center=center.tolist(),scale=scale.tolist(),beta=beta.tolist(),std=std,feature_names=expanded,train_record_ids=[r['id'] for r in rows],ridge=1.,scale_floor=.01)
def heads(visible,names,quadratic=False):
    return dict(marginal=model_fit(visible,names,quadratic,False),conditional=model_fit(visible,names,quadratic,True))
def predict(model,x): return float(model['beta'][0]+((np.asarray(x)-model['center'])/model['scale'])@np.asarray(model['beta'][1:]))
def logpdf(model,x,y):
    mean=predict(model,x);s=model['std']
    return -.5*math.log(2*math.pi)-math.log(s)-.5*((y-mean)/s)**2
def weights(rho):
    if not rho:return {'reference':1.}
    keys=list(rho);logits=np.asarray([0.]+[rho[k]-math.log(len(keys)) for k in keys]);p=np.exp(logits-np.max(logits));p/=p.sum()
    return dict(zip(['reference',*keys],map(float,p)))
def choose(observed,continuation=False):
    eligible=[r for r in observed if (not continuation or r['query_step']>1) and r['response'].get('Y',-math.inf)>=.01 and r['response'].get('NMAE_gain',-math.inf)>=-.01]
    return max(eligible,key=lambda r:(r['response']['Y'],r['id']))['id'] if eligible else 'PARENT'

def frozen_files():
    lock=read(STAGE/'execution_lock.json')
    require(lock['tasks']==[list(t) for t in TASKS] and lock['methods']==METHODS and lock['branches']==BRANCHES,'exact matrix')
    require(lock['queries']==4 and lock['new_full_trainings']==0 and lock['new_LLM_calls']==0,'budget/accounting')
    for name,value in lock['scripts'].items(): require(sha(ROOT/'scripts'/name)==value,'source hash '+name)
    for key,path in [('protocol_sha256',STAGE/'PROTOCOL.md'),('prefixes_sha256',STAGE/'prefixes.json')]: require(sha(path)==lock[key],'frozen '+key)
    for family,value in lock['pools'].items(): require(sha(PREVIOUS/'replay'/f'{family}_augmented_records.json')==value,'pool hash '+family)
    for family,value in lock['definitions'].items(): require(sha(PREVIOUS/'generation'/f'{family}_mismatch/response.json')==value,'measurement hash '+family)
    require(read(STAGE/'source_publication_receipt.json')['status']=='exact_commit_and_tree_verified','source publication')
    gate=read(STAGE/'gate_audit.json');require(gate['status']=='PASS' and gate['code_sha256']==sha(ROOT/'scripts/feedback_forks.py'),'gate matches audited source')
    result=read(STAGE/'results.json')
    require(result['status']=='COMPUTED_AUDIT_PENDING','result status')
    require(result['execution_lock_sha256']==sha(STAGE/'execution_lock.json'),'result lock link')
    require((result['new_full_trainings'],result['cumulative_full_trainings'],result['new_LLM_calls'])==(0,464,0),'final accounting')
    actual=[(r['family'],r['task'],r['method'],r['branch']) for r in result['traces']]
    expected=[(*task,m,b) for task in TASKS for m in METHODS for b in BRANCHES]
    require(actual==expected and len(set(actual))==45,'exact 45 trace identities/order')
    for row in result['traces']:
        filename='_'.join([row['family'],row['task'],row['method'],row['branch']])+'.json'
        require(row['path']=='replay/'+filename,'trace path')
        require(sha(STAGE/row['path'])==row['sha256'],'trace hash '+filename)
        require(np.isfinite(row['seconds']) and row['seconds']>=0,'recorded seconds')
    require(np.isfinite(result['outer_seconds']) and result['outer_seconds']>=0,'outer seconds')
    return dict(trajectories=45,execution_lock_sha256=sha(STAGE/'execution_lock.json'))

def task_audit(family,task):
    source_family='Hydraulic' if family=='ETT' else 'ETT'
    records=read(PREVIOUS/'replay'/f'{source_family}_augmented_records.json')['records']
    source=[public(r,'full') for r in records if r['family']!=family]
    held=[r for r in records if (r['family'],r['task'])==(family,task)];oracle={r['id']:r for r in held}
    names=tuple(sorted(k for k in records[0]['features'] if not k.startswith(('window_','ordinary_measurement_','mismatch_measurement_'))))
    require(len(names)==26 and len(held)>=4,'features/candidates')
    prefix_file=read(STAGE/'prefixes.json');require(prefix_file['selection_uses_target_responses'] is False,'prefix permission declaration')
    matching=[p for p in prefix_file['prefixes'] if (p['family'],p['task'])==(family,task)];require(len(matching)==1,'one prefix per task');prefix=matching[0]
    prefix_model=heads(source,BASE_FEATURES)
    same(prefix['reference_snapshot'],prefix_model,f'{family}/{task}/prefix_model')
    pp={r['id']:predict(prefix_model['marginal'],vector(r,BASE_FEATURES)) for r in held}
    same(prefix['predictions'],pp,f'{family}/{task}/prefix_predictions')
    first=min(held,key=lambda r:(-pp[r['id']],r['id']))['id']
    require(prefix['selected']==first and prefix['response_revealed'] is False,'source-only prefix')
    result_nodes=[]; sequences={}; node_lookup={}
    for method in METHODS:
        for branch in BRANCHES:
            context='_'.join([family,task,method,branch]);folder=STAGE/'replay'/context
            result=read(STAGE/'replay'/(context+'.json'))
            require({k:result[k] for k in ('family','task','method','branch','prefix')}==dict(family=family,task=task,method=method,branch=branch,prefix=first),'trace identity')
            require(len(result['trace'])==4 and len(result['nodes'])==2,'trace/node length')
            visible=copy.deepcopy(source);observed=[];pending=list(held);queried=[];expected_nodes=[]
            rho={} if method=='original_quadratic' else {h:0. for h in SIGNALS}
            for step,row in enumerate(result['trace'],1):
                ctx=f'{context}/{step}';quadratic=method=='original_quadratic';base_names=names if quadratic else BASE_FEATURES
                ref=heads(visible,base_names,quadratic)
                diagnostics={} if quadratic else {h:heads(visible,(*BASE_FEATURES,h)) for h in SIGNALS}
                snapshots=dict(reference=ref,diagnostics=diagnostics);w=weights(rho)
                predictions={r['id']:w['reference']*predict(ref['marginal'],vector(r,base_names,quadratic))+sum(w[h]*predict(m['marginal'],vector(r,(*BASE_FEATURES,h))) for h,m in diagnostics.items()) for r in pending}
                selected=first if step==1 else min(pending,key=lambda r:(-predictions[r['id']],r['id']))['id']
                registration=row['registration'];mode=branch if step==1 else 'full'
                require(registration['step']==step and registration['selected']==selected and registration['report_mode']==mode,ctx+': adaptive selected/report')
                require(registration==read(folder/f'query_{step}_registration.json'),ctx+': persisted registration')
                require(row==read(folder/f'query_{step}_result.json'),ctx+': persisted result')
                same(registration['model_snapshots'],snapshots,ctx+'/models')
                require(registration['model_hash']==digest(registration['model_snapshots']),ctx+': snapshot hash')
                require(registration['visible_hash']==digest(visible),ctx+': visible hash')
                require(registration['visible_before']==[dict(id=r['id'],response_keys=sorted(r['response'])) for r in visible],ctx+': head permissions')
                same(registration['predictions_before'],predictions,ctx+'/predictions');same(registration['weights_before'],w,ctx+'/weights')
                revealed=public(oracle[selected],mode);require(row['visible_report']==revealed,ctx+': exact report visibility')
                before=dict(rho);scores={};applied={};y=revealed['response'].get('Y');z=revealed['response'].get('Z')
                xb=vector(oracle[selected],base_names,quadratic)
                for h,m in diagnostics.items():
                    xh=vector(oracle[selected],(*BASE_FEATURES,h))
                    if 'Z' in revealed['response']:
                        marginal=logpdf(m['marginal'],xh,y)-logpdf(ref['marginal'],xb,y)
                        conditional=logpdf(m['conditional'],[*xh,y],z)-logpdf(ref['conditional'],[*xb,y],z)
                        # Joint computed by separately summing the two log densities, as in the declared factorization.
                        joint=(logpdf(m['marginal'],xh,y)+logpdf(m['conditional'],[*xh,y],z))-(logpdf(ref['marginal'],xb,y)+logpdf(ref['conditional'],[*xb,y],z))
                        scores[h]=dict(marginal=marginal,conditional=conditional,joint=joint)
                        applied[h]=conditional if method.endswith('conditional') else joint
                    elif 'Y' in revealed['response']:
                        marginal=logpdf(m['marginal'],xh,y)-logpdf(ref['marginal'],xb,y)
                        scores[h]=dict(marginal=marginal,conditional=None,joint_observed=marginal)
                        applied[h]=marginal if method.endswith('joint') else 0.
                    else: scores[h]=dict(marginal=None,conditional=None,joint_observed=None);applied[h]=0.
                    rho[h]+=applied[h]
                same(row['event'],dict(scores=scores,applied=applied,before=before,after=dict(rho)),ctx+'/event')
                observed.append(dict(**revealed,query_step=step));visible.append(revealed);queried.append(selected)
                pending=[r for r in pending if r['id']!=selected]
                actual,continuation=choose(observed),choose(observed,True)
                require(row['recommendation']==actual and row['continuation_recommendation']==continuation,ctx+': adoption/continuation')
                if step in (2,4):
                    oracle_gain=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
                    def outcome(key):
                        r=oracle.get(key);gain=r['response']['Y'] if r else 0.
                        return dict(recommendation=key,gain=gain,regret=oracle_gain-gain,NMAE_gain=r['response']['NMAE_gain'] if r else 0.,negative_seed_count=sum(v<0 for v in r['seed_Y']) if r else 0)
                    costs=[oracle[k]['original_full_training_seconds'] for k in queried]
                    node=dict(queries=step,actual=outcome(actual),continuation=outcome(continuation),oracle_gain=oracle_gain,query_seconds=None if None in costs else sum(costs),logical_candidate_full_trainings=3*step,shared_parent_full_trainings=3)
                    expected_nodes.append(node);result_nodes.append(dict(family=family,task=task,method=method,branch=branch,**node));node_lookup[(method,branch,step)]=node
            same(result['nodes'],expected_nodes,context+'/nodes');sequences[(method,branch)]=list(queried)
            close(result['seconds'],next(r['seconds'] for r in read(STAGE/'results.json')['traces'] if (r['family'],r['task'],r['method'],r['branch'])==(family,task,method,branch)),context+'/seconds_link')
    pairs=[]
    for method in METHODS:
        for q in (2,4):
            full=node_lookup[(method,'full',q)]
            for branch in ('global_only','hidden_current'):
                other=node_lookup[(method,branch,q)]
                actual=full['actual']['gain']-other['actual']['gain'];continuation=full['continuation']['gain']-other['continuation']['gain']
                pairs.append(dict(family=family,task=task,method=method,queries=q,comparison='full-minus-'+branch,actual_gain_difference=actual,continuation_gain_difference=continuation,direct_first_adoption_component=actual-continuation,query_divergence_count=sum(a!=b for a,b in zip(sequences[(method,'full')][1:q],sequences[(method,branch)][1:q])),full_query_sequence=sequences[(method,'full')][:q],other_query_sequence=sequences[(method,branch)][:q]))
    return dict(nodes=result_nodes,paired=pairs,trajectories=9)

def aggregate(values):
    require(len(values)==5,'all five task audits required')
    result=read(STAGE/'results.json');nodes=[n for v in values for n in v['nodes']]
    require(len(nodes)==90,'90 independent nodes');same(result['nodes'],nodes,'aggregate_nodes')
    summaries=[];contrasts=[]
    for method in METHODS:
        for q in (2,4):
            for branch in BRANCHES:
                subset=[r for r in nodes if r['method']==method and r['branch']==branch and r['queries']==q]
                require(len(subset)==5,'summary has five tasks')
                summaries.append(dict(method=method,branch=branch,queries=q,actual_gain=float(np.mean([r['actual']['gain'] for r in subset])),continuation_gain=float(np.mean([r['continuation']['gain'] for r in subset]))))
            full=next(r for r in summaries if r['method']==method and r['branch']=='full' and r['queries']==q)
            for branch in ('global_only','hidden_current'):
                other=next(r for r in summaries if r['method']==method and r['branch']==branch and r['queries']==q)
                contrasts.append(dict(method=method,queries=q,comparison='full-minus-'+branch,actual_gain_difference=full['actual_gain']-other['actual_gain'],continuation_gain_difference=full['continuation_gain']-other['continuation_gain']))
    same(result['summaries'],summaries,'summaries');same(result['contrasts'],contrasts,'contrasts')
    pairs=[p for v in values for p in v['paired']]
    lookup={(n['family'],n['task'],n['method'],n['branch'],n['queries']):n for n in nodes}
    analysis=read(STAGE/'analysis.json')
    require(analysis['status']=='DESCRIPTIVE_PAIRED_SUMMARY' and len(analysis['pairs'])==60,'descriptive 60 pairs')
    expected_pairs={}
    for p in pairs:
        family,task,method,q=p['family'],p['task'],p['method'],p['queries'];branch=p['comparison'].removeprefix('full-minus-')
        full,other=lookup[(family,task,method,'full',q)],lookup[(family,task,method,branch,q)]
        expected_pairs[(family,task,method,p['comparison'],q)]=dict(family=family,task=task,method=method,comparison=p['comparison'],queries=q,prefix=p['full_query_sequence'][0],full_queries=p['full_query_sequence'],other_queries=p['other_query_sequence'],subsequent_action_positions_changed=p['query_divergence_count'],subsequent_queried_sets_differ=set(p['full_query_sequence'][1:])!=set(p['other_query_sequence'][1:]),actual_gain_difference=p['actual_gain_difference'],continuation_gain_difference=p['continuation_gain_difference'],actual_recommendation_changed=full['actual']['recommendation']!=other['actual']['recommendation'],continuation_recommendation_changed=full['continuation']['recommendation']!=other['continuation']['recommendation'])
    seen=set()
    for p in analysis['pairs']:
        key=tuple(p[k] for k in ('family','task','method','comparison','queries'))
        require(key in expected_pairs and key not in seen,'unique analysis pair');seen.add(key)
        same(p,expected_pairs[key],'analysis_pair/'+str(key))
    require(seen==set(expected_pairs),'all analysis pairs')
    known=sum(n['query_seconds'] is not None for n in nodes)
    require((analysis['known_cost_nodes'],analysis['nodes'],analysis['new_full_trainings'],analysis['new_LLM_calls'],analysis['cumulative_full_trainings'])==(known,90,0,0,464),'analysis cost/accounting')
    close(analysis['policy_seconds_sum'],sum(t['seconds'] for t in result['traces']),'analysis/policy_seconds_sum');close(analysis['outer_seconds'],result['outer_seconds'],'analysis/outer_seconds')
    descriptive_counts=dict(pairs=60,subsequent_query_sets_differ=sum(p['subsequent_queried_sets_differ'] for p in expected_pairs.values()),actual_gain_differs=sum(p['actual_gain_difference']!=0 for p in expected_pairs.values()),continuation_gain_differs=sum(p['continuation_gain_difference']!=0 for p in expected_pairs.values()),known_cost_nodes=known,direct_first_adoption_nonzero_pairs=sum(p['direct_first_adoption_component']!=0 for p in pairs))
    return dict(trajectories=45,logical_queries=180,nodes=90,summaries=summaries,contrasts=contrasts,per_task_pairs=pairs,descriptive_counts=descriptive_counts,query_divergence_counts=[dict(method=m,comparison='full-minus-'+b,queries=q,count=sum(p['query_divergence_count'] for p in pairs if p['method']==m and p['comparison']=='full-minus-'+b and p['queries']==q)) for m in METHODS for b in ('global_only','hidden_current') for q in (2,4)],cost_note='Recorded historical costs and three-seed logical query accounting; no new training. Policy timing is not an end-to-end equal-budget advantage.',scientific_interpretation='Actual recommendation gains include making the first result directly adoptable. Continuation gains exclude that first result and assess later choices. Zero or negative continuation differences do not establish a positive feedback selection effect.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results-ready',action='store_true');args=parser.parse_args()
    require(args.results_ready,'Parent authorization required: execute only after results ready, then pass --results-ready')
    start=time.perf_counter();audited('frozen_inputs_and_matrix',frozen_files);values=[]
    for family,task in TASKS:
        value=audited(f'task_{family}_{task}',lambda family=family,task=task:task_audit(family,task))
        if value is not None:values.append(value)
    summary=audited('nodes_summaries_and_paired_contrasts',lambda:aggregate(values))
    target=STAGE/'result_audit.json';previous=read(target) if target.exists() else None
    if previous and previous['status']=='PASS':
        previous={k:previous[k] for k in ('status','failures','independent_model_components','numeric_comparisons','max_absolute_numeric_difference','auditor_sha256','results_sha256')}
    report=dict(status='PASS' if not ERRORS else 'FAIL',checks=CHECKS,failures=ERRORS,independent_model_components=COMPONENTS,numeric_comparisons=len(NUMERIC),max_absolute_numeric_difference=max([0.]+[r['max_absolute_error'] for r in NUMERIC]),numeric_comparison_details=NUMERIC,summary=summary,audit_seconds=time.perf_counter()-start,code_sha256=sha(ROOT/'scripts/feedback_forks.py'),auditor_sha256=sha(__file__),results_sha256=sha(STAGE/'results.json') if (STAGE/'results.json').exists() else None,new_full_trainings=0,new_LLM_calls=0,previous_attempt=previous,limitations=['Artifact hashes and persisted registration files establish consistency, not cryptographic proof of historical write order or absence of undocumented outside calls.','The necessary gate checked the runtime ordering and local hidden-response perturbations; this result audit reconstructs all visible histories, two head-specific training subsets, models, scores and choices without importing production code.','Known development assets from two families, one deterministic trajectory per cell; no independent source confirmation, mechanism-label accuracy, or significance claim.','Only locked augmented feature/response pools and r6 artifacts inspected. No raw arrays, D/test, server, new policy runs, LLM calls, or full training.'])
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],checks=len(CHECKS),failures=ERRORS,independent_model_components=COMPONENTS,numeric_comparisons=len(NUMERIC))))
    if ERRORS:raise SystemExit(1)
if __name__=='__main__':main()
