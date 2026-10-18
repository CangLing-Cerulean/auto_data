"""Independent r8 snapshot audit. Run only after parent supplies results ready.

No production imports or fit/predict/policy calls. Ridge is reconstructed by
thin SVD, unlike production's dual linear solve. All checks use saved summaries.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

STAGE=Path(__file__).resolve().parent
ROOT=STAGE.parents[3]
OLD=STAGE.parent/'20261018_measurement_generation_r4'
BASE=('exposure_shift_first','removed_fraction','weighted_implementation','target_weight_second_moment',
      'first_training_mass','parent_B_log_loss','parent_B_target_ratio','parent_B_local_contrast',
      'parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion','parent_B_learning_gain',
      'parent_B_best_fraction','parent_B_curve_reversal','B_input_variance','B_input_roughness',
      'first_input_variance','second_input_variance','first_input_roughness','second_input_roughness',
      'first_B_level_distance','second_B_level_distance')
FIXED=('aligned_level_shift','aligned_variance_shift','aligned_roughness_shift','optimizer_exposure_interaction')
CORE=tuple(sorted(BASE+FIXED))
CORE_COLUMNS=CORE+tuple(f'{CORE[i]}*{CORE[j]}' for i in range(len(CORE)) for j in range(i,len(CORE)))
METHODS={'A00':('fixed','none'),'A01':('fixed','conditional'),'A10':('mismatch','none'),
         'A11':('mismatch','conditional'),'JOINT':('mismatch','joint'),
         'ORDINARY_FIXED':('ordinary','none'),'ORDINARY_CONDITIONAL':('ordinary','conditional')}
CHECKS=[];ERRORS=[];FIT_CACHE={};COMPARISONS=0;HEADS=0;MAX_ERROR=0.;QUERY_COUNT=0

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value,message):
    if not value: raise AssertionError(message)
def close(actual,expected,context):
    global COMPARISONS,MAX_ERROR
    a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),context+': shape/nonfinite')
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    COMPARISONS+=a.size;MAX_ERROR=max(MAX_ERROR,error)
    require(np.allclose(a,b,rtol=2e-9,atol=1e-9),context+f': mismatch max_abs={error}')
def same(actual,expected,context):
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),context+': keys')
        for key,value in expected.items(): same(actual[key],value,context+'/'+str(key))
    elif isinstance(expected,(list,tuple)):
        require(isinstance(actual,(list,tuple)) and len(actual)==len(expected),context+': length')
        if expected and all(type(v) in (float,int) for v in expected): close(actual,expected,context)
        else:
            for i,(a,b) in enumerate(zip(actual,expected)): same(a,b,context+'/'+str(i))
    elif type(expected) in (int,float): close(actual,expected,context)
    else: require(actual==expected,context+': value')
def audited(name,function):
    try:
        details=function();CHECKS.append(dict(check=name,status='PASS',details=details));return details
    except Exception as exc:
        error=dict(check=name,status='FAIL',error=repr(exc),traceback=traceback.format_exc())
        CHECKS.append(error);ERRORS.append(error)

def gate(value):
    if value>=0: return math.sqrt(1/(1+math.exp(-value)))
    return math.exp(value/2)/math.sqrt(1+math.exp(value))
def vector(record,names,norm,rho):
    basic=[float(record['features'][name]) for name in CORE]
    values=basic+[basic[i]*basic[j] for i in range(len(basic)) for j in range(i,len(basic))]
    columns=list(CORE_COLUMNS)
    for name in names:
        signal=math.tanh(float(record['features'][name])/norm[name]) if norm[name]>0 else 0.
        values.append(signal);columns.append('signal:'+name)
        for implementation in ('S','W'):
            values.append(signal*gate(rho.get(name,{}).get(implementation,0.))*int(record['implementation']==implementation))
            columns.append('interaction:'+name+':'+implementation)
    return values,columns

def svd_mean(x,y,groups):
    counts=Counter(groups);w=np.asarray([1/counts[g] for g in groups])
    center=np.sum(w[:,None]*x,axis=0)/sum(w)
    scale=np.sqrt(np.maximum(np.sum(w[:,None]*(x-center)**2,axis=0)/sum(w),1.))
    design=(x-center)/scale;intercept=float(w@y/sum(w))
    a=design*np.sqrt(w)[:,None];target=np.sqrt(w)*(y-intercept)
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    coefficients=vh.T@((s/(1+s*s))*(u.T@target))
    return center,scale,np.r_[intercept,coefficients]

def head(records,names,norm,rho,conditional):
    global HEADS
    key=(tuple(r['id'] for r in records),tuple(names),tuple((q,rho.get(q,{}).get('S',0.),rho.get(q,{}).get('W',0.)) for q in names),conditional)
    if key in FIT_CACHE: return FIT_CACHE[key]
    vectors=[vector(r,names,norm,rho) for r in records]
    x=np.asarray([row[0] for row in vectors]);columns=vectors[0][1]
    y=np.asarray([r['response']['Y'] for r in records]);groups=np.asarray([r['family']+':'+r['task'] for r in records])
    if conditional:
        x=np.column_stack([x,y]);columns=columns+['observed_Y_scoring_only']
        y=np.asarray([r['response']['Z'] for r in records])
    residual=np.empty(len(y))
    for group in sorted(set(groups)):
        train=groups!=group;c,s,b=svd_mean(x[train],y[train],groups[train])
        residual[~train]=y[~train]-(b[0]+((x[~train]-c)/s)@b[1:])
    std=max(math.sqrt(math.fsum(float(np.mean(residual[groups==g]**2)) for g in sorted(set(groups)))/len(set(groups))),.01)
    c,s,b=svd_mean(x,y,groups)
    result=dict(center=c.tolist(),scale=s.tolist(),beta=b.tolist(),std=std,feature_names=columns,
                train_record_ids=[r['id'] for r in records],ridge=1.,scale_floor=.01)
    FIT_CACHE[key]=result;HEADS+=1
    return result
def models(records,names,norm,rho):
    return dict(marginal=head(records,names,norm,rho,False),conditional=head(records,names,norm,rho,True))
def predict(snapshot,x):
    return float(snapshot['beta'][0]+math.fsum((value-center)/scale*beta for value,center,scale,beta in zip(x,snapshot['center'],snapshot['scale'],snapshot['beta'][1:])))
def logdensity(value,mean,std):
    return -.5*math.log(2*math.pi)-math.log(std)-.5*((value-mean)/std)**2
def scores(snapshot,x,y,z):
    marginal=logdensity(y,predict(snapshot['marginal'],x),snapshot['marginal']['std'])
    conditional=logdensity(z,predict(snapshot['conditional'],[*x,y]),snapshot['conditional']['std'])
    return dict(marginal=marginal,conditional=conditional,joint=marginal+conditional)
def chosen(queried):
    candidates=[r for r in queried if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]
    return max(candidates,key=lambda r:(r['response']['Y'],r['id'])) if candidates else None

def validate_lock():
    for name,value in LOCK['scripts'].items(): require(sha(ROOT/'scripts'/name)==value,'source lock '+name)
    for name,value in LOCK['inputs'].items(): require(sha(ROOT/name)==value,'input lock '+name)
    require(sha(STAGE/'PROTOCOL.md')==LOCK['protocol_sha256'],'protocol lock')
    require(RESULTS['execution_lock_sha256']==sha(STAGE/'execution_lock.json'),'result execution lock')
    same(LOCK['methods'],{k:list(v) for k,v in METHODS.items()},'method matrix')
    require(RESULTS['status']=='COMPUTED_AUDIT_PENDING','computed status')
    require(RESULTS['new_full_trainings']==0 and RESULTS['new_LLM_calls']==0 and RESULTS['cumulative_full_trainings']==464,'training/call accounting')
    return dict(scripts=len(LOCK['scripts']),inputs=len(LOCK['inputs']))

def run_audit(entry):
    global QUERY_COUNT
    path=STAGE/entry['path'];require(sha(path)==entry['sha256'],'run sha')
    result=read(path);method=result['method'];mode,rule=METHODS[method]
    require((result['family'],result['task'],method)==(entry['family'],entry['task'],entry['method']),'run identity')
    source=result['source_family'];require(source!=result['family'],'source split')
    pool=POOLS[source];visible=[r for r in pool if r['family']==source]
    held=[r for r in pool if (r['family'],r['task'])==(result['family'],result['task'])]
    oracle={r['id']:r for r in held};remaining=dict(oracle)
    names=FIXED+(() if mode=='fixed' else tuple(f'{mode}_measurement_{i}' for i in range(2)))
    same(result['names'],names,'signal library')
    norm=LOCK['normalizers'][source];rho={q:dict(S=0.,W=0.) for q in names}
    require(len(result['trace'])==len(held),'query inventory')
    queried=[];nodes=[];initial=[]
    for step,trace in enumerate(result['trace'],1):
        context=f'{result["family"]}/{result["task"]}/{method}/{step}'
        regpath=STAGE/trace['registration'];eventpath=STAGE/trace['event']
        require(sha(regpath)==trace['registration_sha256'] and sha(eventpath)==trace['event_sha256'],context+': trace sha')
        reg,event=read(regpath),read(eventpath)
        require(reg['step']==step and event['step']==step and trace['step']==step,context+': event step')
        require(reg['visible_before']==[r['id'] for r in visible],context+': label visibility')
        require(reg['candidate_ids']==sorted(remaining),context+': candidate inventory')
        same(reg['normalizers'],norm,context+': source normalizers')
        same(reg['rho_before'],rho,context+': rho before')
        snapshots=reg['model_snapshots'];require(reg['model_hash']==digest(snapshots),context+': snapshot hash')
        expected=dict(gain=models(visible,names,norm,rho),reference=models(visible,(),norm,{}),
                      diagnostics={q:models(visible,(q,),norm,{}) for q in names})
        same(snapshots,expected,context+': independent SVD fit')
        predictions={identity:predict(snapshots['gain']['marginal'],vector(r,names,norm,rho)[0]) for identity,r in remaining.items()}
        same(reg['predictions_before'],predictions,context+': predictions')
        selected=min(remaining,key=lambda identity:(-predictions[identity],identity))
        require(selected==reg['selected']==trace['selected']==event['selected'],context+': selection')
        close(reg['gain_std'],snapshots['gain']['marginal']['std'],context+': gain std')
        if step==1:
            sealed=read(regpath.parent/'initial_forecasts.json')
            require(sealed['registration_sha256']==sha(regpath),context+': initial sealing')
            initial=[dict(id=r['id'],prediction=predictions[r['id']],std=reg['gain_std']) for r in held]
            same(sealed['rows'],initial,context+': initial forecasts')
        revealed=remaining.pop(selected);y,z=(revealed['response'][k] for k in ('Y','Z'))
        response={k:revealed['response'][k] for k in ('Y','Z','NMAE_gain')}
        same(event['response'],response,context+': revealed response')
        reference=scores(snapshots['reference'],vector(revealed,(),norm,{})[0],y,z)
        evidence={}
        for name in names:
            raw=scores(snapshots['diagnostics'][name],vector(revealed,(name,),norm,{})[0],y,z)
            evidence[name]={key:raw[key]-reference[key] for key in raw}
            close(evidence[name]['joint']-evidence[name]['marginal'],evidence[name]['conditional'],context+': score factorization')
        same(event['scores'],evidence,context+': log ratios')
        if rule!='none':
            for name in names: rho[name][revealed['implementation']]+=evidence[name][rule]
        same(event['rho_after'],rho,context+': bucket update')
        require(event['registration_sha256']==sha(regpath),context+': event registration link')
        require(event['order']=='registered_then_scored_then_updated_then_refit_next_step',context+': order claim')
        queried.append(revealed);visible.append(revealed);QUERY_COUNT+=1
        if step in (2,4,8):
            adopted=chosen(queried);actual=adopted['response']['Y'] if adopted else 0.
            costs=[r['original_full_training_seconds'] for r in queried]
            eligible=[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]
            nodes.append(dict(queries=step,recommendation=adopted['id'] if adopted else 'PARENT',gain=actual,
                              NMAE_gain=adopted['response']['NMAE_gain'] if adopted else 0.,regret=max([0.,*eligible])-actual,
                              negative_seed_count=sum(v<0 for v in adopted['seed_Y']) if adopted else 0,
                              query_negative_mean_count=sum(r['response']['Y']<0 for r in queried),
                              target_harm_queries=sum(r['response']['d_target']<0 for r in queried),
                              complement_harm_queries=sum(r['response']['d_complement']<0 for r in queried),
                              query_seconds=None if any(v is None for v in costs) else math.fsum(costs),
                              logical_candidate_full_trainings=3*step,shared_parent_full_trainings=3))
    same(result['nodes'],nodes,'run nodes');same(entry['nodes'],nodes,'index nodes')
    evaluation=[]
    for forecast in initial:
        actual=oracle[forecast['id']]['response']['Y'];half=1.959963984540054*forecast['std']
        evaluation.append(dict(**forecast,actual=actual,absolute_error=abs(forecast['prediction']-actual),
                               covered=abs(forecast['prediction']-actual)<=half,interval_width=2*half))
    same(result['initial_evaluation'],evaluation,'initial metrics')
    errors=[]
    for i,left in enumerate(evaluation):
        for right in evaluation[i+1:]:
            truth=left['actual']-right['actual'];pred=left['prediction']-right['prediction']
            if truth: errors.append(1. if truth*pred<0 else (.5 if pred==0 else 0.))
    same(result['initial_pairwise_order_error'],math.fsum(errors)/len(errors) if errors else None,'pairwise errors')
    close(entry['seconds'],result['seconds'],'runtime record')
    return dict(queries=len(held),nodes=len(nodes),method=method,task=result['task'])

def summaries_and_counts():
    identities={(r['family'],r['task'],r['method']) for r in RESULTS['runs']}
    require(len(RESULTS['runs'])==len(identities)==35,'35 unique paths')
    require(QUERY_COUNT==224,'224 independently checked queries')
    require(sum(len(r['nodes']) for r in RESULTS['runs'])==91,'91 nodes')
    expected=[]
    for method in METHODS:
        for count in (2,4,8):
            nodes=[n for r in RESULTS['runs'] if r['method']==method for n in r['nodes'] if n['queries']==count]
            require(len(nodes)==(3 if count==8 else 5),'task equal budget count')
            expected.append(dict(method=method,queries=count,tasks=len(nodes),
                                 mean_gain=math.fsum(n['gain'] for n in nodes)/len(nodes),
                                 mean_regret=math.fsum(n['regret'] for n in nodes)/len(nodes)))
    same(RESULTS['summaries'],expected,'macro summaries')
    return dict(paths=35,queries=224,nodes=91,summaries=len(expected))

def analysis_comparisons():
    analysis=read(STAGE/'analysis.json')
    require(analysis['results_sha256']==sha(STAGE/'results.json'),'analysis input hash')
    require(analysis['status']=='DESCRIPTIVE_REGISTERED_COMPARISONS','analysis status')
    runs=[read(STAGE/entry['path']) for entry in RESULTS['runs']]
    lookup={(r['family'],r['task'],r['method']):r for r in runs}
    forecasts=[]
    for method in sorted(METHODS):
        subset=[r for r in runs if r['method']==method]
        def task_average(key):
            return math.fsum(math.fsum(float(p[key]) for p in r['initial_evaluation'])/len(r['initial_evaluation']) for r in subset)/len(subset)
        forecasts.append(dict(method=method,tasks=len(subset),task_equal_MAE=task_average('absolute_error'),
                              task_equal_coverage=task_average('covered'),task_equal_width=task_average('interval_width'),
                              task_equal_pairwise_error=math.fsum(r['initial_pairwise_order_error'] for r in subset)/len(subset)))
    same(analysis['forecast_metrics'],forecasts,'forecast task-equal metrics')
    pairs=[]
    for comparison in ('A10','A01','JOINT','ORDINARY_CONDITIONAL'):
        for family,task in sorted({(r['family'],r['task']) for r in runs}):
            full,other=lookup[family,task,'A11'],lookup[family,task,comparison]
            for count in (2,4,8):
                lhs=next((n for n in full['nodes'] if n['queries']==count),None)
                if lhs is None: continue
                rhs=next(n for n in other['nodes'] if n['queries']==count)
                pairs.append(dict(family=family,task=task,comparison='A11-minus-'+comparison,queries=count,
                                  gain_difference=lhs['gain']-rhs['gain'],recommendation_changed=lhs['recommendation']!=rhs['recommendation'],
                                  query_set_changed=set(s['selected'] for s in full['trace'][:count])!=set(s['selected'] for s in other['trace'][:count])))
    require(len(pairs)==52,'52 registered paired comparisons')
    same(analysis['pairs'],pairs,'registered paired comparisons')
    changes=[]
    for run in runs:
        histories=[]
        source_pool={r['id']:r for r in POOLS[run['source_family']]}
        for trace in run['trace']:
            rho=read(STAGE/trace['event'])['rho_after']
            histories.append(dict(step=trace['step'],implementation=source_pool[trace['selected']]['implementation'],
                                  nonzero_support=sum(value!=0 for context in rho.values() for value in context.values()),
                                  max_abs_support=max(abs(value) for context in rho.values() for value in context.values())))
        changes.append(dict(family=run['family'],task=run['task'],method=run['method'],steps=histories))
    same(analysis['support_changes'],changes,'support histories')
    prior=read(OLD/'replay/summary.json');agent=read(STAGE.parent/'20261018_strong_agent_r5/results.json')
    baselines=[entry for entry in prior['summaries'] if entry['method'] in ('original_quadratic','window_fixed','mismatch_conditional')]
    for count in (2,4):
        decisions=[r['gain'] for r in agent['decisions'] if r['queries']==count]
        random=[r['expected_gain'] for r in agent['uniform_random'] if r['queries']==count]
        require(len(decisions)==len(random)==5,'archived five-task agent baselines')
        baselines.extend([dict(method='strong_agent_r5',queries=count,tasks=5,mean_gain=math.fsum(decisions)/5),
                          dict(method='random_r5',queries=count,tasks=5,mean_gain=math.fsum(random)/5)])
    same(analysis['existing_baselines'],baselines,'archived baseline references')
    nodes=[n for run in runs for n in run['nodes']]
    require(analysis['nodes']==len(nodes)==91,'analysis node count')
    require(analysis['known_cost_nodes']==sum(n['query_seconds'] is not None for n in nodes),'analysis unknown cost preservation')
    close(analysis['policy_seconds_sum'],math.fsum(r['seconds'] for r in runs),'analysis policy overhead')
    close(analysis['outer_seconds'],RESULTS['outer_seconds'],'analysis outer overhead')
    require(analysis['new_full_trainings']==0 and analysis['new_LLM_calls']==0 and analysis['cumulative_full_trainings']==464,'analysis training/call count')
    return dict(pairs=len(pairs),forecasts=len(forecasts),support_history_queries=sum(len(c['steps']) for c in changes),
                archived_baseline_rows=len(baselines),known_cost_nodes=analysis['known_cost_nodes'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results-ready',action='store_true');args=parser.parse_args()
    if not args.results_ready: parser.error('Parent results-ready authorization is required before execution')
    start=time.perf_counter();LOCK=read(STAGE/'execution_lock.json');RESULTS=read(STAGE/'results.json')
    POOLS={source:read(OLD/'replay'/f'{source}_augmented_records.json')['records'] for source in ('ETT','Hydraulic')}
    audited('locked_inputs_and_production_identity',validate_lock)
    for entry in RESULTS['runs']:
        audited(f'{entry["family"]}_{entry["task"]}_{entry["method"]}',lambda entry=entry:run_audit(entry))
    audited('matrix_nodes_and_task_equal_summaries',summaries_and_counts)
    audited('analysis_forecasts_52_pairs_support_cost_and_archived_baselines',analysis_comparisons)
    receipt=dict(status='PASS' if not ERRORS else 'FAIL',checks=CHECKS,numeric_comparisons=COMPARISONS,
                 independently_reconstructed_unique_model_heads=HEADS,queries_checked=QUERY_COUNT,
                 maximum_absolute_numeric_difference=MAX_ERROR,seconds=time.perf_counter()-start,
                 algorithm='task-weighted centered ridge via thin SVD; scalar fsum snapshot predictions and Gaussian log densities',
                 results_sha256=sha(STAGE/'results.json'),execution_lock_sha256=sha(STAGE/'execution_lock.json'),
                 analysis_sha256=sha(STAGE/'analysis.json'),
                 checker_sha256=sha(__file__),new_full_trainings=0,new_LLM_calls=0,
                 limitation='Numerical/event-chain audit, not operating-system label isolation or independent scientific confirmation')
    path=STAGE/'result_audit.json';attempt=2
    while path.exists(): path=STAGE/f'result_audit_attempt_{attempt}.json';attempt+=1
    with path.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],checks=len(CHECKS),failures=len(ERRORS),receipt=str(path))))
    raise SystemExit(0 if not ERRORS else 1)
