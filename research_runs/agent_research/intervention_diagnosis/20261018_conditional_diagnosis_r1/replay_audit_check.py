import json, pathlib, hashlib, csv, math, collections
import numpy as np
root=pathlib.Path.cwd(); stage=root/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'; out=stage/'manual_expert_replay'
data=json.loads((out/'results.json').read_text(encoding='utf-8')); summary=json.loads((out/'summary.json').read_text(encoding='utf-8')); pool=json.loads((stage/'development_pool.json').read_text(encoding='utf-8')); records=pool['records']; byid={r['id']:r for r in records}
base=('exposure_shift_first','removed_fraction','weighted_implementation','target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio','parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion','parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal','B_input_variance','B_input_roughness','first_input_variance','second_input_variance','first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
experts={'level_match_v1':'aligned_level_shift','variance_exposure_v1':'aligned_variance_shift','roughness_exposure_v1':'aligned_roughness_shift','optimizer_exposure_v1':'optimizer_exposure_interaction'}
methods=('reference','all_features','all_features_quadratic','uniform','marginal','joint','conditional'); counts=collections.Counter(); errors=[]; maxima=collections.defaultdict(float)
def check(ok,label):
    counts['assertions']+=1
    if not ok: errors.append(label)
def near(a,b,label,tol=2e-10):
    a=float(a); b=float(b); delta=abs(a-b); maxima[label.split(':')[0]]=max(maxima[label.split(':')[0]],delta)
    check(math.isfinite(a) and math.isfinite(b) and delta<=tol*max(1.,abs(a),abs(b)),label)
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def feature(r,names):
    return np.array([r['features'][n] if '*' not in n else math.prod(r['features'][part] for part in n.split('*')) for n in names],dtype=float)
def mean(s,x):
    return float(np.r_[1.,(x-np.array(s['center']))/np.array(s['scale'])] @ np.array(s['beta']))
def scores(s,r):
    x=feature(r,s['marginal']['feature_names']); y=r['response']['Y']; z=r['response']['Z']
    ans={}
    for kind,xx,v in [('marginal',x,y),('conditional',np.r_[x,y],z)]:
        ss=s[kind]; mu=mean(ss,xx); sd=ss['std']; ans[kind]=-.5*math.log(2*math.pi)-math.log(sd)-.5*((v-mu)/sd)**2
    ans['joint']=ans['marginal']+ans['conditional']; return ans
check(len(byid)==32,'unique32records'); check(len(data['runs'])==70,'70runs'); check(len(base)==22,'22basefeatures')
for key,path in [('protocol_sha256',stage/'PROTOCOL.md'),('design_sha256',stage/'DEVELOPMENT_DESIGN.md'),('pool_sha256',stage/'development_pool.json'),('code_sha256',root/'scripts/replay_conditional_diagnosis.py')]: check(data[key]==hashlib.sha256(path.read_bytes()).hexdigest(),'hash:'+key)
check(summary['results_sha256']==hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),'hash:results')
check(data['summaries']==summary['summaries'],'summaryfilematches')
runkeys=[]; rows=[]; weights_range=[1.,0.]; zero_expert_weights=0
for run in data['runs']:
    key=(run['split'],run['family'],run['task'],run['method']); runkeys.append(key)
    held=[r for r in records if (r['family'],r['task'])==(run['family'],run['task'])]
    training=[r for r in records if (r['family'],r['task'])!=(run['family'],run['task'])] if run['split']=='leave_task_out' else [r for r in records if r['family']!=run['family']]
    visible=[r['id'] for r in training]; remaining={r['id'] for r in held}; queried=[]; rel={h:0. for h in experts} if run['method'] in ('uniform','marginal','joint','conditional') else {}; nodes=[]
    oracle=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]); near(run['oracle_gain'],oracle,'oracle')
    check(len(run['trace'])==len(held),'tracelength:'+str(key))
    for step,t in enumerate(run['trace'],1):
        prefix=str(key)+':'+str(step); counts['steps']+=1
        check(t['step']==step and t['visible_before']==visible,'visible:'+prefix)
        snap=t['model_snapshots']; diagnostics=snap['diagnostics']; check(set(diagnostics)==set(rel),'experts:'+prefix)
        if run['method'].startswith('all_features'):
            ns=tuple(sorted(records[0]['features']))
            expected=ns+tuple(ns[i]+'*'+ns[j] for i in range(len(ns)) for j in range(i,len(ns))) if run['method'].endswith('quadratic') else ns
        else: expected=base
        allmodels={'reference':snap['reference'],**diagnostics}
        for name,s in allmodels.items():
            counts['predictor_snapshots']+=1
            expected_names=expected if name=='reference' else base+(experts[name],)
            for kind in ('marginal','conditional'):
                m=s[kind]; counts['component_snapshots']+=1
                check(m['train_record_ids']==visible,'train_ids:'+prefix+name+kind)
                check(m['feature_names']==list(expected_names)+( ['observed_Y_scoring_only'] if kind=='conditional' else []),'feature_names:'+prefix+name+kind)
                check(m['ridge']==1. and m['scale_floor']==.01 and m['std']>=.01,'model_params:'+prefix)
                check(len(m['beta'])==len(m['feature_names'])+1 and len(m['scale'])==len(m['center'])==len(m['feature_names']),'shape:'+prefix)
                check(np.isfinite(m['beta']).all() and np.isfinite(m['center']).all() and np.isfinite(m['scale']).all() and min(m['scale'])>0,'finite:'+prefix)
        sh=digest(snap); check(t['registration']['model_hash']==sh==t['event']['model_hash'],'snap_hash:'+prefix)
        names=list(diagnostics)
        if names:
            logits=np.array([0.]+[rel[h]-math.log(len(names)) for h in names]); logits-=logits.max(); weights=np.exp(logits);weights/=weights.sum()
        else: weights=np.array([1.])
        weights_range=[min(weights_range[0],float(weights.min())),max(weights_range[1],float(weights.max()))]; zero_expert_weights+=sum(weights[1:]==0)
        near(weights.sum(),1,'weights_sum')
        check(set(t['predictions_before'])==remaining,'prediction_ids:'+prefix)
        computed={}
        for rid in remaining:
            rr=byid[rid]; pred=sum(float(w)*mean(allmodels[name]['marginal'],feature(rr,allmodels[name]['marginal']['feature_names'])) for name,w in zip(['reference']+names,weights))
            computed[rid]=pred; near(t['predictions_before'][rid],pred,'prediction:'+prefix+rid); counts['candidate_predictions']+=1
        selected=min(remaining,key=lambda rid:(-computed[rid],rid)); check(selected==t['selected'],'selected:'+prefix)
        reg=t['registration']; check(reg['event_id']==selected,'registration_id:'+prefix)
        near(reg['reference_gain'],mean(snap['reference']['marginal'],feature(byid[selected],expected)),'registration_reference')
        check(set(reg['gain_predictions'])==set(diagnostics),'registration_experts:'+prefix)
        for h,m in diagnostics.items(): near(reg['gain_predictions'][h],mean(m['marginal'],feature(byid[selected],m['marginal']['feature_names'])),'registration_expert')
        ev=t['event']; rr=byid[selected]; check(ev['event_id']==selected==ev['source_response_id'],'event_id:'+prefix)
        check(ev['order']=='registered_prediction_scored_before_any_refit','event_order:'+prefix)
        check(set(ev['before'])==set(rel)==set(ev['after'])==set(ev['scores']),'event_keys:'+prefix)
        for k in ('Y','Z','NMAE_gain'): near(ev['response'][k],rr['response'][k],'event_response')
        baseline=scores(snap['reference'],rr)
        for h,m in diagnostics.items():
            near(ev['before'][h],rel[h],'reliability_before'); sc=scores(m,rr); delta={k:sc[k]-baseline[k] for k in sc};counts['expert_score_triples']+=1
            for k,v in delta.items(): near(ev['scores'][h][k],v,'logscore:'+k)
            near(delta['conditional'],delta['joint']-delta['marginal'],'Econd_factorization')
            if run['method']!='uniform': rel[h]+=delta[run['method']]
            near(ev['after'][h],rel[h],'reliability_after')
        queried.append(rr); visible.append(selected); remaining.remove(selected)
        eligible=[r for r in queried if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]; chosen=max(eligible,key=lambda r:(r['response']['Y'],r['id'])) if eligible else None; recommendation=chosen['id'] if chosen else 'PARENT'
        check(t['recommendation']==recommendation,'recommendation:'+prefix)
        if step in (2,4,8):
            n=next(n for n in run['nodes'] if n['queries']==step); nodes.append(n)
            check(n['recommendation']==recommendation and n['logical_candidate_full_trainings']==step*3 and n['shared_parent_full_trainings']==3,'node_ids:'+prefix)
            gain=chosen['response']['Y'] if chosen else 0.; nmae=chosen['response']['NMAE_gain'] if chosen else 0.; seeds=chosen['seed_Y'] if chosen else [0.,0.,0.]
            for k,v in [('gain',gain),('NMAE_gain',nmae),('regret',oracle-gain),('min_seed_gain',min(seeds)),('negative_seed_count',sum(y<0 for y in seeds)),('query_negative_mean_count',sum(r['response']['Y']<0 for r in queried))]: near(n[k],v,'node:'+k)
            times=[r['original_full_training_seconds'] for r in queried]
            if any(t is None for t in times): check(n['query_seconds'] is None,'node_cost_null')
            else: near(n['query_seconds'],sum(times),'node_cost')
            check(0<=n['policy_seconds']<=run['seconds'],'node_policy_time')
            rows.append(dict(split=run['split'],family=run['family'],task=run['task'],method=run['method'],**n));counts['nodes']+=1
    check(len(nodes)==len(run['nodes']),'node_count:'+str(key))
check(len(set(runkeys))==70,'runkeys_unique')
for s in data['summaries']:
    subset=[r for r in rows if r['split']==s['split'] and r['queries']==s['queries'] and r['method']==s['method']]; fam=sorted({r['family'] for r in subset})
    check(s['tasks']==len(subset) and s['families']==len(fam),'summary_counts')
    for k,v in [('mean_gain',np.mean([r['gain'] for r in subset])),('family_macro_gain',np.mean([np.mean([r['gain'] for r in subset if r['family']==f]) for f in fam])),('mean_regret',np.mean([r['regret'] for r in subset])),('mean_negative_queries',np.mean([r['query_negative_mean_count'] for r in subset]))]: near(s[k],v,'summary:'+k)
    counts['summaries']+=1
with (out/'decisions.csv').open(encoding='utf-8',newline='') as f: csvrows=list(csv.DictReader(f))
check(len(csvrows)==len(rows),'csv_length')
for a,b in zip(csvrows,rows):
    check(set(a)==set(b),'csv_columns')
    for k,v in b.items():
        if isinstance(v,(int,float)): near(float(a[k]),v,'csv:'+k)
        elif v is None: check(a[k]=='','csv_null:'+k)
        else: check(a[k]==v,'csv_text:'+k)
    counts['csv_rows']+=1
for split in ('leave_task_out','leave_family_out'):
    for family,task in sorted({(r['family'],r['task']) for r in records}):
        rr=[r for r in data['runs'] if (r['split'],r['family'],r['task'])==(split,family,task)]
        check(len({r['trace'][-1]['recommendation'] for r in rr})==1,'terminal_agreement')
        for n in (2,4):
            selected=[next(x for x in r['nodes'] if x['queries']==n) for r in rr if r['method'] in ('conditional','joint','uniform')]
            check(len({(x['recommendation'],x['gain']) for x in selected})==1,'equal_conditional_joint_uniform_node')
print(json.dumps({'status':'PASS' if not errors else 'FAIL','counts':dict(counts),'errors':errors[:30],'error_count':len(errors),'max_abs_differences':dict(maxima),'weights_range':weights_range,'zero_expert_weights':int(zero_expert_weights),'refitting_performed':False,'files_written':0},ensure_ascii=False))
