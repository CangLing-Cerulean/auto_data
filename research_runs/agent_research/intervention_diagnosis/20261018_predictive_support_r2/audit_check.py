"""Independent read-only numerical audit; only audit.json is written.

Rebuilds ridge solutions using a centered weighted dual system (not the
production primal solver), including leave-task-out residual standard errors.
Reads only frozen aggregate development records and r1/r2 result assets.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
STAGE = Path(__file__).resolve().parent
OLD = STAGE.parent / '20261018_conditional_diagnosis_r1'
started = time.perf_counter()
counts = collections.Counter()
errors = []
maxima = collections.defaultdict(float)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(ok, label):
    counts['assertions'] += 1
    if not ok:
        errors.append(label)


def near(a, b, label, tol=2e-9):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        check(False, label + ':shape')
        return
    delta = float(np.max(np.abs(a-b), initial=0.))
    maxima[label.split(':')[0]] = max(maxima[label.split(':')[0]], delta)
    check(np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a-b) <= tol*np.maximum(1., np.maximum(np.abs(a), np.abs(b)))), label)


base = ('exposure_shift_first','removed_fraction','weighted_implementation','target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio','parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion','parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal','B_input_variance','B_input_roughness','first_input_variance','second_input_variance','first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
experts = {'level_match_v1':'aligned_level_shift','variance_exposure_v1':'aligned_variance_shift','roughness_exposure_v1':'aligned_roughness_shift','optimizer_exposure_v1':'optimizer_exposure_interaction'}
methods = ('reference','all_features','all_features_quadratic','uniform','marginal','joint','conditional')
records = read(OLD/'development_pool.json')['records']
byid = {r['id']: r for r in records}
data, summary, lock = (read(STAGE/f) for f in ('results.json','summary.json','execution_lock.json'))


def names_for(method):
    if not method.startswith('all_features'):
        return base
    ns = tuple(sorted(records[0]['features']))
    return ns + tuple(ns[i]+'*'+ns[j] for i in range(len(ns)) for j in range(i,len(ns))) if method.endswith('quadratic') else ns


def feature(record, names):
    return np.array([math.prod(record['features'][p] for p in n.split('*')) for n in names])


def dual_fit(x, y, groups):
    tasks, inverse, size = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1./size[inverse]
    center = np.average(x, axis=0, weights=weight)
    variance = np.average((x-center)**2, axis=0, weights=weight)
    scale = np.sqrt(np.maximum(variance, 1.))
    centered_x = (x-center)/scale
    mean_y = np.average(y, weights=weight)
    a = np.sqrt(weight)[:,None]*centered_x
    slope = a.T @ np.linalg.solve(a@a.T+np.eye(len(y)), np.sqrt(weight)*(y-mean_y))
    return center, scale, np.r_[mean_y, slope]


fit_cache = {}


def rebuild(ids, names, kind):
    key = (tuple(ids), tuple(names), kind)
    if key not in fit_cache:
        rows = [byid[rid] for rid in ids]
        x = np.array([feature(r,names) for r in rows])
        y = np.array([r['response']['Y' if kind=='marginal' else 'Z'] for r in rows])
        if kind == 'conditional':
            x = np.column_stack([x,[r['response']['Y'] for r in rows]])
        groups = np.array([r['family']+':'+r['task'] for r in rows])
        center, scale, beta = dual_fit(x,y,groups)
        residual = np.empty(len(y))
        for held in np.unique(groups):
            train = groups != held
            cc, ss, bb = dual_fit(x[train],y[train],groups[train])
            residual[~train] = y[~train]-np.c_[np.ones((~train).sum()),(x[~train]-cc)/ss]@bb
        std = max(.01,float(np.sqrt(np.mean([np.mean(residual[groups==g]**2) for g in np.unique(groups)]))))
        fit_cache[key] = (center,scale,beta,std)
        counts['independent_fits'] += 1
    return fit_cache[key]


def audit_model(snapshot, ids, names, label):
    for kind in ('marginal','conditional'):
        s = snapshot[kind]
        check(s['train_record_ids']==ids, 'train_ids:'+label+kind)
        check(s['feature_names']==list(names)+(['observed_Y_scoring_only'] if kind=='conditional' else []), 'feature_names:'+label+kind)
        check(s['ridge']==1. and s['scale_floor']==.01, 'ridge_parameters:'+label+kind)
        center,scale,beta,std = rebuild(ids,names,kind)
        for field,value in [('center',center),('scale',scale),('beta',beta),('std',std)]:
            near(s[field],value, 'fit_'+field+':'+label+kind)
        counts['component_snapshots'] += 1


def mean(s,x):
    return float(np.r_[1.,(x-np.array(s['center']))/np.array(s['scale'])]@np.array(s['beta']))


def scores(s,r):
    x=feature(r,s['marginal']['feature_names'])
    out={}
    for kind,xx,yy in [('marginal',x,r['response']['Y']),('conditional',np.r_[x,r['response']['Y']],r['response']['Z'])]:
        ss=s[kind]
        out[kind]=-.5*math.log(2*math.pi)-math.log(ss['std'])-.5*((yy-mean(ss,xx))/ss['std'])**2
    out['joint']=out['marginal']+out['conditional']
    return out


for key,path in [('code_sha256',ROOT/'scripts/conditional_predictive_support.py'),('protocol_sha256',STAGE/'PROTOCOL.md'),('pool_sha256',OLD/'development_pool.json')]:
    check(lock[key]==sha(path),'hash:'+key)
check(data['lock_sha256']==sha(STAGE/'execution_lock.json'),'hash:lock')
check(summary['results_sha256']==sha(STAGE/'results.json'),'hash:results')
check(len(records)==len(byid)==32,'32_unique_records')
check(len(data['runs'])==70,'70_runs')
check(summary['new_full_trainings']==0 and summary['cumulative_full_trainings']==464 and summary['LLM_calls']==0,'training_LLM_counts')
runkeys=[]
rows=[]
for run in data['runs']:
    key=(run['split'],run['family'],run['task'],run['method'])
    runkeys.append(key)
    held=[r for r in records if (r['family'],r['task'])==(run['family'],run['task'])]
    training=[r for r in records if (r['family'],r['task'])!=(run['family'],run['task'])] if run['split']=='leave_task_out' else [r for r in records if r['family']!=run['family']]
    visible=[r['id'] for r in training]
    remaining={r['id'] for r in held}
    queried=[]
    rel={h:0. for h in experts} if run['method'] in ('uniform','marginal','joint','conditional') else {}
    oracle=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
    near(run['oracle_gain'],oracle,'oracle')
    check(len(run['trace'])==len(held),'trace_length:'+str(key))
    for step,t in enumerate(run['trace'],1):
        label=str(key)+':'+str(step)
        counts['steps']+=1
        check(t['step']==step and t['visible_before']==visible,'visible_boundary:'+label)
        snap=t['model_snapshots']
        check(set(snap['diagnostics'])==set(rel),'diagnostic_versions:'+label)
        allmodels={'reference':snap['reference'],**snap['diagnostics']}
        for name,m in allmodels.items():
            audit_model(m,visible,names_for(run['method']) if name=='reference' else base+(experts[name],),label+name)
            counts['predictor_snapshots']+=1
        check(digest(snap)==t['registration']['model_hash']==t['event']['model_hash'],'model_hash:'+label)
        names=list(snap['diagnostics'])
        logits=np.array([0.]+[rel[h]-math.log(len(names)) for h in names]) if names else np.array([0.])
        logits-=logits.max()
        weight=np.exp(logits);weight/=weight.sum()
        check(set(t['predictions_before'])==remaining,'prediction_boundary:'+label)
        computed={}
        for rid in remaining:
            computed[rid]=sum(float(w)*mean(m['marginal'],feature(byid[rid],m['marginal']['feature_names'])) for m,w in zip(allmodels.values(),weight))
            near(t['predictions_before'][rid],computed[rid],'prediction:'+label+rid)
            counts['candidate_predictions']+=1
        selected=min(remaining,key=lambda rid:(-computed[rid],rid))
        check(t['selected']==selected,'selected:'+label)
        reg=t['registration'];ev=t['event'];rr=byid[selected]
        check(reg['event_id']==ev['event_id']==ev['source_response_id']==selected,'event_identity:'+label)
        near(reg['reference_gain'],mean(snap['reference']['marginal'],feature(rr,names_for(run['method']))),'registration_reference')
        check(set(reg['gain_predictions'])==set(rel),'registration_expert_keys:'+label)
        for h,m in snap['diagnostics'].items():
            near(reg['gain_predictions'][h],mean(m['marginal'],feature(rr,m['marginal']['feature_names'])),'registration_expert')
        check(ev['order']=='registered_prediction_scored_before_any_refit','event_order:'+label)
        check(set(ev['before'])==set(rel)==set(ev['after'])==set(ev['scores']),'event_keys:'+label)
        for k in ('Y','Z','NMAE_gain'):near(ev['response'][k],rr['response'][k],'event_response')
        baseline=scores(snap['reference'],rr)
        for h,m in snap['diagnostics'].items():
            near(ev['before'][h],rel[h],'reliability_before')
            sc=scores(m,rr);delta={k:sc[k]-baseline[k] for k in sc}
            for k,v in delta.items():near(ev['scores'][h][k],v,'logscore:'+k)
            near(delta['conditional'],delta['joint']-delta['marginal'],'score_factorization')
            if run['method']!='uniform':rel[h]+=delta[run['method']]
            near(ev['after'][h],rel[h],'reliability_after')
            counts['expert_score_triples']+=1
        queried.append(rr);visible.append(selected);remaining.remove(selected)
        eligible=[r for r in queried if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01]
        chosen=max(eligible,key=lambda r:(r['response']['Y'],r['id'])) if eligible else None
        recommendation=chosen['id'] if chosen else 'PARENT'
        check(t['recommendation']==recommendation,'recommendation:'+label)
        if step in (2,4,8):
            n=next(n for n in run['nodes'] if n['queries']==step)
            check(n['recommendation']==recommendation and n['logical_candidate_full_trainings']==step*3 and n['shared_parent_full_trainings']==3,'node_identity:'+label)
            gain=chosen['response']['Y'] if chosen else 0.
            seeds=chosen['seed_Y'] if chosen else [0.]*3
            for k,v in [('gain',gain),('regret',oracle-gain),('NMAE_gain',chosen['response']['NMAE_gain'] if chosen else 0.),('min_seed_gain',min(seeds)),('negative_seed_count',sum(y<0 for y in seeds)),('query_negative_mean_count',sum(r['response']['Y']<0 for r in queried))]:near(n[k],v,'node_'+k)
            times=[r['original_full_training_seconds'] for r in queried]
            if any(v is None for v in times):check(n['query_seconds'] is None,'missing_cost_preserved')
            else:near(n['query_seconds'],sum(times),'query_seconds')
            check(0<=n['policy_seconds']<=run['seconds'],'policy_seconds')
            rows.append(dict(split=run['split'],family=run['family'],task=run['task'],method=run['method'],**n));counts['nodes']+=1
    check(len(run['nodes'])==sum(n<=len(held) for n in (2,4,8)),'node_count:'+str(key))
check(len(set(runkeys))==70,'unique_runkeys')
check(set(runkeys)=={(split,family,task,method) for split in ('leave_task_out','leave_family_out') for family,task in {(r['family'],r['task']) for r in records} for method in methods},'complete_split_task_method_matrix')
check(counts['steps']==448 and counts['nodes']==182,'expected_steps_nodes')
for s in summary['summaries']:
    subset=[r for r in rows if r['split']==s['split'] and r['queries']==s['queries'] and r['method']==s['method']]
    check(s['tasks']==len(subset),'summary_task_count')
    near(s['mean_gain'],np.mean([r['gain'] for r in subset]),'summary_gain')
    near(s['mean_regret'],np.mean([r['regret'] for r in subset]),'summary_regret')
    counts['summaries']+=1
check(counts['summaries']==42,'42_summaries')
with (STAGE/'decisions.csv').open(encoding='utf-8',newline='') as stream:csvrows=list(csv.DictReader(stream))
check(len(csvrows)==len(rows),'csv_length')
for a,b in zip(csvrows,rows):
    check(set(a)==set(b),'csv_columns')
    for k,v in b.items():
        if isinstance(v,(int,float)):near(float(a[k]),v,'csv_'+k)
        elif v is None:check(a[k]=='','csv_null_'+k)
        else:check(a[k]==v,'csv_text_'+k)
    counts['csv_rows']+=1

static_seen=set()
for s in data['static_predictions']:
    rr=byid[s['id']]
    check((s['family'],s['task'])==(rr['family'],rr['task']),'static_identity')
    ids=[r['id'] for r in records if r['family']!=s['family']]
    ns=names_for(s['method']);audit_model(s['model_snapshot'],ids,ns,'static:'+s['id']+s['method'])
    mu=mean(s['model_snapshot']['marginal'],feature(rr,ns));sd=s['model_snapshot']['marginal']['std']
    for k,v in [('mean',mu),('std',sd),('Y',rr['response']['Y']),('absolute_error',abs(mu-rr['response']['Y']))]:near(s[k],v,'static_'+k)
    check(s['covered95']==(abs(mu-rr['response']['Y'])<=1.959963984540054*sd),'static_coverage')
    static_seen.add((s['id'],s['method']));counts['static_predictions']+=1
check(static_seen=={(r['id'],m) for r in records for m in ('reference','all_features','all_features_quadratic')},'96_static_keys')

packet_stats=[]
for family in ('ETT','Hydraulic'):
    path=STAGE/f'{family}_conditional_source_packet.json';packet=read(path)
    source=[r for r in records if r['family']==family]
    tasks=sorted({r['task'] for r in source});anonymous={t:f'T{i+1}' for i,t in enumerate(tasks)}
    record_lookup={f'R{i+1}':r for i,r in enumerate(source)}
    check(packet['held_target_records_included'] is False and packet['LLM_calls']==0,'packet_scope:'+family)
    seen=set()
    for f in packet['facts']:
        rr=record_lookup[f['record']];ns=names_for('all_features_quadratic') if f['predictor']=='quadratic' else base
        train=[r for r in source if r['task']!=rr['task']]
        check(f['task']==anonymous[rr['task']],'packet_task_identity')
        check(f['train_records']==[f'R{source.index(r)+1}' for r in train],'packet_train_ids')
        for k in ('features','response','seed_Y','seed_Z'):check(f[k]==rr[k],'packet_source_'+k)
        check(len(rr['seed_Y'])==len(rr['seed_Z'])==3,'packet_three_seeds')
        x=np.array([feature(r,ns) for r in train]);y=np.array([r['response']['Y'] for r in train]);z=np.array([r['response']['Z'] for r in train]);groups=np.array([r['task'] for r in train])
        fitted={}
        for kind,xx,yy in [('Y',x,y),('Z',np.c_[x,y],z)]:
            center,scale,beta=dual_fit(xx,yy,groups)
            saved=f['frozen_mean_functions'][kind]
            for field,value in [('center',center),('scale',scale),('beta',beta)]:near(saved[field],value,'packet_fit_'+field)
            fitted[kind]=saved;counts['packet_mean_functions']+=1
        q=feature(rr,ns)
        yp=mean(fitted['Y'],q)
        zp=[mean(fitted['Z'],np.r_[q,sy]) for sy in rr['seed_Y']]
        residual=np.array(rr['seed_Z'])-zp
        qualified=abs(float(np.mean(residual)))>.01 and (all(e>.0025 for e in residual) or all(e<-.0025 for e in residual))
        near(f['Y_prediction'],yp,'packet_Y_prediction')
        near(f['Z_given_observed_seed_Y'],zp,'packet_seed_conditioned_Z')
        near(f['conditional_seed_residual'],residual,'packet_residual')
        check(f['qualified_conditional_mismatch']==qualified,'packet_threshold')
        seen.add((f['record'],f['predictor']));counts['packet_point_predictions']+=1;counts['seed_Z_predictions']+=3
    check(seen=={(rid,kind) for rid in record_lookup for kind in ('linear','quadratic')},'packet_complete:'+family)
    stat=dict(source=family,records=len(source),point_predictions=len(packet['facts']),qualified_linear=sum(f['qualified_conditional_mismatch'] for f in packet['facts'] if f['predictor']=='linear'),qualified_quadratic=sum(f['qualified_conditional_mismatch'] for f in packet['facts'] if f['predictor']=='quadratic'),sha256=sha(path))
    packet_stats.append(stat)
check(data['conditional_packets']==summary['conditional_packets']==packet_stats,'packet_stats_hashes')

prior=read(OLD/'manual_expert_replay/results.json')
decomposition=read(STAGE/'r1_extrapolation_decomposition.json')
check(decomposition['source_sha256']==sha(OLD/'manual_expert_replay/results.json'),'decomposition_source_hash')
for item in decomposition['records']:
    rr=byid[item['id']]
    run=next(r for r in prior['runs'] if r['split']=='leave_family_out' and r['method']=='reference' and (r['family'],r['task'])==(rr['family'],rr['task']))
    m=run['trace'][0]['model_snapshots']['reference']['marginal']
    near(item['intercept'],m['beta'][0],'decomposition_intercept')
    total=item['intercept']
    check(len(item['components'])==len(m['feature_names']),'decomposition_component_count')
    for i,c in enumerate(item['components']):
        name=m['feature_names'][i]
        check(c['name']==name,'decomposition_name')
        near([c['value'],c['source_center'],c['source_scale']],[rr['features'][name],m['center'][i],m['scale'][i]],'decomposition_inputs')
        contribution=(rr['features'][name]-m['center'][i])/m['scale'][i]*m['beta'][i+1]
        near(c['contribution'],contribution,'decomposition_contribution');total+=contribution
    near(item['prediction'],total,'decomposition_prediction');counts['r1_frozen_decompositions']+=1

terminal=[]
for split in ('leave_task_out','leave_family_out'):
    for family,task in sorted({(r['family'],r['task']) for r in records}):
        runs=[r for r in data['runs'] if (r['split'],r['family'],r['task'])==(split,family,task)]
        check(len({r['trace'][-1]['recommendation'] for r in runs})==1,'terminal_agreement')
        terminal.append(dict(split=split,family=family,task=task,recommendation=runs[0]['trace'][-1]['recommendation']))

comparison=[]
for s in summary['summaries']:
    old=next(t for t in prior['summaries'] if (t['split'],t['method'],t['queries'])==(s['split'],s['method'],s['queries']))
    comparison.append(dict(split=s['split'],method=s['method'],queries=s['queries'],r1_mean_gain=old['mean_gain'],r2_mean_gain=s['mean_gain'],difference=s['mean_gain']-old['mean_gain']))
static_magnitudes={m:max(abs(s['mean']) for s in data['static_predictions'] if s['method']==m) for m in ('reference','all_features','all_features_quadratic')}
budget_coverage={str(n):[dict(family=r['family'],task=r['task']) for r in data['runs'] if r['split']=='leave_family_out' and r['method']=='reference' and any(node['queries']==n for node in r['nodes'])] for n in (2,4,8)}
result=dict(status='PASS' if not errors else 'FAIL',counts=dict(counts),error_count=len(errors),errors=errors[:100],max_abs_differences=dict(maxima),packet_stats=packet_stats,terminal_recommendations=terminal,r1_r2_summary_comparison=comparison,max_abs_r2_static_prediction=static_magnitudes,budget_task_coverage=budget_coverage,independent_solver='weighted centered dual ridge; task-macro held-out residual scale',tolerance=2e-9,elapsed_seconds=time.perf_counter()-started,scope='frozen aggregate records only; no original arrays, D/test, servers, keys or old r1 replay execution',limitations=['Wall-clock timings only checked for internal consistency, not remeasured.','No raw-array provenance or independent-source confirmation.','No LLM generation or new measurement interface evaluated.','Source packet IDs are anonymous within their own family; source-family isolation checked by exact aggregate-record mapping.','Eight-query summaries contain only three Hydraulic tasks, while two/four-query summaries contain all five tasks.','Task-macro averages weight three Hydraulic and two ETT tasks, not each source family equally.'])
(STAGE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','counts','error_count','errors','max_abs_differences','max_abs_r2_static_prediction','elapsed_seconds')},ensure_ascii=False))
raise SystemExit(bool(errors))
