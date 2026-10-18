"""Independent r4 asset/snapshot audit. Run only after the parent says results are ready.

Never imports or calls the generation, DSL, fitting, or strategy implementation.
Only replay_audit.json is written. Large quadratic models use observation-space
eigendecomposition and ridge normal-equation checks, never a feature Gram matrix.
"""
import ast
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

STAGE = Path(__file__).resolve().parent
BASE_DIR = STAGE.parent
ROOT = STAGE.parents[3]
REPLAY = STAGE / 'replay'
GEN = STAGE / 'generation'
BASE = ('exposure_shift_first','removed_fraction','weighted_implementation','target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio','parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion','parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal','B_input_variance','B_input_roughness','first_input_variance','second_input_variance','first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
EXPERTS = {'level_match_v1':'aligned_level_shift','variance_exposure_v1':'aligned_variance_shift','roughness_exposure_v1':'aligned_roughness_shift','optimizer_exposure_v1':'optimizer_exposure_interaction'}
INPUT = ('level','level_energy','variance','roughness','endpoint_change','channel_energy_dispersion','high_frequency_fraction','absolute_level')
ERROR = ('loss','target_loss','complement_loss','bias','absolute_error')
METHODS = ('reference','original_quadratic','window_fixed','window_quadratic','manual_uniform','manual_conditional','ordinary_uniform','ordinary_joint','ordinary_conditional','ordinary_combined','mismatch_uniform','mismatch_joint','mismatch_conditional','mismatch_combined')
counts = collections.Counter()
errors = []
maxima = collections.defaultdict(float)
TOL = 2e-9


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def check(ok, label):
    counts['assertions'] += 1
    if not ok:
        errors.append(label)


def near(a, b, label, tol=TOL):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        check(False, label + ':shape')
        return
    delta = float(np.max(np.abs(a-b), initial=0.))
    maxima[label.split(':')[0]] = max(maxima[label.split(':')[0]], delta)
    check(bool(np.isfinite(a).all() and np.isfinite(b).all() and np.all(np.abs(a-b) <= tol*np.maximum(1., np.maximum(np.abs(a), np.abs(b))))), label)


def expression(expression, table, features):
    """Independent, axis-tagged AST interpretation using only permitted parents."""
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 95:
        raise ValueError('AST budget exceeded')
    env = {k:(float(features[k]), 'scalar') for k in BASE[:3]}
    for axis in ('A', 'B'):
        env.update({axis+'_'+name:(table[axis+'_inputs'][:, j],axis) for j,name in enumerate(INPUT)})
    es = sum(table[k] for k in sorted(table) if k.startswith('B_errors_s')) / 3.
    env.update({'E_'+name:(es[:,j],'B') for j,name in enumerate(ERROR)})
    env['E_contrast'] = (es[:,1]-es[:,2], 'B')

    def visit(n):
        if type(n) is ast.Expression:
            return visit(n.body)
        if type(n) is ast.Name and n.id in env:
            return env[n.id]
        if type(n) is ast.Constant and type(n.value) in (int,float) and n.value in (-1,0,.5,1,2):
            return float(n.value), 'scalar'
        if type(n) is ast.UnaryOp and type(n.op) is ast.USub:
            x, axis = visit(n.operand)
            return -x, axis
        if type(n) is ast.BinOp and type(n.op) in (ast.Add,ast.Sub,ast.Mult):
            a, aa = visit(n.left); b, bb = visit(n.right)
            if aa != bb and aa != 'scalar' and bb != 'scalar':
                raise ValueError('Incompatible window axes')
            value = a+b if type(n.op) is ast.Add else a-b if type(n.op) is ast.Sub else a*b
            return value, bb if aa == 'scalar' else aa
        if type(n) is ast.Call and type(n.func) is ast.Name and not n.keywords:
            name = n.func.id
            if name == 'safe_ratio' and len(n.args) == 2:
                a, aa = visit(n.args[0]); b, bb = visit(n.args[1])
                if aa != bb and aa != 'scalar' and bb != 'scalar':
                    raise ValueError('Incompatible ratio axes')
                return a/(1+np.abs(b)), bb if aa == 'scalar' else aa
            if len(n.args) != 1:
                raise ValueError('Invalid arity')
            x, axis = visit(n.args[0])
            if name in ('abs','tanh','square','log1p_abs'):
                return {'abs':np.abs,'tanh':np.tanh,'square':lambda v:v*v,'log1p_abs':lambda v:np.log1p(np.abs(v))}[name](x), axis
            if name in ('mean','std','center','unit') and axis != 'scalar':
                if name == 'mean': return float(np.average(x)), 'scalar'
                if name == 'std': return float(np.sqrt(np.average((x-np.average(x))**2))), 'scalar'
                xx = x-np.average(x)
                return (xx/(1+np.sqrt(np.average(xx**2))) if name == 'unit' else xx), axis
            if name in ('first_mean','second_mean') and axis == 'A':
                mask = table['A_first'] if name == 'first_mean' else np.logical_not(table['A_first'])
                return float(np.average(x[mask])), 'scalar'
        raise ValueError('Forbidden expression node: '+ast.dump(n))

    with np.errstate(over='raise',invalid='raise',divide='raise'):
        value, axis = visit(tree)
    if axis != 'scalar' or not np.isfinite(value):
        raise ValueError('Expression is not a finite scalar')
    return float(value)


def fixed(table, features):
    a, b = table['A_inputs'], table['B_inputs']
    e = sum(table[k] for k in sorted(table) if k.startswith('B_errors_s'))/3.
    e = np.column_stack([e,e[:,1]-e[:,2]])
    centered_e = e-e.mean(axis=0)
    out = {}
    for j, name in enumerate(INPUT):
        cb = b[:,j]-b[:,j].mean()
        delta = features['exposure_shift_first']*(a[table['A_first'],j].mean()-a[~table['A_first'],j].mean())/(1+np.sqrt(np.mean(cb**2)))
        out['window_coverage_'+name] = float(delta)
        for k, error in enumerate((*ERROR,'contrast')):
            denom = np.linalg.norm(cb)*np.linalg.norm(centered_e[:,k])
            corr = float(cb @ centered_e[:,k]/denom) if denom > 0 else 0.
            out['window_correlation_'+name+'_'+error] = corr
            out['window_aligned_'+name+'_'+error] = float(delta*corr)
    return out


def names_for(method, originals, generated):
    names, extras = BASE, {}
    if method == 'original_quadratic': names = originals
    if method.startswith('window_'): names = tuple(sorted((*originals,*WINDOW_NAMES)))
    if method.startswith('manual_'): extras = {k:(*BASE,v) for k,v in EXPERTS.items()}
    for mode in ('ordinary','mismatch'):
        if method.startswith(mode+'_'):
            if method.endswith('_combined'): names = (*BASE,*generated[mode])
            else: extras = {k:(*BASE,k) for k in generated[mode]}
    if method in ('original_quadratic','window_quadratic'):
        names = (*names,*(names[i]+'*'+names[j] for i in range(len(names)) for j in range(i,len(names))))
    return tuple(names), extras


def feature(record, names):
    return np.asarray([math.prod(record['features'][p] for p in name.split('*')) for name in names],dtype=float)


def eigen_fit(x, y, groups):
    """Observation-space eigensolve, distinct from the production linear solve."""
    _, inv, size = np.unique(groups,return_inverse=True,return_counts=True)
    w = 1./size[inv]
    center = (w @ x)/w.sum()
    scale = np.sqrt(np.maximum((w @ ((x-center)**2))/w.sum(),1.))
    d = (x-center)/scale
    ym = float(w @ y/w.sum())
    a = np.sqrt(w)[:,None]*d
    vals, vecs = np.linalg.eigh(a @ a.T)
    slope = a.T @ (vecs @ ((vecs.T @ (np.sqrt(w)*(y-ym)))/(vals+1.)))
    return center, scale, np.r_[ym,slope]


FIT_CACHE = {}


def audit_model(model, ids, names, byid, source, label):
    for kind in ('marginal','conditional'):
        s = model[kind]
        check(s['train_record_ids'] == ids, 'train_ids:'+label+kind)
        check(s['feature_names'] == list(names)+(['observed_Y_scoring_only'] if kind == 'conditional' else []), 'feature_names:'+label+kind)
        check(s['ridge'] == 1. and s['scale_floor'] == .01,'ridge_parameters:'+label+kind)
        key = (source,tuple(sorted(ids)),names,kind)
        if key not in FIT_CACHE:
            rows = [byid[rid] for rid in sorted(ids)]
            x = np.asarray([feature(r,names) for r in rows])
            y = np.asarray([r['response']['Y' if kind == 'marginal' else 'Z'] for r in rows])
            if kind == 'conditional': x = np.column_stack([x,[r['response']['Y'] for r in rows]])
            groups = np.asarray([r['family']+':'+r['task'] for r in rows])
            center,scale,beta = eigen_fit(x,y,groups)
            residual = np.empty(len(y))
            for held in np.unique(groups):
                tr = groups != held
                cc,ss,bb = eigen_fit(x[tr],y[tr],groups[tr])
                residual[~tr] = y[~tr] - (bb[0]+((x[~tr]-cc)/ss) @ bb[1:])
            std = max(.01,float(np.sqrt(np.mean([np.mean(residual[groups == g]**2) for g in np.unique(groups)]))))
            FIT_CACHE[key] = center,scale,beta,std
            counts['independent_component_fits'] += 1
        center,scale,beta,std = FIT_CACHE[key]
        for field,value in [('center',center),('scale',scale),('beta',beta),('std',std)]:
            near(s[field],value,'fit_'+field+':'+label+kind)
        # Independently verify the saved solution's stationarity without d x d.
        rows = [byid[rid] for rid in ids]
        x = np.asarray([feature(r,names) for r in rows])
        yy = np.asarray([r['response']['Y' if kind == 'marginal' else 'Z'] for r in rows])
        if kind == 'conditional': x = np.column_stack([x,[r['response']['Y'] for r in rows]])
        group = [r['family']+':'+r['task'] for r in rows]
        size = collections.Counter(group); w = np.asarray([1./size[g] for g in group])
        d = (x-np.asarray(s['center']))/np.asarray(s['scale'])
        beta_saved = np.asarray(s['beta']); residual = yy-beta_saved[0]-d @ beta_saved[1:]
        near(beta_saved[1:],d.T @ (w*residual),'ridge_stationarity:'+label+kind)
        near(w @ residual,0.,'intercept_stationarity:'+label+kind)
        counts['component_snapshots'] += 1


def mean(model, x):
    return float(model['beta'][0]+((x-np.asarray(model['center']))/np.asarray(model['scale'])) @ np.asarray(model['beta'][1:]))


def score(model, record):
    x = feature(record,model['marginal']['feature_names'])
    out = {}
    for kind,xx,yy in [('marginal',x,record['response']['Y']),('conditional',np.r_[x,record['response']['Y']],record['response']['Z'])]:
        s = model[kind]
        out[kind] = -.5*math.log(2*math.pi)-math.log(s['std'])-.5*((yy-mean(s,xx))/s['std'])**2
    out['joint'] = out['marginal']+out['conditional']
    return out


WINDOW_NAMES = ()


def main():
    global WINDOW_NAMES
    start = time.perf_counter()
    lock = read(STAGE/'evaluation_lock.json')
    genlock = read(GEN/'generation_lock.json')
    seal = read(STAGE/'generated_outputs_lock.json')
    summary = read(REPLAY/'summary.json')
    oldpath = BASE_DIR/'20261018_conditional_diagnosis_r1/development_pool.json'
    records = read(oldpath)['records']
    originals = tuple(sorted(records[0]['features']))
    tasks = sorted({(r['family'],r['task']) for r in records})
    check(len(records) == len({r['id'] for r in records}) == 32,'32_unique_records')
    check(len(originals) == 26 and len(tasks) == 5,'original_26_five_tasks')
    expected_calls = {f'{f}_{m}' for f in ('ETT','Hydraulic') for m in ('ordinary','mismatch')}
    scripts = {'window_measurement_dsl.py','prepare_window_signal_inputs.py','replay_window_signals.py','generate_window_signals.py'}
    check(set(lock['script_sha256']) == scripts,'exact_script_set')
    for name,value in lock['script_sha256'].items(): check(sha(ROOT/'scripts'/name) == value,'code_hash:'+name)
    check(sha(STAGE/'PROTOCOL.md') == lock['protocol_sha256'] == genlock['protocol_sha256'],'protocol_hash')
    check(sha(oldpath) == lock['pool_sha256'],'pool_hash')
    check(lock['methods'] == list(METHODS) and lock['nodes'] == [2,4,8] and lock['new_full_trainings'] == 0,'evaluation_matrix_lock')
    check(lock['source_splits'] == ['ETT_to_Hydraulic','Hydraulic_to_ETT'],'source_split_lock')
    check(genlock['script_sha256'] == lock['script_sha256'],'generation_source_hashes')
    check(genlock['model'] == 'gpt-6.1-sol' and genlock['reasoning'] == 'high' and genlock['max_calls'] == 4 and genlock['no_regeneration'],'generation_model_budget')
    check(len(genlock['calls']) == 4 and {c['name'] for c in genlock['calls']} == expected_calls == set(seal['calls']),'exact_four_calls')
    check(genlock['evaluation_lock_sha256'] == seal['evaluation_lock_sha256'] == summary['evaluation_lock_sha256'] == sha(STAGE/'evaluation_lock.json'),'evaluation_link')
    check(seal['generation_lock_sha256'] == sha(GEN/'generation_lock.json') and genlock['schema_sha256'] == sha(GEN/'schema.json'),'generation_schema_link')
    check(seal['target_measurements_not_yet_evaluated'] is True,'pre_target_seal_claim')
    check(summary['status'] == 'PASS' and summary['new_full_trainings'] == 0 and summary['cumulative_full_trainings'] == 464,'training_count')
    tabledir = BASE_DIR/'20261018_parent_measurements_r3/frequency_corrected'
    check(set(lock['table_sha256']) == {f'{f}_{t}.npz' for f,t in tasks},'exact_five_corrected_tables')
    tables = {}
    for family,task in tasks:
        name = f'{family}_{task}.npz'; path = tabledir/name
        check(sha(path) == lock['table_sha256'][name],'table_hash:'+name)
        correction = read(tabledir/f'{family}_correction.json')
        check(correction['assets'][name]['sha256'] == lock['table_sha256'][name],'correction_link:'+name)
        with np.load(path,allow_pickle=False) as z: tables[(family,task)] = dict(z)
        table = tables[(family,task)]
        check(sum(k.startswith('B_errors_s') for k in table) == 3,'three_parent_error_seeds:'+name)
        counts['corrected_tables'] += 1
    responses = {}; call_details = []; threads = []; facts_hashes = collections.defaultdict(list)
    for call in genlock['calls']:
        name = call['name']; folder = GEN/name; family = call['family']; mode = call['mode']
        check(name == family+'_'+mode,'call_identity:'+name)
        check(set(seal['calls'][name]) == {'response.json','receipt.json','prompt.txt','events.jsonl'},'sealed_files:'+name)
        for filename,h in seal['calls'][name].items(): check(sha(folder/filename) == h,'output_hash:'+name+filename)
        packetpath = STAGE/'corrected_inputs'/f'{family}_facts.json'; packet = read(packetpath)
        check(sha(packetpath) == call['input_sha256'],'corrected_facts_hash:'+name)
        facts_hashes[family].append(call['input_sha256'])
        prompt = (folder/'prompt.txt').read_text(encoding='utf-8')
        check(sha(folder/'prompt.txt') == call['prompt_sha256'],'prompt_hash:'+name)
        check(json.loads(prompt.split('\nFACTS\n',1)[1]) == packet,'prompt_facts:'+name)
        source = [r for r in records if r['family'] == family]
        expected_facts = [{k:v for k,v in row.items() if k != 'frozen_mean_functions'} for row in read(BASE_DIR/'20261018_predictive_support_r2'/f'{family}_conditional_source_packet.json')['facts']]
        check(packet['response_facts'] == expected_facts and packet['held_target_records_included'] is False,'source_fact_mapping:'+name)
        check(packet['input_window_columns'] == list(INPUT) and packet['error_window_columns'] == [*ERROR,'contrast'],'packet_columns:'+name)
        anonymous = {t:f'T{i+1}' for i,t in enumerate(sorted({r['task'] for r in source}))}
        check(set(packet['parent_measurements']) == set(anonymous.values()),'source_only_task_set:'+name)
        for fact in packet['response_facts']:
            rr = source[int(fact['record'][1:])-1]
            check(fact['task'] == anonymous[rr['task']],'source_fact_task:'+name)
            for field in ('features','response','seed_Y','seed_Z'): check(fact[field] == rr[field],'source_fact_'+field+':'+name)
        receipt = read(folder/'receipt.json'); started = read(folder/'started.json'); response = read(folder/'response.json')
        check(receipt['status'] == 'PASS' and receipt['exit_code'] == 0 and receipt['tool_items'] == [],'call_success_no_tools:'+name)
        check(started['model'] == receipt['model_requested'] == genlock['model'] and started['reasoning'] == receipt['reasoning_requested'] == 'high','requested_model:'+name)
        check(receipt['response_sha256'] == sha(folder/'response.json'),'receipt_response_hash:'+name)
        events = [json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        for typ in ('thread.started','turn.started','turn.completed'): check(sum(e['type'] == typ for e in events) == 1,'one_turn:'+name+typ)
        threads.extend(e['thread_id'] for e in events if e['type'] == 'thread.started')
        items = [e['item'] for e in events if 'item' in e]
        check(all(i['type'] in ('agent_message','reasoning','error') for i in items),'event_no_tools:'+name)
        check(receipt['transport_errors'] == [i for i in items if i['type'] == 'error'],'transport_record_preserved:'+name)
        messages = [i for i in items if i['type'] == 'agent_message']
        check(len(messages) == 1 and json.loads(messages[0]['text']) == response,'event_response_match:'+name)
        check(receipt['usage'] == next(e['usage'] for e in events if e['type'] == 'turn.completed'),'usage_match:'+name)
        check(isinstance(receipt['seconds'],(int,float)) and receipt['seconds'] > 0,'generation_seconds:'+name)
        check(set(response) == {'signals','reasoning'} and isinstance(response['reasoning'],str),'response_schema:'+name)
        signals = response['signals']
        check(len(signals) <= 2 and len({s['name'] for s in signals}) == len(signals) and receipt['signals'] == len(signals),'signal_count:'+name)
        for sig in signals:
            check(set(sig) == {'name','expression','measurement_meaning','limitation','falsifying_response'} and all(isinstance(v,str) for v in sig.values()),'signal_schema:'+name)
            for r in records:
                expression(sig['expression'],tables[(r['family'],r['task'])],r['features'])
                counts['independent_expression_evaluations'] += 1
        responses[(family,mode)] = response
        call_details.append(dict(name=name,signals=len(signals),seconds=receipt['seconds'],usage=receipt['usage'],ast_nodes=[len(list(ast.walk(ast.parse(s['expression'],mode='eval')))) for s in signals]))
        counts['generation_calls'] += 1
    check(len(set(threads)) == 4,'four_unique_threads')
    for family,hashes in facts_hashes.items(): check(len(hashes) == 2 and len(set(hashes)) == 1,'same_facts:'+family)

    augmented = {}; timings = {t['source']:t for t in summary['measurement_timings']}
    check(set(timings) == {'ETT','Hydraulic'},'measurement_timing_sources')
    for source in ('ETT','Hydraulic'):
        path = REPLAY/f'{source}_augmented_records.json'; data = read(path)
        check(data['source_family'] == source and data['generated_from_target'] is False,'augmentation_source:'+source)
        check((STAGE/'generated_outputs_lock.json').stat().st_mtime_ns <= path.stat().st_mtime_ns,'seal_mtime_before_augmentation:'+source)
        expected = copy.deepcopy(records)
        for row in expected:
            table = tables[(row['family'],row['task'])]
            measured = fixed(table,row['features']); WINDOW_NAMES = tuple(sorted(measured))
            check(len(WINDOW_NAMES) == 104,'104_fixed_measurements')
            row['features'].update(measured)
            for mode in ('ordinary','mismatch'):
                for j,sig in enumerate(responses[(source,mode)]['signals']):
                    row['features'][f'{mode}_measurement_{j}'] = expression(sig['expression'],table,row['features'])
                    counts['recomputed_generated_measurements'] += 1
            counts['recomputed_fixed_measurements'] += len(measured)
        check(len(data['records']) == 32,'augmented_record_count:'+source)
        for actual,want in zip(data['records'],expected):
            check(set(actual) == set(want),'record_schema:'+source)
            for k in want:
                if k != 'features': check(actual[k] == want[k],'unchanged_record:'+source+k)
            check(set(actual['features']) == set(want['features']),'augmented_feature_set:'+source)
            for k,v in want['features'].items(): near(actual['features'][k],v,'measurement:'+source+want['id']+k)
        augmented[source] = {r['id']:r for r in expected}
        rt = timings[source]['records']
        check(len(rt) == 32 and {t['id'] for t in rt} == set(augmented[source]),'measurement_timing_ids:'+source)
        check(all(isinstance(t['seconds'],(int,float)) and t['seconds'] >= 0 for t in rt),'measurement_seconds:'+source)

    rows = []; keys = []; node_recs = {}; terminal = {}; reference_traces = {}; empty_traces = []
    check(len(summary['runs']) == 70,'70_runs')
    for entry in summary['runs']:
        family,task,method = entry['family'],entry['task'],entry['method']; key = (family,task,method); keys.append(key)
        path = REPLAY/f'{family}_{task}_{method}.json'; run = read(path)
        check(sha(path) == entry['sha256'],'run_hash:'+str(key))
        check((run['family'],run['task'],run['method']) == key,'run_identity:'+str(key))
        source = run['source_family']; check(source in ('ETT','Hydraulic') and source != family,'held_family:'+str(key))
        byid = augmented[source]
        training = [r['id'] for r in records if r['family'] == source]
        held = [r for r in byid.values() if (r['family'],r['task']) == (family,task)]
        visible = list(training); remaining = {r['id'] for r in held}; queried = []
        generated = {m:[f'{m}_measurement_{j}' for j in range(len(responses[(source,m)]['signals']))] for m in ('ordinary','mismatch')}
        names,extras = names_for(method,originals,generated)
        if method == 'window_quadratic': check(len(names) == 8645,'8645_full_quadratic_dimensions')
        rule = method.rsplit('_',1)[-1]; rule = rule if rule in ('joint','conditional') else 'uniform'
        rel = {h:0. for h in extras}
        oracle = max([0.]+[r['response']['Y'] for r in held if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01])
        near(run['oracle_gain'],oracle,'oracle:'+str(key)); near(entry['seconds'],run['seconds'],'run_seconds')
        check(len(run['trace']) == len(held),'trace_length:'+str(key))
        trajectory = []
        for step,t in enumerate(run['trace'],1):
            label = str(key)+':'+str(step); counts['steps'] += 1
            check(t['step'] == step and t['visible_before'] == visible,'visible_boundary:'+label)
            check(t['selected'] not in visible and t['selected'] in remaining,'unrevealed_selection:'+label)
            snap = t['model_snapshots']; check(set(snap['diagnostics']) == set(extras),'expert_set:'+label)
            audit_model(snap['reference'],visible,names,byid,source,label+'reference')
            for h,ns in extras.items(): audit_model(snap['diagnostics'][h],visible,ns,byid,source,label+h)
            reg,ev = t['registration'],t['event']
            check(digest(snap) == reg['model_hash'] == ev['model_hash'],'snapshot_hash:'+label)
            logits = np.asarray([0.]+[rel[h]-math.log(len(extras)) for h in extras]) if extras else np.zeros(1)
            weights = np.exp(logits-logits.max()); weights /= weights.sum()
            models = [snap['reference'],*(snap['diagnostics'][h] for h in extras)]
            check(set(t['predictions_before']) == remaining,'all_unrevealed_candidates_predicted:'+label)
            computed = {}
            for rid in remaining:
                computed[rid] = sum(float(w)*mean(m['marginal'],feature(byid[rid],m['marginal']['feature_names'])) for w,m in zip(weights,models))
                near(t['predictions_before'][rid],computed[rid],'candidate_prediction:'+label+rid)
                counts['candidate_predictions'] += 1
            # Selection uses recorded full-precision predictions; independent predictions
            # are compared above. A separately computed near-tie is explicitly recorded.
            selected = min(remaining,key=lambda rid:(-t['predictions_before'][rid],rid))
            check(t['selected'] == selected,'selected_argmax:'+label)
            independent_selected = min(remaining,key=lambda rid:(-computed[rid],rid))
            if independent_selected != selected:
                check(abs(computed[independent_selected]-computed[selected]) <= TOL,'independent_argmax_disagreement:'+label)
                counts['floating_near_tie_argmax'] += 1
            rr = byid[selected]
            check(reg['event_id'] == ev['event_id'] == ev['source_response_id'] == selected,'event_identity:'+label)
            check(ev['order'] == 'registered_prediction_scored_before_any_refit','registered_before_reveal_claim:'+label)
            near(reg['reference_gain'],mean(snap['reference']['marginal'],feature(rr,names)),'registration_reference')
            check(set(reg['gain_predictions']) == set(extras),'registration_experts:'+label)
            check(set(ev['before']) == set(ev['after']) == set(ev['scores']) == set(extras),'score_keys:'+label)
            for k in ('Y','Z','NMAE_gain'): near(ev['response'][k],rr['response'][k],'event_response:'+label+k)
            baseline = score(snap['reference'],rr)
            for h,m in snap['diagnostics'].items():
                near(reg['gain_predictions'][h],mean(m['marginal'],feature(rr,extras[h])),'registration_expert')
                near(ev['before'][h],rel[h],'reliability_before:'+label+h)
                sc = score(m,rr); delta = {k:sc[k]-baseline[k] for k in sc}
                check(set(ev['scores'][h]) == set(delta),'score_components:'+label+h)
                for k,v in delta.items(): near(ev['scores'][h][k],v,'expert_logscore:'+label+h+k)
                near(delta['conditional'],delta['joint']-delta['marginal'],'factorization')
                if rule != 'uniform': rel[h] += delta[rule]
                near(ev['after'][h],rel[h],'reliability_after:'+label+h)
                counts['expert_score_triples'] += 1
            queried.append(rr); visible.append(selected); remaining.remove(selected)
            eligible = [r for r in queried if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01]
            chosen = max(eligible,key=lambda r:(r['response']['Y'],r['id'])) if eligible else None
            recommendation = chosen['id'] if chosen else 'PARENT'
            check(t['recommendation'] == recommendation,'adoption_recommendation:'+label)
            trajectory.append((selected,recommendation,t['predictions_before']))
            if step in (2,4,8):
                nodes = [n for n in run['nodes'] if n['queries'] == step]
                check(len(nodes) == 1,'unique_budget_node:'+label)
                n = nodes[0]
                check(n['recommendation'] == recommendation and n['logical_candidate_full_trainings'] == 3*step and n['shared_parent_full_trainings'] == 3,'node_identity:'+label)
                gain = chosen['response']['Y'] if chosen else 0.
                for k,v in [('gain',gain),('NMAE_gain',chosen['response']['NMAE_gain'] if chosen else 0.),('negative_seed_count',sum(y < 0 for y in chosen['seed_Y']) if chosen else 0),('regret',oracle-gain)]: near(n[k],v,'node_'+k+':'+label)
                times = [r['original_full_training_seconds'] for r in queried]
                if any(v is None for v in times):
                    check(n['query_seconds'] is None,'unknown_query_cost_preserved:'+label); counts['unknown_cost_nodes'] += 1
                else: near(n['query_seconds'],sum(times),'query_seconds')
                check(0 <= n['policy_seconds'] <= run['seconds'],'policy_seconds:'+label)
                rows.append(dict(family=family,task=task,method=method,**n)); counts['nodes'] += 1
                node_recs[(family,task,method,step)] = recommendation
        check(len(run['nodes']) == sum(n <= len(held) for n in (2,4,8)),'node_count:'+str(key))
        terminal[key] = run['trace'][-1]['recommendation']
        if method == 'reference': reference_traces[(family,task)] = trajectory
        for mode in ('ordinary','mismatch'):
            if method.startswith(mode+'_') and not generated[mode]: empty_traces.append(((family,task),method,trajectory))
        FIT_CACHE.clear()  # Bound memory independently of the number of result files.
    expected_keys = {(f,t,m) for f,t in tasks for m in METHODS}
    check(len(keys) == len(set(keys)) == 70 and set(keys) == expected_keys,'complete_70_task_method_matrix')
    check(counts['steps'] == 448 and counts['nodes'] == 182,'448_steps_182_nodes')
    for key,method,trajectory in empty_traces:
        reference = reference_traces[key]
        check(len(reference) == len(trajectory),'empty_reference_length:'+str(key)+method)
        for a,b in zip(reference,trajectory):
            check(a[:2] == b[:2] and set(a[2]) == set(b[2]),'empty_reference_decisions:'+str(key)+method)
            for rid in a[2]: near(a[2][rid],b[2][rid],'empty_reference_prediction')
        counts['empty_signal_reference_trajectories'] += 1
    for family,task in tasks: check(len({terminal[(family,task,m)] for m in METHODS}) == 1,'terminal_agreement:'+family+task)
    summary_keys = []
    for s in summary['summaries']:
        summary_keys.append((s['method'],s['queries']))
        subset = [r for r in rows if r['method'] == s['method'] and r['queries'] == s['queries']]
        check(s['tasks'] == len(subset) == (3 if s['queries'] == 8 else 5),'summary_task_count')
        near(s['mean_gain'],np.mean([r['gain'] for r in subset]),'summary_gain')
        near(s['mean_regret'],np.mean([r['regret'] for r in subset]),'summary_regret')
        counts['summaries'] += 1
    check(len(summary_keys) == 42 and set(summary_keys) == {(m,n) for m in METHODS for n in (2,4,8)},'42_unique_summaries')
    with (REPLAY/'decisions.csv').open(encoding='utf-8',newline='') as stream: csvrows = list(csv.DictReader(stream))
    check(len(csvrows) == len(rows) == 182,'182_csv_rows')
    for a,b in zip(csvrows,rows):
        check(set(a) == set(b),'csv_columns')
        for k,v in b.items():
            if isinstance(v,(int,float)): near(float(a[k]),v,'csv_'+k)
            elif v is None: check(a[k] == '','csv_unknown_'+k)
            else: check(a[k] == v,'csv_text_'+k)
    # Static, read-only confirmation of the reveal boundary in the frozen source.
    tree = ast.parse((ROOT/'scripts/replay_window_signals.py').read_text(encoding='utf-8'))
    func = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'run')
    register_lines = [n.lineno for n in ast.walk(func) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr == 'register']
    reveal_lines = [n.lineno for n in ast.walk(func) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id == 'oracle']
    check(len(register_lines) == 1 and reveal_lines and register_lines[0] < min(reveal_lines),'source_register_precedes_oracle_lookup')
    counts['csv_rows'] = len(csvrows)
    return dict(status='PASS' if not errors else 'FAIL',counts=dict(counts),error_count=len(errors),errors=errors[:100],max_abs_differences=dict(maxima),generation_calls=call_details,summary_sha256=sha(REPLAY/'summary.json'),generated_outputs_lock_sha256=sha(STAGE/'generated_outputs_lock.json'),audit_source_sha256=sha(Path(__file__)),solver='Independent observation-space symmetric eigensolve plus snapshot ridge stationarity; task-macro leave-task-out residual scale',tolerance=TOL,elapsed_seconds=time.perf_counter()-start,scope='Four frozen source-only calls; five corrected parent tables; all new and fixed measurements; all 70 held-family strategies/448 steps/182 nodes/42 summaries. No production execution, raw arrays, D/test, servers or storage-prohibited paths.',limitations=['Requested model/reasoning verified from command records and locks; server model identity is not independently attested by these event streams.','Generation-before-target order checked through hash-chain, frozen source control flow and local file times; this is not tamper-proof temporal attestation.','Parent table hashes and correction receipt links are checked; original arrays and previously audited r2 source-packet regressions are not recomputed.','Timing checked only for finite/internal consistency; no generation, model fitting policy, original training or end-to-end runtime is rerun.','Interface synthetic probes and prior 14-method simulations are not repeated.','Eight-query summaries cover only three Hydraulic tasks; two/four-query summaries cover all five.','Floating-point argmax disagreements are accepted only within the stated tolerance and counted explicitly.'])


if __name__ == '__main__':
    try:
        result = main()
    except Exception:
        result = dict(status='FAIL',counts=dict(counts),error_count=len(errors)+1,errors=errors[:100],exception=traceback.format_exc(),scope='Audit stopped early; coverage is the completed counters only.')
    (STAGE/'replay_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','counts','error_count','errors')},ensure_ascii=False))
    if 'exception' in result: print(result['exception'])
    raise SystemExit(result['status'] != 'PASS')
