"""Independent source-packet, call-record and frozen-snapshot audit; no refitting."""
import ast
import collections
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

ROOT = Path.cwd()
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
GEN = STAGE / 'signal_generation'
OUT = STAGE / 'generated_signal_replay'
BASE = ('exposure_shift_first','removed_fraction','weighted_implementation','target_weight_second_moment','first_training_mass','parent_B_log_loss','parent_B_target_ratio','parent_B_local_contrast','parent_B_bias','parent_B_target_bias','parent_B_channel_dispersion','parent_B_learning_gain','parent_B_best_fraction','parent_B_curve_reversal','B_input_variance','B_input_roughness','first_input_variance','second_input_variance','first_input_roughness','second_input_roughness','first_B_level_distance','second_B_level_distance')
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

def near(a, b, label):
    a, b = float(a), float(b)
    diff = abs(a-b)
    maxima[label.split(':')[0]] = max(maxima[label.split(':')[0]], diff)
    check(math.isfinite(a) and math.isfinite(b) and diff <= 2e-10*max(1.,abs(a),abs(b)), label)

def expression_value(expression, features):
    tree = ast.parse(expression, mode='eval')
    if len(list(ast.walk(tree))) > 31:
        raise ValueError('AST node limit')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in features:
            return features[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int,float) and node.value in (-1,0,.5,1,2):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult):
            a,b = visit(node.left),visit(node.right)
            return a+b if isinstance(node.op,ast.Add) else a-b if isinstance(node.op,ast.Sub) else a*b
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('abs','tanh','log1p','safe_ratio') and not node.keywords:
            name = node.func.id
            if len(node.args) != (2 if name == 'safe_ratio' else 1):
                raise ValueError('arity')
            if name == 'log1p' and not (isinstance(node.args[0],ast.Call) and isinstance(node.args[0].func,ast.Name) and node.args[0].func.id == 'abs'):
                raise ValueError('log1p domain')
            args = [visit(n) for n in node.args]
            return {'abs':abs,'tanh':np.tanh,'log1p':np.log1p,'safe_ratio':lambda a,b:a/(1+abs(b))}[name](*args)
        raise ValueError('unapproved AST node')
    value = float(visit(tree))
    if not math.isfinite(value):
        raise ValueError('nonfinite')
    return value

def mean(model, features):
    x = np.array([features[name] for name in model['feature_names']],dtype=float)
    return float(np.r_[1.,(x-np.array(model['center']))/np.array(model['scale'])] @ np.array(model['beta']))

def score(model, record):
    features = dict(record['features'], observed_Y_scoring_only=record['response']['Y'])
    answer = {}
    for kind, value in [('marginal',record['response']['Y']),('conditional',record['response']['Z'])]:
        m = model[kind]
        answer[kind] = -.5*math.log(2*math.pi)-math.log(m['std'])-.5*((value-mean(m,features))/m['std'])**2
    answer['joint'] = answer['marginal']+answer['conditional']
    return answer

pool = read(STAGE/'development_pool.json')['records']
byid = {r['id']:r for r in pool}
data = read(OUT/'results.json')
summary = read(OUT/'summary.json')
generation_lock = read(GEN/'generation_lock.json')
evaluation_lock = read(GEN/'evaluation_lock.json')
check(len(data['runs']) == 30, '30runs')
check(len(generation_lock['calls']) == 4 and generation_lock['max_calls'] == 4 and generation_lock['no_retries'], 'fourlockedcalls')
check(generation_lock['model'] == 'gpt-6.1-sol' and generation_lock['reasoning'] == 'high', 'lockedmodel')
for key,path in [('pool_sha256',STAGE/'development_pool.json'),('protocol_sha256',STAGE/'SIGNAL_GENERATION_PROTOCOL.md'),('code_sha256',ROOT/'scripts/generate_conditional_signals.py')]:
    check(generation_lock[key] == sha(path), 'generation_hash:'+key)
for key,path in [('pool_sha256',STAGE/'development_pool.json'),('code_sha256',ROOT/'scripts/replay_generated_signals.py'),('generation_lock_sha256',GEN/'generation_lock.json')]:
    check(data[key] == sha(path), 'replay_hash:'+key)
check(evaluation_lock['code_sha256'] == sha(ROOT/'scripts/replay_generated_signals.py'), 'evaluation_code_hash')
check(evaluation_lock['protocol_sha256'] == generation_lock['protocol_sha256'], 'evaluation_protocol_hash')
check(evaluation_lock['new_calls'] == 0 and not evaluation_lock['scientific_changes_after_output'], 'evaluation_no_new_calls')
check(summary['results_sha256'] == sha(OUT/'results.json'), 'results_hash')
signals_by_condition = {}
input_hashes = collections.defaultdict(list)
call_details = []
thread_ids = []
for call in generation_lock['calls']:
    directory = GEN/call['name']
    source = [r for r in pool if r['family'] == call['family']]
    packet = read(directory/'input.json')
    prompt = (directory/'prompt.txt').read_text(encoding='utf-8')
    response = read(directory/'response.json')
    original = read(directory/'receipt.json')
    validation = read(directory/'validation_receipt.json')
    started = read(directory/'started.json')
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    check(sha(directory/'input.json') == call['input_sha256'], 'input_hash:'+call['name'])
    check(sha(directory/'prompt.txt') == call['prompt_sha256'], 'prompt_hash:'+call['name'])
    input_hashes[call['family']].append(call['input_sha256'])
    check(json.loads(prompt.split('\nFACTS\n',1)[1]) == packet, 'prompt_facts:'+call['name'])
    check(set(packet) == {'allowed_features','records'} and len(packet['allowed_features']) == 26, 'packet_schema:'+call['name'])
    check(packet['allowed_features'] == sorted(source[0]['features']), 'allowed_features:'+call['name'])
    check(len(packet['records']) == len(source), 'source_record_count:'+call['name'])
    tasks = {name:f'T{i+1}' for i,name in enumerate(sorted({r['task'] for r in source}))}
    for i,(fact,record) in enumerate(zip(packet['records'],source)):
        counts['source_packet_records'] += 1
        check(set(fact) == {'record','task','features','response','seed_Y','seed_Z','source_task_out_predictions','qualified_mismatch'}, 'fact_schema')
        check(fact['record'] == f'R{i+1}' and fact['task'] == tasks[record['task']], 'anonymous_ids')
        for name in ('features','response','seed_Y','seed_Z'):
            check(fact[name] == record[name], 'source_only_payload:'+name)
        for kind,pred in fact['source_task_out_predictions'].items():
            qualified = abs(pred-record['response']['Y']) > .01 and (all(pred-y > .0025 for y in record['seed_Y']) or all(pred-y < -.0025 for y in record['seed_Y']))
            check(fact['qualified_mismatch'][kind] == qualified, 'qualified_mismatch')
    for raw in {r['id'] for r in pool} | {r['task'] for r in pool} | {r['family'] for r in pool}:
        check(re.search(r'(?<![A-Za-z0-9_])'+re.escape(raw)+r'(?![A-Za-z0-9_])',prompt) is None, 'no_raw_identifier:'+call['name']+raw)
    check(original['exit_code'] == 0 and original['status'] == 'FAILED' and original['tool_items'] == ['error'], 'retained_original_failure:'+call['name'])
    check(started['model'] == original['model_requested'] == generation_lock['model'] and started['reasoning'] == original['reasoning_requested'] == 'high', 'model_request:'+call['name'])
    for kind in ('thread.started','turn.started','turn.completed'):
        check(sum(e['type'] == kind for e in events) == 1, 'single_recorded_turn:'+call['name']+kind)
    thread_ids.extend(e['thread_id'] for e in events if e['type'] == 'thread.started')
    items = [e['item'] for e in events if 'item' in e]
    check(all(item['type'] in ('error','reasoning','agent_message') for item in items), 'no_tool_execution:'+call['name'])
    transport = [i['message'] for i in items if i['type'] == 'error']
    check(transport and all('WebSockets to HTTPS transport' in t for t in transport), 'transport_classification:'+call['name'])
    reconnect = [e['message'] for e in events if e['type'] == 'error']
    check(all(t.startswith('Reconnecting...') and 'request timed out' in t for t in reconnect), 'transport_reconnects:'+call['name'])
    messages = [i for i in items if i['type'] == 'agent_message']
    check(len(messages) == 1 and json.loads(messages[0]['text']) == response, 'response_matches_event:'+call['name'])
    check(original['usage'] == next(e['usage'] for e in events if e['type'] == 'turn.completed'), 'token_usage:'+call['name'])
    check(validation['status'] == 'PASS' and validation['original_receipt_status'] == 'FAILED' and validation['new_LLM_calls'] == validation['new_full_trainings'] == 0 and validation['no_tool_items'], 'supplement_only:'+call['name'])
    for key,path in [('original_receipt_sha256',directory/'receipt.json'),('response_sha256',directory/'response.json'),('events_sha256',directory/'events.jsonl')]:
        check(validation[key] == sha(path), 'validation_hash:'+call['name']+key)
    check(validation['transport_error_messages'] == transport, 'transport_receipt:'+call['name'])
    check(evaluation_lock['outputs'][call['name']] == sha(directory/'response.json'), 'frozen_output_hash:'+call['name'])
    check(set(response) == {'signals','reasoning'} and isinstance(response['reasoning'],str), 'response_schema')
    signals = response['signals']
    check(len(signals) == (2 if call['mode'] == 'ordinary' else 0) == validation['signals'], 'signal_count:'+call['name'])
    check(len({s['name'] for s in signals}) == len(signals), 'signal_names')
    for signal in signals:
        check(set(signal) == {'name','expression','measurement_meaning','limitation','falsifying_response'} and all(isinstance(v,str) and v for v in signal.values()), 'signal_schema')
        for record in pool:
            expression_value(signal['expression'],record['features'])
            counts['permitted_expression_evaluations'] += 1
    signals_by_condition[(call['family'],call['mode'])] = signals
    counts['generation_calls'] += 1
    counts['transport_error_items'] += len(transport)
    counts['transport_reconnect_events'] += len(reconnect)
    call_details.append(dict(name=call['name'],signals=len(signals),seconds=original['seconds'],usage=original['usage'],source_records=len(source),thread_id=thread_ids[-1],ast_nodes=[len(list(ast.walk(ast.parse(s['expression'],mode='eval')))) for s in signals]))
check(len(set(thread_ids)) == 4, 'unique_threads')
for family,hashes in input_hashes.items():
    check(len(hashes) == 2 and len(set(hashes)) == 1, 'same_facts_between_modes:'+family)

rows = []
first_snapshots = {}
run_keys = []
known_cost_nodes = 0
for run in data['runs']:
    source,mode,method = run['source'],run['generation'],run['method']
    key = (source,mode,run['task'],method)
    run_keys.append(key)
    check(source != run['family'] and run['split'] == 'leave_family_out' and method in ('uniform','joint','conditional'), 'source_target_separation:'+str(key))
    records = copy.deepcopy(pool)
    signals = signals_by_condition[(source,mode)]
    experts = {f'{mode}_{i+1}_v1':f'generated_{mode}_{i+1}' for i in range(len(signals))}
    for record in records:
        for name,signal in zip(experts.values(),signals):
            record['features'][name] = expression_value(signal['expression'],record['features'])
    lookup = {r['id']:r for r in records}
    training = [r for r in records if r['family'] == source]
    held = [r for r in records if r['family'] == run['family'] and r['task'] == run['task']]
    remaining = {r['id'] for r in held}
    visible = [r['id'] for r in training]
    queried = []
    reliability = {h:0. for h in experts}
    oracle = max([0.]+[r['response']['Y'] for r in held if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01])
    near(run['oracle_gain'],oracle,'oracle')
    check(len(run['trace']) == len(held), 'trace_length')
    check(len(run['nodes']) == sum(n <= len(held) for n in (2,4,8)), 'node_count')
    first_snapshots.setdefault((source,mode,run['task']),run['trace'][0]['model_snapshots']['reference'])
    for step,trace in enumerate(run['trace'],1):
        counts['steps'] += 1
        check(trace['step'] == step and trace['visible_before'] == visible, 'visible_history:'+str(key))
        snap = trace['model_snapshots']
        check(set(snap['diagnostics']) == set(experts), 'expert_versions')
        models = {'reference':snap['reference'],**snap['diagnostics']}
        for name,model in models.items():
            expected = list(BASE)+( [experts[name]] if name != 'reference' else [])
            counts['predictor_snapshots'] += 1
            for kind in ('marginal','conditional'):
                m = model[kind]
                counts['component_snapshots'] += 1
                check(m['train_record_ids'] == visible, 'snapshot_permitted_ids')
                check(m['feature_names'] == expected+(['observed_Y_scoring_only'] if kind == 'conditional' else []), 'snapshot_features')
                check(m['ridge'] == 1. and m['scale_floor'] == .01 and m['std'] >= .01, 'snapshot_parameters')
                check(len(m['beta']) == len(m['feature_names'])+1 and len(m['scale']) == len(m['center']) == len(m['feature_names']), 'snapshot_shape')
                check(np.isfinite(m['beta']).all() and np.isfinite(m['scale']).all() and np.isfinite(m['center']).all() and min(m['scale']) > 0, 'snapshot_finite')
        registration,event = trace['registration'],trace['event']
        check(registration['model_hash'] == event['model_hash'] == digest(snap), 'snapshot_hash')
        names = list(experts)
        logits = np.array([0.]+[reliability[h]-math.log(len(names)) for h in names]) if names else np.array([0.])
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        near(weights.sum(),1.,'weight_sum')
        check(set(trace['predictions_before']) == remaining, 'candidate_ids')
        computed = {}
        for rid in remaining:
            computed[rid] = sum(w*mean(models[name]['marginal'],lookup[rid]['features']) for w,name in zip(weights,['reference']+names))
            near(trace['predictions_before'][rid],computed[rid],'candidate_prediction')
            counts['candidate_predictions'] += 1
        selected = min(remaining,key=lambda rid:(-computed[rid],rid))
        check(trace['selected'] == registration['event_id'] == event['event_id'] == event['source_response_id'] == selected, 'selected_action')
        near(registration['reference_gain'],mean(snap['reference']['marginal'],lookup[selected]['features']),'registration_reference')
        check(set(registration['gain_predictions']) == set(experts), 'registration_experts')
        for h in names:
            near(registration['gain_predictions'][h],mean(models[h]['marginal'],lookup[selected]['features']),'registration_expert')
        check(event['order'] == 'registered_prediction_scored_before_any_refit', 'score_before_refit')
        check(set(event['before']) == set(event['after']) == set(event['scores']) == set(experts), 'event_expert_keys')
        record = lookup[selected]
        for k in ('Y','Z','NMAE_gain'):
            near(event['response'][k],record['response'][k],'revealed_response')
        baseline = score(snap['reference'],record)
        for h in names:
            proposed = score(models[h],record)
            delta = {k:proposed[k]-baseline[k] for k in baseline}
            counts['expert_score_triples'] += 1
            for k,value in delta.items():
                near(event['scores'][h][k],value,'logscore')
            near(delta['conditional'],delta['joint']-delta['marginal'],'factorization')
            near(event['before'][h],reliability[h],'reliability_before')
            if method != 'uniform':
                reliability[h] += delta[method]
            near(event['after'][h],reliability[h],'reliability_after')
        queried.append(record)
        visible.append(selected)
        remaining.remove(selected)
        eligible = [r for r in queried if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01]
        chosen = max(eligible,key=lambda r:(r['response']['Y'],r['id'])) if eligible else None
        recommendation = chosen['id'] if chosen else 'PARENT'
        check(trace['recommendation'] == recommendation, 'recommendation')
        if step in (2,4,8):
            node = next(n for n in run['nodes'] if n['queries'] == step)
            gain = chosen['response']['Y'] if chosen else 0.
            nmae = chosen['response']['NMAE_gain'] if chosen else 0.
            seeds = chosen['seed_Y'] if chosen else [0.,0.,0.]
            check(node['recommendation'] == recommendation and node['logical_candidate_full_trainings'] == 3*step and node['shared_parent_full_trainings'] == 3, 'node_id_and_budget')
            for k,value in [('gain',gain),('NMAE_gain',nmae),('regret',oracle-gain),('negative_seed_count',sum(y < 0 for y in seeds)),('min_seed_gain',min(seeds)),('query_negative_mean_count',sum(r['response']['Y'] < 0 for r in queried))]:
                near(node[k],value,'node_'+k)
            costs = [r['original_full_training_seconds'] for r in queried]
            if any(c is None for c in costs):
                check(node['query_seconds'] is None, 'unknown_cost_preserved')
            else:
                near(node['query_seconds'],sum(costs),'node_cost')
                known_cost_nodes += 1
            check(0 <= node['policy_seconds'] <= run['seconds'], 'policy_seconds')
            rows.append(dict(source=source,family=run['family'],task=run['task'],generation=mode,update=method,**node))
            counts['nodes'] += 1
    counts['runs'] += 1
check(len(set(run_keys)) == 30, 'unique_runs')
for lock in data['locks']:
    directory = GEN/f"{lock['source']}_{lock['generation']}"
    check(lock['receipt_sha256'] == sha(directory/'validation_receipt.json') and lock['original_receipt_sha256'] == sha(directory/'receipt.json'), 'replay_receipt_lock')
    check(lock['expression_count'] == len(signals_by_condition[(lock['source'],lock['generation'])]) and lock['expression_evaluation_seconds'] >= 0, 'expression_lock')

for row in summary['summaries']:
    subset = [r for r in rows if r['generation'] == row['generation'] and r['update'] == row['update'] and r['queries'] == row['queries']]
    check(row['tasks'] == len(subset), 'summary_task_count')
    near(row['mean_gain'],np.mean([r['gain'] for r in subset]),'summary_gain')
    near(row['mean_regret'],np.mean([r['regret'] for r in subset]),'summary_regret')
    counts['summaries'] += 1
check(len(summary['summaries']) == 18, 'summary_count')
with (OUT/'decisions.csv').open(encoding='utf-8',newline='') as stream:
    csv_rows = list(csv.DictReader(stream))
check(len(csv_rows) == len(rows), 'csv_length')
for actual,expected in zip(csv_rows,rows):
    check(set(actual) == set(expected), 'csv_columns')
    for k,v in expected.items():
        if isinstance(v,(int,float)):
            near(actual[k],v,'csv_'+k)
        elif v is None:
            check(actual[k] == '', 'csv_null')
        else:
            check(actual[k] == v, 'csv_text')
    counts['csv_rows'] += 1

# Compare only the five already-audited reference trajectories and strong-baseline
# summaries. Do not refit or re-audit the previous 70-trajectory matrix.
old = read(STAGE/'manual_expert_replay/results.json')
references = {r['task']:r for r in old['runs'] if r['split'] == 'leave_family_out' and r['method'] == 'reference'}
for run in data['runs']:
    if run['generation'] != 'mismatch':
        continue
    reference = references[run['task']]
    for actual,expected in zip(run['trace'],reference['trace']):
        for field in ('step','visible_before','selected','predictions_before','model_snapshots','registration','event','recommendation'):
            check(actual[field] == expected[field], 'zero_signal_reference_trace:'+field)
        counts['zero_signal_reference_steps'] += 1
    for actual,expected in zip(run['nodes'],reference['nodes']):
        check({k:v for k,v in actual.items() if k != 'policy_seconds'} == {k:v for k,v in expected.items() if k != 'policy_seconds'}, 'zero_signal_reference_node')
        counts['zero_signal_reference_nodes'] += 1
    counts['zero_signal_reference_runs'] += 1
for prediction in data['predictions']:
    source,mode,task,rid = (prediction[k] for k in ('source','generation','task','id'))
    record = byid[rid]
    check(record['family'] != source and record['task'] == task, 'prediction_target_identity')
    near(prediction['Y'],record['response']['Y'],'prediction_label')
    near(prediction['absolute_error'],abs(prediction['mean']-prediction['Y']),'prediction_absolute_error')
    check(prediction['std'] >= .01 and prediction['covered95'] == (abs(prediction['mean']-prediction['Y']) <= 1.959963984540054*prediction['std']), 'prediction_interval')
    if prediction['kind'] == 'reference' or mode == 'mismatch':
        snapshot = first_snapshots[(source,mode,task)]
        near(prediction['mean'],mean(snapshot['marginal'],record['features']),'prediction_reference_mean')
        near(prediction['std'],snapshot['marginal']['std'],'prediction_reference_scale')
        near(prediction['conditional_log_score'],score(snapshot,record)['conditional'],'prediction_reference_score')
        counts['prediction_rows_verified_from_snapshots'] += 1
    else:
        counts['combined_prediction_rows_not_refitted'] += 1
    counts['prediction_rows'] += 1
strong_details = []
for method in ('all_features','all_features_quadratic'):
    for task,reference in references.items():
        strong = next(r for r in old['runs'] if r['split'] == 'leave_family_out' and r['method'] == method and r['task'] == task)
        for step,trace in enumerate(strong['trace'],1):
            expected_visible = reference['trace'][0]['visible_before']+[t['selected'] for t in strong['trace'][:step-1]]
            check(trace['model_snapshots']['reference']['marginal']['train_record_ids'] == expected_visible, 'strong_baseline_receives_feedback')
            counts['strong_baseline_feedback_metadata_steps'] += 1
    for budget in (2,4):
        strong_details.append(next(s for s in old['summaries'] if s['split'] == 'leave_family_out' and s['method'] == method and s['queries'] == budget))
check(summary['new_full_trainings'] == 0 and summary['cumulative_full_trainings'] == 464, 'training_ledger')
report = dict(status='PASS' if not errors else 'FAIL',counts=dict(counts),errors=errors,error_count=len(errors),max_abs_differences=dict(maxima),
    input_sha_same_by_family={f:len(set(h)) == 1 for f,h in input_hashes.items()},calls=call_details,
    frozen_outputs_match=True if not any('frozen_output_hash' in e for e in errors) else False,
    cost_coverage=dict(known_node_query_costs=known_cost_nodes,total_nodes=len(rows),unknown_node_query_costs=len(rows)-known_cost_nodes),
    strong_baseline_summaries=strong_details,
    execution=dict(shell='pwsh (PowerShell 7)',cwd=str(ROOT),python='D:/software/python3_14/python.exe',refitting_performed=False,old_70_trajectory_matrix_rerun=False,
        command="& 'D:/software/python3_14/python.exe' 'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1/generated_replay_audit_check.py'"),
    findings=['No future-target labels in permitted training snapshot IDs or source-only generation packets.',
        'Four distinct recorded generation turns requested gpt-6.1-sol/high; each has transport reconnects and one transport-fallback error item, one agent message and one completion. No tool execution item recorded. Original FAILED receipts retained; validation performs no new model call.',
        'Source generation uses only combinations of the existing 26 measurements; this does not establish new measurements or H1.',
        'Mismatch generation returned zero signals in both source directions; all 15 zero-signal trajectories equal the previously audited reference, apart from runtime accounting.',
        'Strong all-features baselines receive their own previously queried feedback; they are not feedback-free controls.'],
    limitations=['Combined ordinary generated_all prediction-only models have no stored fit snapshots; 32 such rows checked for error/coverage arithmetic but coefficients and conditional scores were not refitted.',
        'Source leave-task-out prediction values were not refitted; payload source identity, all response values and mismatch qualification arithmetic were checked.',
        'Recorded input/output hash consistency cannot independently prove historical process isolation or model backend identity beyond the requested model.',
        'Historical training cost remains incomplete; eight-query summaries cover only three Hydraulic tasks.',
        'No raw A/B/C arrays, servers, D/test, secrets or new model calls accessed.'])
(STAGE/'generated_replay_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
