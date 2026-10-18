"""Independent r5 result audit. Run only after the main agent says results ready.

Reads frozen pools and result artifacts only; never imports a production fitter,
builder, selector or strategy, and never invokes Codex, prepare or training.
"""
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[3]
PREVIOUS = STAGE.parent / '20261018_measurement_generation_r4'
TASKS = [('ETT', 'ETTh1'), ('ETT', 'ETTh2'), ('Hydraulic', 'G1'), ('Hydraulic', 'G2'), ('Hydraulic', 'G3')]
BASE_FEATURES = ('exposure_shift_first', 'removed_fraction', 'weighted_implementation',
    'target_weight_second_moment', 'first_training_mass', 'parent_B_log_loss', 'parent_B_target_ratio',
    'parent_B_local_contrast', 'parent_B_bias', 'parent_B_target_bias', 'parent_B_channel_dispersion',
    'parent_B_learning_gain', 'parent_B_best_fraction', 'parent_B_curve_reversal',
    'B_input_variance', 'B_input_roughness', 'first_input_variance', 'second_input_variance',
    'first_input_roughness', 'second_input_roughness', 'first_B_level_distance', 'second_B_level_distance')
CHECKS = []
ERRORS = []
MODEL_ERRORS = []


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, context):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), context + ': shape/nonfinite')
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    MODEL_ERRORS.append(dict(context=context, max_absolute_error=difference))
    require(np.allclose(a, b, rtol=2e-9, atol=1e-9), context + f': numerical mismatch {difference}')


def audited(name, function):
    try:
        details = function()
        CHECKS.append(dict(check=name, status='PASS', details=details))
    except Exception as exc:
        ERRORS.append(dict(check=name, error=repr(exc)))
        CHECKS.append(dict(check=name, status='FAIL', error=repr(exc)))


def vector(record, columns, quadratic):
    values = [record['features'][column] for column in columns]
    return values + [values[i]*values[j] for i in range(len(values)) for j in range(i, len(values))] if quadratic else values


def independent_mean_fit(x, y, groups):
    """Weighted ridge normal equations in sample space, ridge=1, free intercept."""
    counts = Counter(groups)
    weights = np.array([1. / counts[group] for group in groups])
    center = np.sum(weights[:, None]*x, axis=0) / np.sum(weights)
    variance = np.sum(weights[:, None]*(x-center)**2, axis=0) / np.sum(weights)
    scale = np.sqrt(np.maximum(variance, 1.))
    design = (x-center)/scale
    intercept = float(weights @ y / weights.sum())
    a = np.sqrt(weights)[:, None]*design
    target = np.sqrt(weights)*(y-intercept)
    coefficients = a.T @ np.linalg.solve(np.eye(len(y)) + a @ a.T, target)
    # Independent stationarity evidence supplements the reconstructed snapshot.
    residual = a @ coefficients-target
    require(np.linalg.norm(a.T @ residual + coefficients) <= 2e-6*(1+np.linalg.norm(coefficients)), 'ridge normal equation residual')
    return center, scale, np.r_[intercept, coefficients]


def independent_model(x, y, groups, record_ids, columns):
    x, y, groups = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
    residuals = np.empty(len(y))
    for group in sorted(set(groups)):
        train = groups != group
        center, scale, beta = independent_mean_fit(x[train], y[train], groups[train])
        residuals[~train] = y[~train] - (beta[0]+((x[~train]-center)/scale) @ beta[1:])
    std = max(float(np.sqrt(np.mean([np.mean(residuals[groups == group]**2) for group in sorted(set(groups))]))), .01)
    center, scale, beta = independent_mean_fit(x, y, groups)
    return dict(center=center.tolist(), scale=scale.tolist(), beta=beta.tolist(), std=std, feature_names=list(columns), train_record_ids=record_ids, ridge=1., scale_floor=.01)


def independent_models(visible, names):
    signals = ('mismatch_measurement_0', 'mismatch_measurement_1')
    specifications = {'reference': (BASE_FEATURES, False), 'fixed_quadratic': (names, True), 'measurement_combined': ((*BASE_FEATURES, *signals), False)}
    result = {}
    groups = [r['family']+':'+r['task'] for r in visible]
    ids = [r['id'] for r in visible]
    y, z = [r['response']['Y'] for r in visible], [r['response']['Z'] for r in visible]
    for key, (columns, quadratic) in specifications.items():
        x = [vector(record, columns, quadratic) for record in visible]
        expanded = list(columns) + [f'{columns[i]}*{columns[j]}' for i in range(len(columns)) for j in range(i, len(columns))] if quadratic else list(columns)
        result[key] = dict(marginal=independent_model(x, y, groups, ids, expanded), conditional=independent_model(np.column_stack([x, y]), z, groups, ids, [*expanded, 'observed_Y_scoring_only']))
    return result, specifications


def recommendation(records):
    eligible = [r for r in records if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01]
    chosen = sorted(eligible, key=lambda r: (-r['response']['Y'], r['id']))[0] if eligible else None
    times = [r['original_full_training_seconds'] for r in records]
    return dict(recommendation=chosen['id'] if chosen else 'PARENT', gain=chosen['response']['Y'] if chosen else 0., NMAE_gain=chosen['response']['NMAE_gain'] if chosen else 0., negative_seed_count=sum(v < 0 for v in chosen['seed_Y']) if chosen else 0, query_seconds=None if None in times else sum(times), logical_candidate_full_trainings=3*len(records), shared_parent_full_trainings=3)


def frozen_files():
    lock = read(STAGE / 'execution_lock.json')
    require(lock['tasks'] == [list(t) for t in TASKS] and lock['max_calls'] == 20 and lock['queries_per_task'] == 4, 'frozen task/budget')
    require((lock['model'], lock['reasoning'], lock['new_full_trainings'], lock['cumulative_full_trainings']) == ('gpt-6.1-sol', 'high', 0, 464), 'frozen model/accounting')
    for name, value in lock['code_sha256'].items():
        require(sha(ROOT / 'scripts' / name) == value, 'code hash '+name)
    require(sha(STAGE / 'PROTOCOL.md') == lock['protocol_sha256'], 'protocol hash')
    for family, value in lock['pool_sha256'].items():
        require(sha(PREVIOUS / 'replay' / f'{family}_augmented_records.json') == value, 'pool hash '+family)
    for family, value in lock['definitions_sha256'].items():
        require(sha(PREVIOUS / 'generation' / f'{family}_mismatch/response.json') == value, 'definition hash '+family)
    require(read(STAGE / 'source_publication_receipt.json')['status'] == 'exact_commit_and_tree_verified', 'source publication receipt')
    require(read(STAGE / 'started.json')['execution_lock_sha256'] == sha(STAGE / 'execution_lock.json'), 'run lock link')
    require(not (STAGE / 'stopped.json').exists(), 'stopped run cannot be treated as complete')
    complete = read(STAGE / 'completed.json')
    require(complete == dict(status='COMPUTED_AUDIT_PENDING', calls=20, tasks=5, new_full_trainings=0, cumulative_full_trainings=464), 'completed accounting')
    expected = {f'{family}_{task}_{step}' for family, task in TASKS for step in range(1, 5)}
    actual = {p.name for p in (STAGE / 'calls').iterdir() if p.is_dir()}
    require(actual == expected, 'exactly twenty call directories')
    return dict(calls=20, execution_lock_sha256=sha(STAGE / 'execution_lock.json'))


def task_audit(family, task):
    source_family = 'Hydraulic' if family == 'ETT' else 'ETT'
    records = read(PREVIOUS / 'replay' / f'{source_family}_augmented_records.json')['records']
    source = [r for r in records if r['family'] == source_family]
    held = [r for r in records if (r['family'], r['task']) == (family, task)]
    idmap = {r['id']: f'S{i+1}' for i, r in enumerate(sorted(source, key=lambda r: r['id']))}
    idmap.update({r['id']: f'Q{i+1}' for i, r in enumerate(sorted(held, key=lambda r: r['id']))})
    reverse = {alias: name for name, alias in idmap.items()}
    names = tuple(sorted(name for name in records[0]['features'] if not name.startswith(('window_', 'ordinary_measurement_', 'mismatch_measurement_'))))
    require(len(names) == 26, 'original feature count')
    permitted = (*names, 'mismatch_measurement_0', 'mismatch_measurement_1')
    taskmap = {name: f'T{i+1}' for i, name in enumerate(sorted({r['task'] for r in source}))}
    definitions = read(PREVIOUS / 'generation' / f'{source_family}_mismatch/response.json')['signals']
    script_ast = ast.parse((ROOT / 'scripts/strong_agent_baseline.py').read_text(encoding='utf-8'))
    function = next(n for n in script_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'prompt')
    prefix = next(n for n in function.body if isinstance(n, ast.Return)).value.left.value
    history, nodes, call_rows, timings = [], [], [], []
    previous_reveal = None
    def facts(record, response):
        value = dict(id=idmap[record['id']], task=taskmap.get(record['task'], 'TARGET'), features={name: record['features'][name] for name in permitted})
        if response:
            value.update(response={name: record['response'][name] for name in ('Y', 'Z', 'NMAE_gain')}, seed_Y=record['seed_Y'], seed_Z=record['seed_Z'])
        return value
    for step in range(1, 5):
        context = f'{family}_{task}_{step}'
        folder = STAGE / 'calls' / context
        seal, packet, snapshots = read(folder / 'input_lock.json'), read(folder / 'input.json'), read(folder / 'models.json')
        require(seal['execution_lock_sha256'] == sha(STAGE / 'execution_lock.json') and seal['step'] == step, context+': input chain')
        require(set(seal['files']) == {'input.json', 'models.json', 'prompt.txt', 'schema.json'}, context+': sealed files')
        for name, value in seal['files'].items():
            require(sha(folder / name) == value, context+': sealed hash '+name)
        expected_top = dict(query_number=step, total_query_budget=4, report_nodes=[2, 4], measurement_definitions={f'mismatch_measurement_{i}': d for i, d in enumerate(definitions)}, source_records=[facts(r, True) for r in source], target_revealed=[dict(**facts(h['record'], True), prior_agent_decision=h['agent_decision']) for h in history])
        require(set(packet) == {*expected_top, 'unqueried_candidates'}, context+': packet schema')
        require({k: packet[k] for k in expected_top} == expected_top, context+': source/history/definition permission')
        revealed_ids = {h['id'] for h in history}
        pending = [r for r in held if r['id'] not in revealed_ids]
        require([row['id'] for row in packet['unqueried_candidates']] == [idmap[r['id']] for r in pending], context+': no replacement')
        expected_models, specs = independent_models(source+[h['record'] for h in history], names)
        require(set(snapshots) == set(expected_models), context+': model list')
        for model_name, model in expected_models.items():
            require(set(snapshots[model_name]) == {'marginal', 'conditional'}, context+': model components')
            for component, expected in model.items():
                actual = snapshots[model_name][component]
                require(set(actual) == set(expected), context+': model fields')
                for field in ('feature_names', 'train_record_ids', 'ridge', 'scale_floor'):
                    require(actual[field] == expected[field], context+': isolated model '+field)
                for field in ('center', 'scale', 'beta', 'std'):
                    close(actual[field], expected[field], context+'/'+model_name+'/'+component+'/'+field)
        for row, record in zip(packet['unqueried_candidates'], pending):
            require(set(row) == {'id', 'task', 'features', 'fitted_suggestions'}, context+': hidden candidate fields')
            require({k: row[k] for k in ('id', 'task', 'features')} == facts(record, False), context+': parent/action features')
            require(set(row['fitted_suggestions']) == set(specs), context+': suggestions')
            for model_name, (columns, quadratic) in specs.items():
                model = expected_models[model_name]['marginal']
                design = (np.array(vector(record, columns, quadratic))-model['center'])/model['scale']
                prediction = model['beta'][0]+design @ model['beta'][1:]
                require(set(row['fitted_suggestions'][model_name]) == {'mean_Y', 'std_Y'}, context+': suggestion fields')
                close([row['fitted_suggestions'][model_name]['mean_Y'], row['fitted_suggestions'][model_name]['std_Y']], [prediction, model['std']], context+'/'+row['id']+'/'+model_name+'/suggestion')
        expected_prompt = prefix+json.dumps(packet, ensure_ascii=False, separators=(',', ':'))
        require((folder / 'prompt.txt').read_text(encoding='utf-8') == expected_prompt, context+': prompt packet')
        schema, decision, receipt = read(folder / 'schema.json'), read(folder / 'response.json'), read(folder / 'receipt.json')
        require(schema['additionalProperties'] is False and set(schema['required']) == {'candidate_id', 'expected_Y', 'expected_Z', 'decision_reason'}, context+': output schema')
        require(schema['properties']['candidate_id']['enum'] == [idmap[r['id']] for r in pending], context+': candidate enum')
        require(set(decision) == set(schema['required']) and decision['candidate_id'] in schema['properties']['candidate_id']['enum'], context+': decision')
        require(all(type(decision[k]) in (int, float) for k in ('expected_Y', 'expected_Z')) and np.isfinite([decision['expected_Y'], decision['expected_Z']]).all(), context+': finite forecasts')
        require(isinstance(decision['decision_reason'], str) and len(decision['decision_reason']) <= 2000, context+': rationale')
        require(receipt['status'] == 'PASS' and receipt['exit_code'] == 0 and receipt['response_sha256'] == sha(folder / 'response.json'), context+': valid response receipt')
        events = [json.loads(line) for line in (folder / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        completed = [e for e in events if e.get('type') == 'turn.completed']
        require(len(completed) == 1 and receipt['usage'] == completed[0].get('usage'), context+': exactly one completed decision and usage')
        messages = [e['item']['text'] for e in events if e.get('item', {}).get('type') == 'agent_message' and e.get('type') == 'item.completed']
        require(bool(messages) and json.loads(messages[-1]) == decision, context+': final event response matches response file')
        tools = [e['item']['type'] for e in events if e.get('item', {}).get('type') not in (None, 'error', 'reasoning', 'agent_message')]
        errors = [e['item'] for e in events if e.get('item', {}).get('type') == 'error']
        require(not tools and receipt['tool_items'] == tools and receipt['transport_errors'] == errors, context+': tools/transport accounting')
        started = read(folder / 'started.json')
        require(started['model'] == receipt['model_requested'] == 'gpt-6.1-sol' and started['reasoning'] == receipt['reasoning_requested'] == 'high', context+': model request')
        if previous_reveal is not None:
            require(started['started'] >= previous_reveal, context+': prior reveal before next call')
        require(np.isfinite([receipt['seconds'], seal['preparation_seconds']]).all() and min(receipt['seconds'], seal['preparation_seconds']) >= 0, context+': costs')
        actual_id = reverse[decision['candidate_id']]
        require(actual_id not in revealed_ids, context+': unique query')
        record = next(r for r in held if r['id'] == actual_id)
        history.append(dict(id=actual_id, record=record, agent_decision=decision))
        revealed = read(folder / 'revealed.json')
        require(revealed['history'] == history and revealed['response_receipt_sha256'] == sha(folder / 'receipt.json'), context+': response/history chain')
        require(revealed['recommendation'] == recommendation([h['record'] for h in history]), context+': adoption rule')
        require(revealed['revealed_at'] >= started['started'], context+': reveal time order')
        previous_reveal = revealed['revealed_at']
        call_rows.append(dict(family=family, task=task, step=step, **receipt))
        timings.append(dict(call=context, request_started=started['started'], revealed_at=previous_reveal, process_seconds=receipt['seconds'], preparation_seconds=seal['preparation_seconds'], usage=receipt['usage'], transport_events=len(errors)))
        if step in (2, 4):
            nodes.append(dict(family=family, task=task, queries=step, **recommendation([h['record'] for h in history])))
    random_rows = []
    for budget in (2, 4):
        gains = [max([0.]+[r['response']['Y'] for r in subset if r['response']['Y'] >= .01 and r['response']['NMAE_gain'] >= -.01]) for subset in itertools.combinations(held, budget)]
        random_rows.append(dict(family=family, task=task, queries=budget, subsets=len(gains), expected_gain=float(sum(gains)/len(gains))))
    return dict(nodes=nodes, random=random_rows, calls=call_rows, timings=timings, model_components=24)


def aggregate(task_results):
    require(len(task_results) == 5, 'all five audited tasks required for aggregate')
    result = read(STAGE / 'results.json')
    expected_nodes = [row for task in task_results for row in task['nodes']]
    expected_random = [row for task in task_results for row in task['random']]
    expected_calls = [row for task in task_results for row in task['calls']]
    require(result['decisions'] == expected_nodes and result['calls'] == expected_calls, 'aggregate decision/call records')
    require(len(result['uniform_random']) == 10, 'random row count')
    for actual, expected in zip(result['uniform_random'], expected_random):
        require({k: actual[k] for k in ('family', 'task', 'queries', 'subsets')} == {k: expected[k] for k in ('family', 'task', 'queries', 'subsets')}, 'random identity/count')
        close(actual['expected_gain'], expected['expected_gain'], 'uniform_random_expected_gain')
    summaries = []
    for budget in (2, 4):
        agent_gain = float(np.mean([r['gain'] for r in expected_nodes if r['queries'] == budget]))
        random_gain = float(np.mean([r['expected_gain'] for r in expected_random if r['queries'] == budget]))
        summaries.append(dict(queries=budget, agent_mean_gain=agent_gain, uniform_random_expected_mean_gain=random_gain))
    require(len(result['summaries']) == 2, 'exactly two aggregate report nodes')
    for actual, expected in zip(result['summaries'], summaries):
        require(actual['queries'] == expected['queries'], 'summary budget')
        close([actual['agent_mean_gain'], actual['uniform_random_expected_mean_gain']], [expected['agent_mean_gain'], expected['uniform_random_expected_mean_gain']], 'summary gains')
    require(result['new_full_trainings'] == 0 and result['cumulative_full_trainings'] == 464, 'final training ledger')
    timings = [row for task in task_results for row in task['timings']]
    usage_totals = Counter()
    for row in timings:
        require(isinstance(row['usage'], dict), 'usage available')
        for key, value in row['usage'].items():
            if isinstance(value, (int, float)):
                require(value >= 0, 'nonnegative usage')
                usage_totals[key] += value
    # Each round must finish its revelations before the next round starts.
    for step in (1, 2, 3):
        previous = [r for r in timings if r['call'].endswith('_'+str(step))]
        following = [r for r in timings if r['call'].endswith('_'+str(step+1))]
        require(max(r['revealed_at'] for r in previous) <= min(r['request_started'] for r in following), 'round barrier')
    generation_costs = []
    for family in ('ETT', 'Hydraulic'):
        for mode in ('ordinary', 'mismatch'):
            receipt = read(PREVIOUS / 'generation' / f'{family}_{mode}/receipt.json')
            generation_costs.append(dict(call=family+'_'+mode, seconds=receipt.get('seconds'), usage=receipt.get('usage'), status=receipt.get('status')))
    return dict(summaries=summaries, calls=20, trajectory_nodes=10, independent_model_components=120, process_seconds_sum=sum(r['process_seconds'] for r in timings), preparation_seconds_sum=sum(r['preparation_seconds'] for r in timings), usage_totals=dict(usage_totals), transport_error_events=sum(r['transport_events'] for r in timings), request_to_last_revelation_seconds=max(r['revealed_at'] for r in timings)-min(r['request_started'] for r in timings), historical_generator_costs=generation_costs, node_costs=expected_nodes, timings=timings)


def main():
    start = time.perf_counter()
    audited('frozen_files_and_completion', frozen_files)
    task_results = []
    for family, task in TASKS:
        def one(family=family, task=task):
            value = task_audit(family, task)
            task_results.append(value)
            return dict(calls=4, nodes=2, model_components=value['model_components'])
        audited(f'trajectory_{family}_{task}', one)
    audited('aggregate_random_cost_tokens_and_rounds', lambda: aggregate(task_results))
    report = dict(status='PASS' if not ERRORS else 'FAIL', checks=CHECKS, failures=ERRORS, numeric_comparisons=len(MODEL_ERRORS), max_absolute_numeric_difference=max([0.]+[r['max_absolute_error'] for r in MODEL_ERRORS]), numeric_comparison_details=MODEL_ERRORS, audit_seconds=time.perf_counter()-start, code_sha256=sha(ROOT / 'scripts/strong_agent_baseline.py'), auditor_sha256=sha(__file__), results_sha256=sha(STAGE / 'results.json') if (STAGE / 'results.json').exists() else None, new_full_trainings=0, actual_LLM_calls_by_auditor=0, limitations=['Artifact hashes establish internal consistency, not cryptographic proof of historical creation order or absence of undocumented outside calls.', 'Recorded model request and event usage cannot independently attest the actual remote model identity, backend compute time, or transport causal behavior.', 'Process seconds and preparation seconds are recorded client measurements; concurrent sums are not elapsed wall time or an equal-compute comparison.', 'Preparation time includes fitting plus serialization and is not a standalone fitting timer.', 'Only the locked parent-feature/response pools and source measurement definitions are inspected; no raw arrays, D/test, new strategy execution, or repeated gate/r4 mathematics audit.', 'Scientific scope remains one adaptive trajectory per task across two previously revealed families.'])
    (STAGE / 'result_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], checks=len(CHECKS), failures=len(ERRORS), numeric_comparisons=len(MODEL_ERRORS))))
    if ERRORS:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
