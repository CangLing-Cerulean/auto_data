"""Create-only tables of audited predictions; no labels or model assets are read."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import math
from pathlib import Path
import re

import fourgroup_policy as policy


OUTPUTS = ('forecast_seed_table.csv', 'forecast_cell_table.csv',
           'forecast_cost_table.csv', 'forecast_recommendations.csv', 'forecast_summary.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read(path):
    raw = path.read_bytes()
    value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=unique_object,
                       parse_constant=lambda token: number(float(token), token))
    return value, hashlib.sha256(raw).hexdigest()


def number(value, name, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value), f'Invalid number: {name}')
    require(not nonnegative or value >= 0, f'Negative cost: {name}')
    return value


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def boolean(value, name):
    require(type(value) is bool, f'Invalid boolean: {name}')
    return value


def indexed(rows, fields, expected, name):
    require(isinstance(rows, list) and len(rows) == len(expected), f'Incomplete {name}')
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f'Duplicate {name}: {key}')
        result[key] = row
    require(set(result) == expected, f'Incorrect {name} coverage')
    return result


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def summarize(stage):
    require(stage.is_absolute() and stage.is_dir(), '--stage must be an absolute local directory')
    # Check the entire output set before reading or creating anything.
    for name in OUTPUTS:
        if (stage / name).exists():
            raise FileExistsError(stage / name)
    plan, plan_hash = read(stage / 'plan.json')
    prediction, prediction_hash = read(stage / 'predictions.json')
    audit, audit_hash = read(stage / 'forecast_audit.json')
    lock, lock_hash = read(stage / 'input_lock.json')
    require(audit['status'] == 'PASS' and audit['mode'] == 'forecast', 'A passing forecast audit is required')
    require(audit['plan_sha256'] == prediction['plan_sha256'] == plan_hash, 'Plan binding mismatch')
    require(audit['predictions_sha256'] == prediction_hash, 'Predictions binding mismatch')
    require(audit['input_lock_sha256'] == plan['input_lock_sha256'] == lock_hash, 'Input lock binding mismatch')
    identities = {}
    for name, key in (('inputs/data/train.csv', 'train_sha256'),
                      ('inputs/data/manifest.json', 'manifest_sha256')):
        matches = [item for item in lock['files'] if item['path'] == name]
        require(len(matches) == 1 and matches[0]['sha256'] == audit[key], f'Data identity mismatch: {name}')
        identities[key] = digest(audit[key])
    require(plan['seeds'] == list(policy.SEEDS) and plan['actions'] == list(policy.ACTIONS)
            and [t['id'] for t in plan['tasks']] == list(policy.TASKS), 'Frozen menu mismatch')
    require(plan['cumulative_before'] == 418, 'Cumulative count mismatch')
    require(prediction['status'] == 'completed' and prediction['new_full_trainings'] == 0
            and prediction['cumulative_full_trainings'] == 418, 'Prediction completion mismatch')
    require(prediction['full_children_started'] is False and prediction['scientific_accuracy'] is None
            and prediction['actual_gain'] is None, 'Expected an unlabelled forecast')
    require(prediction['proxy_rollouts'] == prediction['short_runs'] == 72, 'Forecast count mismatch')
    for source in (prediction, audit):
        require(source['D_access'] is False and source['test_access'] is False, 'Unexpected data access')
    records = indexed(prediction['records'], ('task', 'seed', 'action'),
                      {(t, s, a) for t in policy.TASKS for s in policy.SEEDS for a in policy.ACTIONS}, 'records')
    costs = indexed(prediction['costs'], ('task', 'seed'),
                    {(t, s) for t in policy.TASKS for s in policy.SEEDS}, 'costs')
    audited_costs = indexed(audit['costs'], ('task', 'seed'), set(costs), 'audited costs')
    seed_rows, cell_rows, cost_rows, recommendation_rows = [], [], [], []
    for (task, seed, action), record in sorted(records.items(), key=lambda item:
            (policy.TASKS.index(item[0][0]), policy.SEEDS.index(item[0][1]), policy.ACTIONS.index(item[0][2]))):
        require(set(record['methods']) == set(policy.METHODS), 'Method coverage mismatch')
        row = dict(task=task, seed=seed, action=action)
        for method in policy.METHODS:
            gain = number(record['methods'][method]['gain_percent'], 'gain_percent')
            require(method != 'PARENT' or gain == 0, 'PARENT gain must be zero')
            row[f'{method}_predicted_gain_percent'] = gain
        seed_rows.append(row)
    choices = {m: {} for m in policy.METHODS}
    require(set(prediction['recommendations']) == set(policy.TASKS), 'Recommendation task mismatch')
    for task in policy.TASKS:
        rec_row = dict(task=task)
        require(set(prediction['recommendations'][task]) == set(policy.METHODS), 'Recommendation method mismatch')
        for action in policy.ACTIONS:
            row = dict(task=task, action=action)
            for method in policy.METHODS:
                gains = [records[(task, seed, action)]['methods'][method]['gain_percent'] for seed in policy.SEEDS]
                # Use the frozen policy's exact arithmetic mean implementation.
                mean = policy.recommend({s: {a: records[(task, s, a)]['methods'][method]['gain_percent']
                    for a in policy.ACTIONS} for s in policy.SEEDS}).mean_gains[action]
                row.update({f'{method}_predicted_gain_{kind}_percent': value
                            for kind, value in (('mean', mean), ('min', min(gains)), ('max', max(gains)))})
            cell_rows.append(row)
        for method in policy.METHODS:
            recommendation = asdict(policy.recommend({s: {a: records[(task, s, a)]['methods'][method]['gain_percent']
                for a in policy.ACTIONS} for s in policy.SEEDS}))
            require(prediction['recommendations'][task][method] == recommendation, 'Saved recommendation mismatch')
            require(audit['recommendations'][task][method] == recommendation, 'Audited recommendation mismatch')
            action = recommendation['action']
            choices[method][task] = action
            rec_row[f'{method}_action'] = action
            rec_row[f'{method}_recommended_predicted_gain_percent'] = (0. if action == 'PARENT'
                else recommendation['mean_gains'][action])
        recommendation_rows.append(rec_row)
    for task in policy.TASKS:
        for seed in policy.SEEDS:
            cost, checked = costs[(task, seed)], audited_costs[(task, seed)]
            row = dict(task=task, seed=seed)
            for key in (*policy.METHODS, 'real_parent_seconds', 'allocated_common_preparation_seconds'):
                value = number(cost[key], key, nonnegative=True)
                reference = number(checked[key], key, nonnegative=True)
                require(abs(value-reference) <= 1e-9, 'Audited cost mismatch')
                row[f'{key}_warm_seconds' if key in policy.METHODS else key] = value
            require(cost['PARENT'] == 0, 'PARENT warm cost must be zero')
            require(set(cost['cold_allocated_seconds']) == set(policy.METHODS), 'Cold method coverage mismatch')
            for method in policy.METHODS:
                value = number(cost['cold_allocated_seconds'][method], method, nonnegative=True)
                reference = number(checked['cold_allocated_seconds'][method], method, nonnegative=True)
                expected = cost[method] + cost['real_parent_seconds'] + cost['allocated_common_preparation_seconds']
                require(abs(value-reference) <= 1e-9 and value == expected, 'Cold allocation mismatch')
                row[f'{method}_cold_allocated_seconds'] = value
            gate = cost['M1'] <= cost['SHORT']
            require(boolean(cost['budget_pass'], 'budget_pass') == boolean(checked['budget_pass'], 'audit budget_pass') == gate,
                    'Budget result mismatch')
            row['budget_pass'] = gate
            cost_rows.append(row)
    stop = asdict(policy.recommendation_stop(choices))
    stop['matching_baselines'] = list(stop['matching_baselines'])
    budget = all(row['budget_pass'] for row in cost_rows)
    eligible = budget and not stop['stop']
    require(prediction['identical_recommendation_stop'] == stop and audit['matching_baselines'] == stop['matching_baselines'],
            'Saved recommendation stop mismatch')
    for source in (prediction, audit):
        require(boolean(source['budget_gate'], 'budget_gate') == budget
                and boolean(source['eligible_to_label'], 'eligible_to_label') == eligible, 'Saved gate mismatch')
    numeric_columns = [k for k in cost_rows[0] if k not in ('task', 'seed', 'budget_pass')]
    totals = {k: math.fsum(row[k] for row in cost_rows) for k in numeric_columns}
    task_means = {t: {k: math.fsum(row[k] for row in cost_rows if row['task'] == t)/3
                     for k in numeric_columns} for t in policy.TASKS}
    equal_mean = {k: math.fsum(task_means[t][k] for t in policy.TASKS)/3 for k in numeric_columns}
    counts = {key: number(audit[key], key, nonnegative=True) for key in
              ('audit_updates', 'states_verified', 'residuals_verified', 'moment_files_verified',
               'source_propagation_blocks_verified')}
    counts.update(source_matrix_exponential_probes=len(audit['source_matrix_exponential_probes']),
                  replay_audit_records=len(audit['replay_audit']))
    summary = dict(status='PASS', artifact_kind='audited_predictions_only',
        dataset_id='uci_hydraulic_v1', **identities, plan_sha256=plan_hash,
        predictions_sha256=prediction_hash, forecast_audit_sha256=audit_hash, input_lock_sha256=lock_hash,
        preparation_sha256=digest(audit['preparation_sha256']), auditor_sha256=digest(audit['auditor_sha256']),
        audit_counts=counts, table_rows=dict(seed=72, cell=24, cost=9, recommendations=3),
        cost_seconds=dict(totals=totals, per_task_seed_mean=task_means, task_equal_mean=equal_mean),
        common_recommendations=choices, budget_gate=budget, identical_recommendation_stop=stop,
        eligible_to_label=eligible, new_full_trainings=0, cumulative=418,
        scientific_accuracy=None, actual_gain=None, numerical_audit_status='PASS',
        scientific_gates_evaluated=False, D_access=False, test_access=False,
        interpretation=(('Identical recommendations make this batch\'s frozen increment over all baselines unreachable. '
                         if stop['stop'] else
                         'The identical-recommendation stop condition was not triggered; actual accuracy and gain remain unknown. ')
                        + 'This does not establish general M1 failure or actual M1 versus SHORT gain. '
                        'These tables contain predictions, not measured child performance.'))
    contents = [csv_text(rows) for rows in (seed_rows, cell_rows, cost_rows, recommendation_rows)]
    contents.append(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    # Reserve all names exclusively before writing, including under concurrent invocations.
    handles = []
    try:
        for name in OUTPUTS:
            handles.append((stage / name).open('x', encoding='utf-8', newline=''))
        for stream, content in zip(handles, contents):
            stream.write(content)
    finally:
        for stream in handles:
            stream.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.stage)
    print(json.dumps(dict(status=summary['status'], table_rows=summary['table_rows'])))


if __name__ == '__main__':
    main()
