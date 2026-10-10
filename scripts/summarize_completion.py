"""Rebuild completion-study evidence from saved results without running a model."""
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np
from choose_completion_families import select_families

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'research_runs/completion_study'
RESULTS = ROOT / 'research_runs/results'
BASELINES = ('S000_uniform_s2026', 'S005_uniform_s2027', 'S010_uniform_s2028')
PRIOR_SECONDS = 6566.439622849226


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def result(name):
    return read(RESULTS / name / 'result.json')


def complete(name):
    path = RESULTS / name / 'status.json'
    return path.exists() and read(path)['phase'] == 'completed'


def metric(name):
    value = result(name)
    return value.get('future', value)['groups']['all']


def reduction(candidate, baseline):
    return 100 * (1 - candidate / baseline)


def paired(names, baselines=BASELINES):
    assert len(names) == len(baselines)
    rows = []
    for candidate, baseline in zip(names, baselines):
        a, b = metric(candidate), metric(baseline)
        rows.append({'run_id': candidate, 'baseline': baseline, 'seed': 2026 + len(rows),
            'NMSE': a['NMSE'], 'NMAE': a['NMAE'], 'baseline_NMSE': b['NMSE'], 'baseline_NMAE': b['NMAE'],
            'NMSE_reduction_pct': reduction(a['NMSE'], b['NMSE']), 'NMAE_reduction_pct': reduction(a['NMAE'], b['NMAE'])})
    candidates = [result(name).get('future', result(name)) for name in names]
    references = [result(name).get('future', result(name)) for name in baselines]
    common_groups = set.intersection(*(set(value['groups']) for value in candidates + references))
    groups = []
    for group in sorted(common_groups):
        a = [value['groups'][group] for value in candidates]; b = [value['groups'][group] for value in references]
        assert all(left['windows'] == right['windows'] for left, right in zip(a, b))
        groups.append({'group': group, 'windows': a[0]['windows'],
            'NMSE_reduction_pct': reduction(mean(v['NMSE'] for v in a), mean(v['NMSE'] for v in b)),
            'NMAE_reduction_pct': reduction(mean(v['NMAE'] for v in a), mean(v['NMAE'] for v in b))})
    channels = []
    for column, sensor in enumerate(result(names[0])['sensor_columns']):
        a = mean(value['per_channel_NMSE'][column] for value in candidates)
        b = mean(value['per_channel_NMSE'][column] for value in references)
        channels.append({'sensor': sensor, 'candidate_NMSE': a, 'baseline_NMSE': b,
            'NMSE_reduction_pct': reduction(a, b) if b else None, 'contribution_to_overall_NMSE_change': (a - b) / 15})
    assert abs(sum(row['contribution_to_overall_NMSE_change'] for row in channels)
        - (mean(row['NMSE'] for row in rows) - mean(row['baseline_NMSE'] for row in rows))) < 1e-10
    return {'rows': rows, 'groups': groups, 'channels': channels,
        'mean_NMSE': mean(row['NMSE'] for row in rows), 'mean_NMAE': mean(row['NMAE'] for row in rows),
        'NMSE_reduction_pct': reduction(mean(row['NMSE'] for row in rows), mean(row['baseline_NMSE'] for row in rows)),
        'NMAE_reduction_pct': reduction(mean(row['NMAE'] for row in rows), mean(row['baseline_NMAE'] for row in rows)),
        'NMSE_improving_seeds': sum(row['NMSE_reduction_pct'] > 0 for row in rows), 'seed_count': len(rows)}


def timing_summary():
    plan = read(FOLDER / 'long_plan.json')
    if not all(complete(row['run_id']) for row in plan): return {'phase': 'not_fully_completed'}
    references = {('uniform', 2026): BASELINES[0], ('uniform', 2027): BASELINES[1], ('uniform', 2028): BASELINES[2],
        ('adaptive_is', 2026): 'I001_adaptive_is_numpy_cdf_s2026', ('adaptive_is', 2027): 'C004_adaptive_is_s2027',
        ('adaptive_is', 2028): 'C005_adaptive_is_s2028'}
    rows = []
    for entry in plan:
        name = entry['run_id']; saved = result(name); reference = references[(entry['policy'], entry['seed'])]
        prior = metric(reference); milestone = saved['milestone_3000']['metrics']['groups']['all']
        count_file = 'sampling_weights.npz' if reference.startswith('S') else 'window_draws.npz' if reference.startswith('I') else 'windows.npz'
        with np.load(RESULTS / name / 'draws3000.npz') as current, np.load(RESULTS / reference / count_file) as old:
            counts_identical = np.array_equal(current['draw_count'], old['draw_count'])
        available = [row for row in saved['history'] if row['metric_available_elapsed_seconds'] <= 120]
        selected = min(available, key=lambda row: row['validation']['NMSE']) if available else None
        rows.append({**entry, 'reference_3000': reference, 'draw_counts_at_3000_identical': counts_identical,
            'NMSE_at_3000_absolute_difference': abs(milestone['NMSE'] - prior['NMSE']),
            'best_3000': milestone, 'last_3000': read(RESULTS / name / 'validation_3000.json')['groups']['all'],
            'best_6000': saved['groups']['all'], 'best_step_6000': saved['best_step'],
            'last_6000': read(RESULTS / name / 'validation_6000.json')['groups']['all'],
            'wall_budget_seconds': 120, 'best_within_wall_budget': selected,
            'last_available_step_at_wall_budget': available[-1]['step'] if available else None,
            'scoring_seconds': sum(row['scoring_seconds'] for row in saved['phases']), 'elapsed_seconds': saved['elapsed_seconds']})
    comparisons = []
    for key in ('best_3000', 'last_3000', 'best_6000', 'last_6000', 'wall120'):
        pairs = []
        for seed in (2026, 2027, 2028):
            baseline = next(row for row in rows if row['seed'] == seed and row['policy'] == 'uniform')
            candidate = next(row for row in rows if row['seed'] == seed and row['policy'] == 'adaptive_is')
            b = baseline['best_within_wall_budget']['validation'] if key == 'wall120' and baseline['best_within_wall_budget'] else baseline.get(key)
            a = candidate['best_within_wall_budget']['validation'] if key == 'wall120' and candidate['best_within_wall_budget'] else candidate.get(key)
            if a is not None and b is not None:
                pairs.append({'seed': seed, 'candidate_NMSE': a['NMSE'], 'baseline_NMSE': b['NMSE'],
                    'candidate_NMAE': a['NMAE'], 'baseline_NMAE': b['NMAE'], 'NMSE_reduction_pct': reduction(a['NMSE'], b['NMSE'])})
        comparisons.append({'scope': key, 'pairs': pairs, 'paired_seeds': len(pairs),
            'NMSE_reduction_pct': reduction(mean(r['candidate_NMSE'] for r in pairs), mean(r['baseline_NMSE'] for r in pairs)) if pairs else None,
            'NMAE_reduction_pct': reduction(mean(r['candidate_NMAE'] for r in pairs), mean(r['baseline_NMAE'] for r in pairs)) if pairs else None})
    return {'phase': 'completed', 'rows': rows, 'comparisons': comparisons,
        'first_3000_equivalence_passed': all(row['draw_counts_at_3000_identical'] and row['NMSE_at_3000_absolute_difference'] < 1e-9 for row in rows)}


def main():
    confirmations = {
        'drop_earliest': paired(('C001_drop_earliest_s2026', 'C002_drop_earliest_s2027', 'C003_drop_earliest_s2028')),
        'adaptive_is': paired(('I001_adaptive_is_numpy_cdf_s2026', 'C004_adaptive_is_s2027', 'C005_adaptive_is_s2028')),
        'reshuffle': paired(('C006_reshuffle_s2026', 'C007_reshuffle_s2027', 'C008_reshuffle_s2028'))}
    confirmations['earliest_vs_matched_random'] = paired(('C001_drop_earliest_s2026', 'C002_drop_earliest_s2027', 'C003_drop_earliest_s2028'),
        ('S003_random_drop_s2026', 'S008_random_drop_s2027', 'S013_random_drop_s2028'))
    completed = []
    for path in sorted(RESULTS.glob('*/status.json')):
        name = path.parent.name
        if not (name.startswith('C') or name.startswith(('A100_', 'A101_', 'A102_', 'A103_', 'A104_'))): continue
        status = read(path)
        if status['phase'] == 'completed':
            identity = read(path.parent / 'identity.json')
            completed.append({'run_id': name, 'run_type': identity['run_type'], 'elapsed_seconds': status['elapsed_seconds']})
    summary = {'phase': 'partial_until_all_authorized_plans_resolved', 'evaluation_scope': 'train-internal probes and development validation; no new test',
        'historical_training_runs': 95, 'historical_main_runs': 92, 'historical_auxiliary_training_runs': 3,
        'baseline_seed_means': {'NMSE': mean(metric(name)['NMSE'] for name in BASELINES), 'NMAE': mean(metric(name)['NMAE'] for name in BASELINES)},
        'confirmations': confirmations, 'long_training': timing_summary(), 'completed_local_runs': completed,
        'locally_backed_up_new_main_runs': sum(row['run_type'] == 'main_model' for row in completed),
        'locally_backed_up_new_diagnostic_runs': sum(row['run_type'] != 'main_model' for row in completed),
        'historical_elapsed_seconds': PRIOR_SECONDS, 'new_completed_local_elapsed_seconds': sum(row['elapsed_seconds'] for row in completed),
        'compute_budget_limit_seconds': None}
    matrix_path = FOLDER / 'policy_progress.json'
    if matrix_path.exists():
        matrix = read(matrix_path); summary['two_origin_matrix'] = matrix
        if matrix['phase'] == 'completed_two_origin_matrix': summary['matrix_selection'] = select_families(matrix['rows'])
        attribution = {}
        for row in matrix['rows']:
            if complete(row['run_id']):
                actual = metric(row['run_id'])
                assert actual['NMSE'] == row['NMSE'] and actual['NMAE'] == row['NMAE']
                assert actual['windows'] == row['windows']
                reference = metric(row['baseline'])
                assert abs(reduction(actual['NMSE'], reference['NMSE']) - row['NMSE_reduction_pct']) < 1e-12
                attribution[row['run_id']] = {'origin': row['origin'], 'policy': row['policy'],
                    **paired((row['run_id'],), (row['baseline'],))}
        (FOLDER / 'error_attribution.json').write_text(json.dumps({'completed_comparisons': len(attribution),
            'comparisons': attribution}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        summary['error_attribution'] = {'path': 'error_attribution.json', 'completed_comparisons': len(attribution)}
        if matrix['phase'] == 'completed_two_origin_matrix' and summary['long_training']['phase'] != 'completed':
            summary['phase'] = 'matrix_complete_planned_followups_not_fully_completed'
    quality_path = FOLDER / 'measurement_audit/result.json'
    if quality_path.exists(): summary['measurement_quality_cpu_audit'] = read(quality_path)
    if complete('C301_prefix70_control'):
        summary['additional_origin'] = {'candidate': paired(('C300_prefix70_candidate',), ('C090_prefix70_uniform',)),
            'control': paired(('C301_prefix70_control',), ('C090_prefix70_uniform',))}
    if complete('C406_mlp_candidate_s2028'):
        summary['fixed_mlp_transfer'] = {'candidate': paired(('C401_mlp_candidate_s2026', 'C404_mlp_candidate_s2027', 'C406_mlp_candidate_s2028'),
            ('C400_mlp_uniform_s2026', 'C403_mlp_uniform_s2027', 'C405_mlp_uniform_s2028')),
            'control2026': paired(('C402_mlp_control_s2026',), ('C400_mlp_uniform_s2026',)),
            'new_uniform_vs_old': paired(('C400_mlp_uniform_s2026', 'C403_mlp_uniform_s2027', 'C405_mlp_uniform_s2028'),
                ('M001_residual_mlp_s2026', 'M002_residual_mlp_s2027', 'M003_residual_mlp_s2028'))}
    (FOLDER / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'confirmations': {key: {k: row[k] for k in ('NMSE_reduction_pct', 'NMAE_reduction_pct', 'NMSE_improving_seeds')} for key, row in confirmations.items()},
        'new_backed_up_main_runs': summary['locally_backed_up_new_main_runs'], 'long_training': summary['long_training']['phase']}))


if __name__ == '__main__': main()
