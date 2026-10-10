"""Pure, deterministic selection from the already registered two-origin matrix."""
from statistics import mean

MEMBERS = (20261021, 20261022, 20261023)
FAMILIES = ('oof_window', 'oof_target', 'bilevel', 'coverage_core', 'supervised_core')


def deployed_policy(family):
    return family + '_' + str(MEMBERS[0]) if family.endswith('_core') else family


def control_policy(family):
    return 'coreset_random_' + str(MEMBERS[0]) if family.endswith('_core') else family + '_shuffle'


def select_families(rows):
    assert len(rows) == 30
    indexed = {(row['origin'], row['policy']): row for row in rows}
    assert len(indexed) == 30
    summaries = []
    for family in FAMILIES:
        origins = []
        for origin in (.6, .8):
            names = [family + '_' + str(seed) for seed in MEMBERS] if family.endswith('_core') else [family]
            controls = ['coreset_random_' + str(seed) for seed in MEMBERS] if family.endswith('_core') else [family + '_shuffle']
            candidates = [indexed[(origin, name)] for name in names]
            comparisons = [indexed[(origin, name)] for name in controls]
            candidate_mse, control_mse = mean(r['NMSE'] for r in candidates), mean(r['NMSE'] for r in comparisons)
            improvement = mean(r['NMSE_reduction_pct'] for r in candidates)
            origins.append({'origin': origin, 'candidate_NMSE': candidate_mse, 'control_NMSE': control_mse,
                'candidate_NMAE': mean(r['NMAE'] for r in candidates), 'control_NMAE': mean(r['NMAE'] for r in comparisons),
                'NMSE_reduction_pct': improvement, 'NMAE_reduction_pct': mean(r['NMAE_reduction_pct'] for r in candidates),
                'beats_baseline': improvement > 0, 'beats_control': candidate_mse < control_mse,
                'candidate_run_ids': [r['run_id'] for r in candidates], 'control_run_ids': [r['run_id'] for r in comparisons]})
        summaries.append({'family': family, 'policy': deployed_policy(family), 'control': control_policy(family),
            'origins': origins, 'mean_NMSE_reduction_pct': mean(r['NMSE_reduction_pct'] for r in origins),
            'formal_eligible': all(r['beats_baseline'] and r['beats_control'] for r in origins)})
    summaries.sort(key=lambda row: (-row['mean_NMSE_reduction_pct'], FAMILIES.index(row['family'])))
    return {'ranked_families': summaries, 'formal_candidates': [r['family'] for r in summaries if r['formal_eligible']][:3],
        'additional_origin_and_model_transfer_family': summaries[0]['family'],
        'deployed_member_seed': MEMBERS[0], 'member_selection_by_performance': False, 'test_used': False}
