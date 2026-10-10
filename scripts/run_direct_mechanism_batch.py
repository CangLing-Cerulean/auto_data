"""Five fixed no-donor ablations; do not retune the frozen processing method."""
import json

import numpy as np

from run_direct_processing import STUDY, load, write, sha
from run_direct_own_ablation import run


def main():
    own_results = {}
    records = {}
    for origin, seed in (('0.6', 2026), ('0.8', 2026), ('full', 3026), ('full', 3027), ('full', 3028)):
        result = run(origin, seed)
        own_results[(origin, seed)] = result
        tag = f'DP0{1 if origin == "0.6" else 2}_o{int(float(origin) * 100)}' if origin != 'full' else 'DP03_full'
        candidate = load(STUDY / 'runs' / f'{tag}_source_half_s{seed}' / 'result.json')
        baseline = load(STUDY / 'runs' / f'{tag}_baseline_s{seed}' / 'result.json')
        if result['draw_count_sha256'] != candidate['draw_count_sha256'] or result['draw_count_sha256'] != baseline['draw_count_sha256']:
            raise ValueError('Ablation training exposure not matched')
        control_metrics = result['metrics']['groups']['all']
        candidate_metrics = candidate['metrics']['groups']['all']
        records[f'{origin}/{seed}'] = {'own_half': control_metrics, 'candidate': candidate_metrics,
            'candidate_NMSE_improvement_vs_own_half_pct': 100 * (1 - candidate_metrics['NMSE'] / control_metrics['NMSE']),
            'candidate_NMAE_improvement_vs_own_half_pct': 100 * (1 - candidate_metrics['NMAE'] / control_metrics['NMAE'])}
        write(STUDY / 'mechanism_progress.json', {'status': 'running', 'completed': len(records), 'total': 5, 'last': result['identity']['run_id']})
    own_mean = np.mean([records[f'full/{s}']['own_half']['NMSE'] for s in (3026, 3027, 3028)])
    candidate_mean = np.mean([records[f'full/{s}']['candidate']['NMSE'] for s in (3026, 3027, 3028)])
    gain = float(100 * (1 - candidate_mean / own_mean))
    gates = {'each_origin_candidate_better': all(records[f'{o}/2026']['candidate_NMSE_improvement_vs_own_half_pct'] > 0 for o in ('0.6', '0.8')),
        'each_confirmation_seed_candidate_better': all(records[f'full/{s}']['candidate_NMSE_improvement_vs_own_half_pct'] > 0 for s in (3026, 3027, 3028)),
        'mean_source_increment_at_least_0_25pct': gain >= .25}
    report = {'scope': 'registered_development_mechanism_ablation', 'records': records, 'gates': gates,
        'source_increment_supported': all(gates.values()), 'mean_candidate_NMSE_improvement_vs_own_half_pct': gain,
        'protocol_sha256': sha(STUDY / 'MECHANISM_ABLATION.md'),
        'does_not_prove': ['all_possible_processing_mechanisms', 'independent_dataset_generalization']}
    write(STUDY / 'mechanism_result.json', report)
    write(STUDY / 'mechanism_progress.json', {'status': 'completed', 'completed': 5, 'total': 5, 'source_increment_supported': report['source_increment_supported']})
    print(json.dumps({'mechanism_finished': True, 'source_increment_supported': report['source_increment_supported'], 'mean_increment_pct': gain}), flush=True)


if __name__ == '__main__':
    main()
