"""Execute fixed first-wave gates; retain every negative result."""
from run_direct_processing import STUDY, run, write


def improvement(base, candidate, key='NMSE'):
    return 100 * (base[key] - candidate[key]) / base[key]


def gate(results):
    base = results['baseline']['metrics']
    candidate = results['source_half']['metrics']
    control = results['source_half_scramble']['metrics']
    info = {'NMSE_improvement_pct': improvement(base['groups']['all'], candidate['groups']['all']),
        'NMAE_improvement_pct': improvement(base['groups']['all'], candidate['groups']['all'], 'NMAE'),
        'local_NMSE_improvement_pct': improvement(base['transferred_channels'], candidate['transferred_channels']),
        'candidate_vs_control_NMSE_improvement_pct': improvement(control['groups']['all'], candidate['groups']['all'])}
    info['passes_origin_gate'] = (info['NMSE_improvement_pct'] > 0 and info['NMAE_improvement_pct'] >= -1
        and info['local_NMSE_improvement_pct'] >= 3 and info['candidate_vs_control_NMSE_improvement_pct'] > 0)
    hashes = {r['draw_count_sha256'] for r in results.values()}
    if len(hashes) != 1:
        raise ValueError('Unmatched training exposure')
    return info


def main():
    ledger = {'wave': 1, 'candidate': 'source_half', 'prior_is_historical_reference': True, 'origins': {}, 'status': 'running'}
    for number, origin in enumerate(('0.6', '0.8')):
        results = {}
        for policy in ('baseline', 'source_half', 'source_half_scramble', 'source_prior'):
            run_id = f'DP{number + 1:02d}_o{int(float(origin) * 100)}_{policy}_s2026'
            results[policy] = run(run_id, origin, 2026, policy)
        ledger['origins'][origin] = gate(results)
        write(STUDY / 'wave1_progress.json', ledger)
        if not ledger['origins'][origin]['passes_origin_gate']:
            ledger['status'] = 'candidate_rejected_at_registered_origin_gate'
            write(STUDY / 'wave1_progress.json', ledger)
            return
    average = sum(v['NMSE_improvement_pct'] for v in ledger['origins'].values()) / 2
    ledger['two_origin_average_NMSE_improvement_pct'] = average
    ledger['status'] = 'eligible_for_frozen_full_development_confirmation' if average >= 1 else 'candidate_rejected_below_registered_effect_size'
    write(STUDY / 'wave1_progress.json', ledger)


if __name__ == '__main__':
    main()
