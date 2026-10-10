"""Frozen extra-seed confirmation and paired daily-block sensitivity."""
from datetime import datetime, timezone
import json

import numpy as np

from run_direct_processing import STUDY, PROJECT, run, load, sha, write

SEEDS = (2026, 3026, 3027, 3028)
CONFIRMATION_SEEDS = SEEDS[1:]
POLICIES = ('baseline', 'source_half', 'source_half_scramble')


def pct(base, candidate, key='NMSE'):
    return 100 * (1 - candidate[key] / base[key])


def bootstrap_daily(results):
    days = sorted(results[3026]['baseline']['metrics']['daily'])
    counts = np.array([results[3026]['baseline']['metrics']['daily'][d]['windows'] for d in days])
    series = {}
    for policy in POLICIES:
        rows = []
        for seed in CONFIRMATION_SEEDS:
            daily = results[seed][policy]['metrics']['daily']
            if sorted(daily) != days or not np.array_equal([daily[d]['windows'] for d in days], counts):
                raise ValueError('Daily evaluation support mismatch')
            rows.append([daily[d]['squared_sum'] for d in days])
        series[policy] = np.array(rows).mean(0)
    intervals = {}
    for length in (3, 7, 14):
        rng = np.random.default_rng(20261009)
        blocks = int(np.ceil(len(days) / length))
        starts = rng.integers(0, len(days) - length + 1, size=(10000, blocks))
        selected = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(10000, -1)[:, :len(days)]
        baseline = series['baseline'][selected].sum(1)
        candidate = series['source_half'][selected].sum(1)
        control = series['source_half_scramble'][selected].sum(1)
        gains = 100 * (1 - candidate / baseline)
        control_gains = 100 * (1 - candidate / control)
        intervals[str(length)] = {'block_days': length, 'repetitions': 10000,
            'NMSE_improvement_95pct_interval': np.quantile(gains, [.025, .975]).tolist(),
            'candidate_vs_control_95pct_interval': np.quantile(control_gains, [.025, .975]).tolist()}
    return {'dates': days, 'daily_windows': counts.tolist(), 'intervals': intervals,
        'scope': 'descriptive_development_time_variation_not_independent_or_selection_corrected_inference'}


def summarize(results):
    records = {}
    for seed, policies in results.items():
        hashes = {r['draw_count_sha256'] for r in policies.values()}
        if len(hashes) != 1:
            raise ValueError('Paired exposure changed')
        base, candidate, control = [policies[p]['metrics'] for p in POLICIES]
        records[str(seed)] = {'baseline': base['groups']['all'], 'candidate': candidate['groups']['all'], 'control': control['groups']['all'],
            'NMSE_improvement_pct': pct(base['groups']['all'], candidate['groups']['all']),
            'NMAE_improvement_pct': pct(base['groups']['all'], candidate['groups']['all'], 'NMAE'),
            'local_NMSE_improvement_pct': pct(base['transferred_channels'], candidate['transferred_channels']),
            'candidate_vs_control_NMSE_improvement_pct': pct(control['groups']['all'], candidate['groups']['all'])}
    means = {p: {key: float(np.mean([results[s][p]['metrics']['groups']['all'][key] for s in CONFIRMATION_SEEDS])) for key in ('NMSE', 'NMAE')} for p in POLICIES}
    local = {p: float(np.mean([results[s][p]['metrics']['transferred_channels']['NMSE'] for s in CONFIRMATION_SEEDS])) for p in POLICIES}
    overall_gain = pct(means['baseline'], means['source_half'])
    mae_gain = pct(means['baseline'], means['source_half'], 'NMAE')
    local_gain = 100 * (1 - local['source_half'] / local['baseline'])
    control_gain = pct(means['source_half_scramble'], means['source_half'])
    boot = bootstrap_daily(results)
    gates = {'each_extra_seed_NMSE_improves': all(records[str(s)]['NMSE_improvement_pct'] > 0 for s in CONFIRMATION_SEEDS),
        'mean_NMSE_at_least_1pct': overall_gain >= 1, 'mean_NMAE_degradation_at_most_1pct': mae_gain >= -1,
        'local_NMSE_at_least_3pct': local_gain >= 3, 'candidate_vs_control_at_least_0_25pct': control_gain >= .25,
        'main_7day_interval_positive': boot['intervals']['7']['NMSE_improvement_95pct_interval'][0] > 0}
    return {'scope': 'controlled_researcher_guided_development_only', 'effective_result': all(gates.values()),
        'gates': gates, 'seeds': records, 'confirmation_seed_means': means,
        'confirmation_NMSE_improvement_pct': overall_gain, 'confirmation_NMAE_improvement_pct': mae_gain,
        'confirmation_local_NMSE_improvement_pct': local_gain, 'confirmation_vs_control_improvement_pct': control_gain,
        'daily_block_sensitivity': boot, 'test_access': False, 'API_calls': 0,
        'does_not_establish': ['independent_dataset_generalization', 'row_deletion_or_sampling_effect', 'self_assembled_agent_discovery', 'novelty_or_SOTA']}


def main():
    origin_progress = load(STUDY / 'wave1_progress.json')
    if origin_progress['status'] != 'eligible_for_frozen_full_development_confirmation':
        raise ValueError('Registered two-origin gates did not pass')
    if load(STUDY / 'component_validation.json')['status'] != 'passed':
        raise ValueError('Processing implementation not verified')
    frozen = {'candidate': 'source_half', 'control': 'source_half_scramble', 'origin_progress_sha256': sha(STUDY / 'wave1_progress.json'),
        'protocol_sha256': sha(STUDY / 'EVIDENCE_PROTOCOL.md'), 'worker_sha256': sha(PROJECT / 'scripts/run_direct_processing.py'),
        'seeds': list(SEEDS), 'confirmation_seeds': list(CONFIRMATION_SEEDS), 'no_parameter_or_group_changes_after_confirmation': True,
        'dataset': 'metropt3_v1', 'scope': 'development_only', 'new_data_required_for_this_stage': False}
    freeze_path = STUDY / 'freeze_wave1.json'
    if freeze_path.exists():
        existing = load(freeze_path)
        if existing['identity'] != frozen:
            raise ValueError('Frozen confirmation identity changed')
    else:
        write(freeze_path, {'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'identity': frozen})
    results = {}
    for seed in SEEDS:
        results[seed] = {}
        for policy in POLICIES:
            run_id = f'DP03_full_{policy}_s{seed}'
            results[seed][policy] = run(run_id, 'full', seed, policy)
            write(STUDY / 'confirmation_progress.json', {'status': 'running', 'last_completed_run': run_id,
                'completed_confirmations': sum(len(v) for v in results.values()), 'total_registered': 12})
    result = summarize(results)
    write(STUDY / 'confirmation_result.json', result)
    write(STUDY / 'confirmation_progress.json', {'status': 'effective_controlled_development_result' if result['effective_result'] else 'failed_registered_confirmation_gates',
        'completed_confirmations': 12, 'total_registered': 12, 'result_sha256': sha(STUDY / 'confirmation_result.json')})
    print(json.dumps({'confirmation_finished': True, 'effective_result': result['effective_result'], 'gates': result['gates'],
        'NMSE_improvement_pct': result['confirmation_NMSE_improvement_pct'], 'NMAE_improvement_pct': result['confirmation_NMAE_improvement_pct']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
