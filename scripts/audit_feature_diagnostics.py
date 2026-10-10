"""Post-hoc decomposition of preserved inference evidence; no data/model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

RUNS = ('S000_uniform_s2026', 'S005_uniform_s2027', 'S010_uniform_s2028')


def main(folder):
    started = time.monotonic()
    folder = Path(folder)
    destination = folder / 'posthoc_audit.json'
    if destination.exists():
        raise ValueError('Do not overwrite completed audit')
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    path = folder / 'window_evidence.npz'
    with path.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if source_hash != result['assets']['window_evidence.npz']:
        raise ValueError('Inference evidence changed')
    with np.load(path, allow_pickle=False) as archive:
        evidence = {k: archive[k] for k in archive.files}
    columns = list(result['channels'])
    avg_sq = np.stack([evidence[f'{r}_sq'] for r in RUNS]).mean(0)
    avg_abs = np.stack([evidence[f'{r}_abs'] for r in RUNS]).mean(0)
    total = float(avg_sq.sum())
    analog = evidence['slice_analog_all_constant']
    long = evidence['slice_duration_long']
    special_days = np.isin(evidence['target_start_day'], ('2020-07-21', '2020-07-22'))
    masks = {'analog_constant': analog, 'long_duration': long, 'long_excluding_analog_constant': long & ~analog,
        'analog_constant_and_long': analog & long, 'two_days': special_days, 'outside_two_days': ~special_days,
        'recent_switch': evidence['slice_digital_recent_switch'], 'low_variation': evidence['slice_variation_low']}
    slices = {}
    for name, mask in masks.items():
        slices[name] = {'windows': int(mask.sum()), 'D_share': float(mask.mean()),
            'NMSE_three_seed_mean': float(avg_sq[mask].mean()) if mask.any() else None,
            'NMAE_three_seed_mean': float(avg_abs[mask].mean()) if mask.any() else None,
            'overall_squared_error_share': float(avg_sq[mask].sum() / total),
            'per_channel_NMSE': dict(zip(columns, avg_sq[mask].mean(0).tolist())) if mask.any() else {},
            'per_channel_fraction_of_slice_error': dict(zip(columns, (avg_sq[mask].sum(0) / avg_sq[mask].sum()).tolist())) if mask.any() else {}}
    constant_excess = {}
    for c, name in enumerate(columns):
        mask = evidence['input_constant'][:, c]
        seed_errors = [float(evidence[f'{r}_sq'][mask, c].mean()) for r in RUNS]
        reference = float(evidence['persistence_sq'][mask, c].mean())
        days = evidence['target_start_day']
        daily = {}
        for day in np.unique(days[mask]):
            selected = mask & (days == day)
            if selected.sum() < 100:
                continue
            gaps = [float((evidence[f'{r}_sq'][selected, c] - evidence['persistence_sq'][selected, c]).mean()) for r in RUNS]
            daily[str(day)] = {'windows': int(selected.sum()), 'NMSE_excess_vs_persistence_by_seed': gaps}
        constant_excess[name] = {'windows': int(mask.sum()), 'model_NMSE_by_seed': seed_errors, 'persistence_NMSE': reference,
            'all_seeds_worse': all(v > reference for v in seed_errors),
            'net_contribution_to_overall_NMSE_excess': float(((avg_sq[:, c] - evidence['persistence_sq'][:, c]) * mask).mean() / 15),
            'days_with_at_least_100_windows': len(daily), 'days_all_seeds_worse': sum(all(g > 0 for g in v['NMSE_excess_vs_persistence_by_seed']) for v in daily.values()), 'daily': daily}
    audit = {'scope': 'posthoc_development_description_only', 'source_sha256': source_hash, 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_training_calls': 0, 'new_API_calls': 0, 'fitted_new_thresholds': False,
        'slice_intersections': slices, 'constant_channel_model_excess': constant_excess,
        'interpretation_limits': ['two-day grouping selected after first diagnostic; no independent confirmation', 'persistence difference does not prove improvement attainable through data processing', 'channel-constant feature differs from whole-window low variation', 'days overlap in input and target support; no significance claim'],
        'elapsed_seconds': time.monotonic() - started}
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed_posthoc_audit', 'elapsed_seconds': audit['elapsed_seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    main(parser.parse_args().folder)
