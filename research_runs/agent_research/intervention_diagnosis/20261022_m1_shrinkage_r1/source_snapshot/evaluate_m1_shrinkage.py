"""Label evaluation only after the new forecast lock and numerical audit."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def select(curves):
    return min(sorted(curves, key=int), key=lambda t: curves[t]['B'])


def main(source, stage):
    assert not (stage / 'evaluation.json').exists()
    pred, audit = read(stage / 'predictions.json'), read(stage / 'audit.json')
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256'] == audit['predictions_sha256']
    assert audit['status'] == 'PASS'
    lock = read(stage / 'source_lock.json')
    assert sha(stage / 'source_lock.json') == pred['source_lock_sha256']
    for name, digest in lock['source_metadata_sha256'].items():
        assert sha(source / name) == digest
    for name, digest in lock['evaluation_metadata_sha256'].items():
        assert sha(source / name) == digest
    labels = read(source / 'labels.json')['records']
    spec, old = read(source / 'spec.json'), read(source / 'predictions.json')['records']
    parents = read(source / 'parents.json')['records']
    assert len(labels) == 36
    mean = statistics.mean
    methods = ('RAW', 'M1', 'PREFIX', 'SHORT', 'RAW_PREFIX', 'M1_PREFIX', 'SCALAR_SHRINK', 'CHANNEL_SHRINK')
    rows = []
    for row in pred['records']:
        key = tuple(row[x] for x in ('task', 'seed', 'action'))
        def match(records):
            matches = [r for r in records if tuple(r[x] for x in ('task', 'seed', 'action')) == key]
            assert len(matches) == 1
            return matches[0]
        previous, full, raw = match(old), match(labels), match(audit['raw_records'])
        parent = next(p for p in parents if (p['task'], p['seed']) == key[:2])
        base, ts = parent['selected_losses']['C'], select(full['curves'])
        assert int(ts) == full['selected_step']
        true_gain = 100. * (base - full['curves'][ts]['C']) / base
        item = dict(task=key[0], seed=key[1], action=key[2], true_gain_percent=true_gain,
                    true_selected_step=int(ts), methods={})
        for method in methods:
            curves = (raw if method == 'RAW' else previous['methods'][method] if method in ('M1', 'PREFIX', 'SHORT') else row['methods'][method])['curves']
            selected = select(curves)
            gain = 100. * (base - curves[selected]['C']) / base
            fixed = {t: 100. * (full['curves'][t]['C'] - curves[t]['C']) / base for t in curves if t in full['curves']}
            item['methods'][method] = dict(selected_step=int(selected), gain_percent=gain,
                response_error_pp=gain - true_gain, fixed_step_response_error_pp=fixed)
        rows.append(item)
    summary, recommendations = {}, {}
    for task in ('ALL', *spec['tasks']):
        group = [r for r in rows if task == 'ALL' or r['task'] == task]
        summary[task] = {m: dict(mae_pp=mean(abs(r['methods'][m]['response_error_pp']) for r in group),
            signed_error_pp=mean(r['methods'][m]['response_error_pp'] for r in group),
            checkpoint_mismatches=sum(r['methods'][m]['selected_step'] != r['true_selected_step'] for r in group),
            fixed_step_mae_pp={str(step): mean(abs(r['methods'][m]['fixed_step_response_error_pp'][str(step)]) for r in group)
                for step in spec['grid'] if all(str(step) in r['methods'][m]['fixed_step_response_error_pp'] for r in group)}) for m in methods}
        if task == 'ALL':
            continue
        recommendations[task] = {}
        for method in methods:
            gains = {a: mean(r['methods'][method]['gain_percent'] for r in group if r['action'] == a) for a in spec['actions']}
            action = max(spec['actions'], key=lambda a: gains[a])
            if gains[action] < 1.:
                action = 'PARENT'
            achieved = 0. if action == 'PARENT' else mean(r['true_gain_percent'] for r in group if r['action'] == action)
            recommendations[task][method] = dict(action=action, predicted_means=gains, achieved_gain_percent=achieved)
    gates = {}
    for method in ('SCALAR_SHRINK', 'CHANNEL_SHRINK'):
        precision = {m: summary['ALL'][method]['mae_pp'] < summary['ALL'][m]['mae_pp'] for m in ('RAW', 'M1', 'PREFIX', 'RAW_PREFIX', 'M1_PREFIX')}
        utility = {t: recommendations[t][method]['achieved_gain_percent'] >= recommendations[t]['M1']['achieved_gain_percent'] for t in spec['tasks']}
        gates[method] = dict(precision_comparisons=precision, task_utility_comparisons=utility,
                            pass_all=all(precision.values()) and all(utility.values()))
    passing = [m for m in ('SCALAR_SHRINK', 'CHANNEL_SHRINK') if gates[m]['pass_all']]
    chosen = min(passing, key=lambda m: summary['ALL'][m]['mae_pp']) if passing else None
    original_audit = read(source / 'audit_revision4.json')
    assert original_audit['status'] == 'PASS'
    for method in ('M1', 'PREFIX', 'SHORT'):
        assert abs(summary['ALL'][method]['mae_pp'] - original_audit['mae_pp'][method]) < 1e-10
    result = dict(status='EVALUATED', scope='posthoc_known_source_development_not_independent_confirmation',
        predictions_sha256=sha(stage / 'predictions.json'), audit_sha256=sha(stage / 'audit.json'),
        labels_sha256=sha(source / 'labels.json'), summary=summary, recommendations=recommendations,
        mean_selected_gain_percent={m: mean(recommendations[t][m]['achieved_gain_percent'] for t in spec['tasks']) for m in methods},
        seed_mean_mae_pp={m: mean(abs(mean(r['methods'][m]['response_error_pp'] for r in rows if r['task'] == t and r['action'] == a))
                                for t in spec['tasks'] for a in spec['actions']) for m in methods},
        gates=gates, candidate_for_future_validation=chosen, candidates=rows,
        conditional_reuse_mean_four_candidate_seconds={m: sum(r['methods'][m]['conditional_reuse_ledger_seconds'] for r in pred['records']) / 9. for m in pred['records'][0]['methods']},
        new_full_trainings=0, historical_full_completed=593)
    with (stage / 'evaluation.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('gates', 'candidate_for_future_validation', 'mean_selected_gain_percent', 'seed_mean_mae_pp')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.stage)
