"""Read held-out truth only after all 18 scenarios have been predicted and locked."""
import argparse
import json
from pathlib import Path
import statistics

from m1_shrinkage_inference import read, sha


def main(stage):
    assert not (stage / 'evaluation.json').exists()
    manifest, pred = read(stage / 'fold_manifest.json'), read(stage / 'predictions.json')
    assert sha(stage / 'predictions.json') == read(stage / 'prediction_lock.json')['sha256']
    audit = read(stage / 'audit.json')
    assert audit['status'] == 'PASS' and audit['predictions_sha256'] == sha(stage / 'predictions.json')
    assert sha(stage / 'evaluation_truth.json') == manifest['truth_sha256']
    truth = read(stage / 'evaluation_truth.json')
    methods = ('OVERALL', 'JOINT', 'SEPARATE', 'RIDGE_Y', 'RAW', 'M1', 'PER_CANDIDATE_SHRINK')
    rows, recommendations, cost_rows = [], [], []
    mean = statistics.mean
    for result in pred['records']:
        entry = next(f for f in manifest['folds'] if f['id'] == result['id'])
        fold = read(stage / entry['path'])
        local_rows = []
        for r in fold['targets']:
            predicted = {m: next(p for p in result['methods'][m]['predictions'] if p['id'] == r['id'])['gain_percent'] for m in methods[:4]}
            predicted.update(RAW=r['raw_gain'], M1=r['m1_gain'], PER_CANDIDATE_SHRINK=truth[r['id']]['channel_reference'])
            row = dict(fold=fold['id'], task=fold['task'], id=r['id'], seed=r['seed'], action=r['action'],
                probed=r['probed'], true_gain=truth[r['id']]['y'], predicted=predicted,
                errors={m: predicted[m] - truth[r['id']]['y'] for m in methods})
            rows.append(row)
            local_rows.append(row)
        recommendation = dict(fold=fold['id'], task=fold['task'], probe_actions=fold['probe_actions'], methods={})
        for method in methods:
            gains = {a: mean(r['predicted'][method] for r in local_rows if r['action'] == a) for a in fold['actions']}
            action = max(fold['actions'], key=lambda a: gains[a])
            if gains[action] < 1.:
                action = 'PARENT'
            achieved = 0. if action == 'PARENT' else mean(r['true_gain'] for r in local_rows if r['action'] == action)
            recommendation['methods'][method] = dict(action=action, predicted_means=gains, achieved_gain_percent=achieved)
        recommendations.append(recommendation)
        cost_rows.append(dict(fold=fold['id'], task=fold['task'], **fold['cost'],
            actual_auxiliary_seconds={m: result['methods'][m]['seconds'] for m in methods[:4]}))
    assert len(rows) == 216 and sum(not r['probed'] for r in rows) == 108 and len(recommendations) == 18
    summary = {}
    for task in ('ALL', 'G1', 'G2', 'G3'):
        group = [r for r in rows if task == 'ALL' or r['task'] == task]
        decisions = [r for r in recommendations if task == 'ALL' or r['task'] == task]
        summary[task] = {m: dict(unprobed_mae_pp=mean(abs(r['errors'][m]) for r in group if not r['probed']),
            all_mae_pp=mean(abs(r['errors'][m]) for r in group),
            unprobed_signed_error_pp=mean(r['errors'][m] for r in group if not r['probed']),
            mean_selected_gain_percent=mean(r['methods'][m]['achieved_gain_percent'] for r in decisions),
            harmful_recommendations=sum(r['methods'][m]['achieved_gain_percent'] < 0 for r in decisions)) for m in methods}
    gains_guard = {t: summary[t]['SEPARATE']['mean_selected_gain_percent'] >= summary[t]['M1']['mean_selected_gain_percent'] for t in ('G1', 'G2', 'G3')}
    h1_delta = summary['ALL']['JOINT']['unprobed_mae_pp'] - summary['ALL']['OVERALL']['unprobed_mae_pp']
    h2_delta = summary['ALL']['SEPARATE']['unprobed_mae_pp'] - summary['ALL']['JOINT']['unprobed_mae_pp']
    comparisons = {m: summary['ALL']['SEPARATE']['unprobed_mae_pp'] < summary['ALL'][m]['unprobed_mae_pp'] for m in ('OVERALL', 'JOINT', 'RIDGE_Y', 'M1')}
    result = dict(status='EVALUATED', scope='known_source_grouped_replay_not_independent_confirmation',
        summary=summary, h1_joint_minus_overall_mae_pp=h1_delta, h2_separate_minus_joint_mae_pp=h2_delta,
        h1_accuracy_support=h1_delta < 0, h2_accuracy_support=h2_delta < 0,
        separate_precision_comparisons=comparisons, separate_task_gain_guard=gains_guard,
        full_development_criterion=all(comparisons.values()) and all(gains_guard.values()) and h1_delta < 0,
        recommendations=recommendations, candidates=rows, costs=cost_rows,
        predictions_sha256=sha(stage / 'predictions.json'), audit_sha256=sha(stage / 'audit.json'),
        auxiliary_models=54, auxiliary_updates=54000, ridge_fits=18, new_backbone_trainings=0, historical_full_completed=593)
    with (stage / 'evaluation.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(summary=summary, h1_delta=h1_delta, h2_delta=h2_delta, guard=gains_guard,
                          full_development_criterion=result['full_development_criterion'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
