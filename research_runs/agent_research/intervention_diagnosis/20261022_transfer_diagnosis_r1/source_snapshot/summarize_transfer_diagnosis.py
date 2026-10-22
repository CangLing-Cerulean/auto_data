"""Descriptive tables and figure from the locked, audited replay results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(stage):
    data = json.loads((stage / 'evaluation.json').read_text(encoding='utf-8'))
    pred = json.loads((stage / 'predictions.json').read_text(encoding='utf-8'))
    assert data['status'] == 'EVALUATED'
    methods = list(data['summary']['ALL'])
    folds, actions = [], []
    mean = statistics.mean
    for rec in data['recommendations']:
        rows = [r for r in data['candidates'] if r['fold'] == rec['fold']]
        action_order = list(rec['methods']['M1']['predicted_means'])
        true_means = {a: mean(r['true_gain'] for r in rows if r['action'] == a) for a in action_order}
        true_ranking = sorted(action_order, key=lambda a: -true_means[a])
        for m in methods:
            selected = rec['methods'][m]
            predicted = selected['predicted_means']
            ranking = sorted(action_order, key=lambda a: -predicted[a])
            errors = [predicted[a] - true_means[a] for a in action_order]
            folds.append(dict(fold=rec['fold'], task=rec['task'], method=m,
                unprobed_mae_pp=mean(abs(r['errors'][m]) for r in rows if not r['probed']),
                all_mae_pp=mean(abs(r['errors'][m]) for r in rows),
                seed_mean_mae_pp=mean(abs(e) for e in errors),
                selected_action=selected['action'], selected_gain_percent=selected['achieved_gain_percent'],
                predicted_ranking=' > '.join(ranking), true_ranking=' > '.join(true_ranking)))
            for a, error in zip(action_order, errors):
                actions.append(dict(fold=rec['fold'], task=rec['task'], method=m, action=a,
                    probed=a in rec['probe_actions'], predicted_gain_percent=predicted[a],
                    true_gain_percent=true_means[a], error_pp=error,
                    predicted_rank=ranking.index(a) + 1, true_rank=true_ranking.index(a) + 1))
    write_csv(stage / 'scenario_summary.csv', folds)
    write_csv(stage / 'action_summary.csv', actions)
    write_csv(stage / 'candidate_summary.csv', [dict(fold=r['fold'], task=r['task'], id=r['id'],
        seed=r['seed'], action=r['action'], probed=r['probed'], method=m,
        true_gain_percent=r['true_gain'], predicted_gain_percent=r['predicted'][m], error_pp=r['errors'][m])
        for r in data['candidates'] for m in methods])
    comparisons = {}
    for left, right in (('JOINT', 'OVERALL'), ('SEPARATE', 'JOINT'), ('SEPARATE', 'M1'), ('JOINT', 'M1')):
        deltas = []
        for rec in data['recommendations']:
            by_method = {r['method']: r for r in folds if r['fold'] == rec['fold']}
            deltas.append(by_method[left]['unprobed_mae_pp'] - by_method[right]['unprobed_mae_pp'])
        comparisons[f'{left}_minus_{right}'] = dict(mean_mae_delta_pp=mean(deltas),
            lower=sum(v < 0 for v in deltas), equal=sum(v == 0 for v in deltas), higher=sum(v > 0 for v in deltas),
            minimum=min(deltas), maximum=max(deltas))
    cost_summary = {}
    for task in ('G1', 'G2', 'G3'):
        rows = [r for r in data['costs'] if r['task'] == task]
        cost_summary[task] = {name: dict(minimum=min(r[name] for r in rows), mean=mean(r[name] for r in rows),
            maximum=max(r[name] for r in rows)) for name in ('historical_full_seconds', 'historical_proxy_seconds',
            'target_parent_seconds', 'target_proxy_seconds', 'target_prefix_seconds', 'reference_all_prefix_seconds')}
        cost_summary[task]['target_prefix_saved_mean_seconds'] = mean(r['reference_all_prefix_seconds'] - r['target_prefix_seconds'] for r in rows)
    summary = dict(scope='Descriptive summaries only; reused scenarios are not independent samples',
        comparisons=comparisons, historical_recorded_cost_seconds=cost_summary,
        actual_auxiliary_seconds={m: sum(r['methods'][m]['seconds'] for r in pred['records'])
            for m in ('OVERALL', 'JOINT', 'SEPARATE', 'RIDGE_Y')},
        actual_run_seconds=pred['actual_run_seconds'],
        recommendations={task: {m: dict(Counter(r['methods'][m]['action'] for r in data['recommendations'] if r['task'] == task))
            for m in methods} for task in ('G1', 'G2', 'G3')})
    with (stage / 'descriptive_summary.json').open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 1.25]})
    shown = ('M1', 'OVERALL', 'JOINT', 'SEPARATE')
    colors = ('#8a94a6', '#d99c45', '#246b89', '#6b5ba7')
    vals = [data['summary']['ALL'][m]['unprobed_mae_pp'] for m in shown]
    bars = axes[0].barh(shown, vals, color=colors, height=.62)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 11)
    for bar, value in zip(bars, vals):
        axes[0].text(value + .12, bar.get_y() + bar.get_height() / 2, f'{value:.3f}', va='center', fontsize=10)
    axes[0].set_title('Unprobed candidates: pooled MAE', loc='left', fontsize=11)
    axes[0].set_xlabel('Response error (percentage points)')
    x = np.arange(3)
    for j, (m, color) in enumerate(zip(shown, colors)):
        axes[1].bar(x + (j - 1.5) * .18, [data['summary'][t][m]['unprobed_mae_pp'] for t in ('G1', 'G2', 'G3')],
                    width=.17, label=m, color=color)
    axes[1].set_xticks(x, ('G1', 'G2', 'G3'))
    axes[1].set_title('Task-level error', loc='left', fontsize=11)
    axes[1].set_ylabel('Response error (percentage points)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].grid(axis='x', color='#e5e7eb', linewidth=.6)
    axes[1].grid(axis='y', color='#e5e7eb', linewidth=.6)
    fig.suptitle('Transfer of local diagnostic supervision', x=.065, ha='left', fontsize=14, weight='bold')
    fig.text(.065, .025, '18 fixed replay scenarios | 108 unprobed predictions | Known Hydraulic source | No independent confirmation', fontsize=8, color='#586174')
    fig.tight_layout(rect=(0, .06, 1, .94), w_pad=3)
    (stage / 'figures').mkdir(exist_ok=True)
    fig.savefig(stage / 'figures' / 'prediction_error.png', dpi=180)
    fig.savefig(stage / 'figures' / 'prediction_error.pdf')
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    main(parser.parse_args().stage)
