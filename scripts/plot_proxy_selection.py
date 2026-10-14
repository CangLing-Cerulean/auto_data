"""Render descriptive development results from the audited response matrix only."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.audit.read_text(encoding='utf-8-sig'))
    assert result['status'] == 'PASS'
    args.output.mkdir(parents=True, exist_ok=False)
    methods = ['M1', 'RAW', 'SHORT', 'FIXED']
    tasks = ['G1', 'G2', 'G3']
    colors = ['#176b87', '#d6812b', '#9156a0', '#7b8791']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 12, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout='constrained')
    x = np.arange(len(tasks))
    width = .18
    for i, (method, color) in enumerate(zip(methods, colors)):
        offset = (i - 1.5) * width
        errors = [result['task_MAE_pp'][method][task] for task in tasks]
        gains = [next(d['gain'] for d in result['decisions'] if d['method'] == method and d['task'] == task)
                 for task in tasks]
        axes[0].bar(x + offset, errors, width, color=color, label=method)
        axes[1].bar(x + offset, gains, width, color=color, label=method)
    axes[0].set(title='Response prediction error', ylabel='Mean absolute error (percentage points)')
    axes[1].set(title='Realized gain of the frozen recommendation', ylabel='NMSE improvement relative to parent (%)')
    for ax in axes:
        ax.set_xticks(x, tasks)
        ax.set_xlabel('Previously disclosed development task')
        ax.axhline(0, color='#29353b', linewidth=.7)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(ncols=4, frameon=False, loc='upper left')
    fig.suptitle('Training-data selection: fixed DLinear, four policies, three seeds per task', fontsize=13)
    fig.savefig(args.output / 'selection_summary.png', dpi=200)
    fig.savefig(args.output / 'selection_summary.svg')
    plt.close(fig)

    actions = ['KEEP_FIRST', 'KEEP_SECOND', 'FAVOR_FIRST', 'FAVOR_SECOND']
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 10), layout='constrained')
    width = .15
    for ax, task in zip(axes, tasks):
        cells = [next(c for c in result['cells'] if c['task'] == task and c['action'] == action)
                 for action in actions]
        x = np.arange(len(actions))
        ax.bar(x - 2 * width, [c['truth'] for c in cells], width, color='#202d35', label='Observed')
        for i, (method, color) in enumerate(zip(methods, colors)):
            ax.bar(x + (i - 1) * width, [c['predictions'][method] for c in cells], width,
                   color=color, label=method)
        ax.set_xticks(x, ['Keep first half', 'Keep second half', 'Favor first half', 'Favor second half'])
        ax.set_title(task, loc='left')
        ax.set_ylabel('NMSE improvement (%)')
        ax.axhline(0, color='#29353b', linewidth=.7)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
    axes[0].legend(ncols=5, frameon=False, loc='lower left')
    fig.suptitle('All twelve task-policy cells; mean of three seed-wise responses\nTask panels use separate linear y-axis scales', fontsize=13)
    fig.savefig(args.output / 'all_candidate_responses.png', dpi=180)
    fig.savefig(args.output / 'all_candidate_responses.svg')
    plt.close(fig)
    receipt = {'source_audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
               'scope': 'descriptive development results, not independent-source validation',
               'seed_aggregation': 'arithmetic mean of each seed relative NMSE response',
               'new_training': 0, 'D_access': False, 'test_access': False,
               'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(args.output.iterdir())}}
    (args.output / 'figure_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
