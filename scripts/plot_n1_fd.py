"""Plot every registered finite difference and cross-step-size comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(stage):
    audit = json.loads((stage/'fd_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS'
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.3), sharex=True, layout='constrained')
    colors = {'B': '#2463A0', 'C': '#D86A35'}
    for direction in range(4):
        for split in ('B', 'C'):
            for denominator in (64, 128):
                rows = [r for r in audit['derivative_comparisons']
                        if r['direction']==direction and r['split']==split and r['h_denominator']==denominator]
                assert len(rows)==6
                axes[0,direction].plot([r['step'] for r in rows], [r['normalized_error'] for r in rows],
                    color=colors[split], linestyle='-' if denominator==64 else '--',
                    marker='o' if denominator==64 else 's', markersize=3,
                    label=f'{split}, h=1/{denominator}')
            rows = [r for r in audit['two_h_comparisons'] if r['direction']==direction and r['split']==split]
            assert len(rows)==6
            axes[1,direction].plot([r['step'] for r in rows], [r['normalized_error'] for r in rows],
                color=colors[split], marker='o', markersize=3, label=split)
        axes[0,direction].set_title(f'Direction {direction}: '+('mass' if direction==3 else 'mass conserving'), fontsize=10)
        for row in range(2):
            ax=axes[row,direction]
            ax.axhline(.02, color='#AF2444', linewidth=1, linestyle=':', label='Frozen gate 0.02')
            ax.set_yscale('log')
            ax.grid(True, which='major', alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xticks([500,1500,2500])
        axes[1,direction].set_xlabel('Training step')
    axes[0,0].set_ylabel('FD vs tangent\nnormalized discrepancy')
    axes[1,0].set_ylabel('h=1/64 vs h=1/128\nnormalized discrepancy')
    axes[0,0].legend(fontsize=7, loc='best')
    axes[1,0].legend(fontsize=7, loc='best')
    fig.suptitle('N1 fixed finite differences: all 96 derivative + 48 two-h comparisons', fontsize=13)
    fig.savefig(stage/'fd_comparisons.png', dpi=180)
    fig.savefig(stage/'fd_comparisons.pdf')
    plt.close(fig)
    record={'source':'fd_audit.json','source_sha256':hashlib.sha256((stage/'fd_audit.json').read_bytes()).hexdigest(),
        'derivative_points':96,'two_h_points':48,'gate':.02,'matplotlib_version':matplotlib.__version__,
        'files':{name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in ('fd_comparisons.png','fd_comparisons.pdf')}}
    with (stage/'fd_figure_receipt.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2);stream.write('\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    main(parser.parse_args().stage)
