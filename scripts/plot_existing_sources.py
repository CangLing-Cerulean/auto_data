"""Publication-style descriptive plots from the independently audited matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit=json.loads(args.audit.read_text(encoding='utf-8-sig'))
    assert audit['status']=='PASS' and {c['task'] for c in audit['cells']}=={'M','S'}
    args.output.mkdir(parents=True,exist_ok=False)
    tasks=['M','S'];names={'M':'MetroPT3','S':'SKAB'}
    methods=['M1','RAW','SHORT','FIXED'];colors=['#176b87','#d6812b','#9156a0','#7b8791']
    actions=['KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(11.2,4.5),layout='constrained')
    x=np.arange(2);width=.18
    for i,(method,color) in enumerate(zip(methods,colors)):
        offset=(i-1.5)*width
        axes[0].bar(x+offset,[audit['task_MAE_pp'][method][t] for t in tasks],width,color=color,label=method)
        axes[1].bar(x+offset,[next(d['gain'] for d in audit['decisions'] if (d['task'],d['method'])==(t,method)) for t in tasks],width,color=color)
    axes[0].set(title='Response prediction error',ylabel='MAE (percentage points)')
    axes[1].set(title='Gain of the frozen recommendation',ylabel='NMSE improvement relative to parent (%)')
    for ax in axes:
        ax.set_xticks(x,[names[t] for t in tasks]);ax.axhline(0,color='#263238',linewidth=.7)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    axes[0].legend(ncols=4,frameon=False)
    fig.suptitle('Fixed M1 on two previously used data sources',fontsize=13)
    for ext in ('png','svg'):fig.savefig(args.output/f'source_summary.{ext}',dpi=200)
    plt.close(fig)

    fig,axes=plt.subplots(2,1,figsize=(11.2,8.4),layout='constrained')
    width=.15
    for ax,task in zip(axes,tasks):
        x=np.arange(4)
        for i,(method,color) in enumerate(zip(['Observed']+methods,['#263238']+colors)):
            means=[];position=x+(i-2)*width
            for j,action in enumerate(actions):
                cell=next(c for c in audit['cells'] if (c['task'],c['action'])==(task,action))
                means.append(cell['truth'] if method=='Observed' else cell['predictions'][method])
                rows=sorted([r for r in audit['scalar_records'] if (r['task'],r['action'])==(task,action)],key=lambda r:r['seed'])
                values=[r['actual_gain'] if method=='Observed' else r['predicted'][method] for r in rows]
                ax.scatter(position[j]+np.array([-.025,0,.025]),values,s=14,color=color,edgecolors='white',linewidths=.35,zorder=4)
                ax.vlines(position[j],min(values),max(values),color=color,linewidth=.8,zorder=3)
            ax.bar(position,means,width,color=color,alpha=.65,label=method,zorder=2)
        ax.set_xticks(x,['Keep first half','Keep second half','Favor first half','Favor second half'])
        ax.set_title(names[task],loc='left');ax.set_ylabel('NMSE improvement (%)')
        ax.axhline(0,color='#263238',linewidth=.7);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(ncols=5,frameon=False,loc='best')
    fig.suptitle('All eight source-policy cells\nBars: three-seed means; points and spans: individual seeds and ranges, not confidence intervals',fontsize=11)
    for ext in ('png','svg'):fig.savefig(args.output/f'all_candidate_responses.{ext}',dpi=200)
    plt.close(fig)
    receipt=dict(source_audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        scope='Previously used sources, train-only development; not final test or unseen-source validation',
        seed_aggregation='Arithmetic mean of seed-wise relative NMSE gains; ranges are descriptive, not confidence intervals',
        all_registered_cells_included=True,new_training=0,test_access=False,
        files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir())})
    (args.output/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
