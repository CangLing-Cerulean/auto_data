"""Static scientific figure: all paired means and the three individual seeds."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/agent_research/intervention_diagnosis/20261014_weighted_response_r1'
audit=json.loads((OUT/'final_audit.json').read_text(encoding='utf-8-sig'))
assert audit['status']=='PASS' and len(audit['records'])==36
actions=['KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.labelcolor':'#243246','text.color':'#243246','axes.edgecolor':'#b8c1cd',
                     'xtick.color':'#526174','ytick.color':'#526174','svg.fonttype':'none'})
fig,axes=plt.subplots(3,2,figsize=(11.4,9.3),layout='constrained')
colors={'S':'#2c688f','W':'#bf6843'}
for row,task in enumerate(['G1','G2','G3']):
    ax,gapax=axes[row]
    for col,action in enumerate(actions):
        pairs=[r for r in audit['records'] if r['task']==task and r['action']==action]
        assert len(pairs)==3
        for kind,offset in [('S',-.13),('W',.13)]:
            values=np.array([r[kind+'_gain'] for r in pairs]);mean=values.mean()
            ax.plot([col+offset,col+offset],[values.min(),values.max()],color=colors[kind],lw=1.3)
            ax.scatter(np.full(3,col+offset),values,s=20,facecolors='white',edgecolors=colors[kind],linewidth=.8,zorder=3)
            ax.scatter(col+offset,mean,s=42,color=colors[kind],marker='D',zorder=4,label=kind if col==0 else None)
        gaps=np.array([r['gap_pp'] for r in pairs]);gapax.plot([col,col],[gaps.min(),gaps.max()],color='#637084',lw=1.4)
        gapax.scatter(np.full(3,col),gaps,s=22,color='#a1afbf',zorder=3)
        gapax.scatter(col,gaps.mean(),s=45,color='#253b56',marker='D',zorder=4)
    for axis in [ax,gapax]:
        axis.axhline(0,color='#8c98a8',lw=.8,zorder=0);axis.grid(axis='y',alpha=.17)
        axis.set_xticks(range(4),['KEEP 1','KEEP 2','FAVOR 1','FAVOR 2']);axis.set_xlim(-.55,3.55)
    gapax.axhspan(-.25,.25,color='#dce4ec',alpha=.7,zorder=0)
    ax.set_title(f'{task}  |  Selected-model response',loc='left',fontweight='bold',pad=10)
    gapax.set_title(f'{task}  |  Paired W - S difference',loc='left',fontweight='bold',pad=10)
    ax.set_ylabel('C NMSE improvement (%)');gapax.set_ylabel('Difference (percentage points)')
    if row==0:ax.legend(labels=['Sampling S','Weighting W'],handles=[plt.Line2D([],[],color=colors[k],marker='D',linestyle='') for k in ['S','W']],frameon=False,loc='best')
fig.suptitle('Same mean objective, matched finite-training response',fontsize=16,fontweight='bold')
fig.supxlabel('Diamonds: three-seed means. Dots and ranges: individual seeds, not confidence intervals.\nEach task has its own vertical scale. Shaded band: +/- 0.25 pp. KEEP/FAVOR 1 and 2 refer to the two original A groups.',fontsize=9)
dest=OUT/'figures';dest.mkdir(exist_ok=True)
fig.savefig(dest/'paired_responses.svg',bbox_inches='tight');fig.savefig(dest/'paired_responses.png',dpi=190,bbox_inches='tight');plt.close(fig)
print('Saved paired-response figure for all 12 cells and 36 seeds.')
