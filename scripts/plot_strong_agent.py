"""Render descriptive registered-budget comparisons without pseudo-replication."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnosis_pool_features import read

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_strong_agent_r5'

def main():
    assert read(STAGE/'result_audit.json')['status']=='PASS'
    rows=read(STAGE/'analysis.json')['comparison']
    methods=[('reference','Primitive reference'),('original_quadratic','Fixed quadratic'),('window_fixed','Fixed window features'),('manual_conditional','Manual conditional'),('mismatch_conditional','Generated conditional'),('strong_agent','Strong adaptive agent'),('uniform_random_exact_expectation','Uniform random (exact)')]
    colors=['#94a3b8','#475569','#475569','#0f766e','#0f766e','#c16b28','#a8a29e']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.1,3.7),sharey=True,layout='constrained')
    for ax,q in zip(axes,(2,4)):
        values=[100*next(r['mean_gain'] for r in rows if r['method']==m and r['queries']==q) for m,_ in methods]
        ax.barh(range(len(methods)),values,color=colors,height=.65,zorder=2)
        ax.set_yticks(range(len(methods)),[label for _,label in methods]);ax.set_ylim(len(methods)-.5,-.5)
        ax.set_title(f'{q} candidate queries',loc='left',fontsize=11,fontweight='bold',pad=11)
        ax.set_xlim(0,15);ax.set_xticks([0,5,10,15]);ax.set_xlabel('Mean selected NMSE improvement (%)')
        ax.grid(axis='x',color='#e5e7eb',linewidth=.7,zorder=0)
        for y,value in enumerate(values):ax.text(value+.15,y,f'{value:.2f}',va='center',fontsize=8)
    fig.suptitle('Five development tasks · two source families · one Agent trajectory per task',fontsize=10)
    out=STAGE/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'budget_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(out/'budget_comparison.pdf',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
