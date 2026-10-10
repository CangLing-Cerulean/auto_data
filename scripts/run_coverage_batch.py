"""Explore three full-support policies and a temporal-shuffle control."""
import json
import subprocess
import sys
import time
from run_research import ROOT,write_json
from coverage_sampling import FOLDER,POLICIES,ROUND_LIMIT,used_seconds


def read_metrics(name,formal=False):
    r=json.loads((ROOT/'research_runs/results'/name/'result.json').read_text())
    return (r if formal else r['future'])['groups']['all']


def main():
    path=FOLDER/'progress.json'
    if path.exists(): raise RuntimeError('Inspect existing progress; do not repeat batch')
    progress={'phase':'starting','rows':[],'shortlist':[],'selected_policy':None}
    def save(phase):
        progress.update(phase=phase,budget_used_seconds=used_seconds(),round_limit_seconds=ROUND_LIMIT)
        write_json(path,progress)
    def execute(name,stage,policy,seed=2026):
        allowance=ROUND_LIMIT-used_seconds()-20
        if allowance<100: raise TimeoutError('Insufficient remaining round budget')
        run=ROOT/'research_runs/results'/name
        if run.exists(): raise RuntimeError('Run ID already exists')
        formal=stage==100; script='run_selection.py' if formal else 'run_temporal_probe.py'
        args=['--recipe','coverage_sampling'] if formal else ['--origin',str(stage/100),'--drop-block','-1']
        command=[sys.executable,'scripts/'+script,'--run-id',name,'--seed',str(seed),'--coverage-policy',policy,*args]
        save('running_'+name); begun=time.monotonic()
        try:
            with (FOLDER/f'{name}.log').open('xb') as log: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=allowance)
        except BaseException:
            run.mkdir(exist_ok=True); status=run/'status.json'; old=json.loads(status.read_text()) if status.exists() else {}
            write_json(status,{**old,'phase':'failed_or_budget_stopped','elapsed_seconds':max(old.get('elapsed_seconds',0),time.monotonic()-begun)})
            raise
        values=read_metrics(name,formal)
        baseline=({2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}[seed] if formal else ('T000_prefix60_full' if stage==60 else 'T005_prefix80_full'))
        b=read_metrics(baseline,formal)
        row={'run_id':name,'stage':stage,'policy':policy,'seed':seed,'baseline':baseline,**values,
            'NMSE_reduction_pct':100*(1-values['NMSE']/b['NMSE']),'NMAE_reduction_pct':100*(1-values['NMAE']/b['NMAE'])}
        progress['rows'].append(row); save('completed_'+name); print(json.dumps(row),flush=True)
        return row
    try:
        first=[execute(f'T{23+i:03d}_prefix60_{policy}',60,policy) for i,policy in enumerate(POLICIES)]
        ranked=sorted([r for r in first if r['policy']!='recent_shuffle' and r['NMSE_reduction_pct']>0],key=lambda r:(r['NMSE'],POLICIES.index(r['policy'])))[:2]
        progress['shortlist']=[r['policy'] for r in ranked]
        if not ranked: save('stopped_no_internal_signal'); return
        second=[execute(f'T{27+i:03d}_prefix80_{r["policy"]}',80,r['policy']) for i,r in enumerate(ranked)]
        if any('recent' in r['policy'] for r in ranked): execute('T029_prefix80_recent_shuffle',80,'recent_shuffle')
        candidates=sorted([r for r in second if r['NMSE_reduction_pct']>0],key=lambda r:(r['NMSE'],POLICIES.index(r['policy'])))
        if not candidates: save('stopped_no_transfer_signal'); return
        formal=[execute(f'S{32+i:03d}_{r["policy"]}_s2026',100,r['policy']) for i,r in enumerate(candidates)]
        improved=[r for r in formal if r['NMSE_reduction_pct']>0]
        if not improved: save('stopped_no_formal_signal'); return
        winner=min(improved,key=lambda r:(r['NMSE'],POLICIES.index(r['policy'])))
        progress['selected_policy']=winner['policy']
        confirmations=[winner]
        for number,seed in [(34,2027),(35,2028)]: confirmations.append(execute(f'S{number:03d}_{winner["policy"]}_s{seed}',100,winner['policy'],seed))
        if 'recent' in winner['policy']: execute('S036_recent_shuffle_s2026',100,'recent_shuffle')
        bases=[read_metrics(r['baseline'],True) for r in confirmations]
        avg=lambda key:sum(r[key] for r in confirmations)/3
        mean_nmse=avg('NMSE'); mean_nmae=avg('NMAE'); base_nmse=sum(b['NMSE'] for b in bases)/3; base_nmae=sum(b['NMAE'] for b in bases)/3
        progress.update(mean_NMSE=mean_nmse,mean_NMAE=mean_nmae,mean_NMSE_reduction_pct=100*(1-mean_nmse/base_nmse),mean_NMAE_reduction_pct=100*(1-mean_nmae/base_nmae),
            adopt_default=bool(mean_nmse<=.99*base_nmse and all(r['NMSE_reduction_pct']>0 for r in confirmations) and mean_nmae<=base_nmae))
        save('completed_three_seed_exploration')
    except BaseException as error:
        progress['error']=str(error); save('failed_or_budget_stopped'); raise


if __name__=='__main__': main()
