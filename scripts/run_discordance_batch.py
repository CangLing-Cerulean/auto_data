"""Bounded two-origin, targeted versus shuffled input-discordance study."""
import json
import subprocess
import sys
from run_research import ROOT,write_json


def metrics(run_id):
    return json.loads((ROOT/'research_runs/results'/run_id/'result.json').read_text())['future']['groups']['all']


def main():
    rows=[]; folds=[]
    for origin,number,source,original in [(.6,12,'T010_prefix60_sources','T000_prefix60_full'),(.8,14,'T011_prefix80_sources','T005_prefix80_full')]:
        outcomes=[]
        for offset,policy in enumerate(['targeted','shuffle']):
            name=f'T{number+offset:03d}_prefix{int(origin*100)}_discordance'+('_shuffle' if policy=='shuffle' else '')
            result=ROOT/'research_runs/results'/name/'result.json'
            if not result.exists():
                with (ROOT/'research_runs/selection'/f'{name}.log').open('a') as log:
                    subprocess.run([sys.executable,'scripts/run_temporal_probe.py','--run-id',name,'--origin',str(origin),'--drop-block','-1','--source-map','--discordance',policy],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            values=metrics(name); outcomes.append(values)
            row={'run_id':name,'policy':policy,'origin':origin,**values}; rows.append(row)
            print(json.dumps(row),flush=True)
            write_json(ROOT/'research_runs/selection/discordance_progress.json',{'phase':'running','rows':rows,'folds':folds})
        a,b=outcomes; unweighted=metrics(source); raw=metrics(original)
        checks={'NMAE_reduction_vs_source_at_least_1pct':a['NMAE']<=.99*unweighted['NMAE'],
            'NMSE_vs_source_within_half_pct':a['NMSE']<=1.005*unweighted['NMSE'],
            'NMSE_reduction_vs_original_at_least_1pct':a['NMSE']<=.99*raw['NMSE'],
            'NMAE_vs_original_not_worse':a['NMAE']<=raw['NMAE'],
            'NMAE_better_than_shuffle':a['NMAE']<b['NMAE'],'NMSE_not_worse_than_shuffle':a['NMSE']<=b['NMSE']}
        folds.append({'origin':origin,'source_baseline':source,'original_baseline':original,'checks':checks,
            'NMAE_reduction_vs_source_pct':100*(1-a['NMAE']/unweighted['NMAE']),
            'NMSE_reduction_vs_source_pct':100*(1-a['NMSE']/unweighted['NMSE']),
            'gate_passed':all(checks.values())})
        phase='completed_internal' if origin==.8 else 'first_gate_passed'
        if not folds[-1]['gate_passed']: phase='stopped_internal_gate'
        write_json(ROOT/'research_runs/selection/discordance_progress.json',{'phase':phase,'rows':rows,'folds':folds})
        print(json.dumps(folds[-1]),flush=True)
        if not folds[-1]['gate_passed']: return


if __name__=='__main__': main()
