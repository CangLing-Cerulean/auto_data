"""Execute only the preregistered gated source-selection experiments."""
import json
import subprocess
import sys
from run_research import ROOT, write_json


def execute(run_id,script,args):
    path=ROOT/'research_runs/results'/run_id/'result.json'
    if not path.exists():
        with (ROOT/'research_runs/representation'/f'{run_id}.log').open('a') as log:
            subprocess.run([sys.executable,f'scripts/{script}','--run-id',run_id,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    return json.loads(path.read_text())


def main():
    rows=[]
    for name,origin,baseline in [('T010_prefix60_sources','.6','T000_prefix60_full'),('T011_prefix80_sources','.8','T005_prefix80_full')]:
        result=execute(name,'run_temporal_probe.py',['--origin',origin,'--drop-block','-1','--source-map'])
        ref=json.loads((ROOT/'research_runs/results'/baseline/'result.json').read_text())
        a=result['future']['groups']['all']; b=ref['future']['groups']['all']
        rows.append({'run_id':name,'baseline':baseline,'NMSE':a['NMSE'],'NMAE':a['NMAE'],'NMSE_reduction_pct':100*(1-a['NMSE']/b['NMSE']),'NMAE_not_worse':a['NMAE']<=b['NMAE']})
        write_json(ROOT/'research_runs/representation/leading_progress.json',{'rows':rows,'phase':'internal'})
        print(json.dumps(rows[-1]),flush=True)
    if not(all(r['NMSE_reduction_pct']>=1 for r in rows) and rows[-1]['NMAE_not_worse']):
        write_json(ROOT/'research_runs/representation/leading_progress.json',{'rows':rows,'phase':'stopped_internal_gate'}); return
    for number,seed,base in [(1,2026,'S000_uniform_s2026'),(2,2027,'S005_uniform_s2027'),(3,2028,'S010_uniform_s2028')]:
        name=f'R{number:03d}_sources_s{seed}'
        result=execute(name,'run_selection.py',['--recipe','leading_sources','--seed',str(seed)])
        ref=json.loads((ROOT/'research_runs/results'/base/'result.json').read_text())
        a=result['groups']['all']; b=ref['groups']['all']
        row={'run_id':name,'baseline':base,'NMSE':a['NMSE'],'NMAE':a['NMAE'],'NMSE_reduction_pct':100*(1-a['NMSE']/b['NMSE']),'NMAE_not_worse':a['NMAE']<=b['NMAE']}
        rows.append(row); print(json.dumps(row),flush=True)
        write_json(ROOT/'research_runs/representation/leading_progress.json',{'rows':rows,'phase':'formal'})
        if number==1 and (row['NMSE_reduction_pct']<1 or not row['NMAE_not_worse']):
            write_json(ROOT/'research_runs/representation/leading_progress.json',{'rows':rows,'phase':'stopped_formal_gate'}); return
    write_json(ROOT/'research_runs/representation/leading_progress.json',{'rows':rows,'phase':'completed_three_seeds'})


if __name__=='__main__': main()
