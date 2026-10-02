"""Bounded execution of the user-approved frozen-cluster deletion study."""
import json
import subprocess
import sys
import time
from run_research import ROOT,write_json
from cluster_selection import FOLDER,ROUND_LIMIT,used_seconds,create_mask


def result(run_id,formal=False):
    r=json.loads((ROOT/'research_runs/results'/run_id/'result.json').read_text())
    return (r if formal else r['future'])['groups']['all']


def baseline_gate(a,b): return a['NMSE']<=.99*b['NMSE'] and a['NMAE']<=b['NMAE']
def control_gate(a,b): return a['NMSE']<b['NMSE'] and a['NMAE']<=b['NMAE']


def main():
    progress_path=FOLDER/'progress.json'
    if progress_path.exists(): raise RuntimeError('Existing batch: inspect state and resume individual unfinished run only')
    progress={'phase':'preparing','rows':[],'selected_group':None,'stages':[]}
    def save(phase):
        progress.update(phase=phase,budget_used_seconds=used_seconds(),round_limit_seconds=ROUND_LIMIT)
        write_json(progress_path,progress)
    def execute(run_id,stage,group,random_control=False,seed=2026):
        allowance=ROUND_LIMIT-used_seconds()-60
        if allowance<120: raise TimeoutError('Insufficient round budget for next training plus audit reserve')
        mask=create_mask(stage,group,random_control); formal=stage==100
        script='run_selection.py' if formal else 'run_temporal_probe.py'
        args=['--recipe','frozen_mask'] if formal else ['--origin',str(stage/100),'--drop-block','-1']
        command=[sys.executable,'scripts/'+script,'--run-id',run_id,'--seed',str(seed),'--retention-mask',str(mask),*args]
        if (ROOT/'research_runs/results'/run_id).exists(): raise RuntimeError('Run ID already exists')
        save('running_'+run_id); begun=time.monotonic()
        try:
            with (FOLDER/f'{run_id}.log').open('xb') as log: subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=allowance)
        except BaseException:
            path=ROOT/'research_runs/results'/run_id/'status.json'
            path.parent.mkdir(exist_ok=True)
            old=json.loads(path.read_text()) if path.exists() else {}
            write_json(path,{**old,'phase':'failed_or_budget_stopped','elapsed_seconds':max(old.get('elapsed_seconds',0),time.monotonic()-begun)})
            raise
        values=result(run_id,formal)
        row={'run_id':run_id,'stage':stage,'group':group,'random_control':random_control,'seed':seed,'mask':mask.name,**values}
        progress['rows'].append(row); save('completed_'+run_id); print(json.dumps(row),flush=True)
        return values
    try:
        meta=json.loads((FOLDER/'fit.json').read_text()); base=result('T000_prefix60_full')
        candidates=[]
        for group in meta['eligible_groups']:
            a=execute(f'T{16+group:03d}_prefix60_cluster{group}',60,group)
            if baseline_gate(a,base): candidates.append((a['NMSE'],group,a))
        if not candidates: save('stopped_no_prefix60_candidate'); return
        _,group,a=min(candidates,key=lambda row:(row[0],row[1])); progress['selected_group']=group
        b=execute('T020_prefix60_cluster_random',60,group,True)
        progress['stages'].append({'stage':60,'baseline_gate':True,'control_gate':control_gate(a,b)})
        if not control_gate(a,b): save('stopped_prefix60_control'); return
        try: create_mask(80,group)
        except ValueError as error:
            progress['reason']=str(error); save('stopped_prefix80_ineligible'); return
        a=execute('T021_prefix80_cluster_target',80,group)
        b=execute('T022_prefix80_cluster_random',80,group,True)
        passed=baseline_gate(a,result('T005_prefix80_full')) and control_gate(a,b)
        progress['stages'].append({'stage':80,'baseline_gate':baseline_gate(a,result('T005_prefix80_full')),'control_gate':control_gate(a,b)})
        if not passed: save('stopped_prefix80_gate'); return
        try: create_mask(100,group)
        except ValueError as error:
            progress['reason']=str(error); save('stopped_fulltrain_ineligible'); return
        formal=[]
        for number,seed,base_id in [(26,2026,'S000_uniform_s2026'),(28,2027,'S005_uniform_s2027'),(30,2028,'S010_uniform_s2028')]:
            a=execute(f'S{number:03d}_cluster_target_s{seed}',100,group,False,seed)
            b=execute(f'S{number+1:03d}_cluster_random_s{seed}',100,group,True,seed)
            base=result(base_id,True); formal.append((a,b,base))
            if seed==2026 and not (baseline_gate(a,base) and control_gate(a,b)):
                save('stopped_formal_pilot'); return
        avg=lambda k,j:sum(v[j][k] for v in formal)/3
        progress['development_gate_passed']=bool(avg('NMSE',0)<=.99*avg('NMSE',2) and avg('NMAE',0)<=avg('NMAE',2) and all(a['NMSE']<c['NMSE'] for a,b,c in formal) and avg('NMSE',0)<avg('NMSE',1) and avg('NMAE',0)<=avg('NMAE',1))
        save('completed_formal_confirmation')
    except BaseException as error:
        progress['error']=str(error); save('failed_or_budget_stopped'); raise


if __name__=='__main__': main()
