"""At most five train-internal runs under the shared exploration deadline."""
import json
import subprocess
import sys
import time
from run_research import ROOT,write_json
from coverage_sampling import used_seconds,ROUND_LIMIT
FOLDER=ROOT/'research_runs/learning_sampling'


def main():
    path=FOLDER/'progress.json'
    if path.exists(): raise RuntimeError('Batch exists; inspect before resuming')
    progress={'phase':'starting','rows':[],'winner':None}
    def save(phase):
        progress.update(phase=phase,budget_used_seconds=used_seconds()); write_json(path,progress)
    def run(name,origin,policy,reference=None):
        allowance=ROUND_LIMIT-used_seconds()-20
        if allowance<100: raise TimeoutError('Not enough exploration budget for next run')
        folder=ROOT/'research_runs/results'/name
        if folder.exists(): raise RuntimeError('ID already exists')
        args=[sys.executable,'scripts/run_learning_probe.py','--run-id',name,'--origin',str(origin),'--policy',policy]
        if reference: args+=['--reference',reference]
        save('running_'+name); begun=time.monotonic()
        try:
            with (FOLDER/f'{name}.log').open('xb') as log: subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=allowance)
        except BaseException:
            folder.mkdir(exist_ok=True); status=folder/'status.json'; old=json.loads(status.read_text()) if status.exists() else {}
            write_json(status,{**old,'phase':'failed_or_budget_stopped','elapsed_seconds':max(old.get('elapsed_seconds',0),time.monotonic()-begun)}); raise
        value=json.loads((folder/'result.json').read_text())['future']['groups']['all']
        base='T000_prefix60_full' if origin==.6 else 'T005_prefix80_full'; b=json.loads((ROOT/'research_runs/results'/base/'result.json').read_text())['future']['groups']['all']
        row={'run_id':name,'origin':origin,'policy':policy,'reference':reference,'baseline':base,**value,'NMSE_reduction_pct':100*(1-value['NMSE']/b['NMSE']),'NMAE_reduction_pct':100*(1-value['NMAE']/b['NMAE'])}
        progress['rows'].append(row); save('completed_'+name); print(json.dumps(row),flush=True); return row
    try:
        a=run('T030_prefix60_learning_progress',.6,'progress'); b=run('T031_prefix60_learning_hard',.6,'hard')
        winner=min([a,b],key=lambda r:(r['NMSE'],0 if r['policy']=='progress' else 1)); progress['winner']=winner['policy']
        control=run('T032_prefix60_learning_shuffle',.6,'shuffle',winner['run_id'])
        if winner['NMSE_reduction_pct']<=0: save('stopped_no_prefix60_signal'); return
        future=run(f'T033_prefix80_learning_{winner["policy"]}',.8,winner['policy'])
        control80=run('T034_prefix80_learning_shuffle',.8,'shuffle',future['run_id'])
        progress['internal_signal_confirmed']=winner['NMSE']<control['NMSE'] and future['NMSE_reduction_pct']>0 and future['NMSE']<control80['NMSE']
        save('completed_two_origins')
    except BaseException as error:
        progress['error']=str(error); save('failed_or_budget_stopped'); raise


if __name__=='__main__': main()
