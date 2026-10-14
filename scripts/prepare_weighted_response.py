"""Create the complete matched-weighting plan and immutable copied forecasts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT=BASE/'20261014_weighted_response_r1'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f: json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
h=read(BASE/'20261014_source_hydraulic_r1/plan.json')
old=read(BASE/'20261013_proxy_selection_r1/plan.json')
recovery=BASE/'20261014_proxy_selection_recovery_r1'
archive=read(recovery/'archive_receipt.json')['members_sha256']
labels=read(recovery/'labels.json')
predfile=BASE/'20261013_proxy_selection_r1/predictions.json'
pred=read(predfile)
aggregate='/tmp/auto_data_proxy_selection_recovery_20261014_r1/aggregate'
inputs={};jobs=[];forecasts=[]
for task in h['tasks']:
    paths={task['abc'],*task['probabilities'].values()}
    for item in h['prediction_inputs']:
        if item['path'] in paths: inputs[item['path']]=item['sha256']
    for parent in task['parents']:
        prefix=parent['directory'].removeprefix(old['input_root']+'/')+'/'
        for name,digest in old['inputs_sha256'].items():
            if name.startswith(prefix): inputs[old['input_root']+'/'+name]=digest
        for action in h['actions']:
            ident=f"WR1_{task['id']}_{action}_s{parent['seed']}"
            oldid=f"SELR1_{task['id']}_full_{action}_s{parent['seed']}"
            original=next(r for r in labels['results'] if r['id']==oldid)
            olddir=aggregate+'/runs/'+oldid
            for name,digest in archive.items():
                if name.startswith('runs/'+oldid+'/'): inputs[aggregate+'/'+name]=digest
            jobs.append(dict(id=ident,task=task['id'],action=action,seed=parent['seed'],parent=parent['directory'],sampling_directory=olddir))
            record=next(r for r in pred['records'] if (r['task'],r['seed'],r['action'])==(task['id'],parent['seed'],action))
            m1=record['methods']['M1']
            forecasts.append(dict(id=ident,task=task['id'],seed=parent['seed'],action=action,
                M1_gain_percent=m1['gain_percent'],M1_selected_step=m1['selected_step'],
                source_forecast_key=[task['id'],action,parent['seed']],W_training_started=False))
inputs[old['output_root']+'/predictions.json']=sha(predfile)
inputs[aggregate+'/labels.json']=sha(recovery/'labels.json')
inputs[h['original_core_path']]=h['code_sha256']['run_research.py']
assert sha(ROOT/'scripts/run_research.py')==h['code_sha256']['run_research.py']
names=['run_weighted_response.py','audit_weighted_response.py','launch_weighted_response.py','run_research.py']
code={n:sha(ROOT/'scripts'/n) for n in names};code['PROTOCOL.md']=sha(OUT/'PROTOCOL.md')
plan=dict(stage='weighted_response_r1',output_root='/root/auto_data_experiments/20261014_weighted_response_r1',
    dataset_id='uci_hydraulic_v1',manifest_sha256=h['manifest_sha256'],
    tasks=h['tasks'],seeds=h['seeds'],actions=h['actions'],steps=h['steps'],training=h['training'],jobs=jobs,
    input_sha256=inputs,code_sha256=code,old_predictions=old['output_root']+'/predictions.json',
    old_labels=aggregate+'/labels.json',old_sampling_outcomes_known=True,new_weighting_outcomes_known=False,
    planned_new_full_trainings=36,cumulative_before=373,expected_cumulative_after=409,
    paired_gap_band_pp=.25,adoption_threshold_percent=1.,D_access=False,test_access=False,
    witness_steps=[1,501],preflight_min_free_bytes=1073741824,runtime_min_free_bytes=536870912)
assert len(jobs)==len({j['id'] for j in jobs})==36
write(OUT/'plan.json',plan)
write(OUT/'predictions.json',dict(status='frozen_before_W_training',plan_sha256=sha(OUT/'plan.json'),
    old_predictions_sha256=sha(predfile),records=forecasts,old_sampling_outcomes_known=True,
    new_weighting_labels_observed=False,interpretation='canonical_same_mu_M1_forecast_reuse_not_new_proxy_rollouts'))
write(OUT/'prediction_lock.json',dict(plan_sha256=sha(OUT/'plan.json'),predictions_sha256=sha(OUT/'predictions.json')))
print(json.dumps(dict(jobs=len(jobs),input_files=len(inputs),plan_sha256=sha(OUT/'plan.json'))))
