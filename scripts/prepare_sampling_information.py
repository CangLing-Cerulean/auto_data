"""Register only existing A containers, probabilities and real parent states."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261014_sampling_weight_information_r1';REMOTE='/root/auto_data_experiments/20261014_sampling_weight_information_r1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
h=read(BASE/'20261014_source_hydraulic_r1/plan.json');ms=read(BASE/'20261014_existing_sources_r1/plan.json')
prep=read(BASE/'20261014_existing_sources_r1/preparation.json');interfaces=read(BASE/'20261014_source_adapter_r1/runtime_plan.json')
tasks=[];inputs={}
for t in h['tasks']:
    task=dict(id=t['id'],source='Hydraulic',channels=8,abc=t['abc'],probabilities=t['probabilities'],group_kind='cycle_half',A_cycle_start=t['partitions']['A'][0],parents=t['parents'])
    tasks.append(task)
    paths={t['abc'],*t['probabilities'].values()}
    paths.update(p['directory']+f'/step_{k}.pt' for p in t['parents'] for k in h['steps'])
    for item in h['prediction_inputs']:
        if item['path'] in paths:inputs[item['path']]=item['sha256']
for t in ms['tasks']:
    tid=t['id'];old=ms['output_root'];parents=[dict(seed=p['seed'],directory=old+'/'+p['directory']) for p in interfaces['parents'] if p['task']==tid]
    task=dict(id=tid,source=t['dataset_id'],channels=len(t['sensors']),abc=f'{old}/prepared/{tid}_ABC.npz',
        probabilities={a:f'{old}/prepared/{tid}_{a}.npz' for a in ms['actions']},group_kind='first_flag',parents=parents)
    tasks.append(task)
    for name,digest in prep['assets_sha256'].items():
        if old+'/'+name in {task['abc'],*task['probabilities'].values()}:inputs[old+'/'+name]=digest
    for p in parents:
        for k in h['steps']:
            path=p['directory']+f'/step_{k}.pt';inputs[path]=interfaces['parent_assets_sha256'][path.removeprefix(old+'/')]
assert len(tasks)==5 and len(inputs)==115
code={name:sha(ROOT/'scripts'/name) for name in ('run_sampling_information.py','audit_sampling_information.py','launch_sampling_information.py')}
code['PROTOCOL.md']=sha(STAGE/'PROTOCOL.md')
plan=dict(stage='sampling_weight_information_r1',output_root=REMOTE,tasks=tasks,steps=h['steps'],actions=h['actions'],
    input_sha256=inputs,code_sha256=code,registered_parent_anchors=90,registered_action_pairs=360,registered_identity_controls=90,
    new_full_trainings=0,cumulative_full_trainings=373,BC_values_access=False,D_access=False,test_access=False,
    scope='training_gradient_information_audit_only_no_candidate_response_labels',numerics=dict(moment_atol=1e-10,moment_rtol=1e-10,second_atol=1e-8,second_rtol=1e-8),
    historical_source_commits=dict(H='93cc9c129ccbb4d1227b618ed95fab11e783a373',MS_adapter='26df71363c459706ee502d81cb5e3cff62cee6a7'))
with (STAGE/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps(dict(input_files=len(inputs),tasks=len(tasks),plan_sha256=sha(STAGE/'plan.json'))))
