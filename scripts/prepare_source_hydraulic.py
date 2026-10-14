"""Freeze explicit existing inputs; never reads sensor arrays or candidate outcomes."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261014_source_hydraulic_r1'
REMOTE='/root/auto_data_experiments/20261014_source_hydraulic_r1'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(BASE/'20261013_proxy_selection_r1/plan.json');prep=read(BASE/'20261013_proxy_selection_r1/preparation.json')
engineering=read(BASE/'20261014_source_adapter_r1/runtime_plan.json')
code={name:sha(ROOT/'scripts'/name) for name in ('run_source_hydraulic.py','audit_source_hydraulic.py','launch_source_hydraulic.py','source_quadratic.py','source_dlinear_adapter.py','response_proxy.py')}
for name in ('source_quadratic.py','source_dlinear_adapter.py'):assert code[name]==engineering['code_sha256'][name]
code['run_research.py']=old['core_sha256'];code['PROTOCOL.md']=sha(STAGE/'PROTOCOL.md')
tasks=[];inputs=[]
for task in old['tasks']:
    tid=task['id'];abc=f'prepared/{tid}_ABC.npz'
    item=dict(task,abc=old['input_root']+'/'+abc,probabilities={},parents=[])
    inputs.append(dict(path=item['abc'],sha256=old['inputs_sha256'][abc]))
    for action in old['actions']:
        relative=f'prepared/{tid}_{action}.npz';path=old['output_root']+'/'+relative
        item['probabilities'][action]=path;inputs.append(dict(path=path,sha256=prep['assets_sha256'][relative]))
    for seed in old['seeds']:
        relative=f'runs/DEV_{tid}_full_PARENT_s{seed}'
        item['parents'].append(dict(seed=seed,directory=old['input_root']+'/'+relative))
        for name in ['result.json','timing.json']+[f'step_{step}.pt' for step in old['grid']]:
            key=relative+'/'+name;inputs.append(dict(path=old['input_root']+'/'+key,sha256=old['inputs_sha256'][key]))
    tasks.append(item)
truth=BASE/'20261014_proxy_selection_recovery_r1/final_audit.json'
assert sha(truth)=='2fd17fcb6b5582450d2658328ba8071e11f8565c6aa9e257a37d73a3b4cc3195'
plan=dict(stage='source_hydraulic_r1',retrospective=True,old_outcomes_already_known=True,new_full_trainings=0,cumulative_full_trainings=373,
    output_root=REMOTE,code_sha256=code,prediction_inputs=inputs,analysis_inputs=[dict(path=REMOTE+'/input/old_final_audit.json',sha256=sha(truth))],
    tasks=tasks,actions=old['actions'],seeds=old['seeds'],steps=old['grid'],training=old['training'],
    historical_preparation_seconds=old['prior_preparation_seconds']+prep['seconds'],
    source_adapter_commit='26df71363c459706ee502d81cb5e3cff62cee6a7',source_adapter_result_commit='8b05ffc3a5cc2977716d2fe84ceb687ab487e783',
    original_plan_sha256=sha(BASE/'20261013_proxy_selection_r1/plan.json'),original_preparation_sha256=sha(BASE/'20261013_proxy_selection_r1/preparation.json'),
    manifest_sha256=old['manifest_sha256'],reference_atol=1e-7,reference_rtol=1e-8,D_access=False,test_access=False,
    original_core_path=old['core'],registered_predictions=[f'{t["id"]}_{a}_{s}' for t in tasks for s in old['seeds'] for a in old['actions']])
with (STAGE/'plan.json').open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps(dict(predictions=len(plan['registered_predictions']),prediction_inputs=len(inputs),plan_sha256=sha(STAGE/'plan.json'))))
