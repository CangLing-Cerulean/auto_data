"""Reproduce completed diagnostic facts and audit the supplied processing recipe on real T."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

project=Path('/tmp/auto_data_capability_project_20261010_v3')
root=Path('/tmp/auto_data_capability_acceptance_20261010_v3/known_loop')
sys.path.insert(0,str(project))
from online_ts.agent_research.jobs import prepare_job
from online_ts.agent_research.native import execute_job,run_process
from online_ts.agent_research.runtime import code_identity
from online_ts.agent_research.task import TaskAssets
from online_ts.selection.common import read_json,write_json,file_hash

def processor_worker():
    # Executed in a separate credential-free worker; only T history is sampled.
    import numpy as np
    import torch
    from online_ts.agent_research.transforms import load_processor,apply_processor
    task=TaskAssets(project/'research_runs/agent_research/tasks/metropt3_dev')
    recipe=read_json(project/'research_runs/direct_processing/20261009/processing_recipe.json')
    with np.load(task.export_inputs(),allow_pickle=False) as inputs:
        starts=inputs['start_row'][np.linspace(0,len(inputs['start_row'])-1,128,dtype=np.int64)]
        values=inputs['values'][starts[:,None]+np.arange(32)[None,:]].copy()
    batch=torch.from_numpy(values)
    expected=batch.clone();mapping=recipe['mapping'];changed=recipe['changed_channels'];alpha=recipe['alpha']
    for channel in changed:
        source=mapping[channel]
        expected[:,:,channel]=batch[:,-1:,channel]+alpha*(batch[:,:,channel]-batch[:,-1:,channel])+(1-alpha)*(batch[:,:,source]-batch[:,-1:,source])
    outcomes=[]
    for folder in sorted((root/'processing').glob('*')):
        code=folder/'outputs/processor.py';state=folder/'outputs/processor_state.json'
        if not code.is_file() or not state.is_file():continue
        program=load_processor(code,state);before=batch.clone();actual=apply_processor(batch,program)
        difference=(actual-expected).abs()
        outcomes.append({'candidate_id':folder.name,'code_sha256':file_hash(code),'state_sha256':file_hash(state),
            'T_real_windows':128,'max_absolute_error_from_registered_recipe':float(difference.max()),
            'matches_registered_recipe_on_sample':bool(torch.allclose(actual,expected,rtol=1e-5,atol=1e-6)),
            'input_unmodified':bool(torch.equal(before,batch)),'sample_scope':'128_deterministic_real_T_windows_no_new_sensor_records',
            'not_exhaustive_equivalence_proof':True})
    write_json(root/'audit/processor_worker/processor_comparison.json',{'candidates':outcomes,'new_training_calls':0,'test_access':False})

def main():
    os.environ.pop('DEEPSEEK_API_KEY',None)
    if not any((root/name).is_file() for name in ('result_summary.json','interruption_summary.json')):raise RuntimeError('Loop not finished; do not audit during decisions')
    target=root/'audit'
    if target.exists():raise RuntimeError('Existing audit preserved; no replay')
    target.mkdir()
    task=TaskAssets(project/'research_runs/agent_research/tasks/metropt3_dev')
    inputs=task.export_inputs();guards=task.guards|code_identity()|{str(inputs):file_hash(inputs)}
    with sqlite3.connect(f'file:{root / "runtime.sqlite"}?mode=ro',uri=True) as db:
        diagnostics=[json.loads(row[0]) for row in db.execute("SELECT payload FROM agent_runtime_records WHERE kind='diagnostic_job'")]
    outcomes=[]
    for index,record in enumerate(diagnostics):
        if record['status']!='completed':continue
        decision=record['record_id'].removeprefix('diagnostic:')
        original=root/'diagnostics'/decision
        output=target/('diagnostic_reproduction_'+str(index))
        prepare_job(output,(original/'code/agent.py').read_text(),[{'path':str(inputs),'name':'train_inputs.npz','role':'training','sha256':file_hash(inputs)}])
        execution=execute_job(output,guards,seconds=45)
        reproduced=read_json(output/'outputs/diagnostic.json') if execution['status']=='completed' and (output/'outputs/diagnostic.json').is_file() else None
        outcomes.append({'record_id':record['record_id'],'status':execution['status'],
            'facts_reproduced_exactly':reproduced==record['facts'],'original_facts':record['facts'],
            'reproduced_facts':reproduced,'elapsed_seconds':execution['elapsed_seconds'],
            'scope':'same_code_reproduction_not_independent_mechanism_verification'})
    worker_output=target/'processor_worker';worker_output.mkdir()
    execution=run_process([sys.executable,'-B',str(Path(__file__).resolve()),'--processor-worker'],
        cwd=project,outputs=worker_output,seconds=90,gpu=False)
    write_json(target/'processor_comparison_execution.json',execution)
    value={'status':'completed','diagnostics':outcomes,'processor_comparison_execution':execution['status'],
        'processor_comparison':read_json(worker_output/'processor_comparison.json') if (worker_output/'processor_comparison.json').is_file() else None,
        'new_training_calls':0,'API_calls':0,'test_access':False,'model_selection_feedback':False}
    write_json(target/'summary.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--processor-worker',action='store_true')
    if parser.parse_args().processor_worker:processor_worker()
    else:main()
