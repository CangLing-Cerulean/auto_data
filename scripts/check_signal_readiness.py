"""Real T/CUDA and information-boundary preflight; zero API and optimizer updates."""
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from online_ts.config import GateError
from online_ts.selection.common import file_hash,read_json,write_json
from online_ts.agent_research.experience import ROOT,STUDY
from online_ts.agent_research.jobs import prepare_job
from online_ts.agent_research.memory import MEMORY,verify_memory
from online_ts.agent_research.native import run_process
from online_ts.agent_research.code_boundary import execute_bounded_job
from online_ts.agent_research.signal_study import PILOT,study_code
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.transforms import load_processor,apply_processor

def main():
    task=TaskAssets(STUDY/'tasks/metropt3_dev');inputs=task.export_inputs()
    library=read_json(MEMORY/'library.json');verify_memory(library)
    protocol=read_json(PILOT/'protocol.json');checks=read_json(PILOT/'component_checks.json')
    if checks['status']!='passed' or checks['skipped']:raise GateError('SIGNAL_COMPONENT_CHECKS')
    if protocol['protocol_document_sha256']!=file_hash(PILOT/'PROTOCOL.md') or protocol['memory_sha256']!=library['sha256']:
        raise GateError('SIGNAL_PROTOCOL_OR_MEMORY_BINDING')
    guards=task.guards|study_code()|{str(inputs):file_hash(inputs)}
    root=Path('/tmp/auto_data_signal_preflight_20261010_v3')
    if root.exists():raise GateError('SIGNAL_PREFLIGHT_NOT_REPLAYED')
    root.mkdir();hidden=root/'private_metric.json';hidden.write_text('{"fixture_only":123}',encoding='utf-8')
    asset=[{'path':str(inputs),'name':'train_inputs.npz','role':'training','sha256':file_hash(inputs)}]
    processor='import torch\ndef transform(x,state):\n    return x[:,-1:]+0.9*(x-x[:,-1:])\n'
    builder=f'''import numpy as np,json,sys,pathlib
inp=np.load(pathlib.Path(sys.argv[1])/'train_inputs.npz',allow_pickle=False);out=pathlib.Path(sys.argv[2]);n=len(inp['start_row'])
np.savez_compressed(out/'processing.npz',start_row=inp['start_row'],weight=np.ones(n),sampling_multiplier=np.ones(n))
(out/'processor.py').write_text({processor!r})
(out/'processor_state.json').write_text('{{}}')
(out/'diagnostic.json').write_text(json.dumps({{'legal_windows':n,'mean_variation':float(inp['input_variation'].mean())}}))
'''
    prepare_job(root/'good',builder,asset)
    executed=execute_bounded_job(root/'good',guards,seconds=45)
    if executed['status']!='completed':raise GateError('SIGNAL_PREFLIGHT_BUILDER: '+str(executed.get('error'))+' '+executed['stdout'][-2500:])
    validated=run_process([sys.executable,'-B','-m','online_ts.agent_research.processor_check','--folder',str(root/'good')],
        cwd=ROOT,outputs=root/'good/outputs',seconds=45,gpu=True)
    if validated['status']!='completed':raise GateError('SIGNAL_PREFLIGHT_PROCESSOR: '+validated['stdout'][-2500:])
    forbidden=f'''import pathlib,sys,json
try:
 text=pathlib.Path({str(hidden)!r}).read_text()
except PermissionError:
 text='denied'
pathlib.Path(sys.argv[2],'diagnostic.json').write_text(json.dumps({{'observed':text}}))
'''
    prepare_job(root/'outside_read',forbidden,asset)
    rejected=execute_bounded_job(root/'outside_read',guards)
    if rejected['status']!='failed' or not rejected.get('information_boundary',{}).get('violations'):
        raise GateError('SIGNAL_PREFLIGHT_HIDDEN_READ_NOT_REJECTED')
    with np.load(inputs,allow_pickle=False) as saved:
        ids=saved['start_row'][::max(1,len(saved['start_row'])//17)][:17]
        x=saved['values'][ids[:,None]+np.arange(32)].copy()
    p=read_json(task.base_root/'research_runs/protocol.json')
    spec=importlib.util.spec_from_file_location('signal_preflight_core',task.base_root/'scripts/run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    model=core.DLinear(p).cuda();program=load_processor(root/'good/outputs/processor.py',root/'good/outputs/processor_state.json')
    prediction=model(apply_processor(torch.tensor(x,device='cuda'),program));prediction.square().mean().backward()
    if prediction.shape!=(17,8,15) or sum(v.numel() for v in model.parameters())!=7920:
        raise GateError('SIGNAL_PREFLIGHT_FIXED_MODEL')
    for path,expected in guards.items():
        if file_hash(path)!=expected:raise GateError('SIGNAL_PREFLIGHT_CHANGED_ASSET')
    result={'status':'passed','component_tests':checks['tests_run'],'code':study_code(),
        'protocol_sha256':file_hash(PILOT/'protocol.json'),'base_assets':task.guards,
        'builder_and_cpu_import':'passed','processor_real_T_CPU_CUDA':'passed',
        'fixed_model_forward_backward_without_optimizer':'passed',
        'outside_file_read':'denied_and_violation_preserved_even_if_agent_catches_error',
        'security_sandbox':False,'API_calls':0,'training_calls':0,'test_access':False,
        'full_dataset_verification':'server_T_D_hashes_only_test_not_present',
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name()}}
    write_json(PILOT/'preflight.json',result);print(json.dumps({'status':'passed','checks':checks['tests_run'],'API':0,'training':0}),flush=True)

if __name__=='__main__':main()
