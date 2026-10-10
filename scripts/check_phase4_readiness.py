"""Server-only preflight on real T; no API, optimizer steps or test evaluation."""
import importlib.util
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
from online_ts.agent_research.native import execute_job,run_process
from online_ts.agent_research.phase4 import PILOT,OUTPUT,code_identity,random_code
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.transforms import processor_metadata,load_processor,apply_processor


def main():
    task=TaskAssets(STUDY/'tasks/metropt3_dev');inputs=task.export_inputs()
    library=read_json(MEMORY/'library.json');verify_memory(library)
    protocol=read_json(PILOT/'protocol.json')
    if protocol['protocol_document_sha256']!=file_hash(PILOT/'PROTOCOL.md') or protocol['known_case_sha256']!=file_hash(PILOT/'registered_case.json'):
        raise GateError('PREFLIGHT_PROTOCOL_BINDING')
    for index in range(3):compile(random_code(index,20261101+701+index),'reference.py','exec')
    folder=Path('/tmp/auto_data_phase4_preflight_20261009_open')
    case=read_json(ROOT/'research_runs/direct_processing/20261009/processing_recipe.json')
    source='import torch\ndef transform(x,state):\n    mapping=torch.tensor(state["mapping"],device=x.device)\n    mixed=x[:,-1:]+0.5*(x-x[:,-1:])+0.5*(x[:,:,mapping]-x[:,-1:,mapping])\n    changed=(mapping!=torch.arange(15,device=x.device))[None,None,:]\n    return torch.where(changed,mixed,x)\n'
    state_literal=repr({'mapping':case['mapping']})
    code=f'''import torch,numpy as np,json,pathlib,sys
x=np.load(pathlib.Path(sys.argv[1])/"train_inputs.npz",allow_pickle=False);out=pathlib.Path(sys.argv[2]);n=len(x["start_row"])
np.savez_compressed(out/"processing.npz",start_row=x["start_row"],weight=np.ones(n),sampling_multiplier=np.ones(n))
(out/"processor.py").write_text({source!r})
(out/"processor_state.json").write_text(json.dumps({state_literal}))
'''
    guards=task.guards|code_identity()|{str(inputs):file_hash(inputs)}
    prepare_job(folder,code,[{'path':str(inputs),'name':'train_inputs.npz','role':'training','sha256':file_hash(inputs)}])
    execution=execute_job(folder,guards)
    if execution['status']!='completed':raise GateError('PREFLIGHT_CODE_EXECUTION')
    processor_metadata(folder/'outputs/processor.py',folder/'outputs/processor_state.json')
    checked=run_process([sys.executable,'-B','-m','online_ts.agent_research.processor_check','--folder',str(folder)],
        cwd=ROOT,outputs=folder/'outputs',seconds=45,gpu=True)
    write_json(folder/'processor_execution.json',checked)
    if checked['status']!='completed':raise GateError('PREFLIGHT_INPUT_PROCESSOR: '+checked['stdout'][-1500:])
    with np.load(inputs,allow_pickle=False) as saved:
        starts=saved['start_row'][::max(1,len(saved['start_row'])//17)][:17]
        real=saved['values'][starts[:,None]+np.arange(32)].copy()
    p=read_json(task.base_root/'research_runs/protocol.json')
    spec=importlib.util.spec_from_file_location('preflight_original_core',task.base_root/'scripts/run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    model=core.DLinear(p).cuda();program=load_processor(folder/'outputs/processor.py',folder/'outputs/processor_state.json')
    prediction=model(apply_processor(torch.tensor(real,device='cuda'),program))
    prediction.square().mean().backward()  # Checks gradient flow only; no optimizer or parameter update.
    if prediction.shape!=(17,8,15) or sum(v.numel() for v in model.parameters())!=7920 or any(not torch.isfinite(v.grad).all() for v in model.parameters()):
        raise GateError('PREFLIGHT_FIXED_MODEL_CONTRACT')
    for path,expected in guards.items():
        if file_hash(path)!=expected:raise GateError('PREFLIGHT_SOURCE_CHANGED')
    checks=read_json(PILOT/'component_checks.json')
    if checks['status']!='passed' or checks['skipped']:raise GateError('PREFLIGHT_COMPONENT_CHECKS')
    result={'status':'passed','component_tests':checks['tests_run'],'processor_program_execution':'passed',
        'real_T_history_validation':read_json(folder/'outputs/processor_validation.json'),
        'fixed_DLinear_parameters':7920,'forward_and_gradient_only':'passed_no_optimizer_steps',
        'code':code_identity(),'protocol_sha256':file_hash(PILOT/'protocol.json'),
        'cpu_fit_torch_import':'passed_under_8GiB_virtual_address_limit',
        'API_calls':0,'training_calls':0,'test_access':False,'fixture_scope':'engineering_known_prior_not_agent_candidate',
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'device':torch.cuda.get_device_name()}}
    write_json(PILOT/'preflight.json',result)
    print('Phase4 preflight passed: real T processor, CPU/CUDA, fixed-model gradient, 0 API/training/test',flush=True)


if __name__=='__main__':main()
