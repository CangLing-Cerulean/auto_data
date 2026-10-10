"""Stop only the registered known-loop process after an observed frozen-scope violation."""
import ast
import json
import os
from pathlib import Path
import signal
import time

root=Path('/tmp/auto_data_capability_acceptance_20261010_v3')
loop=root/'known_loop'
source=loop/'processing/src_half_rep_20261010f/code/agent.py'
tree=ast.parse(source.read_text())
if not any(isinstance(node,ast.Import) and any(alias.name=='torch.nn' for alias in node.names) for node in ast.walk(tree)):
    raise RuntimeError('Expected scope violation evidence absent')
launch=json.loads((root/'loop_launch.json').read_text());pid=launch['pid']
stopped=[]
controller=Path('/proc',str(pid),'cmdline')
if controller.is_file():
    command=controller.read_bytes().replace(b'\0',b' ').decode()
    if '/auto_data_capability_project_20261010_v3/scripts/run_capability.py --stage loop' not in command:
        raise RuntimeError('PID no longer belongs to the registered loop')
    os.killpg(pid,signal.SIGTERM) if os.getpgid(pid)==pid else os.kill(pid,signal.SIGTERM)
    stopped.append(pid)
time.sleep(.5)
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(loop)+'/' in command and any(token in command for token in ('/code/agent.py','online_ts.agent_research.trainer','online_ts.agent_research.processor_check')):
        worker=int(entry.name)
        try:os.killpg(worker,signal.SIGTERM) if os.getpgid(worker)==worker else os.kill(worker,signal.SIGTERM)
        except ProcessLookupError:continue
        stopped.append(worker)
target=loop/'interruption_summary.json'
if target.exists():raise RuntimeError('Existing interruption receipt preserved')
value={'status':'interrupted_scope_violation','reason':'generated_processing_program_created_and_attempted_to_train_new_linear_forecast_model',
    'source_evidence':str(source),'previous_result_summary_exists':(loop/'result_summary.json').is_file(),
    'stopped_pids':stopped,'timestamp_ns':time.time_ns(),'counts_as_autonomous_completion':False,
    'scientific_hypothesis_refuted':False,'cumulative_budget_limit_triggered':False,
    'human_intervention':'primary_assistant_stopped_protocol_violation_no_candidate_code_or_strategy_injected',
    'same_ID_replay':False,'all_attempts_and_costs_preserved':True}
target.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value),flush=True)
