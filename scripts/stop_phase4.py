"""Stop only the phase4 v1 processes after an observed platform compatibility fault."""
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.selection.common import write_json

root=Path('/tmp/auto_data_phase4_20261009')
launch=json.loads((root/'launch.json').read_text())
pid=launch['pid'];command=Path('/proc',str(pid),'cmdline')
stopped=[]
if command.is_file():
    text=command.read_bytes().replace(b'\0',b' ').decode()
    if 'online_ts.agent_research.phase4' not in text or os.getpgid(pid)!=pid:raise RuntimeError('Unexpected batch process identity')
    os.killpg(pid,signal.SIGTERM);stopped.append({'pid':pid,'kind':'batch'})
time.sleep(.2)
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:text=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    is_worker=(str(root)+'/runs/' in text and ('online_ts.agent_research.trainer' in text or '/code/agent.py' in text))
    if is_worker:
        worker=int(entry.name)
        try:os.killpg(worker,signal.SIGTERM) if os.getpgid(worker)==worker else os.kill(worker,signal.SIGTERM)
        except ProcessLookupError:pass
        stopped.append({'pid':worker,'kind':'research_worker'})
receipt={'status':'aborted_preserved','reason':'platform_CPU_address_space_limit_prevented_legitimate_torch_import',
         'stopped_processes':stopped,'timestamp_ns':time.time_ns(),'replay_same_ID':False,
         'scientific_effectiveness_of_failed_recipe':'untested','all_assets_preserved':True}
write_json(root/'abort.json',receipt)
write_json(root/'progress.json',{'phase':'aborted_platform_compatibility','queue_active':False,'all_assets_preserved':True})
print(json.dumps(receipt),flush=True)
