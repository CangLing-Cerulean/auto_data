"""Close only the registered batch subtree after verified platform import failure."""
import json
import os
from pathlib import Path
import signal
import time
root=Path('/tmp/auto_data_signal_study_20261010')
launch=json.loads((root/'launch.json').read_text());pid=launch['pid']
expected='/tmp/auto_data_signal_project_20261010/scripts/run_signal_study.py'
command=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
if expected not in command:raise RuntimeError('Registered PID identity changed')
parents={}
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:
        stat=(path/'stat').read_text();parents[int(path.name)]=int(stat[stat.rfind(')')+2:].split()[1])
    except (OSError,ValueError):pass
targets={pid}
while True:
    expanded=targets|{child for child,parent in parents.items() if parent in targets}
    if expanded==targets:break
    targets=expanded
receipt={'status':'closed_for_platform_failure','reason':'NumPy percentile/quantile lazy imports denied by audit',
         'targets':sorted(targets),'source_PID':pid,'closed_ns':time.time_ns(),
         'human_runtime_strategy':False,'maintenance_interruption':True,
         'old_IDs_not_replayed':True,'all_assets_and_costs_retained':True,
         'progress_before_close':json.loads((root/'progress.json').read_text())}
(root/'maintenance_closure.json').write_text(json.dumps(receipt,indent=2)+'\n')
for target in sorted(targets,reverse=True):
    try:os.kill(target,signal.SIGTERM)
    except ProcessLookupError:pass
print(json.dumps(receipt),flush=True)
