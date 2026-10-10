"""Preserve an exited open pilot and correct stale queue metadata before capability acceptance."""
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import tarfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.selection.common import read_json,write_json,file_hash

root=Path('/tmp/auto_data_phase4_20261009_open')
archive=Path('/tmp/phase4_open_closed_20261010.tar')
if archive.exists():raise RuntimeError('Existing archive preserved')
launch=read_json(root/'launch.json');pid=launch['pid']
if Path('/proc',str(pid),'cmdline').is_file():raise RuntimeError('Controller still exists; inspect before closing')
workers=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(root)+'/runs/' in command and any(x in command for x in
        ('online_ts.agent_research.trainer','online_ts.agent_research.processor_check','/code/agent.py')):
        worker=int(entry.name)
        try:os.killpg(worker,signal.SIGTERM) if os.getpgid(worker)==worker else os.kill(worker,signal.SIGTERM)
        except ProcessLookupError:pass
        workers.append(worker)
time.sleep(.3)
runs=[]
for folder in sorted((root/'runs').glob('*')):
    with sqlite3.connect(folder/'runtime.sqlite') as db:
        calls=db.execute('SELECT reserved_tokens,actual_tokens FROM agent_calls').fetchall()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    results=list((folder/'training').glob('*/result.json'))
    for path in results:
        value=read_json(path)
        for name,expected in value['assets'].items():
            if file_hash(path.parent/name)!=expected:raise RuntimeError('Training asset changed')
    runs.append({'run_id':folder.name,'search_completed':(folder/'search_result.json').is_file(),
        'failure':read_json(folder/'failure.json') if (folder/'failure.json').is_file() else None,
        'final_policy':read_json(folder/'final_policy.json') if (folder/'final_policy.json').is_file() else None,
        'completed_trainings':len(results),'API_calls':len(calls),
        'API_actual_known_tokens':sum(row[1] or 0 for row in calls),
        'API_unresolved_calls':sum(row[1] is None for row in calls),
        'API_unresolved_reserved_tokens':sum(row[0] for row in calls if row[1] is None)})
events={}
for path in (Path('/sys/fs/cgroup/memory.events'),Path('/sys/fs/cgroup/memory/memory.oom_control')):
    if path.is_file():events[str(path)]=path.read_text()
summary={'status':'closed_preserved_unexpected_controller_exit','timestamp_ns':time.time_ns(),
    'previous_progress':read_json(root/'progress.json'),'controller_pid':pid,'controller_exists':False,
    'remaining_workers_stopped':workers,'exit_cause':'not_established_from_stale_progress',
    'cgroup_events':events,'runs':runs,'completed_trainings':sum(x['completed_trainings'] for x in runs),
    'API_calls':sum(x['API_calls'] for x in runs),'API_actual_known_tokens':sum(x['API_actual_known_tokens'] for x in runs),
    'API_unresolved_calls':sum(x['API_unresolved_calls'] for x in runs),
    'API_unresolved_reserved_tokens':sum(x['API_unresolved_reserved_tokens'] for x in runs),
    'no_same_ID_replay':True,'formal_comparison_completed':False,'test_access':False}
write_json(root/'closure_summary.json',summary)
write_json(root/'progress.json',{'phase':'closed_incomplete','queue_active':False,'cause':'unexpected_controller_exit',
    'all_attempts_preserved':True,'capability_acceptance_next':True})
files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm')))
manifest={str(p.relative_to(root)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in files}
write_json(root/'closure_asset_manifest.json',manifest);files.append(root/'closure_asset_manifest.json')
with tarfile.open(archive,'w') as output:
    for path in files:output.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt={'archive':str(archive),'sha256':file_hash(archive),'bytes':archive.stat().st_size,'members':len(files),'summary':summary}
write_json(root/'closure_archive_receipt.json',receipt)
print(json.dumps(receipt),flush=True)
