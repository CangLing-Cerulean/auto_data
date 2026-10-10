"""Stop and preserve only the v2 batch after the user's budget-policy change."""
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

root=Path('/tmp/auto_data_phase4_20261009_v2')
archive=Path('/tmp/phase4_v2_superseded_20261009.tar')
if archive.exists():raise RuntimeError('Preserve existing archive; do not repeat this transition')
launch=read_json(root/'launch.json');pid=launch['pid'];stopped=[]
command=Path('/proc',str(pid),'cmdline')
if command.is_file():
    text=command.read_bytes().replace(b'\0',b' ').decode()
    if 'online_ts.agent_research.phase4' not in text or os.getpgid(pid)!=pid:
        raise RuntimeError('Unexpected controller identity')
    os.killpg(pid,signal.SIGTERM);stopped.append({'pid':pid,'kind':'controller'})
time.sleep(.3)
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:text=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(root)+'/runs/' in text and any(name in text for name in
        ('online_ts.agent_research.trainer','online_ts.agent_research.processor_check','/code/agent.py')):
        worker=int(entry.name)
        try:os.killpg(worker,signal.SIGTERM) if os.getpgid(worker)==worker else os.kill(worker,signal.SIGTERM)
        except ProcessLookupError:pass
        stopped.append({'pid':worker,'kind':'worker'})
time.sleep(.5)
for row in stopped:
    path=Path('/proc',str(row['pid']),'stat')
    if path.is_file() and path.read_text().split(') ',1)[1][0]!='Z':
        raise RuntimeError('Target process has not stopped; preserve assets before deployment')
runs=[]
for folder in sorted((root/'runs').glob('*')):
    with sqlite3.connect(folder/'runtime.sqlite') as db:
        calls=db.execute('SELECT reserved_tokens,actual_tokens FROM agent_calls').fetchall()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    results=list((folder/'training').glob('*/result.json'))
    for path in results:
        result=read_json(path)
        for asset,expected in result['assets'].items():
            if file_hash(path.parent/asset)!=expected:raise RuntimeError('Training asset changed')
        for asset,expected in result['identity']['protected_assets'].items():
            if file_hash(asset)!=expected:raise RuntimeError('Protected source changed before superseding')
    runs.append({'run_id':folder.name,'search_completed':(folder/'search_result.json').is_file(),
        'failure':read_json(folder/'failure.json') if (folder/'failure.json').is_file() else None,
        'final_policy':read_json(folder/'final_policy.json') if (folder/'final_policy.json').is_file() else None,
        'completed_trainings':len(results),'training_attempts':len(list((folder/'training').glob('*/job.json'))),
        'API_calls':len(calls),'API_actual_known_tokens':sum(row[1] or 0 for row in calls),
        'API_unresolved_calls':sum(row[1] is None for row in calls),
        'API_unresolved_reserved_tokens':sum(row[0] for row in calls if row[1] is None)})
summary={'status':'superseded_preserved','reason':'user_removed_exploration_caps_20261009',
    'timestamp_ns':time.time_ns(),'stopped_processes':stopped,'runs':runs,
    'started_searches':len(runs),'registered_searches':12,'not_started':12-len(runs),
    'completed_trainings':sum(row['completed_trainings'] for row in runs),
    'API_calls':sum(row['API_calls'] for row in runs),
    'API_actual_known_tokens':sum(row['API_actual_known_tokens'] for row in runs),
    'API_unresolved_calls':sum(row['API_unresolved_calls'] for row in runs),
    'API_unresolved_reserved_tokens':sum(row['API_unresolved_reserved_tokens'] for row in runs),
    'same_ID_replay':False,'old_failures_and_costs_retained':True,'test_access':False,
    'scope':'incomplete_bounded_pilot_not_pooled_with_new_unbounded_runs'}
write_json(root/'superseded_summary.json',summary)
write_json(root/'progress.json',{'phase':'superseded_by_user_budget_change','queue_active':False,
    'preserve_all_run_assets':True,'previous_progress_available_in_local_live_snapshots':True})
files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm')))
manifest={str(p.relative_to(root)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in files}
write_json(root/'superseded_asset_manifest.json',manifest);files.append(root/'superseded_asset_manifest.json')
with tarfile.open(archive,'w') as output:
    for path in files:output.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt={'archive':str(archive),'sha256':file_hash(archive),'bytes':archive.stat().st_size,
    'members':len(files),'summary':summary}
write_json(root/'superseded_archive_receipt.json',receipt)
print(json.dumps(receipt),flush=True)
