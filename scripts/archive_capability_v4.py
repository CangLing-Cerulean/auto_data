"""Preserve finished v4 acceptance, model evidence and validated training assets."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import time

project=Path('/tmp/auto_data_capability_project_20261010_v4')
root=Path('/tmp/auto_data_capability_acceptance_20261010_v4')
archive=Path('/tmp/capability_acceptance_v4_closed_20261010.tar')
sys.path.insert(0,str(project))
from online_ts.agent_research.runtime import code_identity

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

if archive.exists():raise RuntimeError('Existing archive preserved')
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if str(project/'scripts/run_capability.py') in command:raise RuntimeError('Acceptance still running')
summary=read(project/'research_runs/agent_research/capability_acceptance/20261010_v4/bank_summary.json')
if summary['status']!='completed':raise RuntimeError('Bank incomplete')
if (root/'known_loop').exists() and not any((root/'known_loop'/name).is_file() for name in ('result_summary.json','interruption_summary.json')):raise RuntimeError('Loop incomplete')
if code_identity()!=read(root/'known_loop/preregistration.json')['code']:raise RuntimeError('Protected code changed')
runs=[]
for database in sorted(root.rglob('runtime.sqlite')):
    with sqlite3.connect(database) as db:
        calls=db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall()
        records=[(kind,json.loads(payload)) for kind,payload in db.execute('SELECT kind,payload FROM agent_runtime_records')]
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    training=[]
    for result_path in (database.parent/'training').glob('*/result.json'):
        result=read(result_path)
        for name,expected in result['assets'].items():
            if sha(result_path.parent/name)!=expected:raise RuntimeError('Training asset hash changed')
        training.append({'job_id':result_path.parent.name,'metrics':result['metrics']})
    errors=Counter(json.dumps(value.get('validation_errors'),ensure_ascii=False,sort_keys=True)
        for kind,value in records if kind=='format_failure')
    runs.append({'run_id':database.parent.name,'API_attempts':len(calls),
        'actual_known_tokens':sum(row[1] or 0 for row in calls),
        'unresolved_calls':sum(row[1] is None for row in calls),
        'unresolved_reserved_tokens':sum(row[2] for row in calls if row[1] is None),
        'records':dict(Counter(kind for kind,_ in records)),'format_validation_counts':dict(errors),
        'tool_counts':dict(Counter(value.get('selected_action') for kind,value in records if kind=='decision')),
        'completed_trainings':training})
cost={'runs':runs,'API_attempts':sum(row['API_attempts'] for row in runs),
    'actual_known_tokens':sum(row['actual_known_tokens'] for row in runs),
    'unresolved_calls':sum(row['unresolved_calls'] for row in runs),
    'unresolved_reserved_tokens':sum(row['unresolved_reserved_tokens'] for row in runs),
    'new_completed_trainings':sum(len(row['completed_trainings']) for row in runs),
    'invalid_responses':sum(row['records'].get('invalid_response',0) for row in runs),
    'format_failure_records':sum(row['records'].get('format_failure',0) for row in runs)}
write(root/'cost_audit.json',cost)
snapshot=root/'completed_source_snapshot'
snapshot.mkdir()
for folder in ('online_ts','tests','scripts'):
    shutil.copytree(project/folder,snapshot/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
pilot=project/'research_runs/agent_research/capability_acceptance/20261010_v4'
shutil.copytree(pilot,root/'frozen_protocol')
closure={'status':'known_loop_finished_preserved','timestamp_ns':time.time_ns(),'bank_passed':summary['passed'],
    'accepted_bank_inherited_from_v3':True,'new_historical_assessments':0,
    'bank_registered':summary['registered'],'strict_gate_passed':summary['all_structured_checks_passed'],
    'known_loop':read(root/'known_loop/result_summary.json') if (root/'known_loop/result_summary.json').is_file() else None,
    'known_loop_interruption':read(root/'known_loop/interruption_summary.json') if (root/'known_loop/interruption_summary.json').is_file() else None,
    'cost':cost,'active_queue':False,'test_access':False,'same_ID_replay':False}
write(root/'closure_summary.json',closure)
files=sorted(path for path in root.rglob('*') if path.is_file() and not path.name.endswith(('-wal','-shm')))
manifest={str(path.relative_to(root)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in files}
write(root/'closure_asset_manifest.json',manifest);files.append(root/'closure_asset_manifest.json')
with tarfile.open(archive,'w') as output:
    for path in files:output.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,
    'members':len(files),'summary':closure}
write(root/'closure_archive_receipt.json',receipt)
print(json.dumps({key:receipt[key] for key in ('archive','sha256','bytes','members')}|{'cost':{key:value for key,value in cost.items() if key!='runs'}}),flush=True)
