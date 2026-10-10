"""Compact acceptance status, including actual process existence and durable case scores."""
import json
from pathlib import Path
import sqlite3
import sys

root=Path('/tmp/auto_data_capability_acceptance_20261010')
def load(path):return json.loads(path.read_text()) if path.is_file() else None
processes=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if 'scripts/run_capability.py --stage' in command:processes.append({'pid':int(path.name),'stage':'bank' if '--stage bank' in command else 'loop'})
value={'processes':processes,'banks':[],'known_loop':load(root/'known_loop/result_summary.json')}
for folder in sorted(root.glob('bank*')):
    if not folder.is_dir():continue
    calls=[]
    for database in folder.glob('runs/*/runtime.sqlite'):
        with sqlite3.connect(database) as db:calls.extend(db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall())
    value['banks'].append({'folder':folder.name,'summary':load(folder/'summary.json'),
        'completed_scores':len(list(folder.glob('runs/*/score.json'))),
        'scores':[load(path) for path in sorted(folder.glob('runs/*/score.json'))],
        'API_attempts':len(calls),'API_actual_known_tokens':sum(row[1] or 0 for row in calls),
        'unresolved_calls':sum(row[1] is None for row in calls),
        'unresolved_reserved_tokens':sum(row[2] for row in calls if row[1] is None)})
loop=root/'known_loop'
if loop.is_dir():
    value['loop_state']=load(loop/'status.json')
    value['loop_training']=[{'job_id':job.name,'result':load(job/'result.json') is not None,
        'last_step':(load(job/'history.json') or [{'step':0}])[-1]['step']} for job in (loop/'training').glob('*')]
print(json.dumps(value,ensure_ascii=False),flush=True)
