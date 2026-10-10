"""Read-only compact monitoring; excludes raw prompts, credentials and peer feedback."""
import json
from pathlib import Path
import sqlite3
from collections import deque

root=Path('/tmp/auto_data_signal_study_20261010_v2')
def load(path):
    return json.loads(path.read_text()) if path.is_file() else None
processes=[]
for path in Path('/proc').iterdir():
    if not path.name.isdigit():continue
    try:command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if 'scripts/run_signal_study.py' in command and command.startswith('/opt/miniconda3/bin/python'):
        processes.append(int(path.name))
value={'processes':processes,'progress':load(root/'progress.json'),'batch_failure':load(root/'batch_failure.json'),
       'runs':[],'API_attempts':0,'API_actual_known_tokens':0,'unresolved_calls':0,
       'unresolved_reserved_tokens':0,'completed_trainings':0,'failed_trainings':0}
for folder in sorted((root/'runs').glob('*')):
    if not folder.is_dir():continue
    row={'run_id':folder.name,'failure':load(folder/'failure.json'),'final':load(folder/'final_policy.json'),
         'external_score':load(folder/'external_score.json'),'training':[]}
    database=folder/'runtime.sqlite'
    if database.exists():
        with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
            calls=db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall()
            row['API_attempts']=len(calls)
            row['API_actual_known_tokens']=sum(r[1] or 0 for r in calls)
            row['unresolved_calls']=sum(r[1] is None for r in calls)
            row['unresolved_reserved_tokens']=sum(r[2] for r in calls if r[1] is None)
        for key in ('API_attempts','API_actual_known_tokens','unresolved_calls','unresolved_reserved_tokens'):value[key]+=row[key]
    for job in sorted((folder/'training').glob('*')):
        result=load(job/'result.json');execution=load(job/'execution.json');history=load(job/'history.json')
        state='completed' if result else ('failed' if execution and execution['status']!='completed' else 'running')
        row['training'].append({'job':job.name[:12],'status':state,'last_step':history[-1]['step'] if history else None,
            'scope_audit':(load(job/'scope_metrics.json') or {}).get('status')})
        if state=='completed':value['completed_trainings']+=1
        elif state=='failed':value['failed_trainings']+=1
    trace_path=folder/'trace.jsonl'
    if trace_path.exists():
        with trace_path.open() as stream:
            recent=list(deque(stream,maxlen=3))
        row['recent_decisions']=[{'tool':r.get('selected_action'),'reason':r.get('reason_summary'),
             'outcome':r.get('decision_outcome'),'error':(r.get('actual_result') or {}).get('error')}
             for r in (json.loads(line) for line in recent)]
    value['runs'].append(row)
print(json.dumps(value,ensure_ascii=False),flush=True)
