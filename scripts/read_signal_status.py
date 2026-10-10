"""Read-only compact monitoring; excludes raw prompts, credentials and peer feedback."""
import json
from pathlib import Path
import sqlite3

root=Path('/tmp/auto_data_signal_study_20261010')
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
       'completed_trainings':0,'failed_trainings':0}
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
        for key in ('API_attempts','API_actual_known_tokens','unresolved_calls'):value[key]+=row[key]
    for job in sorted((folder/'training').glob('*')):
        result=load(job/'result.json');execution=load(job/'execution.json');history=load(job/'history.json')
        state='completed' if result else ('failed' if execution and execution['status']!='completed' else 'running')
        row['training'].append({'job':job.name[:12],'status':state,'last_step':history[-1]['step'] if history else None,
            'scope_audit':(load(job/'scope_metrics.json') or {}).get('status')})
        if state=='completed':value['completed_trainings']+=1
        elif state=='failed':value['failed_trainings']+=1
    traces=load(folder/'trace.json')
    if isinstance(traces,list) and traces:
        last=traces[-1];row['last_tool']=last.get('selected_action');row['last_reason']=last.get('reason_summary')
    value['runs'].append(row)
print(json.dumps(value,ensure_ascii=False),flush=True)
