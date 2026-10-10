"""Inspect durable progress and real process existence; no API or training."""
import json
from pathlib import Path
import sqlite3

root=Path('/tmp/auto_data_capability_acceptance_20261010_v3')
def read(path):return json.loads(path.read_text()) if path.is_file() else None
processes=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit():continue
    try:command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except (OSError,UnicodeError):continue
    if '/auto_data_capability_project_20261010_v3/scripts/run_capability.py' in command:
        processes.append({'pid':int(entry.name),'stage':'bank' if '--stage bank' in command else 'loop'})
calls=[];runs=[]
for database in sorted(root.rglob('runtime.sqlite')):
    with sqlite3.connect(f'file:{database}?mode=ro',uri=True) as db:
        rows=db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall()
        kinds=dict(db.execute('SELECT kind,count(*) FROM agent_runtime_records GROUP BY kind').fetchall())
    calls.extend(rows)
    score=read(database.parent/'score.json')
    runs.append({'run_id':database.parent.name,'API_attempts':len(rows),'actual_known_tokens':sum(row[1] or 0 for row in rows),
        'invalid_responses':kinds.get('invalid_response',0),'format_failures':kinds.get('format_failure',0),
        'decisions':kinds.get('decision',0),'passed':score.get('passed') if score else None})
progress=read(root/'bank/progress.json');summary=read(root/'bank/summary.json')
loop=root/'known_loop';trainings=[]
for folder in sorted((loop/'training').glob('*')):
    result=read(folder/'result.json');history=read(folder/'history.json') or []
    trainings.append({'job_id':folder.name,'completed':result is not None,
        'last_step':history[-1].get('step') if history else None,'metrics':result.get('metrics') if result else None})
value={'processes':processes,'bank_completed':summary is not None,'bank_done':progress.get('completed',0) if progress else 0,
    'bank_passed':summary.get('passed') if summary else None,'strict_gate_passed':summary.get('all_structured_checks_passed') if summary else None,
    'runs':runs,'API_attempts':len(calls),'API_actual_known_tokens':sum(row[1] or 0 for row in calls),
    'unresolved_calls':sum(row[1] is None for row in calls),'unresolved_reserved_tokens':sum(row[2] for row in calls if row[1] is None),
    'known_loop_result':read(loop/'result_summary.json'),'known_loop_state':read(loop/'status.json'),
    'trainings':trainings,'new_completed_trainings':sum(row['completed'] for row in trainings)}
(root/'live_status.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
compact={key:value[key] for key in ('processes','bank_completed','bank_done','bank_passed','strict_gate_passed','runs',
    'API_attempts','API_actual_known_tokens','unresolved_calls','new_completed_trainings','trainings')}
if value['known_loop_result']:
    compact['known_loop_result']=value['known_loop_result']
if loop.is_dir():
    with sqlite3.connect(f'file:{loop / "runtime.sqlite"}?mode=ro',uri=True) as db:
        evidence=[(kind,json.loads(payload)) for kind,payload in db.execute(
            "SELECT kind,payload FROM agent_runtime_records WHERE kind IN ('diagnostic_job','processing_candidate','decision') ORDER BY created_ns")]
    compact['loop_diagnostics']=[{key:value.get(key) for key in ('record_id','status','purpose','error')} for kind,value in evidence if kind=='diagnostic_job']
    compact['loop_candidates']=[{key:value.get(key) for key in ('candidate_id','status','error','hypothesis_updated')} for kind,value in evidence if kind=='processing_candidate']
    compact['loop_recent_tools']=[{'tool':value['selected_action'],'outcome':value.get('decision_outcome'),
        'error':(value.get('actual_result') or {}).get('error')} for kind,value in evidence if kind=='decision'][-4:]
print(json.dumps(compact,ensure_ascii=False),flush=True)
