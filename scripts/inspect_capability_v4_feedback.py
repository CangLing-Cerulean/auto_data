"""Read feedback records and immutable update IDs, without modifying a live run."""
import json
from pathlib import Path
import sqlite3
root=Path('/tmp/auto_data_capability_acceptance_20261010_v4/known_loop')
with sqlite3.connect(f'file:{root / "runtime.sqlite"}?mode=ro',uri=True) as db:
    decisions=[json.loads(row[0]) for row in db.execute("SELECT payload FROM agent_runtime_records WHERE kind='decision' ORDER BY created_ns")]
result=[]
for row in decisions:
    if row['selected_action'] not in {'update_hypothesis','submit_policy','inspect_results','read_candidate'}:continue
    args=row['arguments'];actual=row.get('actual_result') or {}
    result.append({'tool':row['selected_action'],'candidate_id':args.get('candidate_id'),
        'assessment':args.get('update',{}).get('assessment'),'error':actual.get('error'),
        'recorded_update':actual.get('recorded_update')})
with sqlite3.connect(f'file:{root / "evidence.sqlite"}?mode=ro',uri=True) as db:
    updates=[json.loads(row[0]) for row in db.execute("SELECT payload FROM agent_runtime_records WHERE kind='research_update'")]
print(json.dumps({'recent_decisions':result[-15:],'journal_updates':updates},ensure_ascii=False),flush=True)
