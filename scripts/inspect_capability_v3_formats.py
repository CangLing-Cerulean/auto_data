"""Bounded structural inspection of failed responses, without internal reasoning text."""
import json
from pathlib import Path
import sqlite3

rows=[]
for path in sorted(Path('/tmp/auto_data_capability_acceptance_20261010_v3/bank/runs').glob('*/runtime.sqlite')):
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        invalid=[json.loads(row[0]) for row in db.execute("SELECT payload FROM agent_runtime_records WHERE kind='invalid_response' ORDER BY created_ns DESC LIMIT 2")]
        for item in invalid:
            response=json.loads(db.execute('SELECT payload FROM agent_runtime_records WHERE record_id=?',(item['response_id'],)).fetchone()[0])['response']
            shape={'type':type(response).__name__}
            if isinstance(response,dict):
                shape.update(keys=list(response),action_type=type(response.get('action')).__name__)
                if isinstance(response.get('action'),dict):shape['action_keys']=list(response['action'])
                elif isinstance(response.get('action'),str):shape['action_string']=response['action'][:80]
            elif isinstance(response,str):shape.update(length=len(response),prefix=response[:100])
            rows.append({'run_id':path.parent.name,'error':item['error'],'shape':shape})
print(json.dumps(rows,ensure_ascii=False),flush=True)
