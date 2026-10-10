"""Preserve the first startup-failed bank and source files before entry-point repair."""
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.capability import ROOT,PILOT,OUTPUT
from online_ts.agent_research.runtime import code_identity
from online_ts.selection.common import write_json,file_hash

destination=OUTPUT/'bank/source_snapshot'
if destination.exists():raise RuntimeError('Existing source snapshot preserved')
destination.mkdir()
files=list(code_identity())+[str(ROOT/'scripts/run_capability.py'),str(PILOT/'protocol.json'),str(PILOT/'PROTOCOL.md')]
for value in files:
    path=Path(value);target=destination/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
calls=[]
for path in (OUTPUT/'bank/runs').glob('*/runtime.sqlite'):
    with sqlite3.connect(path) as db:calls.extend(db.execute('SELECT actual_tokens,reserved_tokens FROM agent_calls').fetchall())
summary={'status':'startup_failed_preserved','cause':'CLI_missing_main_guard_reentered_under_spawn_before_transport_worker',
    'registered_cases':12,'API_transport_attempts':len(calls),'API_actual_known_tokens':sum(row[0] or 0 for row in calls),
    'unresolved_ledger_rows':sum(row[0] is None for row in calls),'reserved_tokens':sum(row[1] for row in calls if row[0] is None),
    'provider_contact':'no_worker_request_function_reached_under_observed_spawn_trace',
    'no_case_read_or_scientific_assessment_obtained':True,'new_trainings':0,'same_ID_replay':False,
    'source_snapshot_hashes':{str(Path(value).relative_to(ROOT)):file_hash(value) for value in files}}
write_json(OUTPUT/'bank/startup_failure_summary.json',summary)
print('Startup failure and exact sources preserved',flush=True)
