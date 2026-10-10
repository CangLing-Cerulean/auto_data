"""Preserve the stopped v1 batch and exact accounting before deploying v2."""
import json
from pathlib import Path
import sqlite3
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.selection.common import read_json,write_json,file_hash
from online_ts.agent_research.runtime import ResearchRepository

root=Path('/tmp/auto_data_phase4_20261009')
archive=Path('/tmp/phase4_v1_aborted_20261009.tar')
if archive.exists():raise RuntimeError('Preserve existing archive')
runs=[]
for folder in sorted((root/'runs').glob('*')):
    identity=read_json(folder/'preregistration.json');repo=ResearchRepository(folder.name,identity,folder=folder)
    state=repo.state();state.status='aborted_platform_compatibility';repo.save_state(state,state.decision_version);repo.export()
    with sqlite3.connect(folder/'runtime.sqlite') as db:
        rows=db.execute('SELECT status,reserved_tokens,actual_tokens FROM agent_calls').fetchall()
    db.close()
    results=list((folder/'training').glob('*/result.json'))
    for path in results:
        result=read_json(path)
        for asset,expected in result['assets'].items():
            if file_hash(path.parent/asset)!=expected:raise RuntimeError('Training asset changed')
        for asset,expected in result['identity']['protected_assets'].items():
            if file_hash(asset)!=expected:raise RuntimeError('Protected asset changed before v2 deployment')
    runs.append({'run_id':folder.name,'API_calls':len(rows),'API_actual_known_tokens':sum(r[2] or 0 for r in rows),
        'API_unresolved_calls':sum(r[2] is None for r in rows),'API_unresolved_reserved_tokens':sum(r[1] for r in rows if r[2] is None),
        'completed_trainings':len(results),'formal_success':False,'aborted_not_scientific_recipe_refutation':True})
summary={'status':'aborted_preserved','registered_searches':12,'started_searches':len(runs),'not_started':12-len(runs),
    'runs':runs,'completed_trainings':sum(x['completed_trainings'] for x in runs),
    'API_calls':sum(x['API_calls'] for x in runs),'API_actual_known_tokens':sum(x['API_actual_known_tokens'] for x in runs),
    'API_unresolved_calls':sum(x['API_unresolved_calls'] for x in runs),
    'all_failures_and_costs_preserved':True,'test_access':False,'primary_comparison':'not_completed_due_platform_fault'}
write_json(root/'abort_summary.json',summary)
for path in root.rglob('*.sqlite'):
    connection=sqlite3.connect(path);connection.execute('PRAGMA wal_checkpoint(TRUNCATE)');connection.close()
files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm'))
             and p.name not in ('abort_asset_manifest.json','abort_archive_receipt.json'))
manifest={str(p.relative_to(root)):{'sha256':file_hash(p),'bytes':p.stat().st_size} for p in files}
write_json(root/'abort_asset_manifest.json',manifest);files.append(root/'abort_asset_manifest.json')
with tarfile.open(archive,'w') as output:
    for path in files:output.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt={'archive':str(archive),'sha256':file_hash(archive),'bytes':archive.stat().st_size,'members':len(files),'summary':summary}
write_json(root/'abort_archive_receipt.json',receipt)
print(json.dumps(receipt),flush=True)
