"""Freeze a closed platform-failure cohort without rewriting decisions or call ledgers."""
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
root=Path('/tmp/auto_data_signal_study_20261010')
archive=Path('/tmp/signal_v1_closed_20261010.tar')
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
def load(path):return json.loads(path.read_text())
if archive.exists():raise RuntimeError('Existing archive preserved')
if not (root/'maintenance_closure.json').exists():raise RuntimeError('Closure required')
runs=[]
for folder in sorted((root/'runs').glob('*')):
    with sqlite3.connect(folder/'runtime.sqlite') as db:
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        calls=db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall()
    results=list((folder/'training').glob('*/result.json'))
    for path in results:
        result=load(path)
        for name,h in result['assets'].items():
            if sha(path.parent/name)!=h:raise RuntimeError('Result asset changed')
        for name,h in result['identity']['protected_assets'].items():
            if sha(Path(name))!=h:raise RuntimeError('Protected source asset changed')
    jobs=list((folder/'training').glob('*/job.json'))
    runs.append({'run_id':folder.name,'search_ended':(folder/'search_result.json').exists(),
        'API_attempts':len(calls),'actual_known_tokens':sum(r[1] or 0 for r in calls),
        'unresolved_calls':sum(r[1] is None for r in calls),'unresolved_reserved_tokens':sum(r[2] for r in calls if r[1] is None),
        'completed_trainings':len(results),'interrupted_trainings':len(jobs)-len(results),
        'scientific_success':False,'platform_fault_not_agent_capability_refutation':True})
summary={'status':'closed_platform_failure','registered_searches':9,'started_searches':len(runs),
    'not_started':9-len(runs),'runs':runs,'completed_trainings':sum(r['completed_trainings'] for r in runs),
    'interrupted_trainings':sum(r['interrupted_trainings'] for r in runs),
    'API_attempts':sum(r['API_attempts'] for r in runs),'actual_known_tokens':sum(r['actual_known_tokens'] for r in runs),
    'unresolved_calls':sum(r['unresolved_calls'] for r in runs),'test_access':False,
    'formal_comparison_complete':False,'all_failures_costs_and_original_updates_retained':True}
(root/'closure_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith(('-wal','-shm')))
manifest={str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}
(root/'asset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(root/'asset_manifest.json')
with tarfile.open(archive,'w') as output:
    for path in files:output.add(path,arcname=str(path.relative_to(root)),recursive=False)
receipt={'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'members':len(files),'summary':summary}
(root/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
