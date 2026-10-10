"""Create receipts only for the new model study's completed artifacts."""
import json
from pathlib import Path
from run_research import ROOT, digest, write_json

folder = ROOT / 'research_runs/model_exploration'
progress = json.loads((folder / 'progress.json').read_text())
runs = [r['run_id'] for r in progress['rows']] + ['A015_model_export']
files = {}
for name in runs:
    run = ROOT / 'research_runs/results' / name
    assert json.loads((run / 'status.json').read_text())['phase'] == 'completed'
    hashes = {p.name: digest(p) for p in run.iterdir() if p.is_file() and p.name != 'asset_checksums.json'}
    write_json(run / 'asset_checksums.json', hashes)
    for relative, expected in hashes.items(): files[str((run / relative).relative_to(ROOT)).replace('\\', '/')] = expected
for path in folder.rglob('*'):
    if path.is_file() and path.name != 'download_receipt.json':
        files[str(path.relative_to(ROOT)).replace('\\', '/')] = digest(path)
write_json(folder / 'download_receipt.json', files)
print(json.dumps({'runs': runs, 'files': len(files)}))
