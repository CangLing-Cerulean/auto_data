"""Read-only worker error inspection, without changing or replaying jobs."""
import json
import stat
from pathlib import Path
root=Path('/tmp/auto_data_signal_study_20261010_v2')
rows=[]
for path in root.glob('runs/*/training/*/execution.json'):
    value=json.loads(path.read_text())
    if value['status']!='completed':
        rows.append({'run':path.parents[2].name,'job':path.parent.name,'error':value.get('error'),
                     'stdout':value.get('stdout','')[-5000:],
                     'file_types':[{'path':str(p.relative_to(path.parent)),'mode':stat.filemode(p.lstat().st_mode)}
                         for p in path.parent.rglob('*')]})
print(json.dumps(rows,ensure_ascii=False),flush=True)
