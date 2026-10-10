"""Read-only engineering failure inspection, never fed into the running Agent."""
import json
from pathlib import Path
root=Path('/tmp/auto_data_signal_study_20261010/runs')
rows=[]
for folder in sorted(root.glob('*')):
    for path in sorted(folder.glob('diagnostics/*/execution.json')):
        value=json.loads(path.read_text())
        if value.get('information_boundary',{}).get('violations') or value['status']!='completed':
            rows.append({'run_id':folder.name,'job':path.parent.name,'status':value['status'],
                'error':value.get('error'),'boundary':value.get('information_boundary'),
                'stdout':value.get('stdout','')[-3000:],
                'source':(path.parent/'code/agent.py').read_text()[:4000]})
print(json.dumps(rows,ensure_ascii=False),flush=True)
