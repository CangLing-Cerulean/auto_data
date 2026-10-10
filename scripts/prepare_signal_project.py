"""Create a fresh server project without copying the known source-half answer."""
import json
from pathlib import Path
import shutil

old=Path('/tmp/auto_data_capability_project_20261010_v4')
new=Path('/tmp/auto_data_signal_project_20261010')
study=Path('research_runs/agent_research')
if new.exists():raise RuntimeError('Existing study project will not be overwritten')
folders=('online_ts','tests','scripts',str(study/'memory/v2'),str(study/'tasks/metropt3_dev'))
needed=sum(p.stat().st_size for folder in folders for p in (old/folder).rglob('*') if p.is_file())
if shutil.disk_usage('/tmp').free<needed+6*1024**3:raise RuntimeError('Insufficient study disk reserve')
new.mkdir()
for folder in folders:shutil.copytree(old/folder,new/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(new/'runtime_deps').symlink_to((old/'runtime_deps').resolve(),target_is_directory=True)
for name in ('experience_library.json','experience_specs.json'):shutil.copy2(old/study/name,new/study/name)
library=json.loads((old/study/'experience_library.json').read_text())
sources={row['source']['path'] for row in library['records']}
for value in sources:
    target=new/value;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/value,target)
(new/study/'phase4/20261010_signal').mkdir(parents=True)
print(json.dumps({'status':'created','project':str(new),'previous_project_changed':False,
    'source_half_case_copied':False,'disk_free':shutil.disk_usage('/tmp').free}),flush=True)
