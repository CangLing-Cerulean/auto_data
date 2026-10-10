"""Copy a small isolated code project; large immutable inputs stay at their frozen sources."""
import json
from pathlib import Path
import shutil
import sys

old=Path('/root/data/auto_data_research/agent_system_20261008_native')
new=Path('/tmp/auto_data_phase4_open_project_20261009')
if new.exists():raise RuntimeError('Do not overwrite an existing project')
new.mkdir()
shutil.copytree(old/'online_ts',new/'online_ts',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copytree(old/'tests',new/'tests',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(new/'scripts').mkdir()
for name in ('check_phase4_readiness.py','launch_phase4.py','read_phase4_status.py'):
    shutil.copy2(old/'scripts'/name,new/'scripts'/name)
study=Path('research_runs/agent_research')
for name in ('experience_library.json','experience_specs.json'):
    destination=new/study/name;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(old/study/name,destination)
shutil.copytree(old/study/'memory/v2',new/study/'memory/v2',ignore=shutil.ignore_patterns('__pycache__'))
for source in (old/study/'tasks/metropt3_dev').iterdir():
    if not source.is_file():continue
    destination=new/study/'tasks/metropt3_dev'/source.name;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
library=json.loads((old/study/'experience_library.json').read_text())
sources={row['source']['path'] for row in library['records']}
case=json.loads((old/study/'phase4/20261009_v2/registered_case.json').read_text())
sources.update(case['source'][key] for key in ('path','recipe_path'))
for value in sources:
    source=old/value;destination=new/value;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
(new/study/'phase4/20261009_open').mkdir(parents=True)
print(json.dumps({'status':'created','project':str(new),'old_project_changed':False,
    'large_inputs':'copied_frozen_training_only_assets','source_files':len(sources)}),flush=True)
