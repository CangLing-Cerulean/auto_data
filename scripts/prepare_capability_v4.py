"""New isolated known-case revision after a preserved scope-interrupted run."""
import json
from pathlib import Path
import shutil

old=Path('/tmp/auto_data_capability_project_20261010_v3')
new=Path('/tmp/auto_data_capability_project_20261010_v4')
if new.exists():raise RuntimeError('Existing v4 project preserved')
study=Path('research_runs/agent_research')
folders=('online_ts','tests','scripts',str(study/'memory/v2'),str(study/'tasks/metropt3_dev'))
needed=sum(path.stat().st_size for folder in folders for path in (old/folder).rglob('*') if path.is_file())
if shutil.disk_usage('/tmp').free<needed+2*1024**3:raise RuntimeError('Insufficient space for isolated assets and outputs')
new.mkdir()
for folder in folders:shutil.copytree(old/folder,new/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(new/'runtime_deps').symlink_to((old/'runtime_deps').resolve(),target_is_directory=True)
for name in ('experience_library.json','experience_specs.json'):shutil.copy2(old/study/name,new/study/name)
library=json.loads((old/study/'experience_library.json').read_text())
sources={row['source']['path'] for row in library['records']}
case=json.loads((old/study/'capability_acceptance/20261010_v3/registered_case.json').read_text())
sources.update(case['source'][key] for key in ('path','recipe_path'))
for value in sources:
    target=new/value;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/value,target)
(new/study/'capability_acceptance/20261010_v4').mkdir(parents=True)
print(json.dumps({'project':str(new),'previous_project_unchanged':True,'bytes_copied':needed}),flush=True)
