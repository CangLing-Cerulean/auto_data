"""Create an isolated acceptance project without changing any prior experiment sources."""
import json
from pathlib import Path
import shutil

old=Path('/tmp/auto_data_phase4_open_project_20261009')
new=Path('/tmp/auto_data_capability_project_20261010')
if new.exists():raise RuntimeError('Existing acceptance project preserved')
new.mkdir()
for folder in ('online_ts','tests'):
    shutil.copytree(old/folder,new/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
(new/'runtime_deps').symlink_to((old/'runtime_deps').resolve(),target_is_directory=True)
(new/'scripts').mkdir()
study=Path('research_runs/agent_research')
for name in ('experience_library.json','experience_specs.json'):
    destination=new/study/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/study/name,destination)
shutil.copytree(old/study/'memory/v2',new/study/'memory/v2')
shutil.copytree(old/study/'tasks/metropt3_dev',new/study/'tasks/metropt3_dev')
library=json.loads((old/study/'experience_library.json').read_text())
sources={row['source']['path'] for row in library['records']}
case=json.loads((old/study/'phase4/20261009_open/registered_case.json').read_text())
sources.update(case['source'][key] for key in ('path','recipe_path'))
for value in sources:
    destination=new/value;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/value,destination)
pilot=new/study/'capability_acceptance/20261010';(pilot/'sources').mkdir(parents=True)
shutil.copy2(old/study/'phase4/20261009_open/registered_case.json',pilot/'registered_case.json')
print(json.dumps({'project':str(new),'old_sources_changed':False,'immutable_T_assets_copied':True}),flush=True)
