"""Clone immutable source dependencies, never prior experiments or credentials."""
from pathlib import Path
import json
import shutil

old = Path('/tmp/auto_data_signal_project_20261010_v2')
new = Path('/tmp/auto_data_autonomous_memory_20261012_r2')
study = Path('research_runs/agent_research')
if new.exists():
    raise RuntimeError('Existing project will not be overwritten')
if shutil.disk_usage('/tmp').free < 8 * 1024**3:
    raise RuntimeError('Insufficient disk reserve')
new.mkdir()
for folder in ('online_ts', 'scripts', str(study/'memory/v2'), str(study/'tasks/metropt3_dev')):
    shutil.copytree(old/folder, new/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(new/'runtime_deps').symlink_to((old/'runtime_deps').resolve(), target_is_directory=True)
for name in ('experience_library.json', 'experience_specs.json'):
    shutil.copy2(old/study/name, new/study/name)
library = json.loads((old/study/'experience_library.json').read_text())
for source in {row['source']['path'] for row in library['records']}:
    target = new/source
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old/source, target)
print(json.dumps({'created': str(new), 'previous_project_unchanged': True}))
