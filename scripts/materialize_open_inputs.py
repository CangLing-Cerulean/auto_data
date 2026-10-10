"""Materialize the new project's copied inputs to satisfy the existing no-symlink job contract."""
import hashlib
from pathlib import Path
import shutil

folder=Path('/tmp/auto_data_phase4_open_project_20261009/research_runs/agent_research/tasks/metropt3_dev')
old=Path('/root/data/auto_data_research/agent_system_20261008_native/research_runs/agent_research/tasks/metropt3_dev')
for destination in folder.iterdir():
    if not destination.is_symlink():continue
    source=destination.resolve()
    if source.parent!=old or source.name!=destination.name:raise RuntimeError('Unexpected source outside frozen task')
    temporary=destination.with_suffix(destination.suffix+'.copy')
    if temporary.exists():raise RuntimeError('Preserve unexpected copy')
    expected=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
    shutil.copy2(source,temporary)
    actual=hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()
    if actual!=expected:raise RuntimeError('Copy identity failed')
    temporary.replace(destination)
print('Frozen input copies materialized; original source assets unchanged',flush=True)
