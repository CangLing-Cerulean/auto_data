"""Archive finished acceptance evidence and exact protected source without replay."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import time

project = Path('/tmp/auto_data_capability_project_20261010')
root = Path('/tmp/auto_data_capability_acceptance_20261010')
archive = Path('/tmp/capability_acceptance_closed_20261010.tar')
sys.path.insert(0, str(project))
from online_ts.agent_research.runtime import code_identity

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

if archive.exists():
    raise RuntimeError('Existing archive preserved')
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    except (OSError, UnicodeError):
        continue
    if 'scripts/run_capability.py --stage' in command:
        raise RuntimeError('Acceptance still running')
summary = read(root / 'bank_v2/summary.json')
if summary['status'] != 'completed':
    raise RuntimeError('Bank incomplete')
protected = read(root / 'bank_v2/preregistration.json')['code']
if code_identity() != protected:
    raise RuntimeError('Protected source changed')
snapshot = root / 'completed_source_snapshot'
if snapshot.exists():
    raise RuntimeError('Existing source snapshot preserved')
snapshot.mkdir()
for name, expected in protected.items():
    path = Path(name)
    if sha(path) != expected:
        raise RuntimeError('Protected source hash mismatch')
    target = snapshot / path.relative_to(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
for name in ('online_ts/__init__.py', 'scripts/run_capability_checks.py',
             'tests/test_agent_capability.py'):
    path = project / name
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
shutil.copytree(project / 'research_runs/agent_research/capability_acceptance/20261010',
                root / 'frozen_protocol')
banks = []
for folder in sorted(root.glob('bank*')):
    if not folder.is_dir():
        continue
    calls = []
    for database in folder.glob('runs/*/runtime.sqlite'):
        with sqlite3.connect(database) as db:
            calls.extend(db.execute('SELECT status,actual_tokens,reserved_tokens FROM agent_calls').fetchall())
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    banks.append({'bank': folder.name, 'API_attempts': len(calls),
        'API_actual_known_tokens': sum(row[1] or 0 for row in calls),
        'unresolved_calls': sum(row[1] is None for row in calls),
        'unresolved_reserved_tokens': sum(row[2] for row in calls if row[1] is None)})
closure = {'status': 'completed_bank_gate_not_passed', 'timestamp_ns': time.time_ns(),
    'registered': summary['registered'], 'passed': summary['passed'],
    'all_structured_checks_passed': summary['all_structured_checks_passed'], 'banks': banks,
    'known_loop_started': (root / 'known_loop').exists(), 'new_training_calls': 0,
    'test_access': False, 'active_queue': False, 'scores_unchanged': True,
    'startup_failures': read(root / 'bank/startup_failure_summary.json')}
write(root / 'closure_summary.json', closure)
files = sorted(path for path in root.rglob('*') if path.is_file()
               and not path.name.endswith(('-wal', '-shm')))
manifest = {str(path.relative_to(root)): {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in files}
write(root / 'closure_asset_manifest.json', manifest)
files.append(root / 'closure_asset_manifest.json')
with tarfile.open(archive, 'w') as output:
    for path in files:
        output.add(path, arcname=str(path.relative_to(root)), recursive=False)
receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
           'members': len(files), 'summary': closure}
write(root / 'closure_archive_receipt.json', receipt)
print(json.dumps({'archive': str(archive), 'bytes': receipt['bytes'], 'sha256': receipt['sha256'],
    'members': receipt['members'], 'banks': banks, 'passed': summary['passed']}))
