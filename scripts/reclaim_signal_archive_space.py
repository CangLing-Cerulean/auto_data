"""Remove only three byte-verified redundant server tar copies, retaining all originals."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

def main(manifest_path):
    rows=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    allowed={'phase4_open_closed_20261010.tar','capability_acceptance_v3_closed_20261010.tar',
             'capability_acceptance_v4_closed_20261010.tar'}
    if len(rows)!=3:raise RuntimeError('Expected three registered duplicate archives')
    active=[]
    markers=('auto_data_phase4_open_project_20261009','auto_data_phase4_20261009_open',
        'auto_data_capability_project_20261010_v3','auto_data_capability_acceptance_20261010_v3',
        'auto_data_capability_project_20261010_v4','auto_data_capability_acceptance_20261010_v4')
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=proc.read_bytes().decode(errors='replace')
        except (OSError,PermissionError):continue
        if any(m in cmd for m in markers):active.append(proc.parent.name)
    if active:raise RuntimeError('Old study process remains active: '+','.join(active))
    approved=[]
    for row in rows:
        path=Path(row['server_path'])
        if path.parent!=Path('/tmp') or path.name not in allowed or path.is_symlink() or not row['local_members_verified']:
            raise RuntimeError('Archive target not registered')
        if path.stat().st_size!=row['bytes']:raise RuntimeError('Server duplicate size mismatch')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while block:=stream.read(1024*1024):h.update(block)
        if h.hexdigest()!=row['sha256']:raise RuntimeError('Server/local duplicate hash mismatch')
        approved.append((path,row))
    before=shutil.disk_usage('/tmp').free
    for path,row in approved:path.unlink()
    print(json.dumps({'status':'completed','removed_verified_duplicate_tars':[r for _,r in approved],
        'bytes_removed':sum(r['bytes'] for _,r in approved),'free_before':before,
        'free_after':shutil.disk_usage('/tmp').free,'raw_study_directories_removed':False,
        'local_verified_backup_removed':False,'additional_training':0,'API_calls':0}),flush=True)

if __name__=='__main__':main(sys.argv[1])
