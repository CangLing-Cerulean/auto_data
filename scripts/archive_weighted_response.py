"""Archive a completed and independently accepted phase without deleting assets."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import time
import numpy as np
import torch
import run_weighted_response as run

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
started=time.perf_counter();plan=run.verify(out)
assert run.read(out/'status.json')['status']=='completed'
audit=run.read(out/'final_audit.json');assert audit['status']=='PASS' and audit['cumulative_full_trainings']==409
assert audit['labels_sha256']==run.sha(out/'labels.json') and audit['predictions_sha256']==run.sha(out/'predictions.json')
processes={}
for phase in ['labels','audit']:
    launch=run.read(out/'control'/phase/'launch.json');stat=Path('/proc')/str(launch['pid'])/'stat'
    state=stat.read_text().split()[2] if stat.exists() else 'missing'
    assert state in ['missing','Z'],(phase,state)
    processes[phase]=dict(pid=launch['pid'],process_state=state)
run.write(out/'terminal_status.json',dict(status='completed_no_live_phase_process',processes=processes,observed_unix=time.time(),pending_model_queue=[]))
run.write(out/'runtime_environment.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(0),scope='environment_record_after_accepted_run'))
files=sorted(f for f in out.rglob('*') if f.is_file())
assert all(not f.is_symlink() for f in files)
assert all(f.suffix in ['.py','.md','.json','.log','.lock','.npz','.pt'] for f in files)
required=sum(f.stat().st_size for f in files)+plan['runtime_min_free_bytes']
assert shutil.disk_usage(out).free>=required
members={f.relative_to(out).as_posix():dict(bytes=f.stat().st_size,sha256=run.sha(f)) for f in files}
run.write(out/'archive_members.json',members)
archive=out/'weighted_response_r1.tar.gz'
with tarfile.open(archive,'x:gz') as tar:
    for f in files+[out/'archive_members.json']:tar.add(f,arcname=f.relative_to(out).as_posix(),recursive=False)
with tarfile.open(archive) as tar:
    entries=tar.getmembers();assert len(entries)==len(members)+1 and len({e.name for e in entries})==len(entries)
    assert {e.name for e in entries}==set(members)|{'archive_members.json'}
    for entry in entries:
        assert entry.isfile()
        expected=members[entry.name]['sha256'] if entry.name!='archive_members.json' else run.sha(out/'archive_members.json')
        assert hashlib.file_digest(tar.extractfile(entry),'sha256').hexdigest()==expected
run.write(out/'archive_receipt.json',dict(status='PASS',name=archive.name,bytes=archive.stat().st_size,sha256=run.sha(archive),members=len(members)+1,
    archive_seconds=time.perf_counter()-started,new_full_trainings=36,cumulative_full_trainings=409,old_assets_deleted=False))
print(json.dumps(run.read(out/'archive_receipt.json')),flush=True)
