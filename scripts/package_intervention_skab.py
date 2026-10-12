"""Archive frozen pilot inputs, derived arrays and all run assets without raw CSVs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    target=root/'skab-intervention-20261012-pilot-r1.tar.gz'
    assert not target.exists()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8-sig'))
    core=Path(protocol['frozen_core'])
    with core.open('rb') as stream: assert hashlib.file_digest(stream,'sha256').hexdigest()==protocol['frozen_core_sha256']
    snapshot=root/'frozen_core_snapshot.py';assert not snapshot.exists();shutil.copyfile(core,snapshot)
    names=['protocol.json','run_intervention_skab_pilot.py','audit_intervention_skab.py',
           'package_intervention_skab.py','frozen_core_snapshot.py','acceptance.json','prepared','runs','agent_records']
    files=[]
    for name in names:
        path=root/name;assert path.exists()
        candidates=sorted(path.rglob('*')) if path.is_dir() else [path]
        for f in candidates:
            if f.is_file():
                assert not f.is_symlink() and root.resolve() in f.resolve().parents
                with f.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
                files.append(dict(name=f.relative_to(root).as_posix(),bytes=f.stat().st_size,sha256=digest))
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
        for item in files: tar.add(root/item['name'],arcname=item['name'],recursive=False)
    with target.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    receipt=dict(name=target.name,bytes=target.stat().st_size,sha256=digest,members=files,
                 raw_CSV_included=False,derived_T_D_sensor_arrays_included=True,test_included=False)
    (root/'archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='members'}))
if __name__=='__main__':main()
