"""Package complete r8 registrations and reproducible frozen dependencies."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_interactions_r8'

def main():
    assert read(OUT/'result_audit.json')['status']=='PASS'
    lock=read(OUT/'execution_lock.json')
    files=[p for p in OUT.rglob('*') if p.is_file() and not {'archives','private','__pycache__'}.intersection(p.relative_to(OUT).parts)]
    files += [ROOT/'scripts'/name for name in {*lock['scripts'],'summarize_conditional_interactions.py','package_conditional_interactions.py'}]
    files += [ROOT/name for name in lock['inputs']]
    previous=OUT.parent/'20261018_measurement_generation_r4'
    files += [previous/'replay/summary.json',OUT.parent/'20261018_strong_agent_r5/results.json']
    files += [p for p in (previous/'paper').rglob('*') if p.is_file() and p.suffix in {'.tex','.pdf','.png'}]
    files=sorted(set(files));members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    folder=OUT/'archives';folder.mkdir();archive=folder/'conditional_interactions_r8.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for m in tar.getmembers():
            assert m.isfile() and m.size==members[m.name]['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==members[m.name]['sha256']
    write(OUT/'archive_members.json',dict(status='PASS',count=len(members),members=members))
    write(OUT/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='conditional-interactions-20261018-r8',title='Conditional interactions: complete fixed-pool factorial development',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
