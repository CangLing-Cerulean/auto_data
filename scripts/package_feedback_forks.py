"""Preserve all report interventions, registrations and independently continued traces."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_feedback_forks_r6'

def main():
    assert read(OUT/'result_audit.json')['status']=='PASS'
    folder=OUT/'archives';folder.mkdir();archive=folder/'feedback_forks_r6.tar.gz'
    files=[p for p in OUT.rglob('*') if p.is_file() and not {'archives','private','__pycache__'}.intersection(p.relative_to(OUT).parts)]
    names=set(read(OUT/'execution_lock.json')['scripts'])|{'summarize_feedback_forks.py','package_feedback_forks.py'}
    files+=[ROOT/'scripts'/name for name in sorted(names)]
    previous=OUT.parent/'20261018_measurement_generation_r4'
    files+=[previous/'replay'/f'{source}_augmented_records.json' for source in ('ETT','Hydraulic')]
    files+=[previous/'generation'/f'{source}_mismatch/response.json' for source in ('ETT','Hydraulic')]
    files += [p for p in (previous/'paper').rglob('*') if p.is_file() and p.suffix in {'.tex','.pdf','.png'}]
    members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for m in tar.getmembers():
            assert m.isfile() and m.size==members[m.name]['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==members[m.name]['sha256']
    write(OUT/'archive_members.json',dict(status='PASS',count=len(members),members=members))
    write(OUT/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='feedback-forks-20261018-r6',title='Fixed-prefix feedback interventions: complete development',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
