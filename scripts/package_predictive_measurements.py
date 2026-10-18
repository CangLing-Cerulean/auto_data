"""Package r2 predictions and r3 parent tables without private transport files."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGES=[BASE/'20261018_predictive_support_r2',BASE/'20261018_parent_measurements_r3']
OUT=STAGES[1]

def main():
    assert read(STAGES[0]/'audit.json')['status']=='PASS'
    assert read(OUT/'audit.json')['status']=='PASS_WITH_FINDING'
    assert read(OUT/'correction_audit.json')['status']=='PASS'
    archive_dir=OUT/'archives';archive_dir.mkdir()
    archive=archive_dir/'predictive_support_and_parent_measurements.tar.gz'
    files=[f for p in STAGES for f in p.rglob('*') if f.is_file() and not {'private','archives','__pycache__'}.intersection(f.relative_to(p).parts)]
    files += [ROOT/'scripts'/s for s in ('conditional_predictive_support.py','parent_measurement_tables.py','extract_ett_parent_measurements.py','manage_parent_measurements.py','correct_parent_frequency.py','package_predictive_measurements.py')]
    members={f.relative_to(ROOT).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(files)}
    with tarfile.open(archive,'x:gz') as tar:
        for f in sorted(files):tar.add(f,arcname=f.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for m in tar.getmembers():
            assert m.isfile() and m.size==members[m.name]['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==members[m.name]['sha256']
    write(OUT/'archive_members.json',dict(status='PASS',members=members,count=len(members)))
    write(OUT/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='predictive-support-measurements-20261018-r2-r3',title='Predictive support control and parent measurement interface',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
