"""Package reproducible development records, excluding credentials and private transport."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import sha,write

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_conditional_diagnosis_r1'
archive_dir=STAGE/'archives';archive_dir.mkdir()
archive=archive_dir/'conditional_diagnosis_development_r1.tar.gz'
files=[p for p in STAGE.rglob('*') if p.is_file() and not {'private','archives','control','prepared','__pycache__'}.intersection(p.relative_to(STAGE).parts)]
names=['conditional_diagnosis.py','check_conditional_diagnosis.py','diagnosis_pool_features.py','build_conditional_development.py','extract_ett_diagnosis.py','manage_conditional_diagnosis.py','replay_conditional_diagnosis.py','generate_conditional_signals.py','validate_generated_signals.py','replay_generated_signals.py','package_conditional_development.py']
files += [ROOT/'scripts'/name for name in names]
members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
with tarfile.open(archive,'x:gz') as tar:
    for p in sorted(files):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
with tarfile.open(archive) as tar:
    actual=tar.getmembers();assert len(actual)==len(members)
    for m in actual:
        assert m.isfile() and m.name in members and m.size==members[m.name]['bytes']
        with tar.extractfile(m) as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==members[m.name]['sha256']
write(STAGE/'archive_members.json',dict(status='PASS',members=members,count=len(members),raw_bytes=sum(m['bytes'] for m in members.values())))
manifest=dict(repository='CangLing-Cerulean/auto_data',tag='conditional-diagnosis-development-20261018-r1',title='Conditional diagnosis development: complete retrospective results',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))])
write(STAGE/'release_manifest.json',manifest)
print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))
