"""Archive the complete strong-agent comparator after independent result checks."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_strong_agent_r5'

def main():
    assert read(STAGE/'result_audit.json')['status']=='PASS'
    out=STAGE/'archives';out.mkdir();archive=out/'strong_agent_r5.tar.gz'
    files=[p for p in STAGE.rglob('*') if p.is_file() and not {'archives','private','__pycache__'}.intersection(p.relative_to(STAGE).parts)]
    scripts=set(read(STAGE/'execution_lock.json')['code_sha256'])|{'summarize_strong_agent.py','plot_strong_agent.py','package_strong_agent.py'}
    files+=[ROOT/'scripts'/name for name in sorted(scripts)]
    paper=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_measurement_generation_r4/paper'
    files+=[paper/'main.tex',paper/'images/strong_agent_budgets.pdf']
    members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for m in tar.getmembers():
            assert m.isfile() and m.size==members[m.name]['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==members[m.name]['sha256']
    write(STAGE/'archive_members.json',dict(status='PASS',members=members,count=len(members)))
    write(STAGE/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='strong-agent-20261018-r5',title='Strong adaptive agent: fixed development comparator',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
