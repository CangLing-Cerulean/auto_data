"""Preserve all r4 calls, engineering failures and policy trajectories in one archive."""
import hashlib
from pathlib import Path
import tarfile
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_measurement_generation_r4'

def main():
    assert read(STAGE/'replay_audit.json')['status']=='PASS'
    out=STAGE/'archives';out.mkdir();archive=out/'window_measurement_generation_r4.tar.gz'
    files=[p for p in STAGE.rglob('*') if p.is_file() and not {'archives','private','__pycache__'}.intersection(p.relative_to(STAGE).parts)]
    files+=[ROOT/'scripts'/name for name in ('window_measurement_dsl.py','prepare_window_signal_inputs.py','replay_window_signals.py','generate_window_signals.py','summarize_window_generation.py','summarize_window_conditional_scores.py','package_window_generation.py')]
    members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(files):tar.add(path,arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for m in tar.getmembers():
            assert m.isfile() and m.size==members[m.name]['bytes']
            with tar.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==members[m.name]['sha256']
    write(STAGE/'archive_members.json',dict(status='PASS',members=members,count=len(members)))
    write(STAGE/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='window-measurement-generation-20261018-r4',title='Executable window measurement generation: complete development',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
