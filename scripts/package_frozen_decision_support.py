"""Archive the complete fixed snapshot inputs used by the read-only check."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_decision_support_r7'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    assert result['status']=='PASS'
    files=[p for p in OUT.iterdir() if p.is_file()]
    files += [ROOT/'scripts/check_frozen_decision_support.py',Path(__file__)]
    files += [ROOT/name for name in result['input_sha256'] if name.startswith('research_runs/')]
    files += [ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_measurement_generation_r4/replay/summary.json']
    files=sorted(set(files))
    members={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    folder=OUT/'archives';folder.mkdir();archive=folder/'frozen_decision_support_r7.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(members)
        for item in tar.getmembers():
            with tar.extractfile(item) as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==members[item.name]['sha256']
    write(OUT/'archive_members.json',dict(status='PASS',count=len(members),members=members))
    write(OUT/'release_manifest.json',dict(repository='CangLing-Cerulean/auto_data',tag='decision-support-20261018-r7',title='Frozen decision support and reaffirmed research scope',assets=[dict(name=archive.name,path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=sha(archive))]))
    print(dict(status='PASS',members=len(members),bytes=archive.stat().st_size,sha256=sha(archive)))

if __name__=='__main__':main()
