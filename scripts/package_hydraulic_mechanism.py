"""Check saved diagnostic terms independently, then archive only owned analysis assets."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,obj):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,indent=2,allow_nan=False)
        stream.write('\n')


parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
args=parser.parse_args()
root=Path(args.root)
result=json.loads((root/'decomposition.json').read_text(encoding='utf-8-sig'))
assert result['status']=='PASS' and len(result['records'])==24
assert sha(root/'plan.json')==result['source_plan_sha256']
assert sha(root/'decompose_hydraulic_response.py')==result['script_sha256']
keys={(r['checkpoint'],r['split'],r['action'],r['seed']) for r in result['records']}
assert keys=={(c,s,a,i) for c in ('best','last') for s in ('train','validation') for a in ('P1','CP1') for i in (2026101501,2026101502,2026101503)}
for record in result['records']:
    path=root/record['asset']
    assert sha(path)==result['assets_sha256'][path.name]
    with np.load(path,allow_pickle=False) as saved:
        terms=saved['terms']
        assert terms.shape==(record['windows'],8,6) and np.isfinite(terms).all()
        assert saved['window_start'].shape==saved['cycle'].shape==(record['windows'],)
        np.testing.assert_allclose(terms[:,:,0]+np.sum(terms[:,:,1:5],axis=2),terms[:,:,5],rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(terms.mean(axis=0),record['channel_terms'])
        average=terms.mean(axis=(0,1))
        baseline=average[0]
        gains=[-100*(average[1]+average[2])/baseline,-100*(average[3]+average[4])/baseline,100*(1-average[5]/baseline)]
        expected=[record[k] for k in ('fixed_processing_gain_percent','retraining_adaptation_gain_percent','total_gain_percent')]
        np.testing.assert_allclose(gains,expected,rtol=1e-10,atol=1e-10)
write(root/'terms_audit.json',dict(status='PASS',verified_records=24,new_trainings=0,test_access=False,
    decomposition_sha256=sha(root/'decomposition.json'),auditor_sha256=sha(Path(__file__))))
names=list(result['assets_sha256'])+['decompose_hydraulic_response.py','package_hydraulic_mechanism.py','PROTOCOL.md','plan.json','source_sync_receipt.json','decomposition.json','terms_audit.json','run.log']
output=root/'hydraulic_mechanism_20261013.tar.gz'
with tarfile.open(output,'x:gz') as archive:
    for name in sorted(names):
        archive.add(root/name,arcname=name,recursive=False)
hashes={name:sha(root/name) for name in names}
with tarfile.open(output,'r:gz') as archive:
    assert set(archive.getnames())==set(names)
    for name,expected in hashes.items():
        assert hashlib.file_digest(archive.extractfile(name),'sha256').hexdigest()==expected
receipt=dict(status='PASS',file=output.name,bytes=output.stat().st_size,sha256=sha(output),members_sha256=hashes)
write(root/'archive_receipt.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k!='members_sha256'}),flush=True)
