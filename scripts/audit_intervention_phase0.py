"""Verify saved pair evidence and exercise both adapters on real historical facts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--input',required=True)
    a=p.parse_args();root=Path(a.root);run=root/'runtime_r1';pairs=run/'paired_evaluation'
    out=run/'acceptance'
    if out.exists():raise RuntimeError('Existing acceptance will not be overwritten')
    out.mkdir();started=time.monotonic();receipt=read(pairs/'receipt.json')
    if sha(pairs/'window_identity.npz')!=receipt['window_identity_sha256']:raise RuntimeError('window identity changed')
    with np.load(pairs/'window_identity.npz',allow_pickle=False) as values:
        assert len(values['start_row'])==300535
        assert len(np.unique(values['start_row']))==300535
        assert np.all(np.diff(values['start_row'])>0)
        assert np.array_equal(values['target_start_row'],values['start_row']+32)
    for item in receipt['identities']:
        path=pairs/item['error_file']
        if sha(path)!=item['error_sha256']:raise RuntimeError('error artifact changed')
        with np.load(path,allow_pickle=False) as values:
            error=values['error'];assert error.shape==(300535,8,15)
            assert np.isfinite(error).all()
            assert np.isclose(np.mean(error.astype(np.float64)**2),item['NMSE'],rtol=1e-12,atol=1e-12)
    responses=read(pairs/'response_collection.json')
    for row in responses:
        r=row['response'];assert r['global']['elements']==r['target']['elements']+r['complement']['elements']
        for metric in ('NMSE','NMAE','signed_bias'):
            reconstructed=sum(r[k]['elements']*r[k]['delta'][metric] for k in ('target','complement'))/r['global']['elements']
            assert np.isclose(reconstructed,r['global']['delta'][metric],rtol=1e-10,atol=1e-12)
    driver=root/'scripts/intervention_prototype.py'
    for variant in ('strong_prompt','full'):
        subprocess.run([sys.executable,'-B',str(driver),'prepare','--variant',variant,'--input',a.input,
                        '--output',str(out/(variant+'_context.json'))],check=True)
    baseline=read(out/'strong_prompt_context.json');full=read(out/'full_context.json')
    assert baseline['shared_information_sha256']==full['shared_information_sha256']
    assert baseline['facts']==full['facts'] and baseline['history']==full['history'] and baseline['tools']==full['tools']
    reviewed=[]
    for row in responses:
        if row['action']!='source_sign_flip':continue
        name=str(row['seed']);payload=dict(response=row['response'],facts=dict(parent_id='parent',response_visible=True))
        write(out/(name+'_input.json'),payload)
        for variant in ('strong_prompt','full'):
            subprocess.run([sys.executable,'-B',str(driver),'review','--variant',variant,
                            '--input',str(out/(name+'_input.json')),'--output',str(out/(name+'_'+variant+'.json'))],check=True)
        b=read(out/(name+'_strong_prompt.json'));f=read(out/(name+'_full.json'))
        assert b['raw_response']==f['raw_response']
        assert f['dual_update']['recipe']=='retain_parent'
        assert f['dual_update']['hypothesis']=='retrospective_only'
        reviewed.append(dict(seed=row['seed'],result=f['dual_update']))
    result=dict(status='passed',error_tensors_verified=9,paired_responses_verified=6,
                shared_information_sha256=baseline['shared_information_sha256'],reviews=reviewed,
                new_training_count=0,external_API_calls=0,test_access=False,
                scope='real-history adapter regression, not blind LLM method evaluation',
                process_seconds=time.monotonic()-started,source_sha256=sha(__file__))
    write(run/'acceptance.json',result)
    # All contents are derived errors, public code or bounded acceptance records.
    archive=root/'intervention-phase0-20261012-runtime-r1.tar'
    if archive.exists():raise RuntimeError('Archive already exists')
    members=[]
    with tarfile.open(archive,'w') as tar:
        for path in sorted(run.rglob('*')):
            if path.is_file():
                name=str(path.relative_to(root));tar.add(path,arcname=name,recursive=False)
                members.append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path)))
        for name in read(root/'upload_manifest.json')['files']:
            path=root/name;tar.add(path,arcname='source/'+name,recursive=False)
            members.append(dict(path='source/'+name,bytes=path.stat().st_size,sha256=sha(path)))
        tar.add(Path(__file__),arcname='source/scripts/audit_intervention_phase0.py',recursive=False)
        members.append(dict(path='source/scripts/audit_intervention_phase0.py',bytes=Path(__file__).stat().st_size,sha256=sha(__file__)))
    write(root/'archive_receipt.json',dict(filename=archive.name,bytes=archive.stat().st_size,
                                         sha256=sha(archive),members=members))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
