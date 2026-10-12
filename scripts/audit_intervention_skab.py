"""Read-only scientific acceptance of the frozen SKAB pilot assets on server."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
import torch

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);a=parser.parse_args()
    assert os.name=='posix';root=Path(a.root);start=time.monotonic()
    receipt=read(root/'prepared/receipt.json');protocol=read(root/'protocol.json')
    assert sha(root/'prepared/arrays.npz')==receipt['array_sha256']
    assert sha(root/'run_intervention_skab_pilot.py')==protocol['runner_sha256']
    with np.load(root/'prepared/arrays.npz',allow_pickle=False) as f: arrays={k:f[k] for k in f.files}
    assert not np.any(arrays['target']&arrays['control'])
    for s in np.unique(arrays['strata']):
        m=arrays['strata']==s;assert arrays['target'][m].sum()==arrays['control'][m].sum()
    results=[];checks=[]
    for folder in sorted((root/'runs').iterdir()):
        status=read(folder/'status.json');assert status['status']=='completed' and status['steps_completed']==3000
        result=read(folder/'result.json');results.append(result)
        assert result['prepared_sha256']==receipt['array_sha256']
        assert result['protocol_sha256']==sha(root/'protocol.json')
        assert result['source_sha256']==protocol['runner_sha256']
        for name,identity in result['assets'].items(): assert sha(folder/name)==identity
        with np.load(folder/'best_errors.npz',allow_pickle=False) as f: error=f['error'].astype(np.float64)
        assert error.shape==(8982,8,8) and np.isfinite(error).all()
        for label,mask in [('global',np.ones(len(error),dtype=bool)),('target',arrays['development_target']),('complement',~arrays['development_target'])]:
            values=error[mask];actual=dict(NMSE=float(np.mean(values**2)),NMAE=float(np.mean(abs(values))),signed_bias=float(values.mean()))
            for key,value in actual.items(): assert abs(value-result['metrics'][label][key])<1e-12
        for key in ('NMSE','NMAE','signed_bias'):
            weighted=sum(result['metrics'][s][key]*result['metrics'][s]['elements'] for s in ('target','complement'))/result['metrics']['global']['elements']
            assert abs(weighted-result['metrics']['global'][key])<1e-12
        with np.load(folder/'sampling.npz',allow_pickle=False) as f:
            prob=f['probability'];weight=f['loss_weight'];counts=f['draw_count']
            assert len(prob)==31788 and (prob>0).all() and abs(prob.sum()-1)<1e-12
            assert abs(np.dot(prob,weight)-1)<1e-12 and counts.sum()==3000*256
            assert np.array_equal(f['train_window_start'],arrays['train_windows'][:,0])
            definition=protocol['actions'][result['action']];expected_p=np.ones(len(prob));expected_w=np.ones(len(prob))
            if result['action']!='B0':
                mask=arrays[definition['mask']]
                if definition['family']=='sampling': expected_p[mask]=definition['multiplier']
                else: expected_w[mask]=definition['multiplier']
            expected_p/=expected_p.sum();expected_w/=np.dot(expected_p,expected_w)
            assert np.array_equal(prob,expected_p) and np.array_equal(weight,expected_w)
        checks.append(dict(run=folder.name,assets_verified=True,metrics_reproduced=True,sampling_verified=True))
    for seed in sorted(set(r['seed'] for r in results)):
        paired=[r for r in results if r['seed']==seed];base=next(r for r in paired if r['action']=='B0')
        assert all(r['initialization_sha256']==base['initialization_sha256'] for r in paired)
        for r in paired:
            if protocol['actions'][r['action']]['family']!='sampling':
                assert r['probability_sha256']==base['probability_sha256'] and r['draw_count_sha256']==base['draw_count_sha256']
    out=dict(status='accepted',checks=checks,results=results,new_trainings=len(results),training_seconds=sum(r['process_seconds'] for r in results),audit_seconds=time.monotonic()-start,test_access=False,
             runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0)))
    destination=root/'acceptance.json';assert not destination.exists()
    destination.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('checks','results')}))
if __name__=='__main__':main()
