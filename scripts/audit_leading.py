"""Audit the frozen diagnostic, derivative inputs and paired training receipts."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json
from leading_sources import SourceDLinear,load_mapping


def main():
    run=ROOT/'research_runs/results/A010_leading_audit'
    if run.exists(): raise RuntimeError('Audit already exists')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('running')
    try:
        folder=ROOT/'research_runs/results'; mapping,diagnostic_hash=load_mapping()
        diagnostic=json.loads((folder/'A009_leading_sources/diagnostic.json').read_text())
        reconstructed=np.arange(15)
        for origin in [60,80]:
            with np.load(folder/f'A009_leading_sources/fold_{origin}.npz') as z:
                assert z['train_start'].max()+40<=z['probe_start'].min()
                lookup={tuple(pair):i for i,pair in enumerate(z['pairs'])}
                if origin==60:
                    for i in range(15):
                        j=min(range(15),key=lambda j:z['metrics'][lookup[i,j],0])
                        if z['metrics'][lookup[i,j],0]<.95*z['metrics'][lookup[i,i],0]: reconstructed[i]=j
                assert np.array_equal(z['mapping'],mapping) and np.array_equal(reconstructed,mapping)
                g,h,c=z['gram'],z['cross'],z['coefficient']
                assert np.isfinite(c).all()
                assert (np.linalg.norm(g@c-h,axis=(1,2))/np.maximum(np.linalg.norm(h,axis=(1,2)),1e-30)).max()<1e-6
        p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())
        torch.set_num_threads(4); torch.manual_seed(2026)
        core=DLinear(p); model=SourceDLinear(core,mapping)
        assert sum(v.numel() for v in model.parameters())==7920
        raw=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17),max_rows=256)
        x=torch.tensor(np.stack([raw[i:i+32] for i in range(32)]),dtype=torch.float32)
        transformed=x.clone()
        for i,j in enumerate(mapping):
            if i!=j: transformed[:,:,i]=x[:,:,j]-x[:,-1:,j]+x[:,-1:,i]
        torch.testing.assert_close(model(x),core(transformed),rtol=0,atol=0)
        identity=SourceDLinear(core,list(range(15)))
        torch.testing.assert_close(identity(x),core(x),rtol=0,atol=0)
        rows=json.loads((ROOT/'research_runs/representation/leading_progress.json').read_text())['rows']
        checks=[]
        for row in rows:
            current=folder/row['run_id']; base=folder/row['baseline']
            info=json.loads((current/'identity.json').read_text())
            assert info['source_diagnostic_sha256']==diagnostic_hash and info['source_mapping']==mapping
            assert digest(current/'source_helper.py')==info['source_helper_sha256']
            assert digest(current/'source_protocol.md')==info['source_protocol_sha256']
            assert digest(current/'code_snapshot.py')==info['code_sha256']
            if row['run_id'].startswith('T'):
                with np.load(current/'windows.npz') as a,np.load(base/'windows.npz') as b:
                    for key in ['train_start','probe_start','probability','draw_count']: assert np.array_equal(a[key],b[key]),key
                    assert a['draw_count'].sum()==768000
                assert json.loads((current/'data_audit.json').read_text())==json.loads((base/'data_audit.json').read_text())
            else:
                with np.load(current/'sampling_weights.npz') as a,np.load(base/'sampling_weights.npz') as b:
                    for key in ['sampling_weight','loss_weight','probability','draw_count']: assert np.array_equal(a[key],b[key]),key
                    assert a['draw_count'].sum()==768000
                assert json.loads((current/'data_audit.json').read_text())==json.loads((base/'data_audit.json').read_text())
            checks.append({'run_id':row['run_id'],'baseline':row['baseline'],'identical_actual_draw_counts':True,'identical_data_audit':True,'source_identity_verified':True})
        write_json(ROOT/'research_runs/representation/leading_verification.json',{'diagnostic_sha256':diagnostic_hash,'mapping_reproduced':True,'identity_transform_exact':True,'independent_transform_agrees':True,'parameter_count':7920,'checks':checks})
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes()); status('completed')
    except BaseException:
        status('failed'); raise


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); main()
