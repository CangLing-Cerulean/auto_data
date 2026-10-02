"""A006: replay actual dynamic draws and independently check score gradients."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json
from importance_utils import gradient_norm,verify_norm


def main():
    ledger=ROOT/'research_runs/results/A006_importance_audit'
    if ledger.exists(): raise RuntimeError('Audit already recorded; do not reset cost')
    ledger.mkdir(); begun=time.monotonic()
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in ledger.parent.glob('*/status.json'))
    if used>=7200: raise RuntimeError('Budget exhausted')
    def status(phase): write_json(ledger/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('auditing')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'research_runs/importance'
    records=[]
    for path in sorted(folder.glob('batch*.json')): records.extend(json.loads(path.read_text()))
    p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())
    raw=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17))
    base_audit=json.loads((ROOT/'research_runs/results/S000_uniform_s2026/data_audit.json').read_text())
    assert np.array_equal(raw.mean(0),base_audit['mean'])
    std=raw.std(0); assert np.array_equal(np.where(std==0,1,std),base_audit['scale'])
    data=torch.from_numpy(((raw-base_audit['mean'])/base_audit['scale']).astype(np.float32)).cuda()
    with np.load(ROOT/'research_runs/selection/window_table.npz') as saved: starts=saved['start_row_zero_based']
    n=len(starts); offsets=torch.arange(40,device='cuda'); verified=[]
    for item in records:
        run=ledger.parent/item['id']; identity=json.loads((run/'identity.json').read_text())
        assert json.loads((run/'status.json').read_text())['phase']=='completed'
        audit=json.loads((run/'sampling_audit.json').read_text())
        assert json.loads((run/'data_audit.json').read_text())==base_audit
        torch.manual_seed(item['seed']); model=DLinear(p).cuda()
        initialized=torch.load(run/'initial.pt',map_location='cuda',weights_only=True)
        assert all(torch.equal(v,initialized[k]) for k,v in model.state_dict().items())
        rng=torch.Generator(device='cuda').manual_seed(item['seed']+1000)
        total=np.zeros(n,dtype=np.int64); phase_checks=[]
        for phase in audit['phases']:
            step=phase['start_step']; path=run/f'phase_{step:04d}.npz'
            assert digest(path)==phase['asset_sha256']
            with np.load(path) as saved:
                q=saved['probability']; norm=saved['gradient_norm']; weights=saved['loss_weight']; draws=saved['draw_count']
            assert len(q)==n and np.all(q>0) and abs(q.sum()-1)<1e-12
            expected=np.full(n,1/n) if step==0 else .5/n+.5*norm/norm.sum()
            assert np.allclose(q,expected,rtol=1e-12,atol=0)
            expected_weight=(1/(n*q)).astype(np.float32) if item['recipe']=='adaptive_is' else np.ones(n,dtype=np.float32)
            assert np.array_equal(weights,expected_weight)
            if item['recipe']=='adaptive_is': assert np.max(np.abs(n*q*weights-1))<1e-6
            cdf=torch.from_numpy(q).cuda().cumsum(0); cdf[-1]=1
            count=torch.zeros(n,dtype=torch.int64,device='cuda')
            for _ in range(500):
                u=torch.rand(p['batch_size'],device='cuda',dtype=torch.float64,generator=rng)
                selected=torch.searchsorted(cdf,u,right=True)
                count.scatter_add_(0,selected,torch.ones_like(selected))
            assert np.array_equal(count.cpu().numpy(),draws) and draws.sum()==128000
            total+=draws
            if item['recipe']=='replay_uncorrected':
                with np.load(ledger.parent/item['reference']/path.name) as ref:
                    assert np.array_equal(q,ref['probability']) and np.array_equal(draws,ref['draw_count'])
            error=None
            if step and item['recipe']=='adaptive_is':
                checkpoint=run/f'scoring_{step:04d}.pt'
                assert digest(checkpoint)==phase['score_checkpoint_sha256']
                model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
                chosen=np.r_[np.linspace(0,n-1,16,dtype=np.int64),np.argsort(norm)[-8:]]
                batch=data[torch.from_numpy(starts[chosen]).cuda()[:,None]+offsets]
                calculated=gradient_norm(model,batch[:,:32],batch[:,32:]).cpu().numpy()
                assert np.allclose(calculated,norm[chosen],rtol=1e-5,atol=1e-7)
                error=verify_norm(model,batch[:,:32],batch[:,32:])
            phase_checks.append({'start_step':step,'draws_replayed':True,'positive_support':True,
                'loss_weights_verified':True,'score_autograd_max_relative_error':error})
        with np.load(run/'window_draws.npz') as saved:
            assert np.array_equal(saved['start_row'],starts) and np.array_equal(saved['draw_count'],total)
        latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        assert torch.equal(latest['sampling_rng'],rng.get_state().cpu())
        assert np.array_equal(latest['draw_counts'].numpy(),total) and total.sum()==768000
        baseline={2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}[item['seed']]
        base=json.loads((ledger.parent/baseline/'result.json').read_text())
        result=json.loads((run/'result.json').read_text())
        assert result['history'][0]==base['history'][0],'Uniform warmup did not exactly reproduce baseline'
        verified.append({'run':item['id'],'initialization_reproduced':True,'uniform_warmup_identical':True,
            'all_768000_draws_replayed':True,'phase_checks':phase_checks})
    write_json(folder/'verification.json',verified)
    (ledger/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    status('completed'); print('PASS:',len(verified),'importance runs; exact draw replay, initialization, warmup, weights and gradients')


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
