"""Replay a registered run and preserve a separately costed audit record."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json
from importance_utils import gradient_norm,verify_norm
ACTIVE_LEDGER=None


def main():
    global ACTIVE_LEDGER
    parser=argparse.ArgumentParser()
    parser.add_argument('--audit-id',required=True); parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    assert Path(args.audit_id).name==args.audit_id and Path(args.run_id).name==args.run_id
    ledger=ROOT/'research_runs/results'/args.audit_id
    if ledger.exists(): raise RuntimeError('Audit already recorded; do not reset cost')
    ledger.mkdir(); begun=time.monotonic()
    ACTIVE_LEDGER=ledger
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in ledger.parent.glob('*/status.json'))
    if used>=7200: raise RuntimeError('Budget exhausted')
    def status(phase): write_json(ledger/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('auditing')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    folder=ROOT/'research_runs/importance'
    records=[]
    for path in sorted(folder.glob('batch*.json')): records.extend(json.loads(path.read_text()))
    records=[r for r in records if r['id']==args.run_id]
    assert len(records)==1
    p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())
    raw=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17))
    base_audit=json.loads((ROOT/'research_runs/results/S000_uniform_s2026/data_audit.json').read_text())
    assert np.array_equal(raw.mean(0),base_audit['mean'])
    std=raw.std(0); assert np.array_equal(np.where(std==0,1,std),base_audit['scale'])
    data=torch.from_numpy(((raw-base_audit['mean'])/base_audit['scale']).astype(np.float32)).cuda()
    with np.load(ROOT/'research_runs/selection/window_table.npz') as saved: starts=saved['start_row_zero_based']
    n=len(starts); offsets=torch.arange(40,device='cuda')
    verified=json.loads((folder/'verification.json').read_text()) if (folder/'verification.json').exists() else []
    verified=[r for r in verified if r['run']!=args.run_id]
    for item in records:
        run=ledger.parent/item['id']; identity=json.loads((run/'identity.json').read_text())
        assert json.loads((run/'status.json').read_text())['phase']=='completed'
        audit=json.loads((run/'sampling_audit.json').read_text())
        assert json.loads((run/'data_audit.json').read_text())==base_audit
        torch.manual_seed(item['seed']); model=DLinear(p).cuda()
        initialized=torch.load(run/'initial.pt',map_location='cuda',weights_only=True)
        assert all(torch.equal(v,initialized[k]) for k,v in model.state_dict().items())
        rng=torch.Generator(device='cuda').manual_seed(item['seed']+1000)
        total=np.zeros(n,dtype=np.int64); phase_checks=[]; warmup_different_draws=0
        for phase in audit['phases']:
            if time.monotonic()-begun>=7200-used: raise RuntimeError('Budget exhausted')
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
            cdf=(torch.from_numpy(np.cumsum(q)).cuda() if identity.get('cdf_backend')=='numpy'
                else torch.from_numpy(q).cuda().cumsum(0))
            cdf[-1]=1
            if step==0:
                baseline_cdf=torch.from_numpy(np.cumsum(np.ones(n,dtype=np.float64)/n)).cuda()
                baseline_cdf[-1]=1
            count=torch.zeros(n,dtype=torch.int64,device='cuda')
            for _ in range(500):
                u=torch.rand(p['batch_size'],device='cuda',dtype=torch.float64,generator=rng)
                selected=torch.searchsorted(cdf,u,right=True)
                if step==0:
                    warmup_different_draws+=int(torch.count_nonzero(selected!=torch.searchsorted(baseline_cdf,u,right=True)))
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
        identical=result['history'][0]==base['history'][0]
        if not identical: assert warmup_different_draws>0,'Warmup mismatch unexplained by CDF rounding'
        verified.append({'run':item['id'],'initialization_reproduced':True,'uniform_warmup_identical':identical,
            'warmup_different_draws_vs_numpy_cdf':warmup_different_draws,'warmup_total_draws':128000,
            'warmup_NMSE_difference':result['history'][0]['validation']['NMSE']-base['history'][0]['validation']['NMSE'],
            'numerical_pairing_note':('NumPy CDF matches baseline; uniform warmup verified exactly.' if identical else
                'CUDA versus NumPy CDF rounding; not a bitwise uniform warmup match.'),
            'all_768000_draws_replayed':True,'phase_checks':phase_checks})
    write_json(folder/'verification.json',verified)
    # The separate representation trial uses the original static sampler.
    representation_path=ROOT/'research_runs/representation/verification.json'
    representation=json.loads(representation_path.read_text()) if representation_path.exists() else []
    for run in sorted(ledger.parent.glob('R*')):
        if not (run/'result.json').exists(): continue
        if run.name in {r['run'] for r in representation}: continue
        from channel_rotation import fit_rotation,RotatedDLinear
        identity=json.loads((run/'identity.json').read_text())
        basis,eigen,covariance,rotation_audit=fit_rotation(((raw-base_audit['mean'])/base_audit['scale']).astype(np.float32))
        with np.load(run/'rotation.npz') as saved:
            assert np.array_equal(basis,saved['basis']) and np.array_equal(covariance,saved['covariance'])
            assert np.array_equal(eigen,saved['eigenvalues'])
        baseline={2026:'S000_uniform_s2026',2027:'S005_uniform_s2027',2028:'S010_uniform_s2028'}[identity['seed']]
        with np.load(run/'sampling_weights.npz') as a, np.load(ledger.parent/baseline/'sampling_weights.npz') as b:
            for key in ('sampling_weight','loss_weight','probability','draw_count'): assert np.array_equal(a[key],b[key])
        assert json.loads((run/'data_audit.json').read_text())==base_audit
        wrapped=RotatedDLinear(DLinear(p),basis)
        wrapped.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True))
        assert sum(v.numel() for v in wrapped.parameters())==7920
        assert torch.equal(wrapped.rotation,torch.from_numpy(basis.astype(np.float32)))
        probe=data[torch.from_numpy(starts[::max(1,n//64)]).cuda()[:,None]+offsets].cpu()
        with torch.no_grad():
            x,y=probe[:,:32],probe[:,32:]
            predicted_z=wrapped.core(x@wrapped.rotation)
            physical_loss=(wrapped(x)-y).square().mean().item()
            rotated_loss=(predicted_z-y@wrapped.rotation).square().mean().item()
        assert abs(physical_loss-rotated_loss)<1e-6
        representation.append({'run':run.name,'train_fitted_basis_reproduced':True,
            'same_original_sampling_draws_and_weights':True,'trainable_parameters':7920,
            'MSE_rotation_equivalence_abs_error':abs(physical_loss-rotated_loss),**rotation_audit})
    write_json(ROOT/'research_runs/representation/verification.json',representation)
    (ledger/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    status('completed'); print('PASS:',len(verified),'importance runs; actual draw replay, initialization, weights and gradients; warmup CDF differences quantified')


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    started=time.monotonic()
    try: main()
    except Exception:
        if ACTIVE_LEDGER is not None:
            write_json(ACTIVE_LEDGER/'status.json',{'phase':'audit_failed','elapsed_seconds':time.monotonic()-started})
            (ACTIVE_LEDGER/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
        raise
