"""A005: same DLinear function class, analytic fit as diagnostic ONLY."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,prepare,evaluate,write_json


def main():
    run=ROOT/'research_runs/results/A005_linear_fit_diagnostic'
    if run.exists(): raise RuntimeError('Cannot overwrite diagnostic')
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    if used>=7200: raise RuntimeError('Budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('preparing')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    manifest_path=ROOT/'data/metropt3_v1/manifest.json'; manifest=json.loads(manifest_path.read_text())
    formal=ROOT/'research_runs/selection/protocol.json'; p=json.loads(formal.read_text())
    arrays,indices,groups,audit=prepare(p,manifest)
    table=ROOT/'research_runs/selection/window_table.npz'
    with np.load(table) as saved: assert np.array_equal(saved['start_row_zero_based'],indices['train'])
    data={s:torch.from_numpy(a).cuda() for s,a in arrays.items()}
    starts={s:torch.from_numpy(a).cuda() for s,a in indices.items()}
    gram=torch.zeros((15,33,33),dtype=torch.float64,device='cuda')
    cross=torch.zeros((15,33,8),dtype=torch.float64,device='cuda')
    offsets=torch.arange(40,device='cuda')
    for lo in range(0,len(starts['train']),4096):
        if time.monotonic()-begun>=7200-used: status('budget_exhausted'); raise RuntimeError('Budget exhausted')
        b=data['train'][starts['train'][lo:lo+4096,None]+offsets].double()
        x=b[:,:32].permute(2,0,1); y=b[:,32:].permute(2,0,1)
        x=torch.cat((x,torch.ones((*x.shape[:2],1),device='cuda',dtype=torch.float64)),dim=2)
        gram+=x.transpose(1,2).bmm(x); cross+=x.transpose(1,2).bmm(y)
    g=gram.cpu().numpy()/len(starts['train']); h=cross.cpu().numpy()/len(starts['train'])
    eigen,vectors=np.linalg.eigh(g); keep=eigen>eigen[:,-1,None]*1e-10
    inverse=np.zeros_like(eigen); inverse[keep]=1/eigen[keep]
    coefficient=(vectors*inverse[:,None,:])@vectors.transpose(0,2,1)@h
    residual=np.linalg.norm(g@coefficient-h,axis=(1,2))/np.linalg.norm(h,axis=(1,2))
    assert np.isfinite(coefficient).all() and residual.max()<1e-6
    model=DLinear(p).cuda()
    with torch.no_grad():
        for channel in range(15):
            weight=torch.from_numpy(coefficient[channel,:32].T).to(device='cuda',dtype=torch.float32)
            model.seasonal[channel].weight.copy_(weight); model.trend[channel].weight.copy_(weight)
            model.seasonal[channel].bias.copy_(torch.from_numpy(coefficient[channel,32]).to(device='cuda',dtype=torch.float32))
            model.trend[channel].bias.zero_()
        probe=data['train'][starts['train'][::max(1,len(starts['train'])//256),None]+offsets][:,:32]
        direct=torch.einsum('blc,clh->bhc',probe.double(),torch.from_numpy(coefficient[:,:32]).cuda())+torch.from_numpy(coefficient[:,32].T).cuda()
        difference=(model(probe).double()-direct).abs().max().item()
        assert difference<1e-3
    train_groups={'all':np.ones(len(indices['train']),dtype=bool)}
    fitted_train=evaluate(model,data['train'],starts['train'],train_groups,p,'baseline')
    fitted_val=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
    torch.save(model.state_dict(),run/'diagnostic_model.pt')
    reference=ROOT/'research_runs/results/S000_uniform_s2026/best.pt'
    model.load_state_dict(torch.load(reference,weights_only=True))
    base_train=evaluate(model,data['train'],starts['train'],train_groups,p,'baseline')
    base_val=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
    np.savez_compressed(run/'normal_equations.npz',gram=g,cross=h,coefficient=coefficient,eigenvalues=eigen,retained_eigenvalues=keep)
    report={'scope':'diagnostic only; analytic optimizer differs from formal Adam protocol',
        'manifest_sha256':digest(manifest_path),'protocol_sha256':digest(formal),'core_sha256':digest(ROOT/'scripts/run_research.py'),
        'code_sha256':digest(Path(__file__)),'diagnostic_protocol_sha256':digest(ROOT/'research_runs/importance/FIT_DIAGNOSTIC.md'),
        'window_table_sha256':digest(table),'reference_sha256':digest(reference),'data_audit':audit,
        'rank':keep.sum(1).tolist(),'normal_equation_relative_residual':residual.tolist(),
        'affine_equivalence_max_abs_error':difference,'baseline_train':base_train,'baseline_validation':base_val,
        'analytic_train':fitted_train,'analytic_validation':fitted_val,'sensor_columns':manifest['sensor_columns']}
    write_json(run/'diagnostic.json',report)
    (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (run/'diagnostic_protocol.md').write_bytes((ROOT/'research_runs/importance/FIT_DIAGNOSTIC.md').read_bytes())
    status('completed')
    print(json.dumps({k:report[k]['groups']['all'] for k in ['baseline_train','baseline_validation','analytic_train','analytic_validation']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
