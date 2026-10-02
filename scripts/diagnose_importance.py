"""A004: training-only diagnostic before committing to adaptive sampling."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json
from importance_utils import gradient_norm,distribution,verify_norm


def main():
    run=ROOT/'research_runs/results/A004_importance_diagnostic'
    if run.exists(): raise RuntimeError('Diagnostic cannot overwrite an existing run')
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    if used>=7200: raise RuntimeError('Budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('preparing')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    manifest_path=ROOT/'data/metropt3_v1/manifest.json'; manifest=json.loads(manifest_path.read_text())
    source=manifest_path.parent/'train.csv'
    assert digest(source)==next(s['sha256'] for s in manifest['splits'] if s['name']=='train')
    p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())
    teacher=ROOT/'research_runs/results/S000_uniform_s2026'
    audit=json.loads((teacher/'data_audit.json').read_text())
    raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17))
    assert np.array_equal(raw.mean(0),audit['mean'])
    std=raw.std(0); assert np.array_equal(np.where(std==0,1,std),audit['scale'])
    values=torch.from_numpy(((raw-audit['mean'])/audit['scale']).astype(np.float32)).cuda()
    table=ROOT/'research_runs/selection/window_table.npz'
    with np.load(table) as saved: starts=saved['start_row_zero_based']
    model=DLinear(p).cuda()
    checkpoint=torch.load(teacher/'latest.pt',map_location='cuda',weights_only=False)
    assert checkpoint['step']==3000
    model.load_state_dict(checkpoint['model']); model.eval()
    position=torch.from_numpy(starts).cuda(); offsets=torch.arange(40,device='cuda')
    norms=[]
    for lo in range(0,len(starts),4096):
        if time.monotonic()-begun>=7200-used: status('budget_exhausted'); raise RuntimeError('Budget exhausted')
        batch=values[position[lo:lo+4096,None]+offsets]
        norms.append(gradient_norm(model,batch[:,:32],batch[:,32:]))
    norm=torch.cat(norms); q=distribution(norm); w=1/(len(q)*q)
    chosen=torch.cat((torch.linspace(0,len(starts)-1,32,device='cuda').long(),torch.topk(norm,16).indices))
    batch=values[position[chosen,None]+offsets]
    relative=verify_norm(model,batch[:,:32],batch[:,32:])
    uniform_second=norm.square().mean().item()
    selected_second=(norm.square()/(len(norm)**2*q)).sum().item()
    assert torch.allclose(q*w,torch.full_like(q,1/len(q)),rtol=1e-12,atol=0)
    np.savez_compressed(run/'scores.npz',start_row=starts,gradient_norm=norm.cpu().numpy(),probability=q.cpu().numpy(),loss_weight=w.cpu().numpy())
    report={'teacher':'S000_uniform_s2026/latest.pt','teacher_sha256':digest(teacher/'latest.pt'),
        'manifest_sha256':digest(manifest_path),'source_train_sha256':digest(source),'window_table_sha256':digest(table),
        'code_sha256':digest(Path(__file__)),'helper_sha256':digest(ROOT/'scripts/importance_utils.py'),
        'core_sha256':digest(ROOT/'scripts/run_research.py'),'research_sha256':digest(ROOT/'research_runs/importance/RESEARCH.md'),
        'windows':len(starts),'scoring_forward_presentations':len(starts),
        'verification_windows':len(chosen),'max_autograd_relative_error':relative,
        'uniform_gradient_second_moment':uniform_second,'corrected_gradient_second_moment':selected_second,
        'second_moment_reduction_pct':100*(1-selected_second/uniform_second),
        'gradient_norm_quantiles':torch.quantile(norm,torch.tensor([0,.5,.9,.99,1.],device='cuda',dtype=torch.float64)).cpu().tolist(),
        'top_one_percent_probability_mass':q[torch.topk(norm,int(np.ceil(len(norm)*.01))).indices].sum().item(),
        'loss_weight_min':w.min().item(),'loss_weight_max':w.max().item(),'scores_sha256':digest(run/'scores.npz')}
    write_json(run/'diagnostic.json',report)
    (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (run/'helper_snapshot.py').write_bytes((ROOT/'scripts/importance_utils.py').read_bytes())
    (run/'research_snapshot.md').write_bytes((ROOT/'research_runs/importance/RESEARCH.md').read_bytes())
    status('completed'); print(json.dumps(report),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
