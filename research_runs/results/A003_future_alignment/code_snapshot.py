"""Train-only local gradient alignment from a past teacher to its future probe."""
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json


def main():
    out=ROOT/'research_runs/transfer'; run=ROOT/'research_runs/results/A003_future_alignment'
    if run.exists() or (out/'alignment_scores.npz').exists(): raise RuntimeError('Do not overwrite scoring')
    teacher=ROOT/'research_runs/results/T005_prefix80_full'
    p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())
    audit=json.loads((teacher/'data_audit.json').read_text())
    result=json.loads((teacher/'result.json').read_text())
    assert result['identity']['origin']==.8 and result['identity']['drop_block']==-1
    assert result['steps_completed']==3000
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    remaining=7200-used
    if remaining<=0: raise RuntimeError('Shared budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('preparing'); (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    source=ROOT/'data/metropt3_v1/train.csv'
    assert digest(source)==audit['source_train_sha256']
    raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17))
    mean=np.array(audit['mean']); scale=np.array(audit['scale']); cut=audit['train_raw_rows'][1]
    assert np.array_equal(raw[:cut].mean(0),mean)
    assert np.array_equal(np.where(raw[:cut].std(0)==0,1,raw[:cut].std(0)),scale)
    with np.load(teacher/'windows.npz') as data:
        history=data['train_start']; future=data['probe_start']
    assert history.max()+40<=future.min() and future.max()+40<=len(raw)
    with np.load(ROOT/'research_runs/selection/window_table.npz') as table: base_starts=table['start_row_zero_based']
    source_indices=np.searchsorted(base_starts,history)
    assert np.array_equal(base_starts[source_indices],history)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    values=torch.from_numpy(((raw-mean)/scale).astype(np.float32)).cuda()
    model=DLinear(p).cuda().eval(); model.load_state_dict(torch.load(teacher/'final.pt',map_location='cuda',weights_only=True))
    model.zero_grad(set_to_none=True); offsets=torch.arange(40,device='cuda'); batch_size=p['evaluation_batch_size']
    total_loss=0.
    for lo in range(0,len(future),batch_size):
        batch=values[torch.from_numpy(future[lo:lo+batch_size]).cuda()[:,None]+offsets]
        loss=(model(batch[:,:32])-batch[:,32:]).square().mean()
        factor=len(batch)/len(future); (loss*factor).backward(); total_loss+=float(loss.detach())*factor
    assert np.isclose(total_loss,result['future']['groups']['all']['NMSE'],rtol=1e-5)
    norm=torch.sqrt(sum(parameter.grad.double().square().sum() for parameter in model.parameters()))
    assert torch.isfinite(norm) and norm>0
    direction=DLinear(p).cuda().eval()
    with torch.no_grad():
        for target,parameter in zip(direction.parameters(),model.parameters()): target.copy_(parameter.grad/norm)
        scores=np.empty(len(history),dtype=np.float64)
        for lo in range(0,len(history),batch_size):
            if time.monotonic()-begun>=remaining:
                status('budget_exhausted'); raise RuntimeError('Shared budget exhausted')
            batch=values[torch.from_numpy(history[lo:lo+batch_size]).cuda()[:,None]+offsets]
            residual=(model(batch[:,:32])-batch[:,32:]).double()
            scores[lo:lo+len(batch)]=(2*residual*direction(batch[:,:32]).double()).mean((1,2)).cpu().numpy()
        # Real-data finite difference in double precision, independent of the scorer formula.
        sample=np.linspace(0,len(history)-1,64,dtype=int)
        batch=values[torch.from_numpy(history[sample]).cuda()[:,None]+offsets].double()
        anchor=DLinear(p).double().cuda().eval(); anchor.load_state_dict(model.state_dict())
        vector=DLinear(p).double().cuda().eval(); vector.load_state_dict(direction.state_dict())
        analytic=(2*(anchor(batch[:,:32])-batch[:,32:])*vector(batch[:,:32])).mean((1,2))
        epsilon=1e-4; losses=[]
        for sign in (1,-1):
            shifted=DLinear(p).double().cuda().eval(); shifted.load_state_dict(anchor.state_dict())
            for parameter,delta in zip(shifted.parameters(),vector.parameters()): parameter.add_(delta,alpha=sign*epsilon)
            losses.append((shifted(batch[:,:32])-batch[:,32:]).square().mean((1,2)))
        numerical=(losses[0]-losses[1])/(2*epsilon)
        torch.testing.assert_close(analytic,numerical,rtol=1e-6,atol=1e-8)
        check_error=float((analytic-numerical).abs().max())
        np.testing.assert_allclose(scores[sample],analytic.cpu().numpy(),rtol=1e-4,atol=1e-7)
    assert np.isfinite(scores).all()
    threshold=float(np.quantile(scores,.25)); selected=(scores<=threshold)&(scores<0)
    if not selected.any(): raise RuntimeError('No negative candidates; do not train')
    weights=np.ones(len(base_starts),dtype=np.float64); weights[source_indices[selected]]=.25
    with source.open(newline='') as handle:
        rows=csv.reader(handle); next(rows); months=np.array([row[1][:7] for row in rows])
    months=months[history+31]; shuffled=weights.copy(); rng=np.random.default_rng(20260929)
    for month in np.unique(months):
        members=source_indices[months==month]; shuffled[members]=rng.permutation(weights[members])
        assert np.sum(shuffled[members]==.25)==np.sum(weights[members]==.25)
    np.savez_compressed(out/'alignment_scores.npz',source_eligible_indices=source_indices,source_start_rows=history,
        score=scores,loss_weight=weights,control_loss_weight=shuffled)
    metadata={'scope':'train only; offline training-label-based scoring; no external validation/test',
        'teacher_run':teacher.name,'teacher_sha256':digest(teacher/'final.pt'),'teacher_identity_sha256':digest(teacher/'identity.json'),
        'source_train_sha256':digest(source),'window_table_sha256':digest(ROOT/'research_runs/selection/window_table.npz'),
        'code_sha256':digest(Path(__file__)),'core_sha256':digest(ROOT/'scripts/run_research.py'),
        'alignment_protocol_sha256':digest(out/'ALIGNMENT_PROTOCOL.md'),'source_rows':[0,cut],'future_rows':audit['probe_raw_rows'],
        'scored_windows':len(scores),'eligible_windows':len(base_starts),'negative_scores':int(np.sum(scores<0)),
        'quarter_threshold':threshold,'downweighted_windows':int(selected.sum()),'expected_loss_weight':float(weights.mean()),
        'gradient_norm':float(norm),'finite_difference_max_absolute_error':check_error,'shuffle_seed':20260929,
        'monthly_downweighted':{month:int(np.sum(selected&(months==month))) for month in np.unique(months)},
        'artifact_sha256':digest(out/'alignment_scores.npz')}
    write_json(out/'alignment_recipe.json',metadata); write_json(run/'identity.json',metadata)
    status('completed'); print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
