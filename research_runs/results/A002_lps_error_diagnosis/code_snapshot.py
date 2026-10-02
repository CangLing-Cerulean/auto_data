"""Post-training LPS error decomposition; target states only label diagnostic groups."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,prepare,digest,write_json


@torch.no_grad()
def main():
    run=ROOT/'research_runs/results/A002_lps_error_diagnosis'
    if run.exists(): raise RuntimeError('Diagnosis already started; do not overwrite')
    config=ROOT/'research_runs/selection/protocol.json'
    p=json.loads(config.read_text()); manifest_path=ROOT/'data/metropt3_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    references=[]
    for path in sorted(run.parent.glob('S*/result.json')):
        r=json.loads(path.read_text())
        if r['identity']['recipe'] in ('uniform','month_matched_drop','protect_lps_matched','placebo_matched_swap'):
            references.append((path.parent,r))
    assert len(references)==12,'Wait for all three seeds of all four recipes'
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    remaining=p['session_gpu_wall_budget_seconds']-used
    if remaining<=0: raise RuntimeError('Shared budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase):
        write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('preparing')
    (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
    write_json(run/'identity.json',{'code_sha256':digest(Path(__file__)),'core_sha256':digest(ROOT/'scripts/run_research.py'),
        'protocol_sha256':digest(config),'manifest_sha256':digest(manifest_path)})
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    arrays,indices,_,audit=prepare(p,manifest)
    raw=np.loadtxt(ROOT/'data/metropt3_v1/validation.csv',delimiter=',',skiprows=1,usecols=13)
    assert np.isin(raw,[0,1]).all()
    prefix=np.r_[0,np.cumsum(raw==1)]; starts=indices['validation']
    seen=(prefix[starts+32]-prefix[starts])>0
    future=(prefix[starts+40]-prefix[starts+32])>0
    groups={f'input_any_1_{int(a)}_target_any_1_{int(b)}':(seen==a)&(future==b) for a in (False,True) for b in (False,True)}
    data=torch.from_numpy(arrays['validation']).cuda(); offsets=torch.arange(40,device='cuda')
    positions=torch.from_numpy(starts).cuda(); model=DLinear(p).cuda().eval(); outcomes={}
    for folder,reference in references:
        if time.monotonic()-begun>=remaining:
            status('budget_exhausted'); raise RuntimeError('Shared budget exhausted')
        model.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=True))
        sums={g:0. for g in groups}
        for lo in range(0,len(starts),p['evaluation_batch_size']):
            batch=data[positions[lo:lo+p['evaluation_batch_size'],None]+offsets]
            errors=(model(batch[:,:32])[:,:,11]-batch[:,32:,11]).double().square().mean(1).cpu().numpy()
            for g,mask in groups.items(): sums[g]+=float(errors[mask[lo:lo+len(errors)]].sum())
        overall=sum(sums.values())/len(starts)
        assert np.isclose(overall,reference['per_channel_NMSE'][11],rtol=1e-7,atol=1e-9)
        outcomes[folder.name]={'recipe':reference['identity']['recipe'],'seed':reference['identity']['seed'],
            'checkpoint_sha256':digest(folder/'best.pt'),'LPS_NMSE':overall,
            'groups':{g:{'windows':int(mask.sum()),'LPS_NMSE':sums[g]/int(mask.sum()) if mask.any() else None,
                'contribution_to_overall_LPS_NMSE':sums[g]/len(starts)} for g,mask in groups.items()}}
        status('evaluating')
    output={'scope':'full unchanged validation; target states used only for post-hoc error diagnosis, never selection',
        'identity':json.loads((run/'identity.json').read_text()),'validation_windows':len(starts),'results':outcomes}
    write_json(ROOT/'research_runs/selection/lps_error_diagnosis.json',output)
    status('completed')
    print(json.dumps({'runs':len(outcomes),'group_counts':{g:int(mask.sum()) for g,mask in groups.items()},
        'elapsed_seconds':time.monotonic()-begun}))


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
