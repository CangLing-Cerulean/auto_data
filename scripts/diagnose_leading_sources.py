"""Train-only, frozen two-origin source-selection diagnostic; never a formal candidate."""
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, digest, valid_starts, write_json


def main():
    run=ROOT/'research_runs/results/A009_leading_sources'
    if run.exists(): raise RuntimeError('Immutable diagnostic already exists')
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    allowance=min(600,7200-used)
    if allowance<=0: raise RuntimeError('Budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    def check():
        if time.monotonic()-begun>allowance: raise RuntimeError('Diagnostic budget exhausted')
    status('preparing')
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        source=ROOT/'data/metropt3_v1/train.csv'; mp=source.parent/'manifest.json'
        manifest=json.loads(mp.read_text()); expected=next(s for s in manifest['splits'] if s['name']=='train')
        assert digest(source)==expected['sha256']
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        with source.open(newline='') as handle:
            reader=csv.reader(handle); next(reader)
            times=np.array([r[1] for r in reader],dtype='datetime64[s]')
        assert len(raw)==expected['rows'] and np.isfinite(raw).all()
        protocol=ROOT/'research_runs/representation/LEADING_PROTOCOL.md'
        formal=ROOT/'research_runs/selection/protocol.json'; p=json.loads(formal.read_text())
        starts=valid_starts(times,40,p['max_gap_seconds'])
        table=ROOT/'research_runs/selection/window_table.npz'
        with np.load(table) as saved: assert np.array_equal(starts,saved['start_row_zero_based'])
        reports=[]; mapping=np.arange(15)
        for fold,origin in enumerate([.6,.8]):
            cut=int(len(raw)*origin); end=int(round(len(raw)*(origin+.2)))
            train=starts[starts+40<=cut]; probe=starts[(starts>=cut)&(starts+40<=end)]
            assert train.max()+40<=probe.min()
            mean=raw[:cut].mean(0); std=raw[:cut].std(0); scale=np.where(std==0,1,std)
            data=torch.from_numpy(((raw[:end]-mean)/scale).astype(np.float32)).cuda().double()
            pairs=[(i,j) for i in range(15) for j in range(15)] if fold==0 else sorted(set([(i,i) for i in range(15)]+[(i,int(mapping[i])) for i in range(15)]))
            target=torch.tensor([i for i,j in pairs],device='cuda'); src=torch.tensor([j for i,j in pairs],device='cuda')
            same=target==src; count=len(pairs); offsets=torch.arange(40,device='cuda')
            def batches(ids):
                for lo in range(0,len(ids),1024):
                    check()
                    b=data[torch.from_numpy(ids[lo:lo+1024]).cuda()[:,None]+offsets]
                    own=b[:,:32,target].permute(2,0,1)
                    history=b[:,:32,src].permute(2,0,1)
                    x=history-history[:,:,-1:]+own[:,:,-1:]
                    x[same]=own[same]
                    assert torch.equal(x[same],own[same])
                    x=torch.cat([x,torch.ones((count,len(b),1),device='cuda',dtype=torch.float64)],2)
                    y=b[:,32:,target].permute(2,0,1)
                    yield x,y
            gram=torch.zeros((count,33,33),device='cuda',dtype=torch.float64); cross=torch.zeros((count,33,8),device='cuda',dtype=torch.float64)
            status(f'fit_{int(origin*100)}')
            for x,y in batches(train):
                gram+=x.transpose(1,2).bmm(x); cross+=x.transpose(1,2).bmm(y)
            g=gram.cpu().numpy()/len(train); h=cross.cpu().numpy()/len(train)
            eigen,v=np.linalg.eigh(g); keep=eigen>eigen[:,-1,None]*1e-10
            inv=np.zeros_like(eigen); inv[keep]=1/eigen[keep]
            coefficient=(v*inv[:,None,:])@v.transpose(0,2,1)@h
            residual=np.linalg.norm(g@coefficient-h,axis=(1,2))/np.maximum(np.linalg.norm(h,axis=(1,2)),1e-30)
            assert np.isfinite(coefficient).all() and residual.max()<1e-6
            coef=torch.from_numpy(coefficient).cuda(); sums=torch.zeros((count,2),device='cuda',dtype=torch.float64)
            status(f'evaluate_{int(origin*100)}')
            for x,y in batches(probe):
                error=x.bmm(coef)-y
                sums[:,0]+=error.square().sum((1,2)); sums[:,1]+=error.abs().sum((1,2))
            metrics=(sums/(len(probe)*8)).cpu().numpy()
            lookup={pair:k for k,pair in enumerate(pairs)}
            baseline=np.array([metrics[lookup[(i,i)]] for i in range(15)])
            if fold==0:
                for i in range(15):
                    best=min(range(15),key=lambda j:metrics[lookup[(i,j)],0])
                    if metrics[lookup[(i,best)],0]<baseline[i,0]*.95: mapping[i]=best
            selected=np.array([metrics[lookup[(i,int(mapping[i]))]] for i in range(15)])
            improvement=100*(1-selected[:,0].mean()/baseline[:,0].mean())
            switched=mapping!=np.arange(15)
            gate=bool(improvement>=1 and (fold==0 or (selected[:,1].mean()<=baseline[:,1].mean() and np.all(selected[switched,0]<baseline[switched,0]))))
            np.savez_compressed(run/f'fold_{int(origin*100)}.npz',train_start=train,probe_start=probe,mean=mean,scale=scale,pairs=np.array(pairs),gram=g,cross=h,coefficient=coefficient,metrics=metrics,mapping=mapping)
            reports.append({'origin':origin,'cut':cut,'end':end,'train_windows':len(train),'probe_windows':len(probe),'mapping':mapping.tolist(),'baseline_channels':baseline.tolist(),'selected_channels':selected.tolist(),'baseline_NMSE':baseline[:,0].mean(),'selected_NMSE':selected[:,0].mean(),'baseline_NMAE':baseline[:,1].mean(),'selected_NMAE':selected[:,1].mean(),'NMSE_reduction_pct':improvement,'gate_passed':gate,'max_normal_equation_residual':float(residual.max())})
            print(json.dumps(reports[-1]),flush=True)
        write_json(run/'diagnostic.json',{'scope':'train-only analytic input-source diagnostic, not formal Adam result','manifest_sha256':digest(mp),'train_sha256':digest(source),'protocol_sha256':digest(protocol),'formal_protocol_sha256':digest(formal),'code_sha256':digest(Path(__file__)),'core_sha256':digest(ROOT/'scripts/run_research.py'),'window_table_sha256':digest(table),'sensor_columns':manifest['sensor_columns'],'folds':reports,'gate_passed':all(r['gate_passed'] for r in reports)})
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes()); (run/'protocol.md').write_bytes(protocol.read_bytes())
        status('completed')
    except BaseException:
        status('failed'); raise


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
