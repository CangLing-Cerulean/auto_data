"""Train-only online loss/progress sampling, with frozen probability-shuffle control."""
import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,valid_starts,evaluate,write_json
from coverage_sampling import ROUND_LIMIT,used_seconds

FOLDER=ROOT/'research_runs/learning_sampling'


def distribution(score,cells):
    n=len(score); probability=np.full(n,.5/n,dtype=np.float64)
    for cell in np.unique(cells):
        members=cells==cell; size=int(members.sum()); total=score[members].sum()
        probability[members]+=.5*(size/n)*(score[members]/total if total>0 else np.full(size,1/size))
    probability/=probability.sum()
    return probability


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id',required=True)
    parser.add_argument('--origin',type=float,choices=[.6,.8],required=True)
    parser.add_argument('--policy',choices=['progress','hard','shuffle'],required=True)
    parser.add_argument('--reference'); args=parser.parse_args()
    assert Path(args.run_id).name==args.run_id
    run=ROOT/'research_runs/results'/args.run_id
    if (run/'result.json').exists(): raise RuntimeError('Cannot overwrite completed run')
    source=ROOT/'data/metropt3_v1/train.csv'; manifest_path=source.parent/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    formal=ROOT/'research_runs/selection/protocol.json'; p=json.loads(formal.read_text()); reference=None
    if args.policy=='shuffle':
        assert args.reference and Path(args.reference).name==args.reference
        reference=run.parent/args.reference; ref_result=json.loads((reference/'result.json').read_text())
        assert ref_result['identity']['origin']==args.origin and ref_result['identity']['policy'] in ('progress','hard')
    else: assert args.reference is None
    identity={'policy':args.policy,'origin':args.origin,'seed':2026,'preprocessing':'baseline','dataset_id':manifest['dataset_id'],
        'manifest_sha256':digest(manifest_path),'formal_protocol_sha256':digest(formal),'code_sha256':digest(Path(__file__)),
        'core_sha256':digest(ROOT/'scripts/run_research.py'),'study_protocol_sha256':digest(FOLDER/'PROTOCOL.md'),
        'group_fit_sha256':digest(ROOT/'research_runs/cluster_selection/fit.json'),'group_labels_sha256':digest(ROOT/'research_runs/cluster_selection/window_groups.npz'),
        'reference':args.reference,'reference_result_sha256':digest(reference/'result.json') if reference else None,
        'round_total_limit_seconds':ROUND_LIMIT,'budget_helper_sha256':digest(ROOT/'scripts/coverage_sampling.py')}
    run.mkdir(exist_ok=True)
    if (run/'identity.json').exists(): assert json.loads((run/'identity.json').read_text())==identity
    else:
        write_json(run/'identity.json',identity)
        for f,name in [(Path(__file__),'code_snapshot.py'),(ROOT/'scripts/run_research.py','core_snapshot.py'),(formal,'protocol.json'),(FOLDER/'PROTOCOL.md','study_protocol.md'),(ROOT/'scripts/coverage_sampling.py','budget_helper.py')]: (run/name).write_bytes(f.read_bytes())
    remaining=ROUND_LIMIT-used_seconds()-20
    if remaining<=0: raise RuntimeError('Shared exploration budget exhausted')
    old=json.loads((run/'status.json').read_text()).get('elapsed_seconds',0) if (run/'status.json').exists() else 0
    begun=time.monotonic()
    def status(phase,step=0): write_json(run/'status.json',{'phase':phase,'step':step,'elapsed_seconds':old+time.monotonic()-begun})
    def check(step):
        if time.monotonic()-begun>=remaining: raise TimeoutError(f'Budget exhausted at step {step}')
    status('preparing')
    try:
        assert digest(source)==next(s['sha256'] for s in manifest['splits'] if s['name']=='train')
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        with source.open(newline='') as f:
            reader=csv.reader(f); next(reader); times=np.array([row[1] for row in reader],dtype='datetime64[s]')
        cut=int(len(raw)*args.origin); end=int(round(len(raw)*(args.origin+.2)))
        all_starts=valid_starts(times,40,30); starts=all_starts[all_starts+40<=cut]; future=all_starts[(all_starts>=cut)&(all_starts+40<=end)]
        assert starts.max()+40<=future.min()
        with np.load(ROOT/'research_runs/selection/window_table.npz') as z: assert np.array_equal(all_starts,z['start_row_zero_based'])
        with np.load(ROOT/'research_runs/cluster_selection/window_groups.npz') as z:
            positions=np.searchsorted(z['start_row'],starts); assert np.array_equal(z['start_row'][positions],starts); group=z['group'][positions]
        months=times[starts+31].astype('datetime64[M]').astype(str); unique_months,month_id=np.unique(months,return_inverse=True)
        cells=group*len(unique_months)+month_id
        mean=raw[:cut].mean(0); std=raw[:cut].std(0); scale=np.where(std==0,1,std)
        normalized=((raw[:end]-mean)/scale).astype(np.float32)
        write_json(run/'data_audit.json',{'train_raw_rows':[0,cut],'probe_raw_rows':[cut,end],'train_windows':len(starts),'probe_windows':len(future),'mean':mean.tolist(),'scale':scale.tolist(),'source_train_sha256':digest(source)})
        torch.set_num_threads(4); torch.manual_seed(2026); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()})
        data=torch.from_numpy(normalized).cuda(); positions=torch.from_numpy(starts).cuda(); offsets=torch.arange(40,device='cuda'); n=len(starts)
        model=DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters())==7920
        optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay']); rng=torch.Generator(device='cuda').manual_seed(3026)
        counts=torch.zeros(n,device='cuda',dtype=torch.int64); history=[]; phases=[]; step0=0; previous=None
        if (run/'latest.pt').exists():
            saved=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); rng.set_state(saved['sampling_rng'].cpu())
            counts=saved['draw_counts']; step0=saved['step']; history=saved['history']; phases=saved['phases']; previous=saved['previous_loss']
        for start in range(step0,3000,500):
            check(start); scored=time.monotonic(); current=np.zeros(n,dtype=np.float32); q=np.full(n,1/n,dtype=np.float64)
            if reference is not None:
                with np.load(reference/f'phase_{start:04d}.npz') as z: q=z['probability'].copy(); current=z['current_loss'].copy()
                if start>=1000:
                    shuffle=np.random.default_rng(20261005+start)
                    for cell in np.unique(cells):
                        idx=np.flatnonzero(cells==cell); q[idx]=shuffle.permutation(q[idx])
            elif start>0:
                model.eval(); torch.save(model.state_dict(),run/f'scoring_{start:04d}.pt')
                with torch.no_grad():
                    for lo in range(0,n,2048):
                        check(start); b=data[positions[lo:lo+2048,None]+offsets]
                        current[lo:lo+len(b)]=(model(b[:,:32])-b[:,32:]).square().mean((1,2)).cpu().numpy()
                if start>=1000:
                    score=np.sqrt(current.astype(np.float64)) if args.policy=='hard' else np.sqrt(np.maximum(previous.astype(np.float64)-current,0))
                    q=distribution(score,cells)
                previous=current
            assert np.isfinite(q).all() and q.min()>=(.5/n)*(1-1e-12) and abs(q.sum()-1)<1e-12
            for cell in np.unique(cells): assert abs(q[cells==cell].sum()-(cells==cell).mean())<1e-12
            cdf=torch.from_numpy(np.cumsum(q)).cuda(); cdf[-1]=1
            scoring_seconds=time.monotonic()-scored; phase_counts=torch.zeros_like(counts); losses=[]; model.train()
            for step in range(start+1,start+501):
                check(step); u=torch.rand(256,device='cuda',dtype=torch.float64,generator=rng); selected=torch.searchsorted(cdf,u,right=True)
                phase_counts.scatter_add_(0,selected,torch.ones_like(selected)); batch=data[positions[selected,None]+offsets]
                optimizer.zero_grad(set_to_none=True); loss=(model(batch[:,:32])-batch[:,32:]).square().mean()
                assert torch.isfinite(loss); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            counts+=phase_counts
            asset=run/f'phase_{start:04d}.npz'
            np.savez_compressed(asset,current_loss=current,probability=q,draw_count=phase_counts.cpu().numpy())
            phases.append({'start_step':start,'asset_sha256':digest(asset),'scoring_seconds':scoring_seconds,'scoring_forward_presentations':n if reference is None and start>0 else 0,'min_relative_to_uniform':float(q.min()*n),'max_relative_to_uniform':float(q.max()*n)})
            history.append({'step':start+500,'training_loss':float(np.mean(losses))})
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state(),'draw_counts':counts,'step':start+500,'history':history,'phases':phases,'previous_loss':previous},run/'latest.tmp'); (run/'latest.tmp').replace(run/'latest.pt')
            status('training',start+500); print(json.dumps(history[-1]),flush=True)
        measured=evaluate(model,data,torch.from_numpy(future).cuda(),{'all':np.ones(len(future),dtype=bool)},p,'baseline')
        assert counts.sum().item()==768000
        np.savez_compressed(run/'windows.npz',train_start=starts,probe_start=future,group=group,months=months,cell=cells,draw_count=counts.cpu().numpy())
        torch.save(model.state_dict(),run/'final.pt')
        write_json(run/'sampling_audit.json',{'phases':phases,'retained_windows':n,'removed_windows':0,'total_draws':768000,'stratum_mass_matches_uniform':True,'scoring_forward_presentations':sum(v['scoring_forward_presentations'] for v in phases),'scoring_seconds':sum(v['scoring_seconds'] for v in phases),'windows_sha256':digest(run/'windows.npz')})
        result={'run_id':args.run_id,'identity':identity,'future':measured,'history':history,'steps_completed':3000,'samples_presented':768000,'sensor_columns':manifest['sensor_columns'],'elapsed_seconds':old+time.monotonic()-begun}
        write_json(run/'result.json',result); status('completed',3000); print(json.dumps(measured['groups']['all']),flush=True)
    except BaseException:
        status('failed'); raise


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); main()
