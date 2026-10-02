"""Historical deletion intervention: train.csv only, disjoint forward probe."""
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--origin',type=float,choices=[.6,.8],required=True)
    parser.add_argument('--drop-block',type=int,choices=[-1,0,1,2,3],required=True)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--thin',choices=['none','disjoint','matched_random'],default='none')
    parser.add_argument('--state-balance',action='store_true')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id: raise ValueError('Invalid ID')
    protocol=ROOT/'research_runs/selection/protocol.json'
    p=json.loads(protocol.read_text()); manifest_path=ROOT/'data/metropt3_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text()); source=manifest_path.parent/'train.csv'
    run=ROOT/'research_runs/results'/args.run_id; run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists(): raise RuntimeError('Completed run cannot be overwritten')
    identity={'manifest_sha256':digest(manifest_path),'protocol_sha256':digest(protocol),
        'diagnostic_protocol_sha256':digest(ROOT/'research_runs/transfer/PROTOCOL.md'),
        'code_sha256':digest(Path(__file__)),'core_sha256':digest(ROOT/'scripts/run_research.py'),
        'origin':args.origin,'drop_block':args.drop_block,'seed':args.seed,'thin':args.thin,
        'thinning_protocol_sha256':digest(ROOT/'research_runs/transfer/REDUNDANCY_PROTOCOL.md') if args.thin!='none' else None,
        'scope':'train-only temporal diagnosis; prefix normalization; fixed final checkpoint'}
    if args.state_balance:
        assert args.thin=='none' and args.drop_block==-1
        identity.update(state_balance=True,state_protocol_sha256=digest(ROOT/'research_runs/transfer/STATE_BALANCE_PROTOCOL.md'),
            state_helper_sha256=digest(ROOT/'scripts/state_balance.py'))
    if (run/'identity.json').exists():
        assert json.loads((run/'identity.json').read_text())==identity,'Resume identity mismatch'
    else:
        write_json(run/'identity.json',identity)
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (run/'core_snapshot.py').write_bytes((ROOT/'scripts/run_research.py').read_bytes())
        (run/'protocol.json').write_bytes(protocol.read_bytes())
        (run/'diagnostic_protocol.md').write_bytes((ROOT/'research_runs/transfer/PROTOCOL.md').read_bytes())
        if args.thin!='none': (run/'thinning_protocol.md').write_bytes((ROOT/'research_runs/transfer/REDUNDANCY_PROTOCOL.md').read_bytes())
        if args.state_balance:
            (run/'state_protocol.md').write_bytes((ROOT/'research_runs/transfer/STATE_BALANCE_PROTOCOL.md').read_bytes())
            (run/'state_helper.py').write_bytes((ROOT/'scripts/state_balance.py').read_bytes())
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    old=json.loads((run/'status.json').read_text()).get('elapsed_seconds',0) if (run/'status.json').exists() else 0
    remaining=p['session_gpu_wall_budget_seconds']-used
    if remaining<=0: raise RuntimeError('Shared budget exhausted')
    begun=time.monotonic()
    def status(phase,step): write_json(run/'status.json',{'phase':phase,'step':step,'elapsed_seconds':old+time.monotonic()-begun})
    status('preparing',0)
    expected=next(s for s in manifest['splits'] if s['name']=='train')
    assert digest(source)==expected['sha256']
    raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
    with source.open(newline='') as handle:
        reader=csv.reader(handle); next(reader)
        times=np.array([row[1] for row in reader],dtype='datetime64[s]')
    assert len(raw)==expected['rows'] and np.isfinite(raw).all()
    cut=int(len(raw)*args.origin); end=min(len(raw),int(round(len(raw)*(args.origin+.2))))
    all_starts=valid_starts(times,40,p['max_gap_seconds'])
    with np.load(ROOT/'research_runs/selection/window_table.npz') as frozen:
        assert np.array_equal(all_starts,frozen['start_row_zero_based']),'Base window eligibility changed'
    train_starts=all_starts[all_starts+40<=cut]
    probe_starts=all_starts[(all_starts>=cut)&(all_starts+40<=end)]
    assert len(train_starts)>0 and len(probe_starts)>0
    assert train_starts.max()+40<=probe_starts.min()
    block=np.empty(len(train_starts),dtype=np.int8)
    for i,members in enumerate(np.array_split(np.arange(len(train_starts)),4)): block[members]=i
    retained=np.ones(len(train_starts),dtype=bool) if args.drop_block<0 else block!=args.drop_block
    if args.thin!='none':
        assert args.drop_block==-1
        retained[:]=False
        rng_members=np.random.default_rng(20260930)
        begin=0
        while begin<len(train_starts):
            stop=int(np.searchsorted(train_starts,train_starts[begin]+40))
            chosen=begin if args.thin=='disjoint' else int(rng_members.integers(begin,stop))
            retained[chosen]=True
            begin=stop
        if args.thin=='disjoint': assert np.all(np.diff(train_starts[retained])>=40)
    probability=retained.astype(np.float64)/retained.sum()
    channel_weights=None
    if args.state_balance:
        from state_balance import build_weights
        channel_weights,states,state_audit=build_weights(raw[:cut],train_starts)
        np.savez_compressed(run/'state_weights.npz',start_row=train_starts,loss_weight=channel_weights,state=states)
        state_audit['asset_sha256']=digest(run/'state_weights.npz')
        write_json(run/'state_audit.json',state_audit)
    mean=raw[:cut].mean(0); std=raw[:cut].std(0); scale=np.where(std==0,1,std)
    normalized=((raw[:end]-mean)/scale).astype(np.float32)
    audit={'dataset_id':manifest['dataset_id'],'source_train_sha256':digest(source),'train_raw_rows':[0,cut],
        'probe_raw_rows':[cut,end],'train_windows':len(train_starts),'retained_windows':int(retained.sum()),
        'probe_windows':len(probe_starts),'mean':mean.tolist(),'scale':scale.tolist(),
        'last_train_timestamp':str(times[cut-1]),'first_probe_timestamp':str(times[cut]),
        'last_probe_timestamp':str(times[end-1]),'source_blocks':[
            {'block':i,'windows':int(np.sum(block==i)),'source_start_row':int(train_starts[block==i].min()),
             'source_end_row_exclusive':int(train_starts[block==i].max()+40)} for i in range(4)]}
    write_json(run/'data_audit.json',audit)
    torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()})
    data=torch.from_numpy(normalized).cuda(); positions=torch.from_numpy(train_starts).cuda()
    if channel_weights is not None: channel_weights=torch.from_numpy(channel_weights.astype(np.float32)).cuda()
    cdf=torch.from_numpy(np.cumsum(probability)).cuda(); cdf[-1]=1
    offsets=torch.arange(40,device='cuda'); counts=torch.zeros(len(train_starts),device='cuda',dtype=torch.int64)
    model=DLinear(p).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    rng=torch.Generator(device='cuda').manual_seed(args.seed+1000)
    step0=0; history=[]
    if (run/'latest.pt').exists():
        checkpoint=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        rng.set_state(checkpoint['sampling_rng'].cpu()); counts=checkpoint['draw_counts']
        step0=checkpoint['step']; history=checkpoint['history']
    losses=[]
    for step in range(step0+1,p['steps']+1):
        if time.monotonic()-begun>=remaining:
            status('budget_exhausted',step-1); raise RuntimeError('Shared budget exhausted')
        u=torch.rand(p['batch_size'],device='cuda',dtype=torch.float64,generator=rng)
        chosen=torch.searchsorted(cdf,u,right=True); counts.scatter_add_(0,chosen,torch.ones_like(chosen))
        batch=data[positions[chosen,None]+offsets]
        optimizer.zero_grad(set_to_none=True)
        squared=(model(batch[:,:32])-batch[:,32:]).square()
        loss=squared.mean() if channel_weights is None else (squared.mean(1)*channel_weights[chosen]).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        if step%500==0:
            history.append({'step':step,'training_loss':float(np.mean(losses))}); losses=[]
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state(),
                'draw_counts':counts,'step':step,'history':history},run/'latest.tmp')
            (run/'latest.tmp').replace(run/'latest.pt'); status('training',step)
            print(json.dumps(history[-1]),flush=True)
    model.eval()
    prefix=evaluate(model,data,positions,{'all':np.ones(len(train_starts),dtype=bool),'fitted_members':retained},p,'baseline')
    future=evaluate(model,data,torch.from_numpy(probe_starts).cuda(),{'all':np.ones(len(probe_starts),dtype=bool)},p,'baseline')
    draws=counts.cpu().numpy(); assert draws.sum()==768000 and not np.any(draws[~retained])
    np.savez_compressed(run/'windows.npz',train_start=train_starts,probe_start=probe_starts,block=block,probability=probability,draw_count=draws)
    torch.save(model.state_dict(),run/'final.pt')
    result={'run_id':args.run_id,'identity':identity,'steps_completed':p['steps'],'samples_presented':int(draws.sum()),
        'prefix':prefix,'future':future,'history':history,'windows_sha256':digest(run/'windows.npz'),
        'sensor_columns':manifest['sensor_columns'],'elapsed_seconds':old+time.monotonic()-begun}
    write_json(run/'result.json',result); status('completed',p['steps'])
    print(json.dumps({'completed':args.run_id,'future':future['groups']['all'],'fitted':prefix['groups']['fitted_members']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
