"""Create-only GPU worker for the prospective, new-source cold-start trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from coldstart_model import DLinear, optimizer
from coldstart_expressions import measurement, recipe_weights, _Interpreter, Value, ExpressionError
from coldstart_reference import predict

SEEDS = (18431, 18432, 18433)
LENGTH, HORIZON, BATCH, STEPS = 32, 8, 256, 3000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def position(n):
    return np.linspace(0., 1., n, dtype=np.float64)


def windows(rows):
    return np.lib.stride_tricks.sliding_window_view(rows, LENGTH, axis=0).transpose(0, 2, 1)[:len(rows)-LENGTH-HORIZON+1]


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()


def load(root):
    receipt = read(root/'input_receipt.json')
    assert sha(root/'ABC.npz') == receipt['arrays_sha256']
    with np.load(root/'ABC.npz', allow_pickle=False) as data:
        rows = {part: data[part].copy() for part in 'ABC'}
    assert [v.shape for v in rows.values()] == [(11520,370),(2880,370),(2880,370)]
    assert all(np.isfinite(v).all() for v in rows.values())
    return rows


def tensors(rows):
    return {part: torch.as_tensor(value, device='cuda') for part,value in rows.items()}


def batch(values, starts):
    index = torch.as_tensor(starts, device='cuda', dtype=torch.int64)
    return (values[index[:,None]+torch.arange(LENGTH,device='cuda')],
            values[index[:,None]+LENGTH+torch.arange(HORIZON,device='cuda')])


def errors(model, rows):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(rows)-LENGTH-HORIZON+1,BATCH):
            x,y = batch(rows,np.arange(start,min(start+BATCH,len(rows)-LENGTH-HORIZON+1)))
            values.append((model(x)-y).cpu().numpy())
    result = np.concatenate(values)
    assert np.isfinite(result).all()
    return result


def metrics(error):
    d = error.astype(np.float64)
    return dict(mse=float(np.mean(d*d)),mae=float(np.mean(np.abs(d))),
                residual_sha256=digest(error),shape=list(error.shape))


def checkpoint_model(path):
    model = DLinear(370).cuda()
    model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True))
    return model


def best_model(folder):
    record = read(folder/'result.json')
    return checkpoint_model(folder/f"step_{record['best_step']:04d}.pt")


def train(root, name, mode, weights, seed, gpu):
    folder = root/'runs'/name/str(seed)
    folder.mkdir(parents=True)
    start = time.perf_counter()
    write(folder/'started.json',dict(seed=seed,mode=mode,weights_sha256=digest(weights),started=time.time()))
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model = DLinear(370).cuda(); opt = optimizer(model)
    torch.save(model.state_dict(),folder/'initial.pt')
    rng = np.random.default_rng(seed+100000)
    cdf = np.cumsum(weights,dtype=np.float64); cdf /= cdf[-1]; cdf[-1] = 1.
    n = len(weights); grid = []; best = float('inf'); best_step = None
    for step in range(1,STEPS+1):
        u = rng.random(BATCH)
        ids = np.searchsorted(cdf,u,side='right') if mode=='S' else (u*n).astype(np.int64)
        x,y = batch(gpu['A'],ids)
        model.train(); opt.zero_grad(set_to_none=True)
        losses = ((model(x)-y)**2).mean((1,2))
        if mode=='W': losses = losses*torch.as_tensor(weights[ids],device='cuda',dtype=losses.dtype)
        loss = losses.mean()
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
        if step%500 == 0:
            result = metrics(errors(model,gpu['B']))
            grid.append(dict(step=step,**result))
            torch.save(model.state_dict(),folder/f'step_{step:04d}.pt')
            if result['mse'] < best: best,best_step = result['mse'],step
    torch.save(opt.state_dict(),folder/'last_adam.pt')
    chosen = checkpoint_model(folder/f'step_{best_step:04d}.pt')
    c = metrics(errors(chosen,gpu['C']))
    torch.cuda.synchronize()
    record = dict(status='completed',seed=seed,mode=mode,steps=STEPS,best_step=best_step,
                  B_grid=grid,C=c,seconds=time.perf_counter()-start,weights_sha256=digest(weights))
    record['assets'] = {p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in folder.glob('*.pt')}
    write(folder/'result.json',record)
    return record


def parent(root):
    started = time.perf_counter(); rows=load(root); gpu=tensors(rows)
    weights=np.ones(len(windows(rows['A'])),dtype=np.float64)
    records=[train(root,'PARENT','W',weights,seed,gpu) for seed in SEEDS]
    be=[errors(best_model(root/'runs/PARENT'/str(seed)),gpu['B']).astype(np.float64) for seed in SEEDS]
    b = np.mean([e*e for e in be],axis=0)
    facts=dict(channels=370,channel_indices=list(range(370)),A_rows=11520,A_windows=len(weights),
               B_windows=len(b),C_windows=2841,length=LENGTH,horizon=HORIZON,
               A_raw_mean=read(root/'data_summary.json')['A_raw_mean'],
               A_raw_std=read(root/'data_summary.json')['A_raw_std'],
               A_zero_fraction=read(root/'data_summary.json')['A_zero_fraction'],
               B_input_mean=rows['B'].mean(0,dtype=np.float64).tolist(),
               B_input_std=rows['B'].std(0,dtype=np.float64).tolist(),
               parent_B_mse_by_channel=b.mean((0,1)).tolist(),
               parent_B_mse_by_horizon=b.mean((0,2)).tolist(),
               parents=[dict(seed=r['seed'],B_mse=min(g['mse'] for g in r['B_grid']),C=r['C'],seconds=r['seconds']) for r in records])
    write(root/'parent_facts.json',facts)
    write(root/'parent_receipt.json',dict(status='PASS',new_full_trainings=3,seconds=time.perf_counter()-started))
    return dict(status='PASS',facts=facts)


def registry_values(rows, registry):
    outputs={part:{} for part in rows}; parameters={}
    for entry in registry:
        name,expr=entry['id'],entry['expression']
        assert name in {f'q{i}' for i in range(1,9)} and name not in parameters
        raw={}
        for part,data in rows.items():
            x=windows(data); pos=position(len(x))
            raw[part]=np.concatenate([measurement(expr,x[start:start+128],pos[start:start+128]) for start in range(0,len(x),128)])
        with np.errstate(over='ignore',invalid='ignore'):
            mean=float(raw['A'].mean()); std=float(raw['A'].std())
        if not np.isfinite([mean,std]).all():
            raise ExpressionError('measurement A normalization statistics must be finite')
        scale=std if std>0 else 1.
        parameters[name]=dict(mean=mean,std=std,scale=scale,expression=expr)
        for part in rows:
            with np.errstate(over='ignore',invalid='ignore'):
                normalized=(raw[part]-mean)/scale
            if not np.isfinite(normalized).all():
                raise ExpressionError('normalized measurement must be finite in each authorized partition')
            outputs[part][name]=normalized
    return outputs,parameters


def describe(values):
    return dict(mean=float(np.mean(values)),std=float(np.std(values)),
                quantiles=np.quantile(values,[0,.1,.25,.5,.75,.9,1]).tolist())


def measure(root, request):
    start=time.perf_counter(); rows=load(root)
    # Expression defects are scientific invalid attempts, not infrastructure retries.
    try: q,parameters=registry_values({p:rows[p] for p in 'AB'},request['registry'])
    except ExpressionError as exc: return dict(status='invalid_measurement',error=str(exc),seconds=time.perf_counter()-start)
    gpu=tensors({'B':rows['B']})
    loss=np.mean([np.mean(errors(best_model(root/'runs/PARENT'/str(seed)),gpu['B']).astype(np.float64)**2,axis=(1,2)) for seed in SEEDS],axis=0)
    facts={}
    for name in q['A']:
        v=q['B'][name]; bins=np.quantile(v,[.25,.5,.75]); groups=np.searchsorted(bins,v,side='right')
        corr=float(np.corrcoef(v,loss)[0,1]) if np.std(v)>0 and np.std(loss)>0 else None
        facts[name]=dict(A=describe(q['A'][name]),B=describe(v),parent_B_error_correlation=corr,
                         parent_B_error_quartiles=[dict(count=int(np.sum(groups==i)),mse=float(np.mean(loss[groups==i])) if np.any(groups==i) else None) for i in range(4)])
    return dict(status='PASS',measurements=facts,fit_A=parameters,seconds=time.perf_counter()-start)


def target_mask(spec, q, n):
    variables={name:Value(v,('window',)) for name,v in q.items()}
    variables['position']=Value(position(n),('window',))
    out=_Interpreter(variables,True).run(spec['window_expression'])
    assert out.axes in {(),('window',)}
    selected=np.full(n,out.data.item()) if not out.axes else out.data
    if not np.isfinite(selected).all() or not np.isin(selected,[0,1]).all():
        raise ExpressionError('target expression must be a comparison or 0/1 indicator')
    channels=spec['channels']; horizons=spec['horizons']
    if not (channels and horizons and len(set(channels))==len(channels) and len(set(horizons))==len(horizons)
            and all(type(i)==int and 0<=i<370 for i in channels) and all(type(i)==int and 0<=i<8 for i in horizons)):
        raise ExpressionError('target channels/horizons must be unique nonempty in-range index lists')
    cm=np.isin(np.arange(370),channels); hm=np.isin(np.arange(8),horizons)
    mask=selected.astype(bool)[:,None,None]&hm[None,:,None]&cm[None,None,:]
    if not 0<int(mask.sum())<mask.size: raise ExpressionError('target and complement must both be nonempty')
    return mask


def prequery(root, request, rows, q, weights, stats):
    gpu=tensors({'B':rows['B']}); mask=target_mask(request['target'],q['B'],len(windows(rows['B'])))
    contrasts=[]
    for seed in SEEDS:
        e=errors(best_model(root/'runs/PARENT'/str(seed)),gpu['B']).astype(np.float64)**2
        contrasts.append((float(e[mask].mean())-float(e[~mask].mean()))/float(e.mean()))
    x=[1.,stats['nonzero']/stats['n'],stats['ess']/stats['n'],float(request['mode']=='W'),float(np.tanh(np.mean(contrasts))),float(mask.mean())]
    return dict(x=x,parent_B_contrast_by_seed=contrasts,p_B=float(mask.mean()),weight_stats=stats,
                weights_sha256=digest(weights))


def query(root, request):
    start=time.perf_counter(); rows=load(root)
    folder=root/'queries'/request['arm']/f"{request['round']:02d}"; folder.mkdir(parents=True)
    write(folder/'request.json',request)
    try:
        assert request['mode'] in ('S','W')
        q,parameters=registry_values(rows,request['registry'])
        weights,stats=recipe_weights(request['weight_expression'],q['A'],position(len(windows(rows['A']))))
        cmask=target_mask(request['target'],q['C'],len(windows(rows['C'])))
        before=prequery(root,request,rows,q,weights,stats)
    except (ExpressionError,AssertionError) as exc:
        result=dict(status='invalid_recipe',error=str(exc),seconds=time.perf_counter()-start)
        write(folder/'result.json',result); return result
    identity=hashlib.sha256(request['mode'].encode()+np.ascontiguousarray(weights).tobytes()).hexdigest()
    own=[]
    for p in (root/'queries'/request['arm']).glob('*/prequery.json'):
        own.append(read(p)['recipe_identity'])
    if identity in own:
        result=dict(status='duplicate_own_recipe',recipe_identity=identity,seconds=time.perf_counter()-start)
        write(folder/'result.json',result); return result
    before.update(recipe_identity=identity,fit_A=parameters,p_C=float(cmask.mean()),target_sha256=digest(cmask))
    before['reference_forecast']=predict(request['reference_history'],before['x'])
    # Hash and target are sealed before any child is run or a cached child response read.
    write(folder/'prequery.json',before)
    gpu=tensors(rows); name='R_'+identity; records=[]; reused=(root/'runs'/name).exists()
    for seed in SEEDS:
        if reused:
            record=read(root/'runs'/name/str(seed)/'result.json'); assert record['status']=='completed'
        else: record=train(root,name,request['mode'],weights,seed,gpu)
        records.append(record)
    responses=[]
    for seed,record in zip(SEEDS,records):
        pe=errors(best_model(root/'runs/PARENT'/str(seed)),gpu['C'])
        ce=errors(best_model(root/'runs'/name/str(seed)),gpu['C'])
        assert digest(ce)==record['C']['residual_sha256']
        pp=pe.astype(np.float64)**2; cc=ce.astype(np.float64)**2; denom=float(pp.mean())
        dt=float((pp[cmask]-cc[cmask]).mean()/denom); dc=float((pp[~cmask]-cc[~cmask]).mean()/denom)
        y=float((pp.mean()-cc.mean())/denom); z=dt-dc; p=float(cmask.mean())
        assert abs(y-(p*dt+(1-p)*dc))<1e-12
        responses.append(dict(seed=seed,Y=y,Z=z,target_gain=dt,complement_gain=dc,
                              NMAE_gain=float(1-np.abs(ce.astype(np.float64)).mean()/np.abs(pe.astype(np.float64)).mean()),
                              parent_residual_sha256=digest(pe),child_residual_sha256=digest(ce)))
    result=dict(status='completed',recipe_identity=identity,run=name,prequery=before,responses=responses,
                Y=float(np.mean([r['Y'] for r in responses])),Z=float(np.mean([r['Z'] for r in responses])),
                NMAE_gain=float(np.mean([r['NMAE_gain'] for r in responses])),reused_cross_arm=reused,
                actual_new_full_trainings=0 if reused else 3,logical_full_trainings=3,
                logical_training_seconds=sum(r['seconds'] for r in records),seconds=time.perf_counter()-start)
    write(folder/'result.json',result); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('parent','measure','query'))
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    root=args.root.resolve(); assert root==args.root and root.name=='20261018_cold_start_r10'
    lock=read(root/'execution_lock.json')
    for name,value in lock['runtime_code_sha256'].items(): assert sha(root/name)==value
    configure()
    result=parent(root) if args.action=='parent' else globals()[args.action](root,read(args.request))
    write(args.output,result)


if __name__=='__main__': main()
