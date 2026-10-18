"""Create-only open-processing worker. The r10 experiment is never modified."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
import torch

from coldstart_model import DLinear, optimizer
from coldstart_runtime import (SEEDS, LENGTH, HORIZON, BATCH, STEPS, read, write,
                              sha, digest, windows, configure, tensors, batch,
                              metrics, checkpoint_model, best_model, errors, describe)
from open_processing_materialize import fit_program, apply_program, normalize_weights, ProgramExecutionError
from open_processing_sandbox import encode_state, pack
from open_processing_reference import predict, FEATURES


STUDY='20261018_open_processing_r11'
RMS_FLOOR=1e-12


class InvalidProgram(ValueError):
    pass


def load(root):
    lock=read(root/'execution_lock.json')
    parent=Path(lock['parent_root'])
    assert parent.resolve()==parent and parent.name=='20261018_cold_start_r10'
    assert sha(parent/'ABC.npz')==lock['arrays_sha256']
    assert sha(parent/'development_audit.json')==lock['parent_development_audit_sha256']
    assert read(parent/'development_audit.json')['status']=='PASS'
    for relative,expected in lock['parent_assets'].items():
        path=parent/relative
        assert path.resolve().is_relative_to(parent/'runs/PARENT') and sha(path)==expected
    with np.load(parent/'ABC.npz',allow_pickle=False) as archive:
        rows={part:archive[part].copy() for part in 'ABC'}
    assert [v.shape for v in rows.values()]==[(11520,370),(2880,370),(2880,370)]
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in rows.values())
    return parent,rows


def save_state(folder,state):
    tree,arrays=encode_state(state)
    payload=pack(dict(state=tree),arrays)
    with (folder/'fit_state.npz').open('xb') as stream:stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def save_source(folder,source):
    with (folder/'program.py').open('x',encoding='utf-8',newline='\n') as stream:stream.write(source)


def program_error(exc):
    # Launch/configuration failures stop the experiment; generated-code failures consume an attempt.
    if isinstance(exc,ProgramExecutionError):
        receipt=exc.receipt
        if not (receipt.get('timed_out') or receipt.get('worker_metadata',{}).get('status')=='INVALID_PROGRAM'
                or (isinstance(receipt.get('returncode'),int) and receipt['returncode']<0)):
            raise exc
        return dict(error=str(exc),worker_receipt=receipt)
    return dict(error=f'{type(exc).__name__}: {exc}')


def materialize(folder,source,rows):
    state,fit_receipt=fit_program(source,'pipeline',rows['A'])
    state_hash=save_state(folder,state)
    write(folder/'fit_receipt.json',fit_receipt)
    arrays={};receipts={}
    for part in 'ABC':
        arrays[part],receipts[part]=apply_program(source,'pipeline',rows[part],state,
                                               include_weights=part=='A',include_target=part!='A')
        write(folder/f'{part}_materialize_receipt.json',receipts[part])
    probabilities,loss,stats=normalize_weights(arrays['A']['sampling'],arrays['A']['loss'])
    for part in 'BC':
        mask=arrays[part]['target']
        if not 0<int(mask.sum())<mask.size:raise InvalidProgram(f'{part} target and complement must both be nonempty')
    return arrays,probabilities,loss,stats,state_hash,receipts


def processing_identity(arrays,probabilities,loss):
    # Effective training identity is distinct from source/diagnosis provenance.
    hashes={part:{name:digest(arrays[part][name]) for name in ('model_input','offset','scale')} for part in 'ABC'}
    hashes.update(probabilities=digest(probabilities),loss=digest(loss))
    identity=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return identity,hashes


def prepared_gpu(arrays):
    return {part:{k:torch.as_tensor(value,device='cuda') for k,value in a.items()
                  if k in ('model_input','offset','scale')} for part,a in arrays.items()}


def decoded_errors(model,rows,prepared):
    out=[];model.eval();n=len(prepared['model_input'])
    with torch.no_grad():
        for first in range(0,n,BATCH):
            stop=min(first+BATCH,n);_,y=batch(rows,np.arange(first,stop))
            prediction=model(prepared['model_input'][first:stop])
            prediction=prediction*prepared['scale'][first:stop,None,:]+prepared['offset'][first:stop,None,:]
            out.append((prediction-y).cpu().numpy())
    result=np.concatenate(out)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite decoded residual')
    return result


def train(root,name,identity,probabilities,loss_weights,seed,gpu,prepared):
    folder=root/'runs'/name/str(seed);folder.mkdir(parents=True)
    start=time.perf_counter();step=0
    write(folder/'started.json',dict(seed=seed,processing_identity=identity,started=time.time()))
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=DLinear(370).cuda();opt=optimizer(model)
    torch.save(model.state_dict(),folder/'initial.pt')
    rng=np.random.default_rng(seed+100000);n=len(probabilities)
    uniform=bool(np.all(probabilities==probabilities[0]))
    cdf=np.cumsum(probabilities,dtype=np.float64);cdf/=cdf[-1];cdf[-1]=1.
    grid=[];best=float('inf');best_step=None;updates=0
    try:
        for step in range(1,STEPS+1):
            u=rng.random(BATCH);ids=(u*n).astype(np.int64) if uniform else np.searchsorted(cdf,u,side='right')
            _,original_y=batch(gpu['A'],ids)
            index=torch.as_tensor(ids,device='cuda',dtype=torch.int64)
            x=prepared['A']['model_input'][index]
            offset=prepared['A']['offset'][index,None,:];scale=prepared['A']['scale'][index,None,:]
            y=(original_y-offset)/scale
            model.train();opt.zero_grad(set_to_none=True)
            losses=((model(x)-y)**2).mean((1,2))
            loss=(losses*torch.as_tensor(loss_weights[ids],device='cuda',dtype=losses.dtype)).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite transformed training loss')
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradient')
            opt.step();updates=step
            if step%500==0:
                result=metrics(decoded_errors(model,gpu['B'],prepared['B']))
                grid.append(dict(step=step,**result));torch.save(model.state_dict(),folder/f'step_{step:04d}.pt')
                if result['mse']<best:best,best_step=result['mse'],step
        torch.save(opt.state_dict(),folder/'last_adam.pt')
        c=metrics(decoded_errors(checkpoint_model(folder/f'step_{best_step:04d}.pt'),gpu['C'],prepared['C']))
        torch.cuda.synchronize()
        record=dict(status='completed',seed=seed,steps=STEPS,updates=updates,best_step=best_step,
                    B_grid=grid,C=c,seconds=time.perf_counter()-start,processing_identity=identity)
    except FloatingPointError as exc:
        record=dict(status='invalid_training',seed=seed,attempted_step=step,updates=updates,
                    B_grid=grid,error=str(exc),seconds=time.perf_counter()-start,processing_identity=identity)
    record['assets']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in folder.glob('*.pt')}
    write(folder/'result.json',record)
    return record


def prequery(parent,rows,arrays,probabilities,loss,stats):
    gpu=tensors({'B':rows['B']});mask=arrays['B']['target'];contrasts=[]
    for seed in SEEDS:
        e=errors(best_model(parent/'runs/PARENT'/str(seed)),gpu['B']).astype(np.float64)**2
        contrasts.append(float((e[mask].mean()-e[~mask].mean())/e.mean()))
    original=windows(rows['A']).astype(np.float64)
    denominator=max(float(np.sqrt(np.mean(original**2))),RMS_FLOOR)
    delta=arrays['A']['model_input'].astype(np.float64)-original
    descriptors=dict(input_relative_rms_change=float(np.sqrt(np.mean(delta**2))/denominator),
                     offset_relative_rms=float(np.sqrt(np.mean(arrays['A']['offset'].astype(np.float64)**2))/denominator),
                     mean_abs_log_scale=float(np.mean(np.abs(np.log(arrays['A']['scale'].astype(np.float64))))))
    x=[1.,stats['positive_fraction'],stats['ess_sampling']/stats['n'],
       float(np.tanh(descriptors['input_relative_rms_change'])),float(np.tanh(descriptors['offset_relative_rms'])),
       float(np.tanh(descriptors['mean_abs_log_scale'])),float(np.tanh(stats['loss_std_under_p'])),
       float(np.tanh(np.mean(contrasts))),float(mask.mean())]
    if not np.isfinite(x).all():raise InvalidProgram('Nonfinite reference descriptor')
    return dict(x=x,feature_names=FEATURES,processing_descriptors=descriptors,
                parent_B_contrast_by_seed=contrasts,weight_stats=stats,p_B=float(mask.mean()),
                p_C=float(arrays['C']['target'].mean()),target_B_sha256=digest(mask),
                target_C_sha256=digest(arrays['C']['target']))


def measure(root,request):
    start=time.perf_counter();parent,rows=load(root)
    folder=root/'measurements'/request['arm']/f"{request['round']:02d}";folder.mkdir(parents=True)
    write(folder/'request.json',request)
    if request['measurement'] is None:
        result=dict(status='PASS',measurement=None,seconds=time.perf_counter()-start)
        write(folder/'result.json',result);return result
    source=request['measurement']['source'];save_source(folder,source)
    try:
        state,fit_receipt=fit_program(source,'measurement',rows['A'])
        state_hash=save_state(folder,state);write(folder/'fit_receipt.json',fit_receipt)
        values={};receipts={}
        for part in 'AB':
            output,receipts[part]=apply_program(source,'measurement',rows[part],state)
            values[part]=output['measurement'];write(folder/f'{part}_receipt.json',receipts[part])
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            mean=float(values['A'].mean());std=float(values['A'].std());scale=std if std>0 else 1.
            if not np.isfinite([mean,std]).all():raise InvalidProgram('Nonfinite A measurement normalization')
            normalized={part:(v-mean)/scale for part,v in values.items()}
        if not all(np.isfinite(v).all() for v in normalized.values()):raise InvalidProgram('Nonfinite normalized measurement')
        gpu=tensors({'B':rows['B']})
        loss=np.mean([np.mean(errors(best_model(parent/'runs/PARENT'/str(s)),gpu['B']).astype(np.float64)**2,axis=(1,2)) for s in SEEDS],axis=0)
        v=normalized['B'];bins=np.quantile(v,[.25,.5,.75]);groups=np.searchsorted(bins,v,side='right')
        corr=float(np.corrcoef(v,loss)[0,1]) if np.std(v)>0 and np.std(loss)>0 else None
        result=dict(status='PASS',measurement=dict(A=describe(normalized['A']),B=describe(v),
                    parent_B_error_correlation=corr,parent_B_error_quartiles=[dict(count=int(np.sum(groups==i)),
                    mse=float(loss[groups==i].mean()) if np.any(groups==i) else None) for i in range(4)]),
                    fit_A=dict(mean=mean,std=std,scale=scale,state_sha256=state_hash),
                    source_sha256=sha(folder/'program.py'),seconds=time.perf_counter()-start)
    except (ProgramExecutionError,ValueError,FloatingPointError) as exc:
        result=dict(status='invalid_measurement',**program_error(exc),seconds=time.perf_counter()-start)
    write(folder/'result.json',result);return result


def query(root,request):
    start=time.perf_counter();parent,rows=load(root)
    folder=root/'queries'/request['arm']/f"{request['round']:02d}";folder.mkdir(parents=True)
    write(folder/'request.json',request);save_source(folder,request['source'])
    try:
        arrays,p,loss,stats,state_hash,receipts=materialize(folder,request['source'],rows)
        identity,hashes=processing_identity(arrays,p,loss)
        before=prequery(parent,rows,arrays,p,loss,stats)
        own=[read(path)['processing_identity'] for path in (root/'queries'/request['arm']).glob('*/prequery.json')]
        if identity in own:
            result=dict(status='duplicate_own_program',processing_identity=identity,seconds=time.perf_counter()-start)
            write(folder/'result.json',result);return result
        before.update(processing_identity=identity,effective_array_sha256=hashes,fit_state_sha256=state_hash,
                      source_sha256=sha(folder/'program.py'))
        before['reference_forecast']=predict(request['reference_history'],before['x'])
    except (ProgramExecutionError,ValueError,FloatingPointError) as exc:
        result=dict(status='invalid_program',**program_error(exc),seconds=time.perf_counter()-start)
        write(folder/'result.json',result);return result
    write(folder/'prequery.json',before)
    gpu=tensors(rows);prepared=prepared_gpu(arrays);name='R_'+identity
    records=[];reused=(root/'runs'/name).exists()
    for seed in SEEDS:
        if reused:
            path=root/'runs'/name/str(seed)/'result.json'
            if not path.exists():break
            record=read(path)
        else:record=train(root,name,identity,p,loss,seed,gpu,prepared)
        records.append(record)
        if record['status']!='completed':break
    accounting=dict(processing_identity=identity,run=name,reused_cross_arm=reused,logical_full_trainings=3,
                    actual_new_full_trainings=0 if reused else sum(r['status']=='completed' for r in records),
                    actual_new_partial_updates=0 if reused else sum(r['updates'] for r in records if r['status']!='completed'),
                    logical_training_seconds=sum(r['seconds'] for r in records))
    if len(records)!=3 or any(r['status']!='completed' for r in records):
        result=dict(status='invalid_training',**accounting,seed_records=records,seconds=time.perf_counter()-start)
        write(folder/'result.json',result);return result
    responses=[];mask=arrays['C']['target']
    for seed,record in zip(SEEDS,records):
        pe=errors(best_model(parent/'runs/PARENT'/str(seed)),gpu['C'])
        ce=decoded_errors(best_model(root/'runs'/name/str(seed)),gpu['C'],prepared['C'])
        assert digest(ce)==record['C']['residual_sha256']
        pp=pe.astype(np.float64)**2;cc=ce.astype(np.float64)**2;denom=float(pp.mean())
        dt=float((pp[mask]-cc[mask]).mean()/denom);dc=float((pp[~mask]-cc[~mask]).mean()/denom)
        y=float((pp.mean()-cc.mean())/denom);z=dt-dc;fraction=float(mask.mean())
        assert abs(y-(fraction*dt+(1-fraction)*dc))<1e-12
        responses.append(dict(seed=seed,Y=y,Z=z,target_gain=dt,complement_gain=dc,
                              NMAE_gain=float(1-np.abs(ce.astype(np.float64)).mean()/np.abs(pe.astype(np.float64)).mean()),
                              parent_residual_sha256=digest(pe),child_residual_sha256=digest(ce)))
    result=dict(status='completed',**accounting,prequery=before,responses=responses,
                Y=float(np.mean([r['Y'] for r in responses])),Z=float(np.mean([r['Z'] for r in responses])),
                NMAE_gain=float(np.mean([r['NMAE_gain'] for r in responses])),seconds=time.perf_counter()-start)
    write(folder/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('measure','query'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();assert root==args.root and root.name==STUDY
    lock=read(root/'execution_lock.json');assert lock['execution_enabled']
    for name,expected in lock['runtime_code_sha256'].items():assert sha(root/name)==expected
    preflight=read(root/'preflight_receipt.json');reuse=read(root/'parent_reuse_receipt.json')
    assert preflight['status']==reuse['status']=='PASS'
    assert preflight['execution_lock_sha256']==reuse['execution_lock_sha256']==sha(root/'execution_lock.json')
    assert reuse['preflight_sha256']==sha(root/'preflight_receipt.json')
    assert reuse['parent_root']==lock['parent_root'] and reuse['parent_assets']==lock['parent_assets']
    assert reuse['parents_reused']==preflight['shared_parents']==3
    configure();result=globals()[args.action](root,read(args.request));write(args.output,result)


if __name__=='__main__':main()
