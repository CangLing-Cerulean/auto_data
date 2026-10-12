"""Frozen SKAB common-response pilot: prepare or one fixed DLinear training.

Only train.csv and validation.csv are readable inputs; no test or fault labels.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)
def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_split(folder,manifest,name):
    assert name in ('train','validation')
    identity=next(x for x in manifest['splits'] if x['name']==name)
    path=folder/(name+'.csv')
    if sha(path)!=identity['sha256']:raise RuntimeError('Base split hash mismatch')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        records=list(csv.DictReader(stream))
    assert len(records)==identity['rows']
    values=np.asarray([[float(row[k]) for k in manifest['sensor_columns']] for row in records],dtype=np.float64)
    assert np.isfinite(values).all()
    source=np.asarray([r['source_file'] for r in records]);source_row=np.asarray([int(r['source_row']) for r in records])
    dates=np.asarray([r['datetime'] for r in records],dtype='datetime64[s]')
    windows=[];groups=[]
    for group in sorted(set(source)):
        ix=np.flatnonzero(source==group);ix=ix[np.argsort(source_row[ix])]
        if len(ix)<40:continue
        dt=np.diff(dates[ix]).astype('timedelta64[s]').astype(np.int64)
        assert np.all(dt>0)
        bad=(np.diff(source_row[ix])!=1)|(dt>30)
        prefix=np.r_[0,np.cumsum(bad)];start=np.arange(len(ix)-39)
        start=start[prefix[start+39]==prefix[start]]
        for s in start:windows.append(ix[s:s+40]);groups.append(group)
    windows=np.asarray(windows,dtype=np.int64);groups=np.asarray(groups)
    if not len(windows):raise RuntimeError('No valid forecast windows')
    order=np.argsort(windows[:,0]);windows=windows[order];groups=groups[order]
    assert np.all(source[windows]==source[windows[:,:1]])
    assert np.all(np.diff(source_row[windows],axis=1)==1)
    elapsed=(dates[windows[:,-1]]-dates[windows[:,31]]).astype('timedelta64[s]').astype(np.int64)
    return values,windows,groups,source_row,elapsed


def prepare(root,protocol):
    output=root/'prepared'
    if output.exists():raise RuntimeError('Preparation already exists')
    output.mkdir();started=time.monotonic();data=root/'data/skab_v1'
    manifest=read(data/'manifest.json')
    assert sha(data/'manifest.json')==protocol['manifest_sha256']
    t,tw,tgroups,trows,telapsed=load_split(data,manifest,'train')
    d,dw,dgroups,drows,delapsed=load_split(data,manifest,'validation')
    mean=t.mean(0);std=t.std(0);scale=np.where(std==0,1.,std)
    t=((t-mean)/scale).astype(np.float32);d=((d-mean)/scale).astype(np.float32)
    def variation(values,windows):return np.abs(np.diff(values[windows[:,:32]].astype(np.float64),axis=1)).mean((1,2))
    tv=variation(t,tw);dv=variation(d,dw);threshold=float(np.quantile(tv,.9))
    target=tv>threshold;dtarget=dv>threshold
    assert target.any() and (~target).any() and dtarget.any() and (~dtarget).any()
    # Match source experiment and within-experiment chronological quartile.
    strata=np.empty(len(tw),dtype=np.int64);stratum_id=0
    for group in sorted(set(tgroups)):
        positions=np.flatnonzero(tgroups==group)
        for chunk in np.array_split(positions,4):
            strata[chunk]=stratum_id;stratum_id+=1
    rng=np.random.default_rng(protocol['control_seed']);control=np.zeros(len(tw),dtype=bool);match=[];qualified=True
    for s in np.unique(strata):
        selected=np.flatnonzero((strata==s)&target);pool=np.flatnonzero((strata==s)&~target)
        if len(selected)>len(pool):
            qualified=False;matched=0
        else:
            matched=len(selected)
            if matched:control[rng.choice(pool,matched,replace=False)]=True
        match.append(dict(stratum=int(s),target=len(selected),eligible_control=len(pool),matched=matched))
    if qualified:assert np.sum(target)==np.sum(control) and not (target&control).any()
    else:control[:]=False  # No partial control may be trained or called matched.
    np.savez_compressed(output/'arrays.npz',train=t,development=d,train_windows=tw,development_windows=dw,
                        target=target,control=control,development_target=dtarget,train_variation=tv,development_variation=dv,
                        strata=strata,train_source_row=trows,development_source_row=drows,
                        train_groups=tgroups,development_groups=dgroups)
    info=dict(status='prepared',dataset_id='skab_v1',manifest_sha256=sha(data/'manifest.json'),
        train_rows=len(t),development_rows=len(d),train_windows=len(tw),development_windows=len(dw),
        channels=manifest['sensor_columns'],normalization=dict(mean=mean.tolist(),scale=scale.tolist()),
        input_variation_threshold=threshold,train_target_fraction=float(target.mean()),
        development_target_fraction=float(dtarget.mean()),
        forecast_elapsed_seconds=dict(train_min=int(telapsed.min()),train_max=int(telapsed.max()),
                                      development_min=int(delapsed.min()),development_max=int(delapsed.max())),
        matched_control=dict(qualified=qualified,status='exact_counts_per_source_time_stratum' if qualified else 'infeasible_controls_not_executable',strata=match),
        array_sha256=sha(output/'arrays.npz'),new_trainings=0,test_access=False,process_seconds=time.monotonic()-started)
    write(output/'receipt.json',info);print(json.dumps(info,ensure_ascii=False),flush=True)


def train(root,protocol,action,seed):
    output=root/'runs'/f'{action}_s{seed}'
    if output.exists():raise RuntimeError('Run ID already occupied; never overwrite or repeat')
    output.mkdir(parents=True);started=time.monotonic()
    status=dict(status='running',action=action,seed=seed,steps_completed=0,protocol_sha256=sha(root/'protocol.json'))
    write(output/'status.json',status)
    try:
        receipt=read(root/'prepared/receipt.json');path=root/'prepared/arrays.npz'
        if sha(path)!=receipt['array_sha256']:raise RuntimeError('Prepared arrays changed')
        with np.load(path,allow_pickle=False) as source:arrays={k:source[k] for k in source.files}
        definition=protocol['actions'][action];n=len(arrays['train_windows'])
        if definition.get('mask')=='control' and not receipt['matched_control']['qualified']:
            raise RuntimeError('Matched control infeasible; response must remain unknown')
        weights=np.ones(n);probability=np.ones(n)
        if action!='B0':
            mask=arrays[definition['mask']]
            if definition['family']=='loss_weighting':weights[mask]=definition['multiplier']
            elif definition['family']=='sampling':probability[mask]=definition['multiplier']
            else:raise RuntimeError('Unknown action family')
        probability/=probability.sum();weights/=np.dot(probability,weights)
        p=protocol['training'];assert p['steps']==3000 and p['channels']==8
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
        core_path=Path(protocol['frozen_core'])
        if sha(core_path)!=protocol['frozen_core_sha256']:raise RuntimeError('Frozen DLinear implementation changed')
        spec=importlib.util.spec_from_file_location('frozen_dlinear',core_path);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        model=core.DLinear(p).cuda();assert sum(x.numel() for x in model.parameters())==4224
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        initial_sha=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
        t=torch.tensor(arrays['train'],device='cuda');d=torch.tensor(arrays['development'],device='cuda')
        tw=torch.tensor(arrays['train_windows'],device='cuda');dw=torch.tensor(arrays['development_windows'],device='cuda')
        w=torch.tensor(weights,dtype=torch.float32,device='cuda')
        generator=np.random.default_rng(seed+1000);cdf=np.cumsum(probability);cdf[-1]=1.
        draws=np.zeros(n,dtype=np.int64);history=[];best=float('inf');best_step=None
        def evaluate():
            result=np.empty((len(dw),8,8),dtype=np.float32);model.eval()
            with torch.no_grad():
                for lo in range(0,len(dw),2048):
                    batch=d[dw[lo:lo+2048]];result[lo:lo+len(batch)]=(model(batch[:,:32])-batch[:,32:]).cpu().numpy()
            return result
        for step in range(1,3001):
            chosen=np.searchsorted(cdf,generator.random(p['batch_size']),side='right');np.add.at(draws,chosen,1)
            ids=torch.tensor(chosen,device='cuda');batch=t[tw[ids]];model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=((model(batch[:,:32])-batch[:,32:]).square().mean((1,2))*w[ids]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();optimizer.step()
            if step%500==0:
                error=evaluate()
                if not np.isfinite(error).all():raise RuntimeError('Nonfinite development prediction')
                nmse=float(np.mean(error.astype(np.float64)**2));nmae=float(np.mean(abs(error.astype(np.float64))))
                history.append(dict(step=step,NMSE=nmse,NMAE=nmae));status['steps_completed']=step;write(output/'status.json',status)
                if nmse<best:
                    best=nmse;best_step=step;torch.save(model.state_dict(),output/'best.pt');np.savez_compressed(output/'best_errors.npz',error=error)
        torch.save(model.state_dict(),output/'last.pt')
        np.savez_compressed(output/'sampling.npz',probability=probability,loss_weight=weights,draw_count=draws,
                            train_window_start=arrays['train_windows'][:,0])
        with np.load(output/'best_errors.npz',allow_pickle=False) as saved:error=saved['error'].astype(np.float64)
        metrics={}
        for label,mask in [('global',np.ones(len(error),dtype=bool)),('target',arrays['development_target']),('complement',~arrays['development_target'])]:
            values=error[mask];metrics[label]=dict(windows=int(mask.sum()),elements=int(values.size),
                NMSE=float(np.mean(values**2)),NMAE=float(np.mean(abs(values))),signed_bias=float(values.mean()))
        if sha(root/'protocol.json')!=status['protocol_sha256']:raise RuntimeError('Protocol changed during training')
        result=dict(action=action,seed=seed,metrics=metrics,best_step=best_step,history=history,parameters=4224,
            initialization_sha256=initial_sha,probability_sha256=digest(probability),draw_count_sha256=digest(draws),
            prepared_sha256=receipt['array_sha256'],protocol_sha256=sha(root/'protocol.json'),source_sha256=sha(__file__),
            assets={name:sha(output/name) for name in ('best.pt','last.pt','best_errors.npz','sampling.npz')},
            process_seconds=time.monotonic()-started,test_access=False)
        write(output/'result.json',result);status['status']='completed'
        print(json.dumps(dict(action=action,seed=seed,metrics=metrics,seconds=result['process_seconds'])),flush=True)
    except Exception as error:
        status['status']='failed';status['error']=str(error);raise
    finally:
        status['process_seconds']=time.monotonic()-started;write(output/'status.json',status)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--root',required=True)
    p.add_argument('--action');p.add_argument('--seed',type=int);a=p.parse_args()
    if os.name!='posix':raise RuntimeError('Dedicated research server required')
    root=Path(a.root);protocol=read(root/'protocol.json')
    if sha(__file__)!=protocol['runner_sha256']:raise RuntimeError('Runner differs from frozen protocol')
    if a.command=='prepare':prepare(root,protocol)
    else:
        if a.action not in protocol['actions'] or a.seed not in protocol['seeds']:raise RuntimeError('Unregistered run')
        train(root,protocol,a.action,a.seed)


if __name__=='__main__':main()
