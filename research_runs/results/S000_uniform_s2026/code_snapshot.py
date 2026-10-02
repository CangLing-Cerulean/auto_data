"""Training-window membership/weight/sampling research; immutable raw data."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, digest, prepare, evaluate, write_json


def variation(values, starts, length):
    prefix=np.r_[0,np.cumsum(np.abs(np.diff(values.astype(np.float64),axis=0)).mean(axis=1))]
    return (prefix[starts+length-1]-prefix[starts])/(length-1)


def sampling_policy(recipe, score, threshold):
    sampling=np.ones(len(score),dtype=np.float64)
    loss=np.ones(len(score),dtype=np.float64)
    if recipe=='drop_quiet':
        sampling[score<=threshold]=0
    elif recipe!='uniform':
        raise ValueError(recipe)
    if sampling.sum()<=0:
        raise ValueError('Empty candidate pool')
    probability=sampling/sampling.sum()
    return sampling,loss,probability


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id:
        raise ValueError('Invalid run ID')
    config_path=ROOT/'research_runs/selection/protocol.json'
    p=json.loads(config_path.read_text())
    manifest_path=ROOT/'data/metropt3_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    run=ROOT/'research_runs/results'/args.run_id
    run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():
        raise RuntimeError('Completed run cannot be overwritten')
    identity={'protocol_sha256':digest(config_path),'manifest_sha256':digest(manifest_path),
              'code_sha256':digest(Path(__file__)),'core_sha256':digest(ROOT/'scripts/run_research.py'),
              'recipe':args.recipe,'seed':args.seed,'preprocessing':'baseline'}
    if (run/'identity.json').exists():
        if json.loads((run/'identity.json').read_text())!=identity:
            raise RuntimeError('Resume identity mismatch')
    else:
        write_json(run/'identity.json',identity)
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (run/'core_snapshot.py').write_bytes((ROOT/'scripts/run_research.py').read_bytes())
        (run/'protocol.json').write_bytes(config_path.read_bytes())
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    old=json.loads((run/'status.json').read_text()).get('elapsed_seconds',0) if (run/'status.json').exists() else 0.
    remaining=p['session_gpu_wall_budget_seconds']-used
    if remaining<=0:
        raise RuntimeError('Budget exhausted')
    begun=time.monotonic()
    def status(phase,step):
        write_json(run/'status.json',{'phase':phase,'step':step,'elapsed_seconds':old+time.monotonic()-begun})
    status('preparing',0)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    arrays,indices,groups,audit=prepare(p,manifest)
    scores={split:variation(values,indices[split],p['input_length']) for split,values in arrays.items()}
    threshold=float(np.quantile(scores['train'],.25))
    groups['low_quartile']=scores['validation']<=threshold
    groups['middle']=(scores['validation']>threshold)&~groups['high_input_variation']
    sampling,loss_weight,prob=sampling_policy(args.recipe,scores['train'],threshold)
    table=ROOT/'research_runs/selection/window_table.npz'
    if table.exists():
        with np.load(table) as saved:
            if not np.array_equal(saved['start_row_zero_based'],indices['train']) or not np.array_equal(saved['input_variation'],scores['train']):
                raise RuntimeError('Window metadata changed')
    else:
        np.savez_compressed(table,start_row_zero_based=indices['train'],input_variation=scores['train'])
    quiet=scores['train']<=threshold
    sample_audit={'recipe':args.recipe,'window_table_sha256':digest(table),'source_train_sha256':next(s['sha256'] for s in manifest['splits'] if s['name']=='train'),
                  'eligible_windows':len(prob),'retained_windows':int(np.count_nonzero(prob)),
                  'removed_windows':int(np.count_nonzero(prob==0)),'low_variation_threshold':threshold,
                  'quiet_windows':int(quiet.sum()),'expected_quiet_sampling_mass':float(prob[quiet].sum()),
                  'expected_loss_weight':float(np.dot(prob,loss_weight)),'sampling_method':'common_uniform_inverse_cdf',
                  'mapping':'window_table start row s refers to original train.csv data rows [s,s+40), input [s,s+32), targets [s+32,s+40)'}
    write_json(run/'sampling_audit.json',sample_audit)
    write_json(run/'data_audit.json',audit)
    write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()})
    data={s:torch.from_numpy(v).cuda() for s,v in arrays.items()}
    starts={s:torch.from_numpy(v).cuda() for s,v in indices.items()}
    cdf=torch.from_numpy(np.cumsum(prob)).cuda()
    cdf[-1]=1.
    weights=torch.from_numpy(loss_weight.astype(np.float32)).cuda()
    model=DLinear(p).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    rng=torch.Generator(device='cuda').manual_seed(args.seed+1000)
    counts=torch.zeros(len(prob),device='cuda',dtype=torch.int64)
    step0,best,best_step,history=0,float('inf'),0,[]
    if (run/'latest.pt').exists():
        saved=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['sampling_rng'].cpu())
        counts=saved['draw_counts']
        step0,best,best_step,history=saved['step'],saved['best_score'],saved['best_step'],saved['history']
    offsets=torch.arange(p['input_length']+p['prediction_length'],device='cuda')
    losses=[]
    for step in range(step0+1,p['steps']+1):
        if time.monotonic()-begun>=remaining:
            status('budget_exhausted',step-1)
            raise RuntimeError('Budget exhausted')
        uniform=torch.rand(p['batch_size'],device='cuda',dtype=torch.float64,generator=rng)
        sampled=torch.searchsorted(cdf,uniform,right=True)
        counts.scatter_add_(0,sampled,torch.ones_like(sampled))
        batch=data['train'][starts['train'][sampled,None]+offsets]
        x,y=batch[:,:p['input_length']],batch[:,p['input_length']:]
        optimizer.zero_grad(set_to_none=True)
        per_window=(model(x)-y).square().mean((1,2))
        loss=(per_window*weights[sampled]).mean()/sample_audit['expected_loss_weight']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%p['validation_every']==0:
            metrics=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
            score=metrics['groups']['all']['NMSE']
            entry={'step':step,'train_loss':float(np.mean(losses)),'validation':metrics['groups']['all']}
            losses=[]
            history.append(entry)
            if score<best:
                best,best_step=score,step
                torch.save(model.state_dict(),run/'best.tmp')
                (run/'best.tmp').replace(run/'best.pt')
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state(),
                        'draw_counts':counts,'step':step,'best_score':best,'best_step':best_step,'history':history},run/'latest.tmp')
            (run/'latest.tmp').replace(run/'latest.pt')
            write_json(run/'history.json',history)
            status('training',step)
            print(json.dumps(entry),flush=True)
    model.load_state_dict(torch.load(run/'best.pt',weights_only=True))
    final=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
    draws=counts.cpu().numpy()
    if int(draws.sum())!=p['steps']*p['batch_size'] or np.any(draws[prob==0]):
        raise RuntimeError('Sampling exposure audit failed')
    np.savez_compressed(run/'sampling_weights.npz',sampling_weight=sampling,loss_weight=loss_weight,probability=prob,draw_count=draws)
    sample_audit.update({'unique_windows_seen':int(np.count_nonzero(draws)),'actual_quiet_draws':int(draws[quiet].sum()),
                         'total_draws':int(draws.sum()),'sampling_weights_sha256':digest(run/'sampling_weights.npz')})
    write_json(run/'sampling_audit.json',sample_audit)
    final.update({'run_id':args.run_id,'identity':identity,'best_step':best_step,'steps_completed':p['steps'],
                  'samples_presented':int(draws.sum()),'elapsed_seconds':old+time.monotonic()-begun,
                  'history':history,'sensor_columns':manifest['sensor_columns']})
    write_json(run/'result.json',final)
    status('completed',p['steps'])
    print(json.dumps({'completed':args.run_id,**final['groups']['all']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
