"""New fixed training worker preserving both selected and final-step checkpoints."""
import argparse
import importlib.util
import os
from pathlib import Path
import time

import numpy as np

from ..config import GateError, digest
from ..selection.common import file_hash, read_json, write_json
from .processing import parameters, validate_processing
from .transforms import load_processor, apply_processor


def train(job_path):
    job_path=Path(job_path).resolve(); folder=job_path.parent
    job=read_json(job_path)
    identity=job['identity']
    if digest(identity)!=job['job_id']:
        raise GateError('RESEARCH_TRAINING_IDENTITY')
    root=Path(identity['base_root']); p=read_json(root/'research_runs/protocol.json')
    for path, expected in identity['protected_assets'].items():
        if file_hash(path)!=expected:
            raise GateError('RESEARCH_TRAINING_SOURCE_CHANGED')
    if job['config']!=parameters(job['config']) or job['config']!=identity['parameters'] or p['input_length']!=32 or p['prediction_length']!=8 or p['steps']!=3000:
        raise GateError('RESEARCH_ENGINEERING_PROTOCOL')
    p.update(job['config'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    if torch.__version__!=identity['torch_runtime'] or np.__version__!=identity['numpy_runtime']:
        raise GateError('RESEARCH_TRAINING_RUNTIME_CHANGED')
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
        raise GateError('RESEARCH_4090_REQUIRED')
    torch.set_num_threads(4); torch.manual_seed(identity['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    spec=importlib.util.spec_from_file_location('trusted_original_core',root/'scripts/run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    started=time.monotonic()
    arrays,starts,groups,audit=core.prepare(p,read_json(root/'data/metropt3_v1/manifest.json'))
    with np.load(identity['features_path'],allow_pickle=False) as saved:
        features={k:saved[k].copy() for k in saved.files}
    if not np.array_equal(features['start_row'],starts['train']):
        raise GateError('RESEARCH_TRAINING_WINDOW_MAPPING')
    if identity['processing_path'] is None:
        probability=np.full(len(starts['train']),1/len(starts['train']))
        weights=np.ones(len(starts['train']))
    else:
        if file_hash(identity['processing_path'])!=identity['processing_sha256']:
            raise GateError('RESEARCH_PROCESSING_CHANGED')
        probability,weights,_=validate_processing(identity['processing_path'],features)
    recipe=None
    if identity['input_transform_path']:
        if file_hash(identity['input_transform_path'])!=identity['input_transform_sha256']:
            raise GateError('RESEARCH_INPUT_TRANSFORM_CHANGED')
        if file_hash(identity['input_state_path'])!=identity['input_state_sha256']:raise GateError('RESEARCH_INPUT_STATE_CHANGED')
        recipe=load_processor(identity['input_transform_path'],identity['input_state_path'])
    control=identity['control']
    if control=='group_shuffle':
        random=np.random.default_rng(identity['control_seed'])
        for group in np.unique(features['groups']):
            members=np.flatnonzero(features['groups']==group)
            permutation=random.permutation(members)
            probability[members]=probability[permutation]
            weights[members]=weights[permutation]
        probability/=probability.sum()
        weights/=np.dot(probability,weights)
    elif control not in ('aligned','input_identity'):
        raise GateError('RESEARCH_CONTROL')
    if control=='input_identity' and recipe is None:
        raise GateError('RESEARCH_CONTROL_REQUIRES_TRANSFORM')
    data={s:torch.from_numpy(v).cuda() for s,v in arrays.items()}
    indices={s:torch.from_numpy(v).cuda() for s,v in starts.items()}
    cdf=torch.from_numpy(np.cumsum(probability)).cuda();cdf[-1]=1.
    loss_weights=torch.from_numpy(weights.astype(np.float32)).cuda()
    model=core.DLinear(p).cuda()
    if sum(v.numel() for v in model.parameters())!=7920:
        raise GateError('RESEARCH_FIXED_DLINEAR_STRUCTURE')
    optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    @torch.no_grad()
    def evaluate():
        model.eval()
        sq=torch.zeros(15,dtype=torch.float64,device='cuda');absolute=torch.zeros_like(sq)
        sums={g:[0.,0.,0] for g in groups}
        for lo in range(0,len(indices['validation']),p['evaluation_batch_size']):
            ids=indices['validation'][lo:lo+p['evaluation_batch_size']]
            batch=data['validation'][ids[:,None]+torch.arange(40,device='cuda')]
            pred=model(apply_processor(batch[:,:32],recipe,identity_control=control=='input_identity'))
            error=(pred-batch[:,32:]).double();sq+=error.square().sum((0,1));absolute+=error.abs().sum((0,1))
            ws=error.square().mean((1,2)).cpu().numpy();wa=error.abs().mean((1,2)).cpu().numpy()
            for group,mask in groups.items():
                keep=mask[lo:lo+len(ids)];sums[group][0]+=float(wa[keep].sum());sums[group][1]+=float(ws[keep].sum());sums[group][2]+=int(keep.sum())
        total=len(indices['validation'])*8
        return {'groups':{g:{'NMAE':a/n,'NMSE':s/n,'windows':n} for g,(a,s,n) in sums.items() if n},
                'per_channel_NMSE':(sq/total).cpu().tolist(),'per_channel_NMAE':(absolute/total).cpu().tolist()}
    rng=torch.Generator(device='cuda').manual_seed(identity['seed']+1000)
    offsets=torch.arange(40,device='cuda')
    counts=torch.zeros(len(probability),dtype=torch.int64,device='cuda')
    best,best_step,history=float('inf'),0,[]
    for step in range(1,p['steps']+1):
        if time.monotonic()-started>job['seconds']:
            raise GateError('RESEARCH_TRAINING_TIMEOUT')
        chosen=torch.searchsorted(cdf,torch.rand(p['batch_size'],dtype=torch.float64,device='cuda',generator=rng),right=True)
        counts.scatter_add_(0,chosen,torch.ones_like(chosen))
        batch=data['train'][indices['train'][chosen,None]+offsets]
        optimizer.zero_grad(set_to_none=True)
        model.train()
        transformed=apply_processor(batch[:,:32],recipe,identity_control=control=='input_identity')
        loss=((model(transformed)-batch[:,32:]).square().mean((1,2))*loss_weights[chosen]).mean()
        if not torch.isfinite(loss):raise GateError('RESEARCH_NONFINITE_LOSS')
        loss.backward();optimizer.step()
        if step%p['validation_every']==0:
            metrics=evaluate()
            score=metrics['groups']['all']['NMSE']
            history.append({'step':step,'metrics':metrics['groups']['all']})
            if score<best:
                best,best_step=score,step
                torch.save(model.state_dict(),folder/'best.pt')
            write_json(folder/'history.json',history)
    torch.save(model.state_dict(),folder/'last.pt')
    model.load_state_dict(torch.load(folder/'best.pt',weights_only=True))
    metrics=evaluate()
    draws=counts.cpu().numpy()
    if draws.sum()!=p['steps']*p['batch_size'] or np.any(draws[probability==0]):
        raise GateError('RESEARCH_SAMPLING_EXPOSURE')
    np.savez_compressed(folder/'sampling.npz',start_row=starts['train'],probability=probability,
                        loss_weight=weights,draw_count=draws)
    for path,expected in identity['protected_assets'].items():
        if file_hash(path)!=expected:raise GateError('RESEARCH_SOURCE_CHANGED_DURING_TRAINING')
    for key,hash_key in (('input_transform_path','input_transform_sha256'),('input_state_path','input_state_sha256')):
        if identity[key] and file_hash(identity[key])!=identity[hash_key]:raise GateError('RESEARCH_PROCESSOR_CHANGED_DURING_TRAINING')
    result={'identity':identity,'job_id':job['job_id'],'metrics':metrics['groups']['all'],
        'groups':metrics['groups'],'per_channel_NMSE':metrics['per_channel_NMSE'],
        'per_channel_NMAE':metrics['per_channel_NMAE'],'best_step':best_step,'steps':p['steps'],
        'parameters':7920,'config':job['config'],'elapsed_seconds':time.monotonic()-started,
        'scope':identity['scope'],'independent_E':False,'data_audit':audit,
        'environment':{'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                       'device':torch.cuda.get_device_name()},
        'assets':{name:file_hash(folder/name) for name in ('best.pt','last.pt','sampling.npz','history.json')}}
    write_json(folder/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    train(parser.parse_args().job)
