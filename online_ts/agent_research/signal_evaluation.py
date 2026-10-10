"""Trusted grouped error audit, including last-value reference; no optimizer steps."""
import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
import torch

from ..config import GateError,digest
from ..selection.common import file_hash,read_json,write_json
from .transforms import load_processor,apply_processor

def evaluate(folder):
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    folder=Path(folder);job=read_json(folder/'job.json');identity=job['identity']
    for p,h in identity['protected_assets'].items():
        if file_hash(p)!=h:raise GateError('SIGNAL_AUDIT_ASSET_CHANGED')
    result=read_json(folder/'result.json')
    for name,h in result['assets'].items():
        if file_hash(folder/name)!=h:raise GateError('SIGNAL_CHECKPOINT_CHANGED')
    root=Path(identity['base_root']);p=read_json(root/'research_runs/protocol.json');p.update(job['config'])
    spec=importlib.util.spec_from_file_location('signal_reference_core',root/'scripts/run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    arrays,starts,groups,_=core.prepare(p,read_json(root/'data/metropt3_v1/manifest.json'))
    model=core.DLinear(p).cuda();torch.set_num_threads(4)
    program=load_processor(identity['input_transform_path'],identity['input_state_path']) if identity['input_transform_path'] else None
    data=torch.tensor(arrays['validation'],device='cuda');indices=torch.tensor(starts['validation'],device='cuda')
    offsets=torch.arange(40,device='cuda');started=time.monotonic();out={}
    for checkpoint in ('best.pt','last.pt'):
        model.load_state_dict(torch.load(folder/checkpoint,weights_only=True));model.eval()
        totals={g:{'sq':np.zeros(15),'abs':np.zeros(15),'reference_sq':np.zeros(15),'reference_abs':np.zeros(15),'windows':0} for g in groups}
        with torch.no_grad():
            for lo in range(0,len(indices),p['evaluation_batch_size']):
                ids=indices[lo:lo+p['evaluation_batch_size']];batch=data[ids[:,None]+offsets]
                pred=model(apply_processor(batch[:,:32],program,identity_control=identity['control']=='input_identity'))
                err=(pred-batch[:,32:]).double();ref=(batch[:,31:32]-batch[:,32:]).double()
                sq=err.square().mean(1).cpu().numpy();absolute=err.abs().mean(1).cpu().numpy()
                rsq=ref.square().mean(1).cpu().numpy();rabs=ref.abs().mean(1).cpu().numpy()
                for name,mask in groups.items():
                    keep=mask[lo:lo+len(ids)];r=totals[name];r['windows']+=int(keep.sum())
                    r['sq']+=sq[keep].sum(0);r['abs']+=absolute[keep].sum(0)
                    r['reference_sq']+=rsq[keep].sum(0);r['reference_abs']+=rabs[keep].sum(0)
        out[checkpoint]={g:{'windows':r['windows'],
            'per_channel_NMSE':(r['sq']/r['windows']).tolist(),
            'per_channel_NMAE':(r['abs']/r['windows']).tolist(),
            'persistence_per_channel_NMSE':(r['reference_sq']/r['windows']).tolist(),
            'persistence_per_channel_NMAE':(r['reference_abs']/r['windows']).tolist()} for g,r in totals.items() if r['windows']}
    if out['best.pt']['all']['windows']!=300535:raise GateError('SIGNAL_D_WINDOW_COUNT')
    if not np.isclose(np.mean(out['best.pt']['all']['per_channel_NMSE']),result['metrics']['NMSE'],atol=1e-9,rtol=1e-7):
        raise GateError('SIGNAL_AUDIT_METRIC_MISMATCH')
    write_json(folder/'scope_metrics.json',{'status':'passed','groups':out,'elapsed_seconds':time.monotonic()-started,
        'scope':'development_only','independent_data':False,'additional_training':0,
        'checkpoint_sha256':{n:file_hash(folder/n) for n in ('best.pt','last.pt')},'source_sha256':file_hash(__file__)})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);evaluate(parser.parse_args().folder)
