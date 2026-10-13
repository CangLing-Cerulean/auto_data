"""Posthoc inference-only decomposition of already completed Hydraulic runs."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch


def module(path):
    spec=importlib.util.spec_from_file_location('hydraulic',path)
    loaded=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def category(gain):
    return 'improve' if gain>.25 else 'harm' if gain<-.25 else 'unchanged'


def effective_map(model):
    # The exact real-arithmetic map of replicate-padded moving average DLinear.
    n=32
    smooth=np.zeros((n,n),dtype=np.float64)
    for row in range(n):
        for col in range(row-model.kernel//2,row+model.kernel//2+1):
            smooth[row,min(n-1,max(0,col))]+=1/model.kernel
    weights,biases=[],[]
    for s,t in zip(model.seasonal,model.trend):
        ws=s.weight.detach().cpu().numpy().astype(np.float64)
        wt=t.weight.detach().cpu().numpy().astype(np.float64)
        weights.append(ws@(np.eye(n)-smooth)+wt@smooth)
        biases.append((s.bias.detach().cpu().double()+t.bias.detach().cpu().double()).numpy())
    return np.stack(weights),np.stack(biases)


def evaluate(model,values,starts,proposal,params,action):
    output=[]
    with torch.no_grad():
        for lo in range(0,len(starts),2048):
            batch=values[starts[lo:lo+2048,None]+torch.arange(40,device='cuda')]
            x=batch[:,:32]
            if action!='PARENT':
                x=proposal.transform(x,params,'P1',action=='CP1')
            # Retain original float32 residual convention before float64 analysis.
            output.append((model(x)-batch[:,32:]).cpu().numpy())
    return np.concatenate(output)


def summarize(e,f,c):
    e,f,c=(v.astype(np.float64) for v in (e,f,c))
    d,q=f-e,c-f
    terms=np.stack([np.mean(e*e,axis=1),np.mean(2*e*d,axis=1),np.mean(d*d,axis=1),
                    np.mean(2*(e+d)*q,axis=1),np.mean(q*q,axis=1),np.mean(c*c,axis=1)],axis=-1)
    channel=terms.mean(axis=0)
    global_terms=channel.mean(axis=0)
    residual=terms[:,:,0]+terms[:,:,1:5].sum(axis=2)-terms[:,:,5]
    assert np.all(abs(residual) <= 1e-11*np.maximum(1.,terms[:,:,5]))
    parent,fixed,child=float(np.mean(e*e)),float(np.mean(f*f)),float(np.mean(c*c))
    np.testing.assert_allclose(global_terms[0]+global_terms[1:5].sum(),child,rtol=1e-12,atol=1e-14)
    fixed_gain=100*(1-fixed/parent)
    total_gain=100*(1-child/parent)
    adaptation=100*(fixed-child)/parent
    record=dict(parent_NMSE=parent,fixed_parent_processed_NMSE=fixed,child_processed_NMSE=child,
        fixed_processing_gain_percent=fixed_gain,retraining_adaptation_gain_percent=adaptation,total_gain_percent=total_gain,
        fixed_category=category(fixed_gain),total_category=category(total_gain),
        terms=dict(zip(('parent_error','fixed_alignment','fixed_shift_energy','adaptation_alignment','adaptation_energy','child_error'),global_terms.tolist())),
        channel_terms=channel.tolist(),max_per_window_channel_identity_residual=float(np.max(abs(residual))))
    # Per-window/per-channel terms support later independent checks, not independent-observation inference.
    return record,terms


def projection(model,values,starts,proposal,params,action):
    weights,bias=effective_map(model)
    sums=np.zeros(5,dtype=np.float64)
    n=0
    worst=0.
    with torch.no_grad():
        for lo in range(0,len(starts),2048):
            x=values[starts[lo:lo+2048,None]+torch.arange(32,device='cuda')]
            z=proposal.transform(x,params,'P1',action=='CP1')
            px,pz=model(x).cpu().numpy().astype(np.float64),model(z).cpu().numpy().astype(np.float64)
            x64,z64=x.cpu().numpy().astype(np.float64),z.cpu().numpy().astype(np.float64)
            delta=z64-x64
            mean_delta=delta.mean(axis=1,keepdims=True)
            dm=mean_delta[:,0,None,:]*weights.sum(axis=2).T[None,:,:]
            dc=np.einsum('cht,ntc->nhc',weights,delta-mean_delta,optimize=True)
            predicted=np.einsum('cht,ntc->nhc',weights,x64,optimize=True)+bias.T[None,:,:]
            tolerance=64*np.finfo(np.float32).eps*max(1.,float(np.max(abs(px))),float(np.max(abs(pz))))
            error=max(float(np.max(abs(predicted-px))),float(np.max(abs(dm+dc-(pz-px)))))
            assert error<=tolerance,(error,tolerance)
            worst=max(worst,error)
            sums+=np.array([np.sum(dm*dm),np.sum(dc*dc),np.sum(2*dm*dc),np.sum((pz-px)**2),np.sum((dm+dc)**2)])
            n+=px.size
    result=dict(zip(('mean_projected_energy','centered_projected_energy','projection_cross_term','observed_forecast_change_energy','analytic_forecast_change_energy'),(sums/n).tolist()))
    result['max_affine_reconstruction_absolute_error']=worst
    result['interpretation']='Projected components need not be orthogonal; retain their signed cross term.'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root,out=Path(args.root),Path(args.output)
    h=module(root/'hydraulic_signal_validation.py')
    original=h.verify(root)
    plan=h.read(out/'plan.json')
    assert h.sha(__file__)==plan['script_sha256']
    assert h.sha(out/'PROTOCOL.md')==plan['protocol_text_sha256']
    assert h.read(out/'source_sync_receipt.json')['status']=='exact_commit_and_tree_verified'
    for name,expected in plan['inputs_sha256'].items():
        assert h.sha(root/name)==expected,name
    assert not (out/'decomposition.json').exists()
    assert plan['seeds']==list(h.SEEDS) and plan['actions']==['P1','CP1']
    assert plan['checkpoints']==['best','last'] and plan['splits']==['train','validation']
    assert not list(out.glob('*.npz')), 'Existing diagnostic outputs must never be overwritten'
    with (out/'execution.lock').open('x',encoding='utf-8') as guard:
        guard.write(h.sha(__file__))
    started=time.monotonic()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    arrays,params,_=h.prepared(root,original)
    core=h.module('core',original['frozen_core'])
    proposal=h.module('proposal',root/'proposal_median3.py')
    results,assets=[],{}
    for checkpoint in plan['checkpoints']:
        for seed in h.SEEDS:
            parent=core.DLinear(dict(original['training'],channels=8)).cuda().eval()
            parent.load_state_dict(torch.load(root/'runs'/f'H1_PARENT_s{seed}'/(checkpoint+'.pt'),weights_only=True,map_location='cuda'))
            for split in plan['splits']:
                values=torch.tensor(arrays[split],device='cuda')
                starts=torch.tensor(arrays[split+'_starts'],device='cuda')
                e=evaluate(parent,values,starts,proposal,params,'PARENT')
                if checkpoint=='best' and split=='validation':
                    with np.load(root/'runs'/f'H1_PARENT_s{seed}'/'best_errors.npz',allow_pickle=False) as saved:
                        np.testing.assert_array_equal(e,saved['error'])
                for action in plan['actions']:
                    child=core.DLinear(dict(original['training'],channels=8)).cuda().eval()
                    child.load_state_dict(torch.load(root/'runs'/f'H1_{action}_s{seed}'/(checkpoint+'.pt'),weights_only=True,map_location='cuda'))
                    f=evaluate(parent,values,starts,proposal,params,action)
                    c=evaluate(child,values,starts,proposal,params,action)
                    if checkpoint=='best' and split=='validation':
                        with np.load(root/'runs'/f'H1_{action}_s{seed}'/'best_errors.npz',allow_pickle=False) as saved:
                            np.testing.assert_array_equal(c,saved['error'])
                    record,terms=summarize(e,f,c)
                    projection_result=projection(parent,values,starts,proposal,params,action)
                    name=f'{checkpoint}_{split}_{action}_s{seed}.npz'
                    with (out/name).open('xb') as stream:
                        np.savez_compressed(stream,terms=terms,window_start=arrays[split+'_starts'],
                            cycle=arrays[split+'_cycles'][arrays[split+'_starts']])
                    assets[name]=h.sha(out/name)
                    record.update(checkpoint=checkpoint,split=split,action=action,seed=seed,windows=len(starts),projection=projection_result,asset=name)
                    results.append(record)
    assert len(results)==24
    for name,expected in plan['inputs_sha256'].items():
        assert h.sha(root/name)==expected,'Source changed during analysis: '+name
    h.write(out/'decomposition.json',dict(status='PASS',posthoc=True,records=results,assets_sha256=assets,
        source_plan_sha256=h.sha(out/'plan.json'),script_sha256=h.sha(__file__),process_seconds=time.monotonic()-started,
        new_trainings=0,new_predictions=0,new_thresholds=0,test_access=False,
        checkpoints_used='all three original seeds, best primary and step3000 last secondary',
        interpretation='Adaptation includes changes caused by the full original training and checkpoint-selection procedure; no causal isolation of optimizer or preprocessing mechanism.'))
    print(json.dumps(dict(status='PASS',records=len(results),process_seconds=time.monotonic()-started,
        primary=[{k:r[k] for k in ('action','seed','fixed_processing_gain_percent','retraining_adaptation_gain_percent','total_gain_percent')} for r in results if r['checkpoint']=='best' and r['split']=='validation'])),flush=True)


if __name__=='__main__':
    main()
