"""Adaptive full-support sampling with explicit inverse-probability correction."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,prepare,evaluate,write_json
from importance_utils import gradient_norm,distribution


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--recipe',choices=['adaptive_is','replay_uncorrected'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--reference')
    args=parser.parse_args()
    assert Path(args.run_id).name==args.run_id
    folder=ROOT/'research_runs/importance'; formal=ROOT/'research_runs/selection/protocol.json'
    p=json.loads(formal.read_text()); manifest_path=ROOT/'data/metropt3_v1/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    run=ROOT/'research_runs/results'/args.run_id; run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists(): raise RuntimeError('Cannot overwrite completed run')
    reference=None
    if args.recipe=='replay_uncorrected':
        assert args.reference and Path(args.reference).name==args.reference
        reference=run.parent/args.reference
        source_identity=json.loads((reference/'identity.json').read_text())
        assert source_identity['seed']==args.seed and source_identity['recipe']=='adaptive_is'
        assert (reference/'result.json').exists()
    identity={'recipe':args.recipe,'seed':args.seed,'preprocessing':'baseline',
        'manifest_sha256':digest(manifest_path),'protocol_sha256':digest(formal),
        'sampling_protocol_sha256':digest(folder/'PROTOCOL.md'),'core_sha256':digest(ROOT/'scripts/run_research.py'),
        'code_sha256':digest(Path(__file__)),'helper_sha256':digest(ROOT/'scripts/importance_utils.py'),
        'reference':args.reference,'reference_result_sha256':digest(reference/'result.json') if reference else None,
        'cdf_backend':'numpy','correction_note_sha256':digest(folder/'CORRECTION.md')}
    if (run/'identity.json').exists(): assert json.loads((run/'identity.json').read_text())==identity
    else:
        write_json(run/'identity.json',identity)
        for source,name in [(Path(__file__),'code_snapshot.py'),(ROOT/'scripts/run_research.py','core_snapshot.py'),
            (ROOT/'scripts/importance_utils.py','helper_snapshot.py'),(formal,'protocol.json'),(folder/'PROTOCOL.md','sampling_protocol.md')]:
            (run/name).write_bytes(source.read_bytes())
        (run/'correction_note.md').write_bytes((folder/'CORRECTION.md').read_bytes())
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    old=json.loads((run/'status.json').read_text()).get('elapsed_seconds',0) if (run/'status.json').exists() else 0
    remaining=7200-used
    if remaining<=0: raise RuntimeError('Budget exhausted')
    begun=time.monotonic()
    def status(phase,step): write_json(run/'status.json',{'phase':phase,'step':step,'elapsed_seconds':old+time.monotonic()-begun})
    def budget(step):
        if time.monotonic()-begun>=remaining:
            status('budget_exhausted',step); raise RuntimeError('Budget exhausted')
    status('preparing',0)
    torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    arrays,indices,groups,audit=prepare(p,manifest)
    table=ROOT/'research_runs/selection/window_table.npz'
    with np.load(table) as saved:
        assert np.array_equal(saved['start_row_zero_based'],indices['train'])
        threshold=float(np.quantile(saved['input_variation'],.25))
    values=arrays['validation'].astype(np.float64)
    prefix=np.r_[0,np.cumsum(np.abs(np.diff(values,axis=0)).mean(1))]
    s=indices['validation']; score=(prefix[s+31]-prefix[s])/31
    groups['low_quartile']=score<=threshold
    groups['middle']=(score>threshold)&~groups['high_input_variation']
    write_json(run/'data_audit.json',audit)
    write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
        'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()})
    data={k:torch.from_numpy(v).cuda() for k,v in arrays.items()}
    starts={k:torch.from_numpy(v).cuda() for k,v in indices.items()}
    offsets=torch.arange(40,device='cuda'); n=len(starts['train'])
    model=DLinear(p).cuda(); optimizer=torch.optim.Adam(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
    # Saving initialization makes model-pair verification independent of performance.
    if not (run/'initial.pt').exists(): torch.save(model.state_dict(),run/'initial.pt')
    rng=torch.Generator(device='cuda').manual_seed(args.seed+1000)
    counts=torch.zeros(n,device='cuda',dtype=torch.int64)
    step0,best,best_step,history,phases=0,float('inf'),0,[],[]
    if (run/'latest.pt').exists():
        checkpoint=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        rng.set_state(checkpoint['sampling_rng'].cpu()); counts=checkpoint['draw_counts']
        step0,best,best_step=checkpoint['step'],checkpoint['best_score'],checkpoint['best_step']
        history,phases=checkpoint['history'],checkpoint['phases']
    for phase_start in range(step0,p['steps'],500):
        budget(phase_start); score_started=time.monotonic()
        norm=torch.zeros(n,device='cuda',dtype=torch.float64)
        if reference is not None:
            path=reference/f'phase_{phase_start:04d}.npz'
            expected=next(x['asset_sha256'] for x in json.loads((reference/'sampling_audit.json').read_text())['phases'] if x['start_step']==phase_start)
            assert digest(path)==expected
            with np.load(path) as saved:
                q=torch.from_numpy(saved['probability']).cuda(); norm=torch.from_numpy(saved['gradient_norm']).cuda()
        elif phase_start==0:
            q=torch.full((n,),1/n,device='cuda',dtype=torch.float64)
        else:
            model.eval(); torch.save(model.state_dict(),run/f'scoring_{phase_start:04d}.pt')
            for lo in range(0,n,p['evaluation_batch_size']):
                budget(phase_start)
                batch=data['train'][starts['train'][lo:lo+p['evaluation_batch_size'],None]+offsets]
                norm[lo:lo+len(batch)]=gradient_norm(model,batch[:,:32],batch[:,32:])
            q=distribution(norm)
        assert torch.isfinite(q).all() and torch.all(q>0)
        assert abs(q.sum().item()-1)<1e-12
        inverse=1/(n*q)
        weights=inverse.float() if args.recipe=='adaptive_is' else torch.ones(n,device='cuda')
        cdf=torch.from_numpy(np.cumsum(q.cpu().numpy())).cuda(); cdf[-1]=1
        phase_counts=torch.zeros_like(counts)
        scoring_seconds=time.monotonic()-score_started
        model.train(); losses=[]
        for step in range(phase_start+1,phase_start+501):
            budget(step-1)
            u=torch.rand(p['batch_size'],device='cuda',dtype=torch.float64,generator=rng)
            selected=torch.searchsorted(cdf,u,right=True)
            phase_counts.scatter_add_(0,selected,torch.ones_like(selected))
            batch=data['train'][starts['train'][selected,None]+offsets]
            optimizer.zero_grad(set_to_none=True)
            per_window=(model(batch[:,:32])-batch[:,32:]).square().mean((1,2))
            loss=(per_window*weights[selected]).mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        counts+=phase_counts
        asset=run/f'phase_{phase_start:04d}.npz'
        np.savez_compressed(asset,gradient_norm=norm.cpu().numpy(),probability=q.cpu().numpy(),
            loss_weight=weights.cpu().numpy(),draw_count=phase_counts.cpu().numpy())
        uniform_second=norm.square().mean().item()
        corrected_second=(norm.square()/(n*n*q)).sum().item()
        phases.append({'start_step':phase_start,'end_step':phase_start+500,'asset_sha256':digest(asset),
            'scoring_seconds':scoring_seconds,'scoring_forward_presentations':n if reference is None and phase_start>0 else 0,
            'uniform_gradient_second_moment':uniform_second,'corrected_gradient_second_moment':corrected_second,
            'second_moment_reduction_pct':100*(1-corrected_second/uniform_second) if uniform_second else None,
            'loss_weight_min':weights.min().item(),'loss_weight_max':weights.max().item(),
            'score_checkpoint_sha256':digest(run/f'scoring_{phase_start:04d}.pt') if reference is None and phase_start>0 else None,
            'actual_draws':int(phase_counts.sum()),'distinct_windows':int(torch.count_nonzero(phase_counts))})
        metrics=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
        value=metrics['groups']['all']['NMSE']
        entry={'step':phase_start+500,'train_loss':float(np.mean(losses)),'validation':metrics['groups']['all']}
        history.append(entry)
        if value<best:
            best,best_step=value,phase_start+500
            torch.save(model.state_dict(),run/'best.tmp'); (run/'best.tmp').replace(run/'best.pt')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'sampling_rng':rng.get_state(),
            'draw_counts':counts,'step':phase_start+500,'best_score':best,'best_step':best_step,
            'history':history,'phases':phases},run/'latest.tmp'); (run/'latest.tmp').replace(run/'latest.pt')
        write_json(run/'history.json',history); status('training',phase_start+500)
        print(json.dumps(entry),flush=True)
    assert int(counts.sum())==768000
    model.load_state_dict(torch.load(run/'best.pt',weights_only=True))
    final=evaluate(model,data['validation'],starts['validation'],groups,p,'baseline')
    np.savez_compressed(run/'window_draws.npz',start_row=indices['train'],draw_count=counts.cpu().numpy())
    write_json(run/'sampling_audit.json',{'window_table_sha256':digest(table),'eligible_windows':n,'retained_windows':n,
        'source_train_sha256':next(s['sha256'] for s in manifest['splits'] if s['name']=='train'),
        'unique_windows_seen':int(torch.count_nonzero(counts)),'phases':phases,'total_draws':int(counts.sum()),
        'scoring_forward_presentations':sum(x['scoring_forward_presentations'] for x in phases),
        'scoring_seconds':sum(x['scoring_seconds'] for x in phases),'window_draws_sha256':digest(run/'window_draws.npz'),
        'mapping':'start_row s: original train.csv data rows [s,s+40), input [s,s+32), target [s+32,s+40)'})
    final.update({'run_id':args.run_id,'identity':identity,'steps_completed':3000,'samples_presented':int(counts.sum()),
        'best_step':best_step,'history':history,'sensor_columns':manifest['sensor_columns'],
        'elapsed_seconds':old+time.monotonic()-begun})
    write_json(run/'result.json',final); status('completed',3000)
    print(json.dumps({'completed':args.run_id,**final['groups']['all']}),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
