"""Training-window membership/weight/sampling research; immutable raw data."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, digest, prepare, evaluate, write_json

QUIET_LOSS_WEIGHT=0.25
RANDOM_SELECTION_SEED=20260926
HIGH_VARIATION_SAMPLING_WEIGHT=2.0
SWAP_REMOVAL_SEED=20260927
PLACEBO_RESTORE_SEED=20260928
MONTH_RECIPES=('month_matched_drop','protect_lps_matched','placebo_matched_swap')
ALIGNMENT_RECIPES=('future_conflict_downweight','future_conflict_shuffled')


def variation(values, starts, length):
    prefix=np.r_[0,np.cumsum(np.abs(np.diff(values.astype(np.float64),axis=0)).mean(axis=1))]
    return (prefix[starts+length-1]-prefix[starts])/(length-1)


def training_months(starts, input_length):
    with (ROOT/'data/metropt3_v1/train.csv').open(newline='') as handle:
        rows=csv.reader(handle)
        next(rows)
        months=np.array([row[1][:7] for row in rows])
    return months[starts+input_length-1]


def training_lps_mask(starts, input_length):
    values=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=13)
    if not np.isin(values,[0,1]).all():
        raise ValueError('Unexpected LPS states')
    prefix=np.r_[0,np.cumsum(values==1)]
    return (prefix[starts+input_length]-prefix[starts])>0


def alignment_loss_weights(recipe):
    folder=ROOT/'research_runs/transfer'
    metadata=json.loads((folder/'alignment_recipe.json').read_text())
    assert metadata['artifact_sha256']==digest(folder/'alignment_scores.npz')
    assert metadata['window_table_sha256']==digest(ROOT/'research_runs/selection/window_table.npz')
    key='loss_weight' if recipe=='future_conflict_downweight' else 'control_loss_weight'
    with np.load(folder/'alignment_scores.npz') as saved: return saved[key]


def sampling_policy(recipe, score, threshold, months=None, rare=None, aligned_loss=None):
    sampling=np.ones(len(score),dtype=np.float64)
    loss=np.ones(len(score),dtype=np.float64)
    if recipe=='drop_quiet':
        sampling[score<=threshold]=0
    elif recipe=='downweight_quiet':
        loss[score<=threshold]=QUIET_LOSS_WEIGHT
    elif recipe=='random_drop':
        count=int(np.count_nonzero(score<=threshold))
        chosen=np.random.default_rng(RANDOM_SELECTION_SEED).choice(len(score),count,replace=False)
        sampling[chosen]=0
    elif recipe=='oversample_high':
        sampling[score>np.quantile(score,.9)]=HIGH_VARIATION_SAMPLING_WEIGHT
    elif recipe in ALIGNMENT_RECIPES:
        if aligned_loss is None or len(aligned_loss)!=len(score) or not np.isin(aligned_loss,[.25,1]).all():
            raise ValueError('Frozen alignment loss weights required')
        loss=aligned_loss.copy()
    elif recipe in MONTH_RECIPES:
        if months is None or len(months)!=len(score):
            raise ValueError('Training month labels required')
        rng=np.random.default_rng(RANDOM_SELECTION_SEED)
        for month in np.unique(months):
            members=np.flatnonzero(months==month)
            count=int(np.count_nonzero(score[members]<=threshold))
            sampling[rng.choice(members,count,replace=False)]=0
        if recipe!='month_matched_drop':
            if rare is None or len(rare)!=len(score):
                raise ValueError('Training LPS input mask required')
            original=sampling.copy()
            remove_rng=np.random.default_rng(SWAP_REMOVAL_SEED)
            restore_rng=np.random.default_rng(PLACEBO_RESTORE_SEED)
            for month in np.unique(months):
                members=months==month
                rare_removed=np.flatnonzero(members & rare & (original==0))
                count=len(rare_removed)
                remove_pool=np.flatnonzero(members & ~rare & (original>0))
                remove=remove_rng.choice(remove_pool,count,replace=False)
                if recipe=='protect_lps_matched':
                    restore=rare_removed
                else:
                    restore_pool=np.flatnonzero(members & ~rare & (original==0))
                    restore=restore_rng.choice(restore_pool,count,replace=False)
                sampling[remove]=0
                sampling[restore]=1
    elif recipe not in ('uniform','pca_rotation','leading_sources','frozen_mask'):
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
    parser.add_argument('--retention-mask')
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
    assert bool(args.retention_mask)==(args.recipe=='frozen_mask')
    if args.retention_mask:
        from cluster_selection import mask_identity
        identity.update(mask_identity(args.retention_mask))
    if args.recipe=='pca_rotation':
        identity.update(preprocessing='train_pca_rotation',
            representation_protocol_sha256=digest(ROOT/'research_runs/representation/PROTOCOL.md'),
            representation_helper_sha256=digest(ROOT/'scripts/channel_rotation.py'))
    alignment=None
    if args.recipe=='leading_sources':
        from leading_sources import load_mapping
        mapping,source_hash=load_mapping()
        identity.update(preprocessing='anchored_source_history',source_mapping=mapping,source_diagnostic_sha256=source_hash,
            source_protocol_sha256=digest(ROOT/'research_runs/representation/LEADING_TRAINING_PROTOCOL.md'),
            source_helper_sha256=digest(ROOT/'scripts/leading_sources.py'))
    if args.recipe in ALIGNMENT_RECIPES:
        recipe_path=ROOT/'research_runs/transfer/alignment_recipe.json'
        alignment=json.loads(recipe_path.read_text())
        assert alignment['source_train_sha256']==next(s['sha256'] for s in manifest['splits'] if s['name']=='train')
        identity.update(selection_recipe_sha256=digest(recipe_path),selection_asset_sha256=digest(ROOT/'research_runs/transfer/alignment_scores.npz'))
        assert identity['selection_asset_sha256']==alignment['artifact_sha256']
    if (run/'identity.json').exists():
        if json.loads((run/'identity.json').read_text())!=identity:
            raise RuntimeError('Resume identity mismatch')
    else:
        write_json(run/'identity.json',identity)
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
        (run/'core_snapshot.py').write_bytes((ROOT/'scripts/run_research.py').read_bytes())
        (run/'protocol.json').write_bytes(config_path.read_bytes())
        if args.retention_mask:
            from cluster_selection import snapshot_mask
            snapshot_mask(args.retention_mask,run)
        if alignment is not None: (run/'selection_recipe.json').write_bytes(recipe_path.read_bytes())
        if args.recipe=='leading_sources':
            (run/'source_protocol.md').write_bytes((ROOT/'research_runs/representation/LEADING_TRAINING_PROTOCOL.md').read_bytes())
            (run/'source_helper.py').write_bytes((ROOT/'scripts/leading_sources.py').read_bytes())
        if args.recipe=='pca_rotation':
            (run/'representation_protocol.md').write_bytes((ROOT/'research_runs/representation/PROTOCOL.md').read_bytes())
            (run/'representation_helper.py').write_bytes((ROOT/'scripts/channel_rotation.py').read_bytes())
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
    old=json.loads((run/'status.json').read_text()).get('elapsed_seconds',0) if (run/'status.json').exists() else 0.
    remaining=p['session_gpu_wall_budget_seconds']-used
    if args.retention_mask:
        from cluster_selection import ROUND_LIMIT
        remaining=min(remaining,ROUND_LIMIT-used-60)
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
    basis=None
    if args.recipe=='pca_rotation':
        from channel_rotation import fit_rotation
        basis,eigen,covariance,rotation_audit=fit_rotation(arrays['train'])
        np.savez_compressed(run/'rotation.npz',basis=basis,eigenvalues=eigen,covariance=covariance)
        rotation_audit['asset_sha256']=digest(run/'rotation.npz')
        write_json(run/'rotation_audit.json',rotation_audit)
    scores={split:variation(values,indices[split],p['input_length']) for split,values in arrays.items()}
    threshold=float(np.quantile(scores['train'],.25))
    groups['low_quartile']=scores['validation']<=threshold
    groups['middle']=(scores['validation']>threshold)&~groups['high_input_variation']
    months=training_months(indices['train'],p['input_length']) if args.recipe in MONTH_RECIPES else None
    rare=training_lps_mask(indices['train'],p['input_length']) if args.recipe in MONTH_RECIPES[1:] else None
    aligned_loss=alignment_loss_weights(args.recipe) if alignment is not None else None
    sampling,loss_weight,prob=sampling_policy(args.recipe,scores['train'],threshold,months,rare,aligned_loss)
    if args.retention_mask:
        from cluster_selection import load_mask
        retained,prob,mask_meta=load_mask(args.retention_mask,indices['train'])
        assert mask_meta['stage']==100
        sampling=retained.astype(np.float64)
    table=ROOT/'research_runs/selection/window_table.npz'
    if table.exists():
        with np.load(table) as saved:
            if not np.array_equal(saved['start_row_zero_based'],indices['train']) or not np.array_equal(saved['input_variation'],scores['train']):
                raise RuntimeError('Window metadata changed')
    else:
        np.savez_compressed(table,start_row_zero_based=indices['train'],input_variation=scores['train'])
    quiet=scores['train']<=threshold
    high=scores['train']>np.quantile(scores['train'],.9)
    sample_audit={'recipe':args.recipe,'window_table_sha256':digest(table),'source_train_sha256':next(s['sha256'] for s in manifest['splits'] if s['name']=='train'),
                  'eligible_windows':len(prob),'retained_windows':int(np.count_nonzero(prob)),
                  'removed_windows':int(np.count_nonzero(prob==0)),'low_variation_threshold':threshold,
                  'quiet_windows':int(quiet.sum()),'expected_quiet_sampling_mass':float(prob[quiet].sum()),
                  'expected_loss_weight':float(np.dot(prob,loss_weight)),'sampling_method':'common_uniform_inverse_cdf',
                  'expected_quiet_loss_mass':float(np.dot(prob[quiet],loss_weight[quiet])/np.dot(prob,loss_weight)),
                  'high_variation_threshold':float(np.quantile(scores['train'],.9)),
                  'high_variation_windows':int(high.sum()),'expected_high_sampling_mass':float(prob[high].sum()),
                  'mapping':'window_table start row s refers to original train.csv data rows [s,s+40), input [s,s+32), targets [s+32,s+40)'}
    if months is not None:
        sample_audit['month_matching']={m:{'eligible':int(np.sum(months==m)),
            'quiet_reference_removed':int(np.sum((months==m)&quiet)),
            'actual_removed':int(np.sum((months==m)&(prob==0)))} for m in np.unique(months)}
        assert all(v['quiet_reference_removed']==v['actual_removed'] for v in sample_audit['month_matching'].values())
    if rare is not None:
        original,_,_=sampling_policy('month_matched_drop',scores['train'],threshold,months)
        sample_audit['lps_support']={'definition':'any raw LPS=1 in 32 input records; training only',
            'eligible':int(rare.sum()),'retained':int(np.sum(rare&(prob>0))),
            'expected_draws':float(prob[rare].sum()*p['steps']*p['batch_size']),
            'restored_windows':int(np.sum((original==0)&(sampling>0))),
            'replacement_removed_windows':int(np.sum((original>0)&(sampling==0))),
            'swap_removal_seed':SWAP_REMOVAL_SEED,'placebo_restore_seed':PLACEBO_RESTORE_SEED}
    if alignment is not None:
        assert np.isclose(sample_audit['expected_loss_weight'],alignment['expected_loss_weight'])
        sample_audit['alignment']={'recipe_sha256':identity['selection_recipe_sha256'],
            'asset_sha256':identity['selection_asset_sha256'],'downweighted_windows':int(np.sum(loss_weight<1)),
            'teacher_run':alignment['teacher_run'],'source_rows':alignment['source_rows'],'future_rows':alignment['future_rows']}
    write_json(run/'sampling_audit.json',sample_audit)
    write_json(run/'data_audit.json',audit)
    write_json(run/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()})
    data={s:torch.from_numpy(v).cuda() for s,v in arrays.items()}
    starts={s:torch.from_numpy(v).cuda() for s,v in indices.items()}
    cdf=torch.from_numpy(np.cumsum(prob)).cuda()
    cdf[-1]=1.
    weights=torch.from_numpy(loss_weight.astype(np.float32)).cuda()
    model=DLinear(p).cuda()
    if args.recipe=='leading_sources':
        from leading_sources import SourceDLinear
        model=SourceDLinear(model,mapping).cuda()
        assert sum(v.numel() for v in model.parameters())==7920
    if basis is not None:
        from channel_rotation import RotatedDLinear
        model=RotatedDLinear(model,basis).cuda()
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
                         'actual_high_draws':int(draws[high].sum()),
                         'total_draws':int(draws.sum()),'sampling_weights_sha256':digest(run/'sampling_weights.npz')})
    if rare is not None:
        sample_audit['lps_support'].update(actual_draws=int(draws[rare].sum()),distinct_seen=int(np.sum(rare&(draws>0))))
    if alignment is not None:
        sample_audit['alignment'].update(actual_downweighted_draws=int(draws[loss_weight<1].sum()),
            actual_mean_loss_weight=float(np.dot(draws,loss_weight)/draws.sum()))
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
