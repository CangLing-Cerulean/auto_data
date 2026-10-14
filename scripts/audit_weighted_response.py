"""Independent replay, source identity and paired response audit; no D/test."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import run_research as core

ROOT=Path('/root/auto_data_experiments/20261014_weighted_response_r1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def equal(a,b):
    if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)
def coords(state):
    return torch.stack([torch.cat((state[f'seasonal.{c}.weight'].T,state[f'seasonal.{c}.bias'][None],state[f'trend.{c}.weight'].T,state[f'trend.{c}.bias'][None])) for c in range(8)])
def snap(directory,step):return torch.load(directory/('initial.pt' if step==0 else f'step_{step}.pt'),map_location='cpu',weights_only=True)
def mse(error):return float(np.mean(error.astype(np.float64)**2))
def verify(out,plan):
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in plan['input_sha256'].items():assert sha(name)==digest,name
def category(v):return 'improve' if v>.25 else 'hurt' if v<-.25 else 'unchanged'

def audit(out):
    begin=time.perf_counter();assert out==ROOT and out.resolve()==ROOT
    plan,labels,pred,lock,source=(read(out/n) for n in ['plan.json','labels.json','predictions.json','prediction_lock.json','source_sync_receipt.json'])
    assert labels['status']==read(out/'status.json')['status']=='completed'
    assert source['status']=='exact_commit_and_tree_verified'
    assert source['plan_sha256']==labels['plan_sha256']==pred['plan_sha256']==lock['plan_sha256']==sha(out/'plan.json')
    assert source['predictions_sha256']==labels['predictions_sha256']==lock['predictions_sha256']==sha(out/'predictions.json')
    verify(out,plan);write(out/'audit.lock',dict(labels_sha256=sha(out/'labels.json')))
    expected=[j['id'] for j in plan['jobs']]
    assert len(expected)==len(set(expected))==36
    assert [r['id'] for r in labels['results']]==expected==[r['id'] for r in pred['records']]
    assert {p.name for p in (out/'runs').iterdir()}==set(expected)
    oldpred=read(plan['old_predictions']);oldlabels=read(plan['old_labels'])
    assert pred['old_predictions_sha256']==sha(plan['old_predictions']) and not pred['new_weighting_labels_observed']
    rows=[];fixed=[];states=0;residuals=0;replays=[];costs=[]
    for task in plan['tasks']:
        tid=task['id'];values={};starts={}
        with np.load(task['abc'],allow_pickle=False) as z:
            for split in ('A','B','C'):
                lo,hi=task['partitions'][split]
                np.testing.assert_array_equal(z[split+'_starts'],(np.arange(hi-lo)[:,None]*60+np.arange(21)).reshape(-1))
                np.testing.assert_array_equal(z[split+'_cycles'],np.repeat(np.arange(lo,hi),60))
                np.testing.assert_array_equal(z[split+'_base_rows'],np.arange(lo*60,hi*60))
                values[split]=torch.from_numpy(z[split+'_values']).cuda();starts[split]=torch.from_numpy(z[split+'_starts']).cuda()
            window_cycles=z['A_cycles'][z['A_starts']];window_base_rows=z['A_base_rows'][z['A_starts']]
            first=window_cycles<task['partitions']['A'][0]+154
        n=len(first);assert n==6468 and first.sum()==3234
        weights={}
        for action,path in task['probabilities'].items():
            favored=first if action.endswith('FIRST') else ~first
            w=np.where(favored,2. if action.startswith('KEEP') else 1.5,0. if action.startswith('KEEP') else .5)
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(w,n*z['probability'])
                np.testing.assert_array_equal(window_cycles,z['window_cycles'])
                np.testing.assert_array_equal(window_base_rows,z['window_base_rows'])
            weights[action]=torch.tensor(w,dtype=torch.float32,device='cuda')
        offset=torch.arange(40,device='cuda');cdf=np.full(n,1/n).cumsum();cdf[-1]=1.
        model=core.DLinear(dict(plan['training'],channels=8)).cuda()
        @torch.no_grad()
        def errors(state,split):
            model.load_state_dict(state);model.eval();chunks=[]
            for lo in range(0,len(starts[split]),2048):
                batch=values[split][starts[split][lo:lo+2048,None]+offset]
                chunks.append((model(batch[:,:32])-batch[:,32:]).cpu().numpy())
            e=np.concatenate(chunks);assert np.isfinite(e).all();return e
        for job in (j for j in plan['jobs'] if j['task']==tid):
            directory=out/'runs'/job['id'];seed=job['seed'];action=job['action']
            result=read(directory/'result.json');cost=read(directory/'timing.json')['seconds']
            assert next(r for r in labels['results'] if r['id']==job['id'])==dict(result,seconds=cost)
            assert result['status']==read(directory/'status.json')['status']=='completed' and result['steps']==result['steps_completed']==3000
            assert result['checkpoints']==plan['steps'] and result['witness_steps']==[1,501]
            for k,v in job.items():assert result[k]==v
            assert result['plan_sha256']==sha(out/'plan.json') and result['predictions_sha256']==sha(out/'predictions.json')
            for name,digest in result['assets_sha256'].items():assert sha(directory/name)==digest
            assert set(result['assets_sha256'])=={'initial.pt','sampling.npz','selected_B_errors.npz','selected_C_errors.npz',*[f'step_{k}.pt' for k in plan['steps']], 'witness_1.pt','witness_501.pt'}
            parentdir=Path(job['parent']);parent=read(parentdir/'result.json');initial=snap(parentdir,0)
            rng=np.random.default_rng(seed+1000);draws=np.zeros(n,dtype=np.int64);rng_states={0:rng.bit_generator.state}
            for step in range(1,3001):
                ids=np.searchsorted(cdf,rng.random(256),side='right');np.add.at(draws,ids,1)
                if step in plan['steps']+[1,501]:rng_states[step]=rng.bit_generator.state
            with np.load(directory/'sampling.npz',allow_pickle=False) as z:np.testing.assert_array_equal(draws,z['draw_count'])
            for step in [0]+plan['steps']+[1,501]:
                current=torch.load(directory/f'witness_{step}.pt',weights_only=True,map_location='cpu') if step in [1,501] else snap(directory,step)
                assert current['step']==step;packed=coords(current['model']);equal(packed,current['original_coordinates'])
                assert torch.isfinite(packed).all();equal(current['rng_numpy'],rng_states[step])
                if step==0:
                    equal(current['model'],initial['model']);equal(current['optimizer'],initial['optimizer'])
                    equal(current['rng_torch'],initial['rng_torch']);equal(current['rng_cuda'],initial['rng_cuda'])
                else:
                    assert len(current['optimizer']['state'])==32
                    for val in current['optimizer']['state'].values():
                        assert val['step'].item()==step and torch.isfinite(val['exp_avg']).all() and torch.isfinite(val['exp_avg_sq']).all()
                states+=1
            for anchor in [0,500]:
                state=snap(directory,anchor);model.load_state_dict(state['model']);model.train()
                opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.);opt.load_state_dict(state['optimizer'])
                torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state(state['rng_cuda']);one_rng=np.random.default_rng();one_rng.bit_generator.state=state['rng_numpy']
                ids=np.searchsorted(cdf,one_rng.random(256),side='right');index=torch.from_numpy(ids).cuda()
                batch=values['A'][starts['A'][index,None]+offset];opt.zero_grad(set_to_none=True)
                error=model(batch[:,:32])-batch[:,32:]
                # Explicit full-element mean preserves fixed batch denominator, including zeros.
                loss=(error.square()*weights[action][index,None,None]).mean();loss.backward();opt.step()
                witness=torch.load(directory/f'witness_{anchor+1}.pt',weights_only=True,map_location='cpu')
                equal(model.state_dict(),witness['model']);equal(opt.state_dict(),witness['optimizer'])
                equal(one_rng.bit_generator.state,witness['rng_numpy']);equal(torch.get_rng_state(),witness['rng_torch']);equal(torch.cuda.get_rng_state(),witness['rng_cuda'])
                replays.append(dict(id=job['id'],from_step=anchor,to_step=anchor+1,bitexact=True))
            baseline=mse(errors(snap(parentdir,parent['selected_step'])['model'],'C'));assert baseline==parent['selected_losses']['C']>0
            olddir=Path(job['sampling_directory']);oldresult=read(olddir/'result.json')
            registered=next(r for r in oldlabels['results'] if r['id']==olddir.name)
            assert registered['assets_sha256']==oldresult['assets_sha256'] and registered['selected_losses']==oldresult['selected_losses']
            equal(snap(olddir,0)['model'],initial['model'])
            curves={}
            for name,path,record in [('W',directory,result),('S',olddir,oldresult)]:
                curve={}
                for step in plan['steps']:
                    st=snap(path,step)['model'];b=mse(errors(st,'B'));c=mse(errors(st,'C'))
                    assert b==record['selection_losses'][str(step)];curve[step]=dict(B=b,C=c,gain=100*(baseline-c)/baseline)
                selected=min(plan['steps'],key=lambda k:curve[k]['B']);assert selected==record['selected_step']
                for split in ['B','C']:
                    e=errors(snap(path,selected)['model'],split)
                    with np.load(path/f'selected_{split}_errors.npz',allow_pickle=False) as z:np.testing.assert_array_equal(e,z['error'])
                    assert mse(e)==record['selected_losses'][split];residuals+=1
                curves[name]=curve
            forecast=next(r for r in pred['records'] if r['id']==job['id'])
            oldforecast=next(r for r in oldpred['records'] if (r['task'],r['seed'],r['action'])==(tid,seed,action))['methods']['M1']
            assert forecast['M1_gain_percent']==oldforecast['gain_percent'] and forecast['M1_selected_step']==oldforecast['selected_step'] and not forecast['W_training_started']
            s=curves['S'][oldresult['selected_step']]['gain'];w=curves['W'][result['selected_step']]['gain'];z=forecast['M1_gain_percent']
            rows.append(dict(task=tid,seed=seed,action=action,id=job['id'],S_gain=s,W_gain=w,gap_pp=w-s,
                S_selected_step=oldresult['selected_step'],W_selected_step=result['selected_step'],M1_gain=z,
                mean_only_joint_absolute_error_lower_bound_pp=abs(w-s)/2,M1_joint_absolute_error_pp=(abs(z-s)+abs(z-w))/2))
            for step in plan['steps']:
                fixed.append(dict(task=tid,seed=seed,action=action,step=step,S_gain=curves['S'][step]['gain'],W_gain=curves['W'][step]['gain'],gap_pp=curves['W'][step]['gain']-curves['S'][step]['gain']))
            costs.append(dict(id=job['id'],seconds=cost));print(json.dumps(dict(audited=len(rows),id=job['id'])),flush=True)
    assert len(rows)==36 and len(fixed)==216 and states==324 and len(replays)==72 and residuals==144
    cells=[];decisions=[]
    for task in plan['tasks']:
        tid=task['id']
        for action in plan['actions']:
            subset=[r for r in rows if r['task']==tid and r['action']==action];assert len(subset)==3
            s,w,z=(float(np.mean([r[k] for r in subset])) for k in ['S_gain','W_gain','M1_gain']);gaps=[r['gap_pp'] for r in subset]
            stable='W_better' if min(gaps)>.25 else 'W_worse' if max(gaps)<-.25 else 'opposite_beyond_band' if min(gaps)<-.25 and max(gaps)>.25 else 'no_stable_material_gap'
            cells.append(dict(task=tid,action=action,S_gain=s,W_gain=w,gap_pp=w-s,seed_gaps_pp=gaps,gap_class=stable,
                S_category=category(s),W_category=category(w),S_eligible=s>=1,W_eligible=w>=1,M1_gain=z,
                mean_only_joint_absolute_error_lower_bound_pp=abs(w-s)/2,M1_S_absolute_error_pp=abs(z-s),M1_W_absolute_error_pp=abs(z-w)))
        group=[c for c in cells if c['task']==tid];best=max(group,key=lambda c:c['M1_gain']);choice=best['action'] if best['M1_gain']>=1 else 'PARENT'
        decisions.append(dict(task=tid,M1_choice=choice,**{name:dict(realized_gain=0. if choice=='PARENT' else best[name+'_gain'],oracle_gain=max([0.]+[c[name+'_gain'] for c in group])) for name in ['S','W']}))
        for name in ['S','W']:
            decisions[-1][name]['regret_pp']=decisions[-1][name]['oracle_gain']-decisions[-1][name]['realized_gain']
    verify(out,plan);assert sha(out/'labels.json')==read(out/'audit.lock')['labels_sha256']
    write(out/'final_audit.json',dict(status='PASS',new_full_trainings=36,cumulative_full_trainings=409,states_verified=states,
        selected_residuals_verified=residuals,one_step_replays=replays,diagnostic_optimizer_updates=72,records=rows,fixed_step_records=fixed,cells=cells,decisions=decisions,
        stable_material_cells=sum(c['gap_class'] in ['W_better','W_worse'] for c in cells),M1_S_MAE_pp=float(np.mean([c['M1_S_absolute_error_pp'] for c in cells])),
        M1_W_MAE_pp=float(np.mean([c['M1_W_absolute_error_pp'] for c in cells])),
        M1_joint_MAE_pp=float(np.mean([c[f'M1_{name}_absolute_error_pp'] for c in cells for name in ['S','W']])),
        M1_worst_absolute_error_pp=max(c[f'M1_{name}_absolute_error_pp'] for c in cells for name in ['S','W']),
        mean_only_joint_MAE_lower_bound_pp=float(np.mean([c['mean_only_joint_absolute_error_lower_bound_pp'] for c in cells])),
        run_costs=costs,training_seconds=sum(c['seconds'] for c in costs),audit_seconds=time.perf_counter()-begin,original_inputs_unchanged=True,
        plan_sha256=sha(out/'plan.json'),labels_sha256=sha(out/'labels.json'),predictions_sha256=sha(out/'predictions.json'),D_access=False,test_access=False,
        independent_source_claim=False,new_method_claim=False))
    print(json.dumps(dict(status='PASS',states=states,records=36)),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    try:audit(args.output)
    except Exception:
        write(args.output/'audit_failure.json',dict(status='failed',error=traceback.format_exc()));raise
