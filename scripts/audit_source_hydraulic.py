"""Independent full-pool SOURCE audit; no import of the response implementation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/root/auto_data_experiments/20261014_source_hydraulic_r1')
ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
METHODS=('M1','RAW','FIXED','SHORT','SOURCE')
GRID=(500,1000,1500,2000,2500,3000)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def category(x):return 'improve' if x>.25 else 'hurt' if x<-.25 else 'unchanged'
def close(a,b,atol=1e-9):assert abs(a-b)<=atol,(a,b)
def verify(out,plan,pred):
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for item in plan['prediction_inputs']+plan['analysis_inputs']:assert sha(item['path'])==item['sha256'],item['path']
    for name,digest in pred['assets_sha256'].items():assert sha(out/name)==digest,name

def pack(values):
    return np.stack([np.concatenate([values[f'seasonal.{c}.weight'].T,values[f'seasonal.{c}.bias'][None],
        values[f'trend.{c}.weight'].T,values[f'trend.{c}.bias'][None]]) for c in range(8)])

def parent_state(path,step):
    s=torch.load(path,map_location='cpu',weights_only=True);assert s['step']==step
    names=[f'{kind}.{c}.{part}' for kind in ('seasonal','trend') for c in range(8) for part in ('weight','bias')]
    assert list(s['model'])==names
    group,=s['optimizer']['param_groups'];assert group['params']==list(range(32))
    assert group['lr']==.001 and group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    theta=pack({k:v.numpy().astype(np.float64) for k,v in s['model'].items()})
    assert np.array_equal(theta,s['original_coordinates'].numpy())
    diagonal={}
    for i,name in enumerate(names):
        state=s['optimizer']['state'][i];assert float(state['step'])==step
        v=state['exp_avg_sq'].numpy().astype(np.float64)
        assert v.shape==tuple(s['model'][name].shape) and np.isfinite(v).all() and (v>=0).all()
        diagonal[name]=1/(np.sqrt(v/(1-.999**step))+1e-8)
    return theta,pack(diagonal)

@torch.no_grad()
def audit_task(out,task,pred,core):
    tid=task['id'];z=np.load(task['abc'],allow_pickle=False)
    values={s:torch.tensor(z[f'{s}_values'],device='cuda') for s in 'ABC'}
    starts={s:torch.tensor(z[f'{s}_starts'],device='cuda') for s in 'ABC'}
    first=z['A_cycles'][z['A_starts']]<task['partitions']['A'][0]+154
    assert len(first)==6468 and first.sum()==3234
    masses={'PARENT':np.full(6468,1/6468)}
    for action in ACTIONS:
        mass=(first if action=='KEEP_FIRST' else ~first if action=='KEEP_SECOND' else np.where(first,3.,1.) if action=='FAVOR_FIRST' else np.where(first,1.,3.)).astype(np.float64)
        masses[action]=mass/mass.sum()
        with np.load(task['probabilities'][action],allow_pickle=False) as p:
            assert np.array_equal(p['probability'],masses[action])
            assert np.array_equal(p['window_base_rows'],z['A_base_rows'][z['A_starts']])
            assert np.array_equal(p['window_cycles'],z['A_cycles'][z['A_starts']])
    offset=torch.arange(40,device='cuda')
    def batch(split,lo):return values[split][starts[split][lo:lo+2048,None]+offset]
    rebuilt={a:[torch.zeros((8,66,66),device='cuda',dtype=torch.float64),torch.zeros((8,66,8),device='cuda',dtype=torch.float64),torch.zeros((8,8),device='cuda',dtype=torch.float64)] for a in masses}
    for lo in range(0,6468,2048):
        data=batch('A',lo);x=data[:,:32].transpose(1,2)
        trend=F.avg_pool1d(F.pad(x,(12,12),mode='replicate'),25,stride=1)
        one=torch.ones_like(x[...,:1]);phi=torch.cat([x-trend,one,trend,one],-1).double().permute(1,0,2)
        y=data[:,32:].double().permute(2,0,1)
        for a,mass in masses.items():
            w=torch.tensor(mass[lo:lo+len(data)],device='cuda',dtype=torch.float64)[None,:,None]
            weighted=phi*w
            rebuilt[a][0]+=phi.transpose(1,2)@weighted
            rebuilt[a][1]+=weighted.transpose(1,2)@y
            rebuilt[a][2]+=(y*y*w).sum(1)
    moments={};moment_error=0.
    for a,terms in rebuilt.items():
        moments[a]=dict(zip(('gram','cross','target_square'),[v.cpu().numpy() for v in terms]))
        with np.load(out/'moments'/f'{tid}_{a}.npz',allow_pickle=False) as saved:
            for k,v in moments[a].items():
                error=float(np.max(np.abs(saved[k]-v)));moment_error=max(moment_error,error)
                assert error<=1e-10+1e-10*np.max(np.abs(v)),(tid,a,k,error)
    model=core.DLinear(dict(input_length=32,prediction_length=8,moving_average_kernel=25,channels=8)).cuda().eval()
    def errors(coords,split):
        for c in range(8):
            for p,v in [(model.seasonal[c].weight,coords[c,:32].T),(model.seasonal[c].bias,coords[c,32]),
                        (model.trend[c].weight,coords[c,33:65].T),(model.trend[c].bias,coords[c,65])]:
                p.copy_(torch.tensor(v,device='cuda'))
        e=np.concatenate([(model(batch(split,lo)[:,:32])-batch(split,lo)[:,32:]).cpu().numpy() for lo in range(0,len(starts[split]),2048)])
        return dict(shape=list(e.shape),dtype=str(e.dtype),byte_sha256=hashlib.sha256(e.tobytes()).hexdigest(),
            nmse=float(np.mean(e.astype(np.float64)**2)),nmae=float(np.mean(np.abs(e.astype(np.float64)))))
    checked=[];costs=[]
    for parent in task['parents']:
        seed=parent['seed'];pdir=Path(parent['directory']);states={step:parent_state(pdir/f'step_{step}.pt',step) for step in GRID}
        parent_result=read(pdir/'result.json');baseline=parent_result['selected_losses']['C']
        # Independent B-selected parent readout binds the response denominator.
        parent_losses={step:{s:errors(states[step][0].astype(np.float32),s)['nmse'] for s in 'BC'} for step in GRID}
        parent_best=min(GRID,key=lambda step:parent_losses[step]['B'])
        assert parent_best==parent_result['selected_step'] and parent_losses[parent_best]['C']==baseline
        for a in ACTIONS:
            directory=out/'forecasts'/f'{tid}_{a}_{seed}';result=read(directory/'result.json')
            expected=next(r for r in pred['records'] if (r['task'],r['action'],r['seed'])==(tid,a,seed))
            assert {k:v for k,v in expected.items() if k!='seconds'}==result
            delta=np.zeros((8,66,8));max_error=0.;max_ratio=0.;curves={}
            for step in GRID:
                theta,p=states[step];gp=moments['PARENT'];ga=moments[a]
                forcing=(2/64)*((ga['gram']-gp['gram'])@theta-(ga['cross']-gp['cross']))
                h=(2/64)*gp['gram']
                generator=np.zeros((8,8,67,67))
                generator[:,:,:66,:66]=-p.transpose(0,2,1)[:,:,:,None]*h[:,None,:,:]
                generator[:,:,:66,66]=-(p*forcing).transpose(0,2,1)
                exp=torch.linalg.matrix_exp(torch.from_numpy(generator.reshape(64,67,67)*.5)).numpy().reshape(8,8,67,67)
                vectors=np.concatenate([delta.transpose(0,2,1),np.ones((8,8,1))],axis=2)
                delta=(exp@vectors[:,:,:,None])[:,:,:66,0].transpose(0,2,1)
                assert np.isfinite(delta).all()
                with np.load(directory/f'step_{step}.npz',allow_pickle=False) as saved:
                    err=np.max(np.abs(saved['delta']-delta),axis=1);bound=1e-7+1e-8*np.max(np.abs(delta),axis=1)
                    assert (err<=bound).all(),(tid,a,seed,step,float(np.max(err/bound)))
                    max_error=max(max_error,float(err.max()));max_ratio=max(max_ratio,float((err/bound).max()))
                    coords=saved['coordinates'];assert np.array_equal(coords,(theta+saved['delta']).astype(np.float32))
                curves[str(step)]={}
                for split in 'BC':
                    summary=errors(coords,split);assert summary==read(directory/f'step_{step}_{split}.json')
                    curves[str(step)][split]=summary['nmse']
            selected=min(GRID,key=lambda step:curves[str(step)]['B']);gain=100*(baseline-curves[str(selected)]['C'])/baseline
            assert result['curves']==curves and result['selected_step']==selected and result['baseline_C']==baseline
            assert result['gain_percent']==gain and result['category']==category(gain)
            for name,digest in result['assets_sha256'].items():assert sha(directory/name)==digest
            assert expected['seconds']==read(directory/'timing.json')['seconds']
            checked.append(dict(task=tid,action=a,seed=seed,gain=gain,max_absolute_delta_error=max_error,max_error_bound_ratio=max_ratio))
            print(json.dumps(dict(audited=len(checked),task=tid,action=a,seed=seed,max_error_bound_ratio=max_ratio)),flush=True)
        c=next(c for c in pred['costs'] if (c['task'],c['seed'])==(tid,seed));setup=read(out/f'{tid}_setup.json')
        close(c['moment_allocation_seconds'],setup['moment_seconds']/3)
        close(c['action_seconds'],sum(r['seconds'] for r in pred['records'] if (r['task'],r['seed'])==(tid,seed)))
        close(c['SOURCE'],c['moment_allocation_seconds']+c['parent_state_seconds']+c['action_seconds'])
        close(c['real_parent_seconds'],read(pdir/'timing.json')['seconds'])
        costs.append(c)
    z.close();return checked,costs,moment_error

def main(out):
    started=time.perf_counter();assert out==ROOT and out.resolve()==ROOT
    plan=read(out/'plan.json');pred=read(out/'predictions.json');lock=read(out/'prediction_sync_receipt.json')
    assert lock['status']=='exact_commit_and_tree_verified' and lock['predictions_sha256']==sha(out/'predictions.json')
    assert pred['plan_sha256']==sha(out/'plan.json') and pred['status']=='completed'
    source=read(out/'source_sync_receipt.json');assert source['plan_sha256']==sha(out/'plan.json') and source['status']=='exact_commit_and_tree_verified'
    verify(out,plan,pred);write(out/'audit.lock',dict(predictions_sha256=sha(out/'predictions.json')))
    spec=importlib.util.spec_from_file_location('original_core',out/'run_research.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    checked=[];costs=[];moment_errors=[]
    for task in plan['tasks']:
        rows,cs,error=audit_task(out,task,pred,core);checked+=rows;costs+=cs;moment_errors.append(error)
        for c in cs:
            setup=read(out/f"{task['id']}_setup.json")
            close(c['allocated_common_preparation_seconds'],plan['historical_preparation_seconds']/9+setup['load_seconds']/3)
            close(c['cold_total_seconds'],c['SOURCE']+c['real_parent_seconds']+c['allocated_common_preparation_seconds'])
    assert len(checked)==36 and len(costs)==9
    close(pred['cold_total_seconds'],sum(c['cold_total_seconds'] for c in costs))
    booked=sum(read(out/f"{t['id']}_setup.json")[k] for t in plan['tasks'] for k in ('load_seconds','moment_seconds'))+sum(c[k] for c in costs for k in ('parent_state_seconds','action_seconds'))
    close(pred['booked_execution_seconds'],booked);close(pred['other_phase_seconds'],pred['phase_seconds']-booked)
    old=read(out/'input/old_final_audit.json');assert old['status']=='PASS'
    rows=[]
    for oldrow in old['scalar_records']:
        row=dict(oldrow);row['predicted']=dict(oldrow['predicted'])
        row['predicted']['SOURCE']=next(r['gain'] for r in checked if (r['task'],r['seed'],r['action'])==(row['task'],row['seed'],row['action']))
        rows.append(row)
    assert len(rows)==36
    cells=[];decisions=[];task_mae={m:{} for m in METHODS};realized={m:[] for m in METHODS}
    for task in plan['tasks']:
        tid=task['id']
        for a in ACTIONS:
            subset=[r for r in rows if (r['task'],r['action'])==(tid,a)];assert len(subset)==3
            truth=float(np.mean([r['actual_gain'] for r in subset]));estimates={m:float(np.mean([r['predicted'][m] for r in subset])) for m in METHODS}
            cells.append(dict(task=tid,action=a,truth=truth,predictions=estimates,actual_category=category(truth),predicted_categories={m:category(v) for m,v in estimates.items()}))
        tc=[c for c in cells if c['task']==tid];oracle=max([0.]+[c['truth'] for c in tc])
        for m in METHODS:
            task_mae[m][tid]=float(np.mean([abs(c['truth']-c['predictions'][m]) for c in tc]))
            means={c['action']:c['predictions'][m] for c in tc};choice=max(ACTIONS,key=lambda a:means[a]);choice=choice if means[choice]>=1 else 'PARENT'
            gain=0. if choice=='PARENT' else next(c['truth'] for c in tc if c['action']==choice);realized[m].append(gain)
            decisions.append(dict(task=tid,method=m,action=choice,gain=gain,oracle_gain=oracle,regret=oracle-gain,accepted_harm=choice!='PARENT' and gain<-.25))
            if m=='SOURCE':assert pred['recommendations'][tid]==dict(action=choice,means=means)
    mae={m:float(np.mean(list(task_mae[m].values()))) for m in METHODS};mean_gain={m:float(np.mean(realized[m])) for m in METHODS}
    for m in METHODS[:-1]:
        close(mae[m],old['prediction_MAE_pp'][m]);close(mean_gain[m],old['mean_recommendation_gain'][m])
        for task in task_mae[m]:close(task_mae[m][task],old['task_MAE_pp'][m][task])
    for c in cells:
        oc=next(o for o in old['cells'] if (o['task'],o['action'])==(c['task'],c['action']));close(c['truth'],oc['truth'])
        for m in METHODS[:-1]:close(c['predictions'][m],oc['predictions'][m])
    noninferior=sum(task_mae['M1'][t]<=task_mae['SOURCE'][t] for t in task_mae['M1'])
    descriptive_error=mae['M1']<=.8*mae['SOURCE'] and mae['SOURCE']-mae['M1']>=.05 and noninferior>=2
    descriptive_selection=mean_gain['M1']>=mean_gain['SOURCE']+.10 and min(realized['M1'])>=-.25
    verify(out,plan,pred);assert sha(out/'predictions.json')==lock['predictions_sha256']
    result=dict(status='PASS',retrospective=True,independent_confirmation=False,new_full_trainings=0,cumulative_full_trainings=373,
        D_access=False,test_access=False,original_inputs_unchanged=True,propagation_blocks_verified=36*6*64,residual_summaries_verified=36*6*2,
        parent_readout_summaries_verified=9*6*2,moment_files_verified=15,max_moment_error=max(moment_errors),checked=checked,
        scalar_records=rows,cells=cells,prediction_MAE_pp=mae,task_MAE_pp=task_mae,decisions=decisions,mean_recommendation_gain=mean_gain,
        recommendation_gains=realized,per_seed_MAE_pp={m:{str(seed):float(np.mean([abs(r['actual_gain']-r['predicted'][m]) for r in rows if r['seed']==seed])) for seed in plan['seeds']} for m in METHODS},
        category_accuracy={m:sum(c['actual_category']==c['predicted_categories'][m] for c in cells)/12 for m in METHODS},
        M1_vs_SOURCE_descriptive=dict(noninferior_tasks=noninferior,error_scale_pass=bool(descriptive_error),selection_scale_pass=bool(descriptive_selection)),
        costs=costs,batch_cold_total_seconds={**old['batch_cold_total_seconds'],'SOURCE':pred['cold_total_seconds']},
        forecast_phase_seconds=pred['phase_seconds'],forecast_other_phase_seconds=pred['other_phase_seconds'],audit_seconds=time.perf_counter()-started,
        source_method_under_historical_60_seconds=[c['SOURCE']<=60 for c in costs],strict_budget_matched_comparison=False)
    write(out/'final_audit.json',result);print(json.dumps(dict(status='PASS',prediction_MAE_pp=mae,mean_recommendation_gain=mean_gain)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);torch.set_num_threads(4);main(args.output)
    except Exception:
        write(args.output/'audit_failure.json',dict(status='failed',error=traceback.format_exc(),new_full_trainings=0));raise
