"""Retrospective SOURCE-formula forecasts on the complete registered Hydraulic pool."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import response_proxy as moments_api
import source_dlinear_adapter as adapter

ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
GRID=(500,1000,1500,2000,2500,3000)
ROOT=Path('/root/auto_data_experiments/20261014_source_hydraulic_r1')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value,replace=False):
    with Path(path).open('w' if replace else 'x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def arrays(path,**items):
    with Path(path).open('xb') as f:np.savez_compressed(f,**items)
def tick():torch.cuda.synchronize();return time.perf_counter()
def category(v):return 'improve' if v>.25 else 'hurt' if v<-.25 else 'unchanged'
def module(path):
    spec=importlib.util.spec_from_file_location('fixed_core',path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def verify(out):
    assert out==ROOT and out.resolve()==ROOT
    plan=read(out/'plan.json');sync=read(out/'source_sync_receipt.json')
    assert sync['status']=='exact_commit_and_tree_verified' and sync['plan_sha256']==sha(out/'plan.json')
    assert plan['retrospective'] and plan['new_full_trainings']==0
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for item in plan['prediction_inputs']:
        assert sha(item['path'])==item['sha256'],item['path']
    return plan


class Context:
    def __init__(self,out,plan,task):
        self.out,self.plan,self.task=out,plan,task['id'];self.core=module(out/'run_research.py')
        with np.load(task['abc'],allow_pickle=False) as z:
            self.values={s:torch.tensor(z[f'{s}_values'],device='cuda') for s in ('A','B','C')}
            self.starts={s:torch.tensor(z[f'{s}_starts'],device='cuda') for s in ('A','B','C')}
        self.n=len(self.starts['A']);assert self.n==6468
        self.probabilities={'PARENT':np.full(self.n,1/self.n)}
        for action,path in task['probabilities'].items():
            with np.load(path,allow_pickle=False) as z:self.probabilities[action]=z['probability'].copy()
        self.offset=torch.arange(40,device='cuda')
        self.model=self.core.DLinear(dict(input_length=32,prediction_length=8,moving_average_kernel=25,channels=8)).cuda()

    def batch(self,split,start,end):return self.values[split][self.starts[split][start:end,None]+self.offset]

    @torch.no_grad()
    def build_moments(self):
        result={action:None for action in ('PARENT',)+ACTIONS}
        for lo in range(0,self.n,2048):
            batch=self.batch('A',lo,min(lo+2048,self.n));phi=moments_api.features(self.model,batch[:,:32])
            for action in result:
                mass=torch.tensor(self.probabilities[action][lo:lo+len(batch)],device='cuda',dtype=torch.float64)
                item=moments_api.batch_moments(phi,batch[:,32:],mass)
                if result[action] is None:result[action]=item
                else:result[action].add_(item)
        packed={}
        for action,m in result.items():
            packed[action]={k:getattr(m,k).cpu().numpy() for k in ('gram','cross','target_square')}
            arrays(self.out/'moments'/f'{self.task}_{action}.npz',**packed[action])
        return packed

    @torch.no_grad()
    def evaluate(self,coordinates,split):
        assert split in ('B','C')
        moments_api.load_parameters(self.model,torch.tensor(coordinates,device='cuda'))
        self.model.eval();error=[]
        for lo in range(0,len(self.starts[split]),2048):
            batch=self.batch(split,lo,min(lo+2048,len(self.starts[split])))
            error.append((self.model(batch[:,:32])-batch[:,32:]).cpu().numpy())
        error=np.concatenate(error);assert np.isfinite(error).all()
        return dict(shape=list(error.shape),dtype=str(error.dtype),byte_sha256=hashlib.sha256(error.tobytes()).hexdigest(),
            nmse=float(np.mean(error.astype(np.float64)**2)),nmae=float(np.mean(np.abs(error.astype(np.float64)))))


def run(out):
    started=tick();plan=verify(out)
    write(out/'forecast.lock',dict(plan_sha256=sha(out/'plan.json'),retrospective=True))
    (out/'moments').mkdir();(out/'forecasts').mkdir();records=[];costs=[];setups=[]
    for task in plan['tasks']:
        begin=tick();ctx=Context(out,plan,task);load_seconds=tick()-begin
        begin=tick();moments=ctx.build_moments();moment_seconds=tick()-begin
        setup=dict(task=ctx.task,load_seconds=load_seconds,moment_seconds=moment_seconds)
        write(out/f'{ctx.task}_setup.json',setup);setups.append(setup)
        for seed in plan['seeds']:
            parent=next(p for p in task['parents'] if p['seed']==seed)
            begin=tick();states={}
            cpu_model=ctx.core.DLinear(dict(input_length=32,prediction_length=8,moving_average_kernel=25,channels=8))
            for step in GRID:
                snapshot=torch.load(Path(parent['directory'])/f'step_{step}.pt',map_location='cpu',weights_only=True)
                states[step]=adapter.parent_coordinates(cpu_model,snapshot)
            parent_result=read(Path(parent['directory'])/'result.json')
            baseline=parent_result['selected_losses']['C'];state_seconds=tick()-begin
            action_seconds=0.
            for action in ACTIONS:
                directory=out/'forecasts'/f'{ctx.task}_{action}_{seed}';directory.mkdir()
                begin=tick();delta=np.zeros_like(states[500]['theta']);curves={};diag=[]
                for step in GRID:
                    delta,checks=adapter.quadratic_segment(moments['PARENT']['gram'],moments['PARENT']['cross'],
                        moments[action]['gram'],moments[action]['cross'],states[step],delta)
                    coordinates=(states[step]['theta']+delta).astype(np.float32)
                    arrays(directory/f'step_{step}.npz',delta=delta,coordinates=coordinates)
                    curves[step]={}
                    for split in ('B','C'):
                        summary=ctx.evaluate(coordinates,split);write(directory/f'step_{step}_{split}.json',summary)
                        curves[step][split]=summary['nmse']
                    diag.append(dict(step=step,clipped_negative_eigenvalues=sum(c['clipped_negative_eigenvalues'] for c in checks),
                        max_roundoff_tolerance=max(c['roundoff_tolerance'] for c in checks)))
                selected=min(GRID,key=lambda k:curves[k]['B']);gain=100.*(baseline-curves[selected]['C'])/baseline
                result=dict(task=ctx.task,seed=seed,action=action,selected_step=selected,baseline_C=baseline,
                    gain_percent=gain,category=category(gain),curves=curves,numerical_diagnostics=diag,
                    assets_sha256={p.name:sha(p) for p in sorted(directory.iterdir())},retrospective=True)
                write(directory/'result.json',result);seconds=tick()-begin
                write(directory/'timing.json',dict(seconds=seconds,boundary='propagation through result/assets/hash; clock receipt excluded'))
                records.append(dict(result,seconds=seconds));action_seconds+=seconds
                write(out/'forecast_status.json',dict(status='running',completed=len(records),registered=36,new_full_trainings=0),replace=True)
                print(json.dumps(dict(completed=len(records),task=ctx.task,seed=seed,action=action)),flush=True)
            method_seconds=moment_seconds/3+state_seconds+action_seconds
            real_parent=read(Path(parent['directory'])/'timing.json')['seconds']
            common=plan['historical_preparation_seconds']/9+load_seconds/3
            costs.append(dict(task=ctx.task,seed=seed,moment_allocation_seconds=moment_seconds/3,parent_state_seconds=state_seconds,
                action_seconds=action_seconds,SOURCE=method_seconds,real_parent_seconds=real_parent,
                allocated_common_preparation_seconds=common,cold_total_seconds=method_seconds+real_parent+common))
        del ctx
    assert len(records)==36 and len(costs)==9
    recommendations={}
    for task in plan['tasks']:
        tid=task['id'];means={a:float(np.mean([r['gain_percent'] for r in records if r['task']==tid and r['action']==a])) for a in ACTIONS}
        choice=max(ACTIONS,key=lambda a:means[a]);choice=choice if means[choice]>=1. else 'PARENT'
        recommendations[tid]=dict(action=choice,means=means)
    assets={p.relative_to(out).as_posix():sha(p) for sub in ('moments','forecasts') for p in sorted((out/sub).rglob('*')) if p.is_file()}
    verify(out);elapsed=tick()-started
    booked=sum(s['load_seconds']+s['moment_seconds'] for s in setups)+sum(c['parent_state_seconds']+c['action_seconds'] for c in costs)
    write(out/'predictions.json',dict(status='completed',plan_sha256=sha(out/'plan.json'),records=records,costs=costs,
        recommendations=recommendations,assets_sha256=assets,phase_seconds=elapsed,booked_execution_seconds=booked,
        other_phase_seconds=elapsed-booked,phase_clock_excludes='final batch JSON/clock receipt write; Python startup and network',
        cold_total_seconds=sum(c['cold_total_seconds'] for c in costs),retrospective=True,old_outcomes_already_known=True,
        new_full_trainings=0,cumulative_full_trainings=373,D_access=False,test_access=False))
    write(out/'forecast_status.json',dict(status='completed',completed=36,registered=36,new_full_trainings=0),replace=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
        run(args.output)
    except Exception:
        write(args.output/'forecast_failure.json',dict(status='failed',error=traceback.format_exc(),new_full_trainings=0))
        raise
