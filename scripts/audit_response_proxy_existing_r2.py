"""Audit revision 2: tolerate bookkeeping roundoff and verify aggregate costs."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F

GRID=(500,1000,1500,2000,2500,3000)
ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
METHODS=('M1','RAW','FIXED','SHORT')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def category(v):return 'improve' if v>.25 else 'hurt' if v<-.25 else 'unchanged'
def mse(v):return float(np.mean(v.astype(np.float64)**2))
def ident(t,e,a,s):return f'EXSR1_{t}_{e}_{a}_s{s}'
def coordinates(state):
    channels=len([k for k in state if k.startswith('seasonal.') and k.endswith('.bias')])
    return torch.stack([torch.cat([state[f'seasonal.{c}.weight'].T,state[f'seasonal.{c}.bias'][None],
        state[f'trend.{c}.weight'].T,state[f'trend.{c}.bias'][None]]) for c in range(channels)])
def snap(directory,step):return torch.load(directory/('initial.pt' if step==0 else f'step_{step}.pt'),weights_only=True,map_location='cpu')
def gamma(n,dtype):
    u=np.finfo(dtype).eps/2;return n*u/(1-n*u)


def verify_inputs(out):
    plan=read(out/'plan.json')
    assert read(out/'source_sync_receipt.json')['status']=='exact_commit_and_tree_verified'
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    assert sha(plan['core'])==plan['core_sha256']
    for task in plan['tasks']:
        assert sha(task['manifest'])==task['manifest_sha256']
        assert sha(task['train'])==task['train_sha256']
    prep=read(out/'preparation.json')
    assert prep['status']=='PASS' and prep['plan_sha256']==sha(out/'plan.json')
    for name,digest in prep['assets_sha256'].items():assert sha(out/name)==digest,name
    return plan,prep


def reconstruct(task,out):
    """Reconstruct every partition directly from raw train rows, independent of preparer."""
    manifest=read(task['manifest']);sensors=manifest['sensor_columns']
    with Path(task['train']).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==next(s['rows'] for s in manifest['splits'] if s['name']=='train')
    metro=manifest['dataset_id']=='metropt3_v1'
    raw=np.array([[float(r[c]) for c in sensors] for r in rows],dtype=np.float64)
    times=np.array([r['timestamp' if metro else 'datetime'] for r in rows],dtype='datetime64[s]').astype(np.int64)
    files=np.array(['metropt3']*len(rows) if metro else [r['source_file'] for r in rows])
    source=np.arange(len(rows),dtype=np.int64) if metro else np.array([int(r['source_row']) for r in rows])
    original=np.array([int(r['']) for r in rows]) if metro else source.copy()
    del rows
    membership=np.empty(len(raw),dtype=np.int8);halves=np.zeros(len(raw),dtype=bool)
    partition_ids={s:[] for s in ('A','B','C')}
    for name in sorted(set(files)):
        ids=np.flatnonzero(files==name);ids=ids[np.argsort(source[ids],kind='stable')]
        assert np.all(np.diff(source[ids])==1) and np.all(np.diff(times[ids])>0)
        a,b=len(ids)*3//5,len(ids)*4//5
        for code,(split,subset) in enumerate(zip(partition_ids,(ids[:a],ids[a:b],ids[b:]))):
            membership[subset]=code;partition_ids[split].extend(subset.tolist())
        halves[ids[:a//2]]=True
    a_ids=np.array(partition_ids['A']);mean=raw[a_ids].mean(0);scale=raw[a_ids].std(0);scale[scale==0]=1.
    with np.load(out/'prepared'/f'{task["id"]}_ABC.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['mean'],mean);np.testing.assert_array_equal(saved['scale'],scale)
        arrays={k:saved[k] for k in saved.files}
    for code,split in enumerate(partition_ids):
        ids=np.array(partition_ids[split],dtype=np.int64)
        np.testing.assert_array_equal(arrays[f'{split}_base_rows'],ids)
        assert np.all(membership[ids]==code)
        for name,expected in [('times',times[ids]),('files',files[ids]),('source_rows',source[ids]),
            ('original_indices',original[ids]),('first',halves[ids] if split=='A' else np.zeros(len(ids),dtype=bool)),
            ('values',((raw[ids]-mean)/scale).astype(np.float32))]:
            np.testing.assert_array_equal(arrays[f'{split}_{name}'],expected)
        # Independent interval intersection: discard a start if any of its 39 edges is invalid.
        starts=np.arange(len(ids)-39,dtype=np.int64);valid=np.ones(len(starts),dtype=bool)
        for offset in range(39):
            left,right=ids[starts+offset],ids[starts+offset+1]
            valid&=(files[left]==files[right])&(source[right]-source[left]==1)&(times[right]>times[left])&(times[right]-times[left]<=30)
            if split=='A':valid&=halves[left]==halves[right]
        np.testing.assert_array_equal(arrays[f'{split}_starts'],starts[valid])
    first=arrays['A_first'][arrays['A_starts']]
    probabilities={}
    for action in ('PARENT',)+ACTIONS:
        if action=='PARENT':p=np.full(len(first),1./len(first))
        else:
            chosen=first if action.endswith('FIRST') else ~first
            total=1. if action.startswith('KEEP') else .75
            p=np.empty(len(first),dtype=np.float64);p[chosen]=total/chosen.sum();p[~chosen]=(1.-total)/(~chosen).sum()
        with np.load(out/'prepared'/f'{task["id"]}_{action}.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['probability'],p)
        probabilities[action]=p
    return arrays,probabilities


class AuditContext:
    def __init__(self,out,plan,task):
        self.out,self.plan,self.task=out,plan,task['id']
        self.arrays,self.probabilities=reconstruct(task,out)
        self.values={s:torch.tensor(self.arrays[f'{s}_values'],device='cuda') for s in ('A','B','C')}
        self.starts={s:torch.tensor(self.arrays[f'{s}_starts'],device='cuda') for s in self.values}
        core=module('audit_core',plan['core']);self.channels=len(task['sensors'])
        self.model=core.DLinear(dict(plan['training'],channels=self.channels)).cuda().eval()
        self.states_checked=self.residuals_checked=self.moments_checked=0
        self.moment_reference={}

    def batches(self,split):
        for lo in range(0,len(self.starts[split]),2048):
            yield lo,self.values[split][self.starts[split][lo:lo+2048,None]+torch.arange(40,device='cuda')]

    @torch.no_grad()
    def errors(self,state,split):
        self.model.load_state_dict(state)
        values=[(self.model(batch[:,:32])-batch[:,32:]).cpu().numpy() for _,batch in self.batches(split)]
        result=np.concatenate(values);assert np.isfinite(result).all();return result

    def check_error(self,error,path):
        with np.load(path,allow_pickle=False) as z:
            assert set(z.files)=={'error_shape','error_dtype','error_sha256','nmse','nmae'}
            assert tuple(z['error_shape'])==error.shape and z['error_dtype'].item()==error.dtype.str
            assert z['error_sha256'].item()==hashlib.sha256(error.tobytes()).hexdigest()
            assert z['nmse'].item()==mse(error) and z['nmae'].item()==np.mean(np.abs(error.astype(np.float64)))
        self.residuals_checked+=1

    @staticmethod
    def features(batch):
        x=batch[:,:32].transpose(1,2)
        trend=F.avg_pool1d(F.pad(x,(12,12),mode='replicate'),25,stride=1)
        one=torch.ones_like(x[...,:1])
        return torch.cat([x-trend,one,trend,one],dim=2).double()

    @torch.no_grad()
    def moments(self,action):
        if action not in self.moment_reference:
            sums=None
            for lo,batch in self.batches('A'):
                phi=self.features(batch).permute(1,0,2);target=batch[:,32:].double().permute(2,0,1)
                mass=torch.tensor(self.probabilities[action][lo:lo+len(batch)],device='cuda')
                weighted=phi*mass[None,:,None]
                terms=[phi.transpose(1,2)@weighted,weighted.transpose(1,2)@target,(target.square()*mass[None,:,None]).sum(1)]
                if sums is None:sums=terms
                else:
                    for s,t in zip(sums,terms):s+=t
            self.moment_reference[action]=[s.cpu().numpy() for s in sums]
        return self.moment_reference[action]

    def run(self,engine,action,seed):
        directory=self.out/'runs'/ident(self.task,engine,action,seed)
        result=read(directory/'result.json');status=read(directory/'status.json')
        assert result['status']==status['status']=='completed'
        assert result['id']==directory.name and result['plan_sha256']==sha(self.out/'plan.json')
        assert (result['task'],result['engine'],result['action'],result['seed'])==(self.task,engine,action,seed)
        for name,digest in result['assets_sha256'].items():assert sha(directory/name)==digest
        parent_dir=self.out/'runs'/ident(self.task,'full','PARENT',seed)
        initial=coordinates(snap(parent_dir,0)['model'])
        if engine=='full' and action=='PARENT':
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            core=module('audit_seed_core',self.plan['core'])
            registered_model=core.DLinear(dict(self.plan['training'],channels=self.channels))
            assert torch.equal(initial,coordinates(registered_model.state_dict()))
        steps=result['checkpoints']
        assert steps==(list(GRID) if engine!='short' else sorted(set([k for k in GRID if k<=result['steps']]+[result['steps']])))
        assert result['steps']==3000 if engine!='short' else 0<result['steps']<3000
        for step in [0]+steps:
            saved=snap(directory,step);packed=coordinates(saved['model'])
            assert saved['step']==step and torch.equal(saved['original_coordinates'],packed) and bool(torch.isfinite(packed).all())
            if step==0:
                assert torch.equal(initial,packed) and not saved['optimizer']['state']
                assert hashlib.sha256(packed.numpy().tobytes()).hexdigest()==result['initial_sha256']
            else:
                assert len(saved['optimizer']['state'])==4*self.channels
                for state in saved['optimizer']['state'].values():
                    assert state['step'].item()==step and bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all())
            self.states_checked+=1
        if engine=='proxy':
            with np.load(directory/'moments.npz',allow_pickle=False) as z:
                for key,expected in zip(('gram','cross','target_square'),self.moments(action)):np.testing.assert_array_equal(z[key],expected)
            assert result['selected_step'] is None and result['selection_losses']=={}
            self.moments_checked+=1
            return result,{}
        cdf=np.cumsum(self.probabilities[action]);cdf[-1]=1.
        rng=np.random.default_rng(seed+1000);counts=np.zeros(len(cdf),dtype=np.int64)
        for _ in range(result['steps']):np.add.at(counts,np.searchsorted(cdf,rng.random(256),side='right'),1)
        with np.load(directory/'sampling.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['draw_count'],counts)
        assert saved['rng_numpy']==rng.bit_generator.state
        # Forward/Adam contain no stochastic layers after initialization.
        assert torch.equal(saved['rng_torch'],snap(directory,0)['rng_torch'])
        assert torch.equal(saved['rng_cuda'],snap(directory,0)['rng_cuda'])
        curves={}
        for step in steps:
            state=snap(directory,step)['model']
            curves[step]={s:mse(self.errors(state,s)) for s in ('B','C')}
            assert curves[step]['B']==result['selection_losses'][str(step)]
            if engine=='short':
                self.check_error(self.errors(state,'C'),directory/f'step_{step}_C_errors.npz')
                assert curves[step]==result['short_curves'][str(step)]
        selected=min(steps,key=lambda k:curves[k]['B']);assert selected==result['selected_step']
        for split in ('B','C'):
            error=self.errors(snap(directory,selected)['model'],split)
            self.check_error(error,directory/f'selected_{split}_errors.npz')
            assert mse(error)==result['selected_losses'][split]
        return result,curves

    @torch.no_grad()
    def energy(self,seed,action,step):
        paths={e:self.out/'runs'/ident(self.task,e,'PARENT',seed) for e in ('full','proxy')}
        real=snap(paths['full'],step)['model'];reference=snap(paths['proxy'],step)['model']
        child=snap(self.out/'runs'/ident(self.task,'proxy',action,seed),step)['model']
        truth=snap(self.out/'runs'/ident(self.task,'full',action,seed),step)['model']
        paired={k:real[k]+(child[k]-reference[k]) for k in child}
        pc,pr,pp,combo=[coordinates(v).double().cuda() for v in (child,real,reference,paired)]
        total=np.zeros(4,dtype=np.float64);count=0;gap_max=0.
        for _,batch in self.batches('C'):
            errors=[]
            for state in (child,real,reference,paired,truth):
                self.model.load_state_dict(state)
                errors.append((self.model(batch[:,:32])-batch[:,32:]).double())
            raw,e_real,e_ref,corr,true=errors;e0=e_ref-e_real;et=raw-true;corrected=corr-true
            phi=self.features(batch);target=batch[:,32:].double()
            pbound=torch.einsum('nci,cij->njc',phi.abs(),(combo-(pr+pc-pp)).abs())
            magnitude=sum(torch.einsum('nci,cij->njc',phi.abs(),v.abs())+target.abs() for v in (pc,pr,pp,combo))
            bound=pbound+gamma(70,np.float32)*magnitude+gamma(256,np.float64)*(magnitude+pbound)
            gap=(corr-(raw-e0)).abs();assert bool((gap<=bound).all())
            gap_max=max(gap_max,float(gap.max()))
            total+=np.array([float((et**2).sum()),float((e0**2).sum()),float((et*e0).sum()),float((corrected**2).sum())])
            count+=et.numel()
        et,e0,cross,corrected=total/count
        return dict(task=self.task,seed=seed,action=action,step=step,raw_energy=et,parent_energy=e0,cross=cross,
            paired_energy=corrected,alignment=cross/math.sqrt(et*e0) if et*e0>0 else None,
            rounding_bound_pass=True,identity_gap_max=gap_max)


def preflight(out):
    plan,prep=verify_inputs(out)
    trees=[]
    for name in ('response_proxy_existing.py','response_proxy_selection.py'):
        trees.append({n.name:n for n in ast.parse((out/name).read_text()).body if isinstance(n,ast.FunctionDef)})
    for name in ('train','score_proxy'):assert ast.dump(trees[0][name])==ast.dump(trees[1][name])
    for name in plan['code_sha256']:
        if name.endswith('.py'):compile((out/name).read_text(),name,'exec')
    tasks=[]
    for task in plan['tasks']:
        arrays,probabilities=reconstruct(task,out)
        tasks.append(dict(task=task['id'],windows={s:len(arrays[f'{s}_starts']) for s in ('A','B','C')},
            group_mass={a:float(p[arrays['A_first'][arrays['A_starts']]].sum()) for a,p in probabilities.items()}))
        del arrays,probabilities
    write(out/'preflight_audit.json',dict(status='PASS',tasks=tasks,frozen_update_and_score_AST_equal=True,
        all_train_rows_and_windows_reconstructed=True,test_access=False,new_training=0,plan_sha256=sha(out/'plan.json'),
        preparation_sha256=sha(out/'preparation.json')))


def audit_parents(out):
    plan,prep=verify_inputs(out);cal=read(out/'calibration.json')
    assert cal['status']=='completed' and cal['plan_sha256']==sha(out/'plan.json')
    for name,digest in cal['assets_sha256'].items():assert sha(out/name)==digest
    counters=np.zeros(3,dtype=np.int64);results=[]
    for task in plan['tasks']:
        ctx=AuditContext(out,plan,task)
        for seed in plan['seeds']:
            for engine in ('full','proxy'):
                result,curves=ctx.run(engine,'PARENT',seed);results.append(result)
            for step in GRID:
                state=snap(out/'runs'/ident(ctx.task,'full','PARENT',seed),step)['model']
                for split in ('B','C'):ctx.check_error(ctx.errors(state,split),out/'calibration'/f'{ctx.task}_{seed}'/f'step_{step}_{split}.npz')
        timings=[m for m in cal['measures'] if m['task']==ctx.task]
        p=max(read(out/'runs'/ident(ctx.task,'proxy','PARENT',s)/'timing.json')['seconds'] for s in plan['seeds'])
        f=min(read(out/'runs'/ident(ctx.task,'full','PARENT',s)/'timing.json')['seconds'] for s in plan['seeds'])
        s=max(m['six_checkpoint_seconds'] for m in timings);e=max(max(m['checkpoint_seconds']) for m in timings)
        b,r=math.ceil(1.25*(1.25*p+s)),math.ceil(max(1.,2*e))
        assert cal['budgets'][ctx.task]==dict(short_budget_seconds=b,short_training_boundary_seconds=b-r,
            selection_budget_seconds=4*b,reserve_seconds=r,max_proxy_parent_seconds=p,max_six_checkpoint_seconds=s,
            max_checkpoint_seconds=e,min_full_parent_seconds=f,gate=bool(b>r and b<f))
        counters+=np.array([ctx.states_checked,ctx.residuals_checked,ctx.moments_checked])
        del ctx
    assert cal['budget_gate']==all(v['gate'] for v in cal['budgets'].values())
    verify_inputs(out)
    write(out/'parents_audit.json',dict(status='PASS',budget_gate=cal['budget_gate'],calibration_sha256=sha(out/'calibration.json'),
        states_verified=int(counters[0]),residuals_verified=int(counters[1]),moments_verified=int(counters[2]),
        full_parent_trainings=6,cumulative_full_trainings=373,test_access=False))


def audit_final(out):
    started=time.perf_counter();plan,prep=verify_inputs(out)
    revision=read(out/'audit_revision_lock.json')
    assert revision['auditor_sha256']==sha(Path(__file__))
    assert revision['original_auditor_sha256']==sha(out/'audit_response_proxy_existing.py')
    assert revision['plan_sha256']==sha(out/'plan.json')
    prediction,labels,lock,sync=(read(out/n) for n in ('predictions.json','labels.json','prediction_lock.json','prediction_sync_receipt.json'))
    parent_audit=read(out/'parents_audit.json');cal=read(out/'calibration.json')
    assert parent_audit['status']=='PASS' and parent_audit['calibration_sha256']==sha(out/'calibration.json')
    assert sync['status']=='exact_commit_and_tree_verified'
    assert sync['audit_revision_lock_sha256']==sha(out/'audit_revision_lock.json')
    assert lock['predictions_sha256']==labels['predictions_sha256']==sync['predictions_sha256']==sha(out/'predictions.json')
    assert sync['prediction_lock_sha256']==sha(out/'prediction_lock.json')
    assert lock['plan_sha256']==prediction['plan_sha256']==sha(out/'plan.json')
    assert prediction['calibration_sha256']==sha(out/'calibration.json') and prediction['preparation_sha256']==sha(out/'preparation.json')
    assert labels['status']=='completed' and prediction['budget_gate'] and not prediction['full_children_started']
    for name,digest in prediction['assets_sha256'].items():assert sha(out/name)==digest
    for name,digest in cal['assets_sha256'].items():assert sha(out/name)==digest
    assert {p.name for p in (out/'runs').iterdir()}=={j['id'] for j in plan['jobs']} and len(plan['jobs'])==84
    assert {r['id'] for r in labels['results']}=={j['id'] for j in plan['jobs'] if j['phase']=='labels'}
    for item in labels['results']:
        expected=dict(item);seconds=expected.pop('seconds');directory=out/'runs'/item['id']
        assert read(directory/'result.json')==expected and read(directory/'timing.json')['seconds']==seconds
    records={(r['task'],r['seed'],r['action']):r for r in prediction['records']};assert len(records)==24
    rows,costs,energies,diagnostics,run_costs=[],[],[],[],[]
    counters=np.array([parent_audit[k] for k in ('states_verified','residuals_verified','moments_verified')])
    for task in plan['tasks']:
        ctx=AuditContext(out,plan,task);tid=ctx.task
        for seed in plan['seeds']:
            pd=out/'runs'/ident(tid,'full','PARENT',seed);pp=out/'runs'/ident(tid,'proxy','PARENT',seed)
            parent=read(pd/'result.json');baseline=parent['selected_losses']['C']
            cost=dict(task=tid,seed=seed,real_parent_seconds=read(pd/'timing.json')['seconds'],M1=read(pp/'timing.json')['seconds'],RAW=0.,SHORT=0.,FIXED=0.)
            for action in ACTIONS:
                results,truths={},{}
                for engine in ('proxy','short','full'):
                    result,curves=ctx.run(engine,action,seed);results[engine]=result
                    seconds=read(out/'runs'/ident(tid,engine,action,seed)/'timing.json')['seconds']
                    run_costs.append(dict(id=result['id'],engine=engine,task=tid,steps=result['steps'],seconds=seconds))
                    if engine=='proxy':cost['M1']+=seconds;cost['RAW']+=seconds
                    elif engine=='short':cost['SHORT']+=seconds
                    else:truths=curves
                record=records[(tid,seed,action)];true_step=results['full']['selected_step']
                actual=100.*(baseline-truths[true_step]['C'])/baseline
                rows.append(dict(task=tid,seed=seed,action=action,actual_gain=actual,predicted={m:record['methods'][m]['gain_percent'] for m in METHODS}))
                for method in ('M1','RAW','FIXED'):
                    directory=out/'forecasts'/f'{tid}_{action}_{seed}_{method}';local=read(directory/'result.json')
                    for key,value in local.items():assert record['methods'][method][key]==value
                    for name,digest in local['assets_sha256'].items():assert sha(directory/name)==digest
                    steps=[parent['selected_step']] if method=='FIXED' else list(GRID);curves={}
                    for step in steps:
                        if method=='FIXED':state=snap(pd,step)['model']
                        else:
                            child=snap(out/'runs'/ident(tid,'proxy',action,seed),step)['model'];state=child
                            if method=='M1':
                                real,reference=snap(pd,step)['model'],snap(pp,step)['model']
                                state={k:real[k]+(child[k]-reference[k]) for k in child}
                        curves[step]={}
                        for split in ('B','C'):
                            error=ctx.errors(state,split);ctx.check_error(error,directory/f'step_{step}_{split}_errors.npz')
                            curves[step][split]=mse(error);assert curves[step][split]==local['curves'][str(step)][split]
                    selected=min(steps,key=lambda k:curves[k]['B'])
                    assert selected==local['selected_step'] and local['gain_percent']==100.*(baseline-curves[selected]['C'])/baseline
                    assert record['methods'][method]['category']==category(local['gain_percent'])
                    cost[method]+=read(directory/'timing.json')['seconds']
                    if method!='FIXED':diagnostics.append(dict(task=tid,seed=seed,action=action,method=method,hit=selected==true_step,
                        B_selection_regret=truths[selected]['B']-truths[true_step]['B'],
                        C_prediction_term=curves[selected]['C']-truths[selected]['C'],C_selection_term=truths[selected]['C']-truths[true_step]['C']))
                    else:assert local['gain_percent']==0.
                short=record['methods']['SHORT'];actual_short=results['short']
                assert short['gain_percent']==100.*(baseline-actual_short['selected_losses']['C'])/baseline
                assert short['selected_step']==actual_short['selected_step'] and short['steps']==actual_short['steps']
                assert short['curves']==actual_short['short_curves'] and short['category']==category(short['gain_percent'])
                energies.extend(ctx.energy(seed,action,k) for k in GRID)
                print(json.dumps(dict(audit='final',completed_records=len(rows),task=tid,action=action,seed=seed)),flush=True)
            common=prep['seconds']/6+(read(out/f'{tid}_parent_setup.json')['seconds']+read(out/f'{tid}_setup.json')['seconds'])/3
            cost['allocated_common_preparation_seconds']=common
            cost['cold_total_seconds']={m:cost[m]+cost['real_parent_seconds']+common for m in METHODS}
            registered=next(c for c in prediction['costs'] if (c['task'],c['seed'])==(tid,seed))
            assert cost.keys()==registered.keys()
            assert (cost['task'],cost['seed'])==(registered['task'],registered['seed'])
            for key in cost:
                if key in ('task','seed'):continue
                if key=='cold_total_seconds':
                    assert cost[key].keys()==registered[key].keys()
                    assert all(abs(cost[key][m]-registered[key][m])<=1e-9 for m in METHODS)
                else:assert abs(cost[key]-registered[key])<=1e-9
            costs.append(cost)
        counters+=np.array([ctx.states_checked,ctx.residuals_checked,ctx.moments_checked]);del ctx
    assert prediction['batch_cold_total_seconds'].keys()==set(METHODS)
    assert all(abs(sum(c['cold_total_seconds'][m] for c in costs)-prediction['batch_cold_total_seconds'][m])<=1e-9 for m in METHODS)
    cells,decisions=[],[];task_mae={m:{} for m in METHODS};realized={m:[] for m in METHODS}
    for task in plan['tasks']:
        tid=task['id']
        for action in ACTIONS:
            subset=[r for r in rows if (r['task'],r['action'])==(tid,action)]
            truth=float(np.mean([r['actual_gain'] for r in subset]));pred={m:float(np.mean([r['predicted'][m] for r in subset])) for m in METHODS}
            cells.append(dict(task=tid,action=action,truth=truth,predictions=pred,actual_category=category(truth),
                predicted_categories={m:category(v) for m,v in pred.items()},
                unstable=min(r['actual_gain'] for r in subset)<-.25 and max(r['actual_gain'] for r in subset)>.25))
        subset=[c for c in cells if c['task']==tid];oracle=max([0.]+[c['truth'] for c in subset])
        for method in METHODS:
            task_mae[method][tid]=float(np.mean([abs(c['truth']-c['predictions'][method]) for c in subset]))
            means={c['action']:c['predictions'][method] for c in subset};choice=max(ACTIONS,key=lambda a:means[a])
            if means[choice]<1.:choice='PARENT'
            assert prediction['recommendations'][tid][method]==dict(action=choice,means=means,categories={a:category(v) for a,v in means.items()})
            gain=0. if choice=='PARENT' else next(c['truth'] for c in subset if c['action']==choice)
            realized[method].append(gain)
            decisions.append(dict(task=tid,method=method,action=choice,gain=gain,oracle_gain=oracle,regret=oracle-gain,
                accepted_harm=gain<-.25,effective_actions=sum(c['truth']>=1. for c in subset),
                covered_effective_actions=sum(c['truth']>=1. and c['predictions'][method]>=1. for c in subset)))
    mae={m:float(np.mean(list(v.values()))) for m,v in task_mae.items()};best=min(('RAW','SHORT'),key=lambda m:mae[m])
    noninferior=sum(task_mae['M1'][t]<=task_mae[best][t] for t in task_mae['M1'])
    error_gate=mae['M1']<=.8*mae[best] and mae[best]-mae['M1']>=.05 and noninferior==2
    mean_gain={m:float(np.mean(v)) for m,v in realized.items()}
    selection_gate=mean_gain['M1']>=max(mean_gain[b] for b in ('RAW','SHORT','FIXED'))+.10 and min(realized['M1'])>=-.25
    budget_gate=all(c[m]<=cal['budgets'][c['task']]['selection_budget_seconds'] for c in costs for m in ('M1','SHORT'))
    budget_gate=budget_gate and all(r['seconds']<=cal['budgets'][r['task']]['short_budget_seconds'] for r in run_costs if r['engine']=='short')
    assert budget_gate and len(rows)==24 and len(energies)==144 and len(run_costs)==72
    assert int(counters[2])==30
    verify_inputs(out)
    write(out/'final_audit.json',dict(status='PASS',new_full_trainings=30,cumulative_full_trainings=397,
        audit_revision=2,auditor_sha256=sha(Path(__file__)),audit_revision_lock_sha256=sha(out/'audit_revision_lock.json'),
        states_verified=int(counters[0]),residuals_verified=int(counters[1]),moments_verified=int(counters[2]),
        scalar_records=rows,cells=cells,prediction_MAE_pp=mae,task_MAE_pp=task_mae,best_dynamic_baseline=best,
        noninferior_sources=noninferior,decisions=decisions,mean_recommendation_gain=mean_gain,recommendation_gains=realized,
        checkpoint_diagnostics=diagnostics,error_energy=energies,costs=costs,run_costs=run_costs,
        budget_gate=budget_gate,error_gate=bool(error_gate),selection_gate=bool(selection_gate),
        advance=bool(budget_gate and error_gate and selection_gate),test_access=False,base_validation_access=False,
        unseen_source_claim=False,audit_seconds=time.perf_counter()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('preflight','parents','final'));p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
    {'preflight':preflight,'parents':audit_parents,'final':audit_final}[args.phase](out)
