"""Conditional, fixed 16-run N1 finite differences; no scientific proposals."""
import argparse
import hashlib
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import audit_fourgroup_m1 as independent
import fourgroup_runtime as rt
import n1_path_runtime as base
import response_tangent as reference

PHASES=('fd','audit_fd')
SEED=2026101701
TASK='G1'


def specifications():
    return [dict(id=f'N1FD_G1_d{j}_h{den}_{side}_s{SEED}',direction=j,
        denominator=den,h=1./den,sign=sign,task=TASK,seed=SEED)
        for j in range(4) for den in (64,128) for side,sign in (('MINUS',-1),('PLUS',1))]


def verify(out):
    plan=base.verify(out)
    assert not (out/'stage_failure.json').exists() and not (out/'fd_failure.json').exists()
    fdplan=rt.read(out/'fd_plan.json')
    assert fdplan['stage']=='n1_fixed_finite_differences' and fdplan['runs']==specifications()
    assert fdplan['base_plan_sha256']==rt.sha(out/'plan.json')
    assert fdplan['cumulative_before']==418 and fdplan['maximum_new_full_trainings']==16
    assert fdplan['normalized_tolerance']==.02 and fdplan['low_signal_parent_factor']==1e-6
    assert not fdplan['scientific_candidates_enabled']
    audit=rt.read(out/'paths_audit.json')
    assert audit['status']=='PASS' and audit['all_cost_lower_bounds_pass']
    assert audit['paths_sha256']==rt.sha(out/'paths.json')
    prior=rt.read(out/'paths_sync_receipt.json')
    assert prior['status']=='stage_results_and_release_verified'
    assert prior['paths_audit_sha256']==rt.sha(out/'paths_audit.json')==fdplan['paths_audit_sha256']
    assert fdplan['paths_sync_receipt_sha256']==rt.sha(out/'paths_sync_receipt.json')
    source=rt.read(out/'fd_source_sync_receipt.json')
    assert source['status']=='exact_commit_and_tree_verified'
    assert source['fd_plan_sha256']==rt.sha(out/'fd_plan.json')
    for name,digest in fdplan['code_sha256'].items():
        assert rt.sha(rt.safe_file(out,name))==digest,name
    return plan,fdplan


def groups(out):
    abc=independent.load_abc(out,TASK)
    result=abc['A_cycles'][abc['A_starts']]//77
    np.testing.assert_array_equal(np.bincount(result),np.full(4,1617))
    return result


def train(out,plan,spec,ctx,group_ids):
    assert shutil.disk_usage(out).free>=plan['runtime_min_free_bytes']
    directory=out/'fd_runs'/spec['id']
    directory.mkdir(parents=True)
    started=rt.tick()
    status=dict(spec,status='running',steps_completed=0,new_full_trainings=0)
    rt.write(directory/'status.json',status)
    try:
        # No renormalization after float32 conversion; all four realized components survive.
        group_weights=(1.+spec['sign']*spec['h']*reference.direction_basis()[:,spec['direction']]).astype(np.float32)
        weights=group_weights[group_ids]
        rt.arrays(directory/'weights.npz',group_weights=group_weights,weights=weights,groups=group_ids)
        device_weights=torch.tensor(weights,device='cuda')
        model=ctx.model(SEED)
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(SEED+1000)
        rt.equal(rt.snapshot(model,opt,rng,0),rt.load_snapshot(ctx.parent('full',SEED),0))
        rt.save_snapshot(directory/'initial.pt',model,opt,rng,0)
        draws=np.zeros(ctx.n,dtype=np.int64)
        for step in range(1,3001):
            model.train();opt.zero_grad(set_to_none=True)
            ids=np.searchsorted(ctx.cdf,rng.random(256),side='right')
            np.add.at(draws,ids,1)
            index=torch.tensor(ids,device='cuda')
            batch=ctx.batch('A',index)
            loss=((model(batch[:,:32])-batch[:,32:]).square()*device_weights[index,None,None]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite FD training loss')
            loss.backward();opt.step()
            status['steps_completed']=step
            if step in rt.GRID:
                rt.save_snapshot(directory/f'step_{step}.pt',model,opt,rng,step)
                rt.write(directory/f'step_{step}_metrics.json',{split:ctx.evaluate(model,split) for split in ('B','C')})
                rt.write(directory/'status.json',status,update=True)
        rt.arrays(directory/'sampling.npz',draw_count=draws)
        assets={p.name:rt.sha(p) for p in sorted(directory.iterdir()) if p.name!='status.json'}
        result=dict(status,status='completed',new_full_trainings=1,assets_sha256=assets,
            fd_plan_sha256=rt.sha(out/'fd_plan.json'),diagnostic_only=True)
        rt.write(directory/'result.json',result)
        rt.write(directory/'status.json',dict(status,status='completed',new_full_trainings=1),update=True)
        elapsed=rt.tick()-started
        rt.write(directory/'timing.json',dict(seconds=elapsed,boundary='initialization through states, B/C metrics, hashes, result and status'))
        print(dict(id=spec['id'],status='completed',seconds=elapsed),flush=True)
        return dict(spec,status='completed',seconds=elapsed,result_sha256=rt.sha(directory/'result.json'))
    except Exception:
        rt.write(directory/'status.json',dict(status,status='failed',error=traceback.format_exc(),
            seconds=time.perf_counter()-started),update=True)
        raise


def execute(out,plan,fdplan):
    started=rt.tick()
    ctx=rt.Context(out,plan,TASK)
    group_ids=groups(out)
    results=[]
    for spec in specifications():
        results.append(train(out,plan,spec,ctx,group_ids))
        rt.write(out/'fd_status.json',dict(status='running',completed=len(results),
            new_full_trainings=len(results),cumulative_full_trainings=418+len(results)),update=(out/'fd_status.json').exists())
    result=dict(status='completed_pending_audit',runs=results,new_full_trainings=16,cumulative_full_trainings=434,
        fd_plan_sha256=rt.sha(out/'fd_plan.json'),seconds=rt.tick()-started,scientific_candidates=0)
    rt.write(out/'fd.json',result)
    rt.write(out/'fd_status.json',dict(status='completed_pending_audit',completed=16,
        new_full_trainings=16,cumulative_full_trainings=434),update=True)
    return result


@torch.no_grad()
def residual(ctx,model,split):
    model.eval()
    values=[]
    for lo in range(0,len(ctx.starts[split]),2048):
        batch=ctx.batch(split,torch.arange(lo,min(lo+2048,len(ctx.starts[split])),device='cuda'))
        values.append((model(batch[:,:32])-batch[:,32:]).cpu().numpy())
    return np.concatenate(values)


def audit_run(out,plan,spec):
    directory=out/'fd_runs'/spec['id']
    result=rt.read(directory/'result.json')
    assert result['status']=='completed' and result['steps_completed']==3000
    assert result['fd_plan_sha256']==rt.sha(out/'fd_plan.json')
    for name,digest in result['assets_sha256'].items():
        assert rt.sha(rt.safe_file(directory,name))==digest
    for name,value in spec.items():
        assert result[name]==value
    saved=independent.arrays(directory/'weights.npz')
    # Reconstruct W independently; do not call the training weight helper.
    columns=np.array([[1.,-1.,0.,0.],[1.,1.,-2.,0.],[1.,1.,1.,-3.],[1.,1.,1.,1.]]).T
    columns/=np.sqrt(np.array([2.,6.,12.,4.]))
    expected=(1.+spec['sign']*spec['h']*columns[:,spec['direction']]).astype(np.float32)
    group_ids=groups(out)
    np.testing.assert_array_equal(saved['group_weights'],expected)
    np.testing.assert_array_equal(saved['weights'],expected[group_ids])
    np.testing.assert_array_equal(saved['groups'],group_ids)
    ctx=rt.Context(out,plan,TASK)
    model=independent.new_model(plan,SEED)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    rng=np.random.default_rng(SEED+1000)
    rt.equal(rt.snapshot(model,opt,rng,0),rt.load_snapshot(directory,0))
    rt.equal(rt.load_snapshot(directory,0),rt.load_snapshot(ctx.parent('full',SEED),0))
    weights=torch.tensor(expected[group_ids],device='cuda')
    draws=np.zeros(ctx.n,dtype=np.int64)
    checks=0
    for step in range(1,3001):
        ids=np.searchsorted(ctx.cdf,rng.random(256),side='right')
        np.add.at(draws,ids,1)
        index=torch.tensor(ids,device='cuda')
        batch=ctx.batch('A',index)
        model.train();opt.zero_grad(set_to_none=True)
        loss=((model(batch[:,:32])-batch[:,32:]).square()*weights[index,None,None]).mean()
        loss.backward();opt.step()
        if step in rt.GRID:
            rt.equal(rt.snapshot(model,opt,rng,step),rt.load_snapshot(directory,step))
            recorded=rt.read(directory/f'step_{step}_metrics.json')
            for split in ('B','C'):
                rt.equal(recorded[split],independent.metric(residual(ctx,model,split)))
                checks+=1
    np.testing.assert_array_equal(draws,independent.arrays(directory/'sampling.npz')['draw_count'])
    np.testing.assert_array_equal(draws,independent.arrays(ctx.parent('full',SEED)/'sampling.npz')['draw_count'])
    print(dict(audited_id=spec['id'],states=7,residual_sets=checks,status='PASS'),flush=True)
    return dict(id=spec['id'],states=7,residual_sets=checks)


def error_comparison(left,right,parent,factor):
    a,b=np.linalg.norm(left),np.linalg.norm(right)
    floor=factor*np.linalg.norm(parent)
    difference=np.linalg.norm(left-right)
    denominator=max(a,b,floor)
    # Zero parent and zero signals have exactly zero discrepancy, without inventing a floor.
    normalized=0. if denominator==0 and difference==0 else float(difference/denominator)
    return dict(normalized_error=normalized,absolute_error=float(difference),left_norm=float(a),
        right_norm=float(b),parent_norm=float(np.linalg.norm(parent)),denominator=float(denominator),
        low_signal=bool(max(a,b)<=floor))


def audit(out,plan,fdplan):
    started=rt.tick()
    completed=rt.read(out/'fd.json')
    assert completed['status']=='completed_pending_audit' and completed['new_full_trainings']==16
    assert completed['fd_plan_sha256']==rt.sha(out/'fd_plan.json')
    assert [r['id'] for r in completed['runs']]==[r['id'] for r in specifications()]
    for row in completed['runs']:
        assert row['status']=='completed'
        assert row['result_sha256']==rt.sha(out/'fd_runs'/row['id']/'result.json')
    replays=[audit_run(out,plan,spec) for spec in specifications()]
    ctx=rt.Context(out,plan,TASK)
    model=independent.new_model(plan,SEED)
    directions=reference.direction_basis()
    comparisons,cross_h=[],[]
    for step in rt.GRID:
        parent=out/'paths'/f'N1_G1_s{SEED}'
        saved=independent.arrays(parent/f'step_{step}_tangent.npz')
        model.load_state_dict(rt.load_snapshot(parent,step)['model'])
        parent_errors={split:residual(ctx,model,split).astype(np.float64) for split in ('B','C')}
        jacobians={}
        for split in ('B','C'):
            parts=[]
            for lo in range(0,len(ctx.starts[split]),2048):
                batch=ctx.batch(split,torch.arange(lo,min(lo+2048,len(ctx.starts[split])),device='cuda'))
                parts.append(independent.features(batch[:,:32]).double().cpu().numpy())
            jacobians[split]=np.einsum('bcd,cdhj->bhcj',np.concatenate(parts),saved['S'].reshape(*base.SHAPE,4))
        for j in range(4):
            values={}
            for den in (64,128):
                errors,weights={},{}
                for side in ('MINUS','PLUS'):
                    path=out/'fd_runs'/f'N1FD_G1_d{j}_h{den}_{side}_s{SEED}'
                    model.load_state_dict(rt.load_snapshot(path,step)['model'])
                    errors[side]={split:residual(ctx,model,split).astype(np.float64) for split in ('B','C')}
                    weights[side]=independent.arrays(path/'weights.npz')['group_weights'].astype(np.float64)
                effective=directions.T@(weights['PLUS']-weights['MINUS'])/(2./den)
                for split in ('B','C'):
                    fd=(errors['PLUS'][split]-errors['MINUS'][split])/(2./den)
                    predicted=jacobians[split]@effective
                    item=error_comparison(fd,predicted,parent_errors[split],fdplan['low_signal_parent_factor'])
                    comparisons.append(dict(step=step,split=split,direction=j,h_denominator=den,
                        realized_direction=effective.tolist(),passed=item['normalized_error']<=fdplan['normalized_tolerance'],**item))
                    values[(den,split)]=fd
            for split in ('B','C'):
                item=error_comparison(values[(64,split)],values[(128,split)],parent_errors[split],fdplan['low_signal_parent_factor'])
                cross_h.append(dict(step=step,split=split,direction=j,
                    passed=item['normalized_error']<=fdplan['normalized_tolerance'],**item))
    assert len(comparisons)==96 and len(cross_h)==48
    result=dict(status='PASS',replays=replays,derivative_comparisons=comparisons,two_h_comparisons=cross_h,
        all_derivative_comparisons_pass=all(r['passed'] for r in comparisons),
        all_two_h_comparisons_pass=all(r['passed'] for r in cross_h),
        numerical_gate_pass=all(r['passed'] for r in comparisons+cross_h),
        new_full_trainings=16,cumulative_full_trainings=434,scientific_candidates=0,
        seconds=rt.tick()-started,fd_plan_sha256=rt.sha(out/'fd_plan.json'),fd_sha256=rt.sha(out/'fd.json'),
        D_access=False,test_access=False)
    rt.write(out/'fd_audit.json',result)
    return result


def validate(out,phase):
    plans=verify(out)
    if phase=='fd':
        assert not (out/'fd_execution.lock').exists()
    else:
        assert (out/'fd.json').is_file() and not (out/'audit_fd_execution.lock').exists()
    return plans


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=PHASES)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--launch',action='store_true')
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    out,phase=args.output,args.phase
    rt.configure()
    plan,fdplan=validate(out,phase)
    control=out/'control'/phase
    if args.launch:
        assert not args.supervise
        control.mkdir(parents=True)
        command=[sys.executable,'-B',str(Path(__file__).resolve()),phase,'--output',str(out),'--supervise']
        with (control/'run.log').open('xb') as log:
            child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
        rt.write(control/'launch.json',dict(pid=child.pid,phase=phase,started_unix=time.time()))
        print(dict(status='supervisor_launched',phase=phase,pid=child.pid),flush=True)
        return
    if args.supervise:
        assert control.is_dir() and not (control/'child.json').exists() and not (control/'exit.json').exists()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),phase,'--output',str(out)]
        started=time.time()
        child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL)
        rt.write(control/'child.json',dict(pid=child.pid,supervisor_pid=os.getpid(),command=command,started_unix=started))
        code=child.wait()
        rt.write(control/'exit.json',dict(exit_code=code,started_unix=started,finished_unix=time.time(),phase=phase))
        raise SystemExit(code)
    rt.write(out/f'{phase}_execution.lock',dict(started_unix=time.time(),fd_plan_sha256=rt.sha(out/'fd_plan.json')))
    try:
        result=execute(out,plan,fdplan) if phase=='fd' else audit(out,plan,fdplan)
        print(dict(phase=phase,status=result['status']),flush=True)
    except Exception:
        rt.write(out/'fd_failure.json',dict(phase=phase,status='failed_closed',error=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
