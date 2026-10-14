"""Independent fourth-moment and automatic-differentiation cross-check."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/root/auto_data_experiments/20261014_sampling_weight_information_r1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def check(a,b,atol,rtol):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape
    error=float(np.max(np.abs(a-b)));bound=atol+rtol*float(np.max(np.abs(b)))
    assert np.isfinite(a).all() and np.isfinite(b).all() and error<=bound,(error,bound)
    return error,bound
def verify(out,plan,result):
    for name,digest in plan['input_sha256'].items():assert sha(name)==digest,name
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in result['statistics_sha256'].items():assert sha(out/name)==digest,name

def unpack(snapshot,channels,step):
    assert snapshot['step']==step
    names=[f'{kind}.{c}.{part}' for kind in ('seasonal','trend') for c in range(channels) for part in ('weight','bias')]
    assert list(snapshot['model'])==names
    m={k:v.numpy().astype(np.float64) for k,v in snapshot['model'].items()}
    theta=np.stack([np.concatenate((m[f'seasonal.{c}.weight'].T,m[f'seasonal.{c}.bias'][None],m[f'trend.{c}.weight'].T,m[f'trend.{c}.bias'][None])) for c in range(channels)])
    assert np.array_equal(theta,snapshot['original_coordinates'].numpy())
    return theta

def audit_task(out,plan,task,result):
    tid=task['id'];channels=task['channels'];scale=2/(channels*8)
    expected_anchors=[dict(seed=p['seed'],step=step) for p in task['parents'] for step in plan['steps']]
    expected_keys={(p['seed'],step,a) for p in task['parents'] for step in plan['steps'] for a in ['PARENT']+plan['actions']}
    actual_keys=[(r['seed'],r['step'],r['action']) for r in result['records']]
    assert len(actual_keys)==90 and len(set(actual_keys))==90 and set(actual_keys)==expected_keys
    assert result['anchors']==expected_anchors and result['source']==task['source']
    assert result['statistics_sha256']==sha(out/'statistics'/f'{tid}.npz')
    with np.load(task['abc'],allow_pickle=False) as z:
        values=torch.from_numpy(z['A_values']).cuda();starts=torch.from_numpy(z['A_starts']).cuda()
        mask=z['A_first'][z['A_starts']] if task['group_kind']=='first_flag' else z['A_cycles'][z['A_starts']]<task['A_cycle_start']+154
    n=len(mask);counts=[int(mask.sum()),int((~mask).sum())];flags=torch.tensor(mask,device='cuda');offset=torch.arange(40,device='cuda')
    grams=torch.zeros((2,channels,66,66),device='cuda',dtype=torch.float64);crosses=torch.zeros((2,channels,66,8),device='cuda',dtype=torch.float64)
    fourth=torch.zeros_like(grams);third=torch.zeros_like(crosses);targets=torch.zeros((2,channels),device='cuda',dtype=torch.float64)
    fixtures=[]
    with torch.no_grad():
        for lo in range(0,n,2048):
            data=values[starts[lo:lo+2048,None]+offset];x=data[:,:32].transpose(1,2)
            smooth=F.avg_pool1d(torch.cat((x[...,:1].expand(-1,-1,12),x,x[...,-1:].expand(-1,-1,12)),dim=2),25,stride=1)
            phi=torch.cat((x-smooth,torch.ones_like(x[...,:1]),smooth,torch.ones_like(x[...,:1])),dim=2).double()
            y=data[:,32:].transpose(1,2).double();local=flags[lo:lo+len(data)]
            if lo==0:fixtures.append((phi[0].cpu(),y[0].cpu()))
            if lo+len(data)==n:fixtures.append((phi[-1].cpu(),y[-1].cpu()))
            for group,select in enumerate((local,~local)):
                for c in range(channels):
                    px=phi[select,c];py=y[select,c];norm=(px*px).sum(1);weighted=px*norm[:,None]
                    grams[group,c]+=px.T@px;crosses[group,c]+=px.T@py
                    fourth[group,c]+=px.T@weighted;third[group,c]+=weighted.T@py
                    targets[group,c]+=(py.square().sum(1)*norm).sum()
    g,r,j,v,z=(a.cpu().numpy() for a in (grams,crosses,fourth,third,targets))
    saved=np.load(out/'statistics'/f'{tid}.npz',allow_pickle=False)
    check(saved['gram'],g,1e-10,1e-10);check(saved['cross'],r,1e-10,1e-10)
    coefficients={'PARENT':np.array([1/n,1/n])}
    masses={'KEEP_FIRST':[1.,0.],'KEEP_SECOND':[0.,1.],'FAVOR_FIRST':[.75,.25],'FAVOR_SECOND':[.25,.75]}
    for action,mass in masses.items():
        coefficients[action]=np.asarray(mass)/counts
        with np.load(task['probabilities'][action],allow_pickle=False) as p:
            check(p['probability'],np.where(mask,coefficients[action][0],coefficients[action][1]),1e-15,1e-12)
    norm_checks=[];rows=[];k=0;max_second_error_ratio=0.
    for parent in task['parents']:
        for step in plan['steps']:
            theta=unpack(torch.load(Path(parent['directory'])/f'step_{step}.pt',map_location='cpu',weights_only=True),channels,step)
            assert np.array_equal(theta,saved['theta'][k])
            second=np.array([scale**2*(np.sum(theta*(j[group]@theta))-2*np.sum(theta*v[group])+np.sum(z[group])) for group in range(2)])
            error,bound=check(saved['gradient_norm_square_sum'][:,k],second,1e-8,1e-8);max_second_error_ratio=max(max_second_error_ratio,error/bound)
            for fixture,(px,py) in enumerate(fixtures):
                params=torch.tensor(theta,requires_grad=True);residual=torch.einsum('cf,cfh->ch',px,params)-py
                loss=residual.square().mean();gradient,=torch.autograd.grad(loss,params)
                expected=scale**2*float((px.square().sum(1)*residual.detach().square().sum(1)).sum())
                actual=float(gradient.square().sum());err,bnd=check(actual,expected,1e-8,1e-8)
                norm_checks.append(dict(task=tid,seed=parent['seed'],step=step,fixture=fixture,error=err,bound=bnd))
            for action,q in coefficients.items():
                row=next(rw for rw in result['records'] if (rw['seed'],rw['step'],rw['action'])==(parent['seed'],step,action))
                gs=q[0]*g[0]+q[1]*g[1];rs=q[0]*r[0]+q[1]*r[1];w=q*n
                gw=(w[0]*g[0]+w[1]*g[1])/n;rw=(w[0]*r[0]+w[1]*r[1])/n
                matrix_gap=max(float(np.max(np.abs(gs-gw))),float(np.max(np.abs(rs-rw))))
                check(row['moment_max_gap'],matrix_gap,1e-10,1e-10)
                assert matrix_gap<=1e-10+1e-10*max(float(np.max(np.abs(gs))),float(np.max(np.abs(rs))))
                mean=scale*(gs@theta-rs);mean_weighted=scale*(gw@theta-rw)
                gradient_gap=float(np.max(np.abs(mean-mean_weighted)))
                check(row['expected_gradient_max_gap'],gradient_gap,1e-10,1e-10)
                assert gradient_gap<=1e-10+1e-10*float(np.max(np.abs(mean)))
                mean2=float(np.sum(mean*mean));es=float(q@second);ew=float((q*q*n)@second);vs,vw=es-mean2,ew-mean2
                assert min(vs,vw)>=0
                expected=dict(mean_gradient_norm_square=mean2,sampling_second_moment=es,weighting_second_moment=ew,
                    sampling_covariance_trace=vs,weighting_covariance_trace=vw,sampling_batch256_trace=vs/256,weighting_batch256_trace=vw/256)
                for name,value in expected.items():check(row[name],value,1e-8,1e-8)
                assert row['task']==tid and row['group_counts']==counts and row['window_count']==n and row['source']==task['source']
                check(row['group_window_probability'],q,1e-15,1e-12);check(row['uniform_loss_weights'],q*n,1e-12,1e-12)
                if vs>0:check(row['weighting_over_sampling_variance'],vw/vs,1e-8,1e-8)
                else:assert row['weighting_over_sampling_variance'] is None
                assert row['identity_control']==(action=='PARENT')
                if action=='PARENT':check(vs,vw,1e-8,1e-8)
                rows.append(row)
            k+=1
    assert k==18 and len(rows)==90 and len(norm_checks)==36;saved.close()
    return dict(task=tid,records=90,norm_checks=norm_checks,max_second_error_bound_ratio=max_second_error_ratio)

def main(out):
    started=time.perf_counter();assert out==ROOT and out.resolve()==ROOT
    plan=read(out/'plan.json');result=read(out/'diagnostic.json');lock=read(out/'diagnostic_result_lock.json')
    source=read(out/'source_sync_receipt.json');assert source['status']=='exact_commit_and_tree_verified' and source['plan_sha256']==sha(out/'plan.json')
    assert result['status']=='completed' and result['plan_sha256']==sha(out/'plan.json') and lock['diagnostic_sha256']==sha(out/'diagnostic.json')
    assert [t['task'] for t in result['tasks']]==[t['id'] for t in plan['tasks']]
    assert result['action_pairs']==360 and result['identity_controls']==90
    verify(out,plan,result);write(out/'audit.lock',lock);checks=[]
    for task in plan['tasks']:
        current=next(t for t in result['tasks'] if t['task']==task['id'])
        assert current==read(out/f"{task['id']}_diagnostic.json")
        checked=audit_task(out,plan,task,current);checks.append(checked)
        print(json.dumps({k:v for k,v in checked.items() if k!='norm_checks'}),flush=True)
    assert sum(t['records'] for t in checks)==450 and sum(len(t['norm_checks']) for t in checks)==180
    verify(out,plan,result);assert sha(out/'diagnostic.json')==lock['diagnostic_sha256']
    output=dict(status='PASS',parent_anchors_verified=90,paired_action_records_verified=360,identity_controls_verified=90,
        autograd_checks_verified=180,checks=checks,original_inputs_unchanged=True,diagnostic_sha256=sha(out/'diagnostic.json'),
        audit_seconds=time.perf_counter()-started,new_full_trainings=0,cumulative_full_trainings=373,BC_values_access=False,D_access=False,
        test_access=False,actual_child_response_assessed=False,conclusion_scope='gradient_information_only_not_model_response_or_Adam_update_covariance')
    write(out/'final_audit.json',output);print(json.dumps(dict(status='PASS',audit_seconds=output['audit_seconds'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True);torch.set_num_threads(4);main(args.output)
    except Exception:
        write(args.output/'audit_failure.json',dict(status='failed',error=traceback.format_exc(),new_full_trainings=0));raise
