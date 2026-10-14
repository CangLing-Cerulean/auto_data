"""Training-only gradient information audit: direct sample norm accumulation."""
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
ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def tick():torch.cuda.synchronize();return time.perf_counter()
def verify(out):
    assert out==ROOT and out.resolve()==ROOT
    plan=read(out/'plan.json');receipt=read(out/'source_sync_receipt.json')
    assert receipt['status']=='exact_commit_and_tree_verified' and receipt['plan_sha256']==sha(out/'plan.json')
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in plan['input_sha256'].items():assert sha(name)==digest,name
    return plan
def configure():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)

@torch.no_grad()
def task_run(out,plan,task):
    begin=tick();tid=task['id'];channels=task['channels'];scale=2/(channels*8)
    with np.load(task['abc'],allow_pickle=False) as z:
        values=torch.tensor(z['A_values'],device='cuda');starts=torch.tensor(z['A_starts'],device='cuda')
        first=z['A_first'][z['A_starts']] if task['group_kind']=='first_flag' else z['A_cycles'][z['A_starts']]<task['A_cycle_start']+154
    n=len(first);counts=[int(first.sum()),int((~first).sum())];assert min(counts)>0 and n==len(starts)
    probabilities={'PARENT':np.full(n,1/n)}
    for action,path in task['probabilities'].items():
        with np.load(path,allow_pickle=False) as p:probabilities[action]=p['probability'].copy()
    coefficients={}
    for action,p in probabilities.items():
        assert p.shape==(n,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-12
        q=[float(p[first][0]),float(p[~first][0])]
        assert np.all(p[first]==q[0]) and np.all(p[~first]==q[1])
        expected={'PARENT':[counts[0]/n,counts[1]/n],'KEEP_FIRST':[1.,0.],'KEEP_SECOND':[0.,1.],
                  'FAVOR_FIRST':[.75,.25],'FAVOR_SECOND':[.25,.75]}[action]
        assert np.allclose(np.array(q)*counts,expected,atol=1e-12,rtol=1e-12)
        coefficients[action]=q
    anchors=[];coords=[]
    for parent in task['parents']:
        for step in plan['steps']:
            path=Path(parent['directory'])/f'step_{step}.pt';snap=torch.load(path,map_location='cpu',weights_only=True)
            assert snap['step']==step
            theta=snap['original_coordinates'].numpy().astype(np.float64)
            assert theta.shape==(channels,66,8) and np.isfinite(theta).all()
            coords.append(theta);anchors.append(dict(seed=parent['seed'],step=step))
    theta=torch.tensor(np.stack(coords),device='cuda');assert len(theta)==18
    gram=torch.zeros((2,channels,66,66),device='cuda',dtype=torch.float64)
    cross=torch.zeros((2,channels,66,8),device='cuda',dtype=torch.float64)
    squares=torch.zeros((2,18),device='cuda',dtype=torch.float64)
    flags=torch.tensor(first,device='cuda');offset=torch.arange(40,device='cuda')
    for lo in range(0,n,1024):
        data=values[starts[lo:lo+1024,None]+offset];x=data[:,:32].transpose(1,2)
        trend=F.avg_pool1d(F.pad(x,(12,12),mode='replicate'),25,stride=1);one=torch.ones_like(x[...,:1])
        phi=torch.cat((x-trend,one,trend,one),-1).double();y=data[:,32:].transpose(1,2).double()
        errors=torch.einsum('bcf,scfh->bsch',phi,theta)-y[:,None]
        norm2=scale**2*(phi.square().sum(-1)[:,None,:]*errors.square().sum(-1)).sum(-1)
        local=flags[lo:lo+len(data)]
        for group,mask in enumerate((local,~local)):
            px=phi[mask].transpose(0,1);py=y[mask].transpose(0,1)
            gram[group]+=px.transpose(1,2)@px;cross[group]+=px.transpose(1,2)@py
            squares[group]+=norm2[mask].sum(0)
    g,r,s=gram.cpu().numpy(),cross.cpu().numpy(),squares.cpu().numpy();th=np.stack(coords)
    path=out/'statistics'/f'{tid}.npz'
    with path.open('xb') as f:np.savez_compressed(f,gram=g,cross=r,gradient_norm_square_sum=s,theta=th)
    records=[]
    for action,q in coefficients.items():
        w=np.asarray(q)*n;gs=q[0]*g[0]+q[1]*g[1];rs=q[0]*r[0]+q[1]*r[1]
        gw=(w[0]*g[0]+w[1]*g[1])/n;rw=(w[0]*r[0]+w[1]*r[1])/n
        matrix_gap=max(float(np.max(np.abs(gs-gw))),float(np.max(np.abs(rs-rw))))
        matrix_bound=1e-10+1e-10*max(float(np.max(np.abs(gs))),float(np.max(np.abs(rs))))
        assert matrix_gap<=matrix_bound
        for j,anchor in enumerate(anchors):
            mean=scale*(gs@th[j]-rs);weighted=scale*(gw@th[j]-rw)
            gap=float(np.max(np.abs(mean-weighted)));assert gap<=1e-10+1e-10*np.max(np.abs(mean))
            mean2=float(np.sum(mean**2));es=float(q[0]*s[0,j]+q[1]*s[1,j]);ew=float(n*(q[0]**2*s[0,j]+q[1]**2*s[1,j]))
            vs,vw=es-mean2,ew-mean2;assert min(vs,vw)>=0 and np.isfinite([mean2,es,ew,vs,vw]).all()
            if action=='PARENT':assert abs(vs-vw)<=1e-8+1e-8*abs(vs)
            records.append(dict(task=tid,source=task['source'],**anchor,action=action,window_count=n,group_counts=counts,
                group_window_probability=q,uniform_loss_weights=w.tolist(),mean_gradient_norm_square=mean2,
                expected_gradient_max_gap=gap,moment_max_gap=matrix_gap,sampling_second_moment=es,weighting_second_moment=ew,
                sampling_covariance_trace=vs,weighting_covariance_trace=vw,sampling_batch256_trace=vs/256,weighting_batch256_trace=vw/256,
                weighting_over_sampling_variance=None if vs==0 else vw/vs,identity_control=action=='PARENT'))
    return dict(task=tid,source=task['source'],anchors=anchors,records=records,statistics_sha256=sha(path),seconds=tick()-begin)

def main(out):
    started=tick();plan=verify(out);write(out/'diagnostic.lock',dict(plan_sha256=sha(out/'plan.json')));(out/'statistics').mkdir()
    tasks=[]
    for task in plan['tasks']:
        result=task_run(out,plan,task);write(out/f"{task['id']}_diagnostic.json",result);tasks.append(result)
        print(json.dumps(dict(task=task['id'],records=len(result['records']),seconds=result['seconds'])),flush=True)
    assert sum(len(t['records']) for t in tasks)==450
    verify(out)
    result=dict(status='completed',plan_sha256=sha(out/'plan.json'),tasks=tasks,action_pairs=360,identity_controls=90,
        phase_seconds=tick()-started,new_full_trainings=0,cumulative_full_trainings=373,BC_values_access=False,D_access=False,test_access=False,
        actual_child_response_assessed=False,statistics_sha256={f"statistics/{t['task']}.npz":t['statistics_sha256'] for t in tasks})
    write(out/'diagnostic.json',result);write(out/'diagnostic_result_lock.json',dict(diagnostic_sha256=sha(out/'diagnostic.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:configure();main(args.output)
    except Exception:
        write(args.output/'diagnostic_failure.json',dict(status='failed',error=traceback.format_exc(),new_full_trainings=0));raise
