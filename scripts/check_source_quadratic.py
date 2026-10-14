"""Independent algebraic checks; generated matrices are not sensor records."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import source_quadratic as source


def matrix_exp_taylor(matrix):
    # Independent augmented-generator reference, scaling/squaring plus series.
    size=max(0,int(np.ceil(np.log2(max(1.,np.linalg.norm(matrix,ord=np.inf))))))
    a=matrix/(2.**size);term=np.eye(len(a));answer=term.copy()
    for order in range(1,100):
        term=term@a/order;answer+=term
        if np.linalg.norm(term,ord=np.inf)<1e-18:break
    else:raise AssertionError('Reference series did not converge')
    for _ in range(size):answer=answer@answer
    return answer


def main(output):
    started=time.perf_counter();rng=np.random.default_rng(2026101407);checks=[]
    def compare(name,actual,expected,atol=2e-11,rtol=2e-11):
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        assert np.allclose(actual,expected,atol=atol,rtol=rtol),name
        checks.append(dict(name=name,max_abs=float(np.max(np.abs(actual-expected))),atol=atol,rtol=rtol))
    for rank in (0,2,5):
        x=rng.normal(size=(rank,5));h=x.T@x/10;p=np.exp(rng.uniform(-2,2,5))
        g=rng.normal(size=5);initial=rng.normal(size=5);duration=.7
        observed=source.segment_response(h,p,g,initial,learning_rate=.01,steps=70)
        generator=np.zeros((6,6));generator[:5,:5]=-p[:,None]*h;generator[:5,5]=-p*g
        expected=matrix_exp_taylor(duration*generator)@np.r_[initial,1.]
        compare(f'augmented_generator_rank_{rank}',observed.delta,expected[:5])
        compare(f'zero_action_rank_{rank}',source.segment_response(h,p,np.zeros(5),np.zeros(5),learning_rate=.01,steps=70).delta,np.zeros(5))
        if rank==0:compare('zero_eigenvalue_limit',observed.delta,initial-duration*p*g)
        left=source.segment_response(h,p,g,initial,learning_rate=.01,steps=30)
        right=source.segment_response(h,p,g,left.delta,learning_rate=.01,steps=40)
        compare(f'segment_composition_rank_{rank}',right.delta,observed.delta)
    compare('zero_duration',source.segment_response(np.eye(3),np.array([.2,2.,5.]),np.ones(3),np.arange(3.),learning_rate=.01,steps=0).delta,np.arange(3.))
    # Normalized group deletion changes both the removed and retained masses.
    n,channels,width,horizon=12,2,5,3
    x=rng.normal(size=(channels,n,width));target=rng.normal(size=(channels,n,horizon));theta=rng.normal(size=(channels,width,horizon))
    parent=np.full(n,1/n);child=np.r_[np.full(n//2,2/n),np.zeros(n//2)]
    def moments(prob):return np.einsum('cni,n,cnj->cij',x,prob,x),np.einsum('cni,n,cnh->cih',x,prob,target)
    gp,cp=moments(parent);gc,cc=moments(child)
    dg=source.quadratic_gradient(gc,cc,theta)-source.quadratic_gradient(gp,cp,theta)
    residual=np.einsum('cnf,cfh->cnh',x,theta)-target
    direct=2/(channels*horizon)*np.einsum('cnf,n,cnh->cfh',x,child-parent,residual)
    compare('normalized_group_gradient',dg,direct)
    direction=rng.normal(size=theta.shape);eps=1e-5
    def objective(value,prob):return np.sum((np.einsum('cnf,cfh->cnh',x,value)-target)**2*prob[None,:,None])/(channels*horizon)
    numeric=((objective(theta+eps*direction,child)-objective(theta+eps*direction,parent))-
             (objective(theta-eps*direction,child)-objective(theta-eps*direction,parent)))/(2*eps)
    compare('normalized_group_finite_difference',np.array(np.sum(dg*direction)),np.array(numeric),atol=2e-9,rtol=2e-9)
    twice_x=np.repeat(x,2,axis=1);twice_y=np.repeat(target,2,axis=1);twice_p=np.repeat(child,2)/2
    g2=np.einsum('cni,n,cnj->cij',twice_x,twice_p,twice_x);c2=np.einsum('cni,n,cnh->cih',twice_x,twice_p,twice_y)
    compare('duplication_no_extra_inverse_N',source.quadratic_gradient(g2,c2,theta),source.quadratic_gradient(gc,cc,theta))
    # Direct simulation of Adam's second-moment recurrence, without reuse of its extraction formula.
    beta=.999;v=np.zeros(5);square=rng.uniform(.1,1.,5)
    for step in range(1,501):v=beta*v+(1-beta)*square
    compare('adam_constant_gradient_bias_correction',source.adam_diagonal(v,500),1/(np.sqrt(square)+1e-8))
    compare('adam_zero_second_moment',source.adam_diagonal(np.zeros(3),1),np.full(3,1e8),atol=0,rtol=0)
    # A negative mode, invalid dimensions and invalid Adam state must fail explicitly.
    rejected=[]
    cases={
        'negative_curvature':lambda:source.segment_response(np.diag([1.,-.01]),np.ones(2),np.ones(2),np.zeros(2),learning_rate=.01,steps=10),
        'asymmetric_hessian':lambda:source.segment_response(np.array([[1.,.2],[0.,1.]]),np.ones(2),np.ones(2),np.zeros(2),learning_rate=.01,steps=10),
        'nonpositive_preconditioner':lambda:source.segment_response(np.eye(2),np.array([0.,1.]),np.ones(2),np.zeros(2),learning_rate=.01,steps=10),
        'nonfinite_gradient':lambda:source.segment_response(np.eye(2),np.ones(2),np.array([np.nan,1.]),np.zeros(2),learning_rate=.01,steps=10),
        'negative_adam_moment':lambda:source.adam_diagonal(np.array([-.1]),500),
        'uninitialized_adam':lambda:source.adam_diagonal(np.zeros(2),0),
    }
    for name,case in cases.items():
        try:case()
        except ValueError:rejected.append(name)
        else:raise AssertionError(f'Invalid input accepted: {name}')
    result=dict(status='PASS',checks=checks,rejected_inputs=rejected,new_full_trainings=0,sensor_records_generated=0,
        base_data_access=False,parent_checkpoint_access=False,candidate_label_access=False,
        source_implementation_reproduction=False,scope='algebraic operators only; not validation of Adam trajectory approximation',
        code_sha256={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,source.__file__)},seconds=time.perf_counter()-started)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status='PASS',numerical_checks=len(checks),rejected_inputs=len(rejected))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
