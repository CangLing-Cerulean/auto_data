"""Synthetic GPU checks; no dataset or checkpoint files are read."""
import copy
import json
import time
import numpy as np
import torch
import open_processing_runtime as runtime


def main():
    runtime.configure();start=time.perf_counter();checks=[]
    rng=np.random.default_rng(48219)
    rows={part:rng.normal(size=(320,370)).astype(np.float32) for part in 'ABC'}
    gpu=runtime.tensors(rows);n=281
    arrays={part:dict(model_input=runtime.windows(v).copy(),offset=np.zeros((n,370),np.float32),
                      scale=np.ones((n,370),np.float32)) for part,v in rows.items()}
    prepared=runtime.prepared_gpu(arrays)
    p,loss,stats=runtime.normalize_weights(np.ones(n),np.ones(n))
    assert np.all(p==1/n) and np.array_equal(loss,np.ones(n))
    checks.append('identity_uniform_probabilities_and_unit_loss')
    for seed in runtime.SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        original=runtime.DLinear(370).cuda();changed=copy.deepcopy(original)
        oldopt=runtime.optimizer(original);newopt=runtime.optimizer(changed)
        sampling=np.random.default_rng(seed+100000)
        for step in range(3):
            ids=(sampling.random(runtime.BATCH)*n).astype(np.int64)
            x,y=runtime.batch(gpu['A'],ids)
            index=torch.as_tensor(ids,device='cuda',dtype=torch.int64)
            encoded=prepared['A']['model_input'][index]
            target=(y-prepared['A']['offset'][index,None,:])/prepared['A']['scale'][index,None,:]
            assert torch.equal(x,encoded) and torch.equal(y,target)
            oldopt.zero_grad(set_to_none=True);newopt.zero_grad(set_to_none=True)
            left=((original(x)-y)**2).mean((1,2)).mean()
            right=(((changed(encoded)-target)**2).mean((1,2))*torch.as_tensor(loss[ids],device='cuda',dtype=y.dtype)).mean()
            assert torch.equal(left,right);left.backward();right.backward()
            assert all(torch.equal(a.grad,b.grad) for a,b in zip(original.parameters(),changed.parameters()))
            oldopt.step();newopt.step()
            assert all(torch.equal(a,b) for a,b in zip(original.parameters(),changed.parameters()))
            for a,b in zip(oldopt.state.values(),newopt.state.values()):
                assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
        for part in 'BC':
            np.testing.assert_array_equal(runtime.errors(original,gpu[part]),runtime.decoded_errors(changed,gpu[part],prepared[part]))
    checks.append('three_seed_three_step_identity_forward_gradient_adam_and_BC_errors_bitwise')
    offset=np.linspace(-.5,.5,370,dtype=np.float32);scale=np.linspace(.5,2.,370,dtype=np.float32)
    transformed={part:dict(model_input=((runtime.windows(v)-offset)/scale).astype(np.float32),
                           offset=np.broadcast_to(offset,(n,370)).copy(),scale=np.broadcast_to(scale,(n,370)).copy()) for part,v in rows.items()}
    mapped=runtime.prepared_gpu(transformed)
    with torch.no_grad():
        expected=[]
        for first in range(0,n,runtime.BATCH):
            stop=min(first+runtime.BATCH,n);_,truth=runtime.batch(gpu['C'],np.arange(first,stop))
            value=changed(mapped['C']['model_input'][first:stop])*torch.as_tensor(scale,device='cuda')+torch.as_tensor(offset,device='cuda')
            expected.append((value-truth).cpu().numpy())
    np.testing.assert_array_equal(runtime.decoded_errors(changed,gpu['C'],mapped['C']),np.concatenate(expected))
    checks.append('nontrivial_output_mapping_compared_with_original_truth')
    first,_=runtime.processing_identity(arrays,p,loss)
    second,_=runtime.processing_identity(transformed,p,loss);assert first!=second
    altered={part:dict(v) for part,v in arrays.items()};altered['B']['offset']=arrays['B']['offset'].copy();altered['B']['offset'][0,0]=1
    assert runtime.processing_identity(altered,p,loss)[0]!=first
    p2=p.copy();p2[0]*=2;p2/=p2.sum()
    assert runtime.processing_identity(arrays,p2,loss)[0]!=first
    l2=loss.copy();l2[0]=2
    assert runtime.processing_identity(arrays,p,l2)[0]!=first
    checks.append('processing_identity_covers_inputs_output_mapping_and_both_weights')
    prior=runtime.predict([], [1,1,1,0,0,0,0,0,.5]);assert prior['df']==2 and prior['loc']==[0.,0.]
    np.testing.assert_allclose(prior['scale'],np.eye(2)*(1+3.25)/2,rtol=0,atol=0)
    checks.append('nine_feature_reference_prior')
    print(json.dumps(dict(status='PASS',checks=checks,seconds=time.perf_counter()-start,
                         real_data_access=False,full_trainings=0,synthetic_adam_updates=18)))


if __name__=='__main__':main()
