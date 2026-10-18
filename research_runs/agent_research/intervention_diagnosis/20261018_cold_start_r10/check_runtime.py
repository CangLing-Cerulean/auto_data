"""Generated-input integration check, streamed to the existing GPU; no source values."""
import numpy as np
import torch
import coldstart_runtime as r


def check():
    r.configure();checks=[]
    rows=np.arange(100*370,dtype=np.float32).reshape(100,370)/1000
    w=r.windows(rows);assert w.shape==(61,32,370)
    assert np.array_equal(w[0],rows[:32]) and np.array_equal(w[-1],rows[60:92])
    checks.append('window_row_coordinates_and_no_cross_boundary')
    x,y=r.batch(torch.as_tensor(rows,device='cuda'),np.array([0,60]))
    assert np.array_equal(x.cpu(),w[[0,60]]) and np.array_equal(y.cpu(),np.stack([rows[32:40],rows[92:100]]))
    checks.append('GPU_window_and_horizon_indexing')
    registry=[dict(id='q1',expression='mean_channels(mean_time(x))')]
    q,params=r.registry_values({'A':rows,'B':rows+3},registry)
    assert abs(q['A']['q1'].mean())<1e-14 and abs(q['A']['q1'].std()-1)<1e-14
    assert np.allclose(q['B']['q1']-q['A']['q1'],3/params['q1']['scale'],rtol=1e-6)
    checks.append('measurement_A_fit_and_B_transfer')
    target=dict(window_expression='position >= 0.5',channels=[0,369],horizons=[4,7])
    mask=r.target_mask(target,q['B'],61)
    assert mask.shape==(61,8,370) and mask.sum()==31*2*2 and not mask[:30].any()
    checks.append('target_window_channel_horizon_intersection')
    for expr in ('1','0'):
        try:r.target_mask(dict(window_expression=expr,channels=list(range(370)),horizons=list(range(8))),{},61)
        except r.ExpressionError:pass
        else:raise AssertionError('empty target/complement accepted')
    checks.append('target_empty_and_full_rejected')
    torch.manual_seed(18421);model=r.DLinear(370).cuda()
    residual=r.errors(model,torch.as_tensor(rows,device='cuda'))
    with torch.no_grad():
        manual=model(torch.as_tensor(np.array(w),device='cuda')).cpu().numpy()-np.stack([rows[i+32:i+40] for i in range(61)])
    assert np.array_equal(residual,manual)
    checks.append('full_partition_residual_independent_index_reference')
    rng=np.random.default_rng(18421);u=rng.random(10000);n=11481
    cdf=np.cumsum(np.ones(n));cdf/=cdf[-1];cdf[-1]=1
    assert np.array_equal(np.searchsorted(cdf,u,side='right'),(u*n).astype(np.int64))
    checks.append('uniform_S_and_W_common_random_indices')
    a=residual.astype(np.float64)**2;b=(residual.astype(np.float64)+.03)**2
    p=mask.mean();Y=(a.mean()-b.mean())/a.mean();dt=(a[mask]-b[mask]).mean()/a.mean();dc=(a[~mask]-b[~mask]).mean()/a.mean()
    assert abs(Y-p*dt-(1-p)*dc)<1e-12
    checks.append('Y_Z_common_denominator_identity')
    return dict(status='PASS',checks=checks,count=len(checks),torch=torch.__version__,numpy=np.__version__,real_data_access=False,new_full_trainings=0)
