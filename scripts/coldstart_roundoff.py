"""Absolute forward-error bound for the frozen DLinear float32 arithmetic.

The bound depends on operands and operation counts, not observed disagreement.
TF32 must be disabled; this does not certify optimizer trajectories.
"""
import numpy as np


def gamma(operations, unit_roundoff):
    assert operations*unit_roundoff < 1
    return operations*unit_roundoff/(1-operations*unit_roundoff)


def forward_bound(state, values):
    x=np.asarray(values,dtype=np.float64)
    assert x.ndim==3 and x.shape[1]==32
    sw,sb,tw,tb=(np.asarray(state[name],dtype=np.float64) for name in
                 ('seasonal_weight','seasonal_bias','trend_weight','trend_bias'))
    u=np.finfo(np.float32).eps/2
    padded=np.pad(x,((0,0),(12,12),(0,0)),mode='edge')
    window=np.lib.stride_tricks.sliding_window_view(padded,25,axis=1)
    trend=window.mean(-1)
    # 25-term sum and rounded reciprocal/division: 26 operation bound.
    trend_error=gamma(26,u)*np.mean(np.abs(window),axis=-1)
    seasonal=x-trend
    seasonal_error=trend_error+u*(np.abs(x)+np.abs(trend)+trend_error)

    def dot(a,w):return np.einsum('nlc,clh->nhc',a,w,optimize=True)

    def branch(a,a_error,w,b):
        product=dot(a,w)
        magnitude=dot(np.abs(a),np.abs(w))
        propagated=dot(a_error,np.abs(w))
        # 32 products plus 32 additions bounds ordinary FP32 or FMA accumulation.
        error=propagated+gamma(64,u)*(magnitude+propagated)
        bias=b.T[None]
        result=product+bias
        error=error+u*(np.abs(product)+error+np.abs(bias))
        return result,error,magnitude+np.abs(bias)

    s,se,sm=branch(seasonal,seasonal_error,sw,sb)
    t,te,tm=branch(trend,trend_error,tw,tb)
    reference=s+t
    bound=se+te+u*(np.abs(s)+np.abs(t)+se+te)
    # Also cover the negligible float64 reference arithmetic instead of treating
    # it as exact, using a conservative 128-operation path coefficient.
    bound+=gamma(128,np.finfo(np.float64).eps/2)*(sm+tm+np.abs(reference))
    assert np.isfinite(reference).all() and np.isfinite(bound).all()
    return reference,bound
