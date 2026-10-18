"""Prequential bivariate reference for the fixed open-program descriptors."""
import numpy as np
from coldstart_reference import evidence


FEATURES=('intercept','effective_positive_fraction','sampling_ess_fraction',
          'input_relative_rms_change_tanh','offset_relative_rms_tanh','mean_abs_log_scale_tanh',
          'loss_std_under_sampling_tanh','parent_B_target_contrast_tanh','target_B_fraction')


def predict(history,x):
    d=len(FEATURES);x=np.asarray(x,dtype=np.float64)
    design=np.asarray([r['x'] for r in history],dtype=np.float64).reshape(-1,d)
    targets=np.asarray([r['y'] for r in history],dtype=np.float64).reshape(-1,2)
    if x.shape!=(d,) or not all(np.isfinite(v).all() for v in (x,design,targets)):
        raise ValueError('Invalid fixed reference features or responses')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        precision=np.eye(d)+design.T@design
        mean=np.linalg.solve(precision,design.T@targets)
        psi=np.eye(2)+targets.T@targets-mean.T@precision@mean
        np.linalg.cholesky(psi)
        df=len(history)+2;location=x@mean
        scale=(1+x@np.linalg.solve(precision,x))*psi/df
        np.linalg.cholesky(scale)
    if not all(np.isfinite(v).all() for v in (precision,mean,psi,location,scale)):
        raise ValueError('Nonfinite reference posterior')
    return dict(n=len(history),df=df,loc=location.tolist(),scale=scale.tolist(),
                posterior=dict(K=precision.tolist(),M=mean.tolist(),Psi=psi.tolist(),nu=len(history)+3))
