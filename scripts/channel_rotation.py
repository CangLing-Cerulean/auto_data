"""Frozen reversible train-fitted data basis around the unchanged DLinear."""
import numpy as np
import torch


def fit_rotation(normalized_train):
    values=normalized_train.astype(np.float64)
    covariance=values.T@values/len(values)
    eigen,basis=np.linalg.eigh(covariance)
    order=np.argsort(eigen)[::-1]; eigen=eigen[order]; basis=basis[:,order]
    pivots=np.argmax(np.abs(basis),axis=0)
    basis*=np.where(basis[pivots,np.arange(len(pivots))]<0,-1,1)
    orthogonal_error=float(np.max(np.abs(basis.T@basis-np.eye(len(eigen)))))
    sample=values[::max(1,len(values)//1024)]
    roundtrip_error=float(np.max(np.abs(sample@basis@basis.T-sample)))
    energy_error=float(abs(np.square(sample@basis).sum()/np.square(sample).sum()-1))
    assert orthogonal_error<1e-12 and roundtrip_error<1e-10 and energy_error<1e-12
    return basis,eigen,covariance,{'fitted_rows':len(values),'retained_components':len(eigen),
        'orthogonality_max_abs_error':orthogonal_error,'roundtrip_max_abs_error':roundtrip_error,
        'squared_norm_relative_error':energy_error,'whitening':False,'dimensionality_reduction':False}


class RotatedDLinear(torch.nn.Module):
    def __init__(self,core,basis):
        super().__init__(); self.core=core
        self.register_buffer('rotation',torch.from_numpy(basis.astype(np.float32)))

    def forward(self,x):
        return self.core(x@self.rotation)@self.rotation.T
