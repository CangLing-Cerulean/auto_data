"""SOURCE Appendix C operators for a quadratic objective and fixed diagonal P.

This is a formula adaptation, not the authors' Adam implementation and not an
exact derivative through Adam state. No data, checkpoints, or labels are loaded.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class SegmentResult:
    delta: np.ndarray
    propagation: np.ndarray
    response: np.ndarray
    eigenvalues: np.ndarray
    clipped_negative_eigenvalues: int
    roundoff_tolerance: float


def _finite(value, name):
    result=np.asarray(value,dtype=np.float64)
    if not np.isfinite(result).all():raise ValueError(f'{name} must be finite')
    return result


def segment_response(hessian, preconditioner, gradient_change, initial_delta, *, learning_rate, steps):
    """Propagate delta'=-P H delta-P gradient_change for learning_rate*steps.

    H is a symmetric PSD matrix, P a strictly positive diagonal vector. The
    forcing gradient already contains probability/loss weights; no extra 1/N
    is introduced. The zero eigenvalue response is learning_rate*steps.
    """
    h=_finite(hessian,'hessian');p=_finite(preconditioner,'preconditioner')
    g=_finite(gradient_change,'gradient_change');delta=_finite(initial_delta,'initial_delta')
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or not h.shape[0]:raise ValueError('H must be nonempty and square')
    n=h.shape[0]
    if p.shape!=(n,) or g.shape!=(n,) or delta.shape!=(n,):raise ValueError('Coordinate shapes differ')
    if (p<=0).any():raise ValueError('P must be strictly positive')
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Learning rate must be positive')
    if isinstance(steps,bool) or not isinstance(steps,(int,np.integer)) or steps<0:raise ValueError('Steps must be a nonnegative integer')
    duration=float(learning_rate)*int(steps)
    if not math.isfinite(duration):raise ValueError('Nonfinite segment duration')
    eps=np.finfo(np.float64).eps
    h_tolerance=64*n*eps*max(1.,float(np.linalg.norm(h,ord=2)))
    if float(np.max(np.abs(h-h.T)))>h_tolerance:raise ValueError('H is not symmetric')
    root=np.sqrt(p);m=root[:,None]*((h+h.T)/2)*root[None,:]
    if not np.isfinite(m).all():raise FloatingPointError('Preconditioned Hessian overflow')
    eigenvalues,q=np.linalg.eigh(m)
    tolerance=64*n*eps*max(1.,float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -tolerance:raise ValueError('H is not PSD within floating-point roundoff')
    clipped=int(np.sum(eigenvalues<0));eigenvalues=np.maximum(eigenvalues,0.)
    decay=np.exp(-duration*eigenvalues)
    response_values=np.full_like(eigenvalues,duration)
    positive=eigenvalues>0
    response_values[positive]=-np.expm1(-duration*eigenvalues[positive])/eigenvalues[positive]
    propagation=root[:,None]*((q*decay)@q.T)/root[None,:]
    response=root[:,None]*((q*response_values)@q.T)*root[None,:]
    predicted=propagation@delta-response@g
    if not all(np.isfinite(x).all() for x in (propagation,response,predicted)):
        raise FloatingPointError('Nonfinite segment response')
    return SegmentResult(predicted,propagation,response,eigenvalues,clipped,tolerance)


def quadratic_gradient(gram, cross, coordinates):
    """Gradient of mean squared loss in [channel, feature, horizon] coordinates."""
    g=_finite(gram,'gram');c=_finite(cross,'cross');theta=_finite(coordinates,'coordinates')
    if theta.ndim!=3 or min(theta.shape)<=0:raise ValueError('Invalid coordinate shape')
    channels,width,horizon=theta.shape
    if g.shape!=(channels,width,width) or c.shape!=theta.shape:raise ValueError('Moment shapes differ')
    return (2./(channels*horizon))*(g@theta-c)


def adam_diagonal(exp_avg_sq, step, *, beta2=.999, epsilon=1e-8):
    """Frozen denominator 1/(sqrt(v/(1-beta2**step))+epsilon).

    This retains bias correction for v. It does not differentiate v or model
    first-moment dynamics; no heavy-ball learning-rate multiplier is applied.
    """
    v=_finite(exp_avg_sq,'exp_avg_sq')
    if (v<0).any():raise ValueError('Second moment must be nonnegative')
    if isinstance(step,bool) or not isinstance(step,(int,np.integer)) or step<=0:raise ValueError('A positive Adam step is required')
    if not math.isfinite(beta2) or not 0<=beta2<1:raise ValueError('Invalid beta2')
    if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Invalid epsilon')
    correction=1. if beta2==0 else -math.expm1(int(step)*math.log(beta2))
    result=1./(np.sqrt(v/correction)+epsilon)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite Adam preconditioner')
    return result
