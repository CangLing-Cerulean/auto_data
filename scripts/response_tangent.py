"""Float64 N1 sensitivities in the original DLinear coordinates.

No assets, datasets, optimizer execution, or training are loaded here. The caller
executes the frozen parent Adam and supplies its *before-update* state and batch
gradient. Flattening uses C order for (channel, 66, horizon[, direction]); the
two bias coordinates (32 and 65) remain distinct. The derivative is of the real
Adam mapping along the supplied parent path, not of IEEE rounding operations.
"""
from dataclasses import dataclass
import numpy as np


BETA1 = 0.9
BETA2 = 0.999
LEARNING_RATE = 0.001
EPS = 1e-8
COORDINATES = 66
DIRECTIONS = 4


def _finite(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def direction_basis():
    """W=[three fixed Helmert columns, ones/2]; returns a fresh array."""
    return np.column_stack((np.array([1., -1., 0., 0.]) / np.sqrt(2.),
                            np.array([1., 1., -2., 0.]) / np.sqrt(6.),
                            np.array([1., 1., 1., -3.]) / np.sqrt(12.),
                            np.ones(4) / 2.))


def actual_coordinates(actual_u):
    """Decompose actual float32 weight-minus-one values, retaining all columns.

    The caller must subtract one from the saved weights before this call; no
    projection, renormalization, clipping, or scientific-domain check occurs.
    """
    u = _finite(actual_u, 'actual_u')
    if u.shape != (4,):
        raise ValueError('actual_u must have shape (4,)')
    return direction_basis().T @ u


@dataclass(frozen=True)
class TangentState:
    """Parameter, first-moment and second-moment derivatives, each p by 4."""
    S: np.ndarray
    M: np.ndarray
    V: np.ndarray

    @classmethod
    def zeros(cls, parameter_count):
        if isinstance(parameter_count, bool) or not isinstance(parameter_count, int) or parameter_count <= 0:
            raise ValueError('parameter_count must be a positive integer')
        return cls(*(np.zeros((parameter_count, DIRECTIONS), dtype=np.float64)
                     for _ in range(3)))


def tangent_step(parent_theta, parent_m, parent_v, parent_step, gradient,
                 hvp_s, forcing, tangent, *, history_all_zero,
                 updated_parent_m=None, updated_parent_v=None):
    """Advance S/M/V once, without updating the real parent.

    All parent vectors and ``gradient`` have shape (p,). ``parent_step`` is the
    number of completed parent updates, so bias correction uses step+1.
    ``hvp_s`` is H(theta_before) @ S_before, and ``forcing`` is C(theta_before),
    both (p,4). Update tangent M/V first, then use *updated* parent m/v and
    tangent M/V in the parameter update. The optional pair updated_parent_m/v
    binds evaluation to actual frozen float32 parent moments after its update;
    otherwise moments are computed by the float64 real Adam recurrence.

    ``history_all_zero`` is a required boolean (p,) certificate that all parent
    gradients BEFORE this update were zero at those coordinates. A zero updated
    v is accepted only with this certificate, zero current gradient and zero m.
    At that joint zero the first derivative is -eta*Mbar/eps; the sqrt mapping
    alone is not differentiable there. The caller advances the certificate with
    ``history_all_zero & (gradient == 0)``. No epsilon is put under the sqrt.
    """
    theta = _finite(parent_theta, 'parent_theta')
    if theta.ndim != 1 or not theta.size:
        raise ValueError('parent_theta must be a nonempty vector')
    p = theta.size
    if isinstance(parent_step, (bool, np.bool_)) or not isinstance(parent_step, (int, np.integer)) or parent_step < 0:
        raise ValueError('parent_step must be a nonnegative integer')
    m, v, g = (_finite(x, name) for x, name in
               ((parent_m, 'parent_m'), (parent_v, 'parent_v'), (gradient, 'gradient')))
    if any(x.shape != (p,) for x in (m, v, g)):
        raise ValueError('parent state and gradient shapes differ')
    history = np.asarray(history_all_zero)
    if history.shape != (p,) or history.dtype != np.bool_:
        raise ValueError('history_all_zero must be a boolean vector of shape (p,)')
    if np.any(v < 0) or np.any((v == 0) & (m != 0)):
        raise ValueError('inconsistent or negative parent second moment')
    if np.any(history & ((m != 0) | (v != 0))):
        raise ValueError('zero-gradient history contradicts parent moments')
    if parent_step == 0 and (np.any(m != 0) or np.any(v != 0) or not history.all()):
        raise ValueError('initial Adam state must be zero with zero history')
    S, M, V, hs, c = (_finite(x, name) for x, name in
                      ((tangent.S, 'S'), (tangent.M, 'M'), (tangent.V, 'V'),
                       (hvp_s, 'hvp_s'), (forcing, 'forcing')))
    if any(x.shape != (p, DIRECTIONS) for x in (S, M, V, hs, c)):
        raise ValueError('tangent, HVP and forcing must have shape (p,4)')
    if np.any(V[history] != 0):
        raise ValueError('zero-gradient history must have zero V tangent')
    if parent_step == 0 and any(np.any(x != 0) for x in (S, M, V)):
        raise ValueError('initial tangent state must be zero')
    if (updated_parent_m is None) != (updated_parent_v is None):
        raise ValueError('updated parent moments must be supplied together')
    if updated_parent_m is None:
        m_new = BETA1 * m + (1. - BETA1) * g
        v_new = BETA2 * v + (1. - BETA2) * g * g
    else:
        m_new = _finite(updated_parent_m, 'updated_parent_m')
        v_new = _finite(updated_parent_v, 'updated_parent_v')
        if m_new.shape != (p,) or v_new.shape != (p,):
            raise ValueError('updated parent moments must have shape (p,)')
        # Actual float32 moments need not equal a float64 recurrence bitwise.
        # Bound rounding by operand magnitudes, including cancellation in m.
        # This is a state-input guard, not an algebra/FD acceptance tolerance.
        for value, left, right in ((m_new, BETA1 * m, (1. - BETA1) * g),
                                   (v_new, BETA2 * v, (1. - BETA2) * g * g)):
            bound = 8. * np.finfo(np.float32).eps * (np.abs(left) + np.abs(right))
            if np.any(np.abs(value - (left + right)) > bound):
                raise ValueError('updated moments disagree with Adam recurrence')
    if np.any(v_new < 0):
        raise ValueError('negative updated second moment')
    zero = v_new == 0
    if np.any(zero & (~history | (g != 0) | (m_new != 0))):
        raise ValueError('zero v lacks all-zero parent gradient history')
    gd = hs + c
    M_new = BETA1 * M + (1. - BETA1) * gd
    V_new = BETA2 * V + 2. * (1. - BETA2) * g[:, None] * gd
    step = int(parent_step) + 1
    mbar = m_new / (1. - BETA1 ** step)
    vbar = v_new / (1. - BETA2 ** step)
    Mbar = M_new / (1. - BETA1 ** step)
    Vbar = V_new / (1. - BETA2 ** step)
    derivative = Mbar / EPS  # joint-zero expression, overwritten where v>0
    positive = ~zero
    root = np.sqrt(vbar[positive])
    denom = root + EPS
    derivative[positive] = (Mbar[positive] / denom[:, None]
                            - (mbar[positive] / (2. * root * denom ** 2))[:, None]
                            * Vbar[positive])
    result = TangentState(S - LEARNING_RATE * derivative, M_new, V_new)
    if not all(np.isfinite(x).all() for x in (result.S, result.M, result.V)):
        raise FloatingPointError('nonfinite tangent update')
    return result


def batch_terms(phi, targets, theta, group_ids, S):
    """Return parent gradient, H@S and four weight-forcing columns.

    phi: (batch,channels,66), targets: (batch,horizon,channels), theta:
    (channels,66,horizon), S: (channels*66*horizon,4). Two explicit ones
    columns at 32 and 65 preserve redundant biases. All terms use the fixed
    denominator batch*channels*horizon, including the common mass direction.
    Only per-channel 66x66 Gram blocks are constructed, never a p by p Hessian.
    """
    x, y, t, s = (_finite(a, name) for a, name in
                  ((phi, 'phi'), (targets, 'targets'), (theta, 'theta'), (S, 'S')))
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] != COORDINATES:
        raise ValueError('phi must have shape (nonempty batch,channels,66)')
    n, channels, _ = x.shape
    if t.ndim != 3 or t.shape[:2] != (channels, COORDINATES) or t.shape[2] == 0:
        raise ValueError('theta must have shape (channels,66,nonempty horizon)')
    horizon = t.shape[2]
    if y.shape != (n, horizon, channels) or s.shape != (t.size, DIRECTIONS):
        raise ValueError('targets or S shape mismatch')
    if not np.all(x[:, :, 32] == 1.) or not np.all(x[:, :, 65] == 1.):
        raise ValueError('original-coordinate bias features must both be one')
    groups = np.asarray(group_ids)
    if (groups.shape != (n,) or not np.issubdtype(groups.dtype, np.integer)
            or np.any(groups < 0) or np.any(groups >= 4)):
        raise ValueError('group_ids must be integer batch vector in [0,4)')
    factor = 2. / (n * channels * horizon)
    residual = np.einsum('bcd,cdh->bch', x, t) - y.transpose(0, 2, 1)
    gradient = factor * np.einsum('bcd,bch->cdh', x, residual)
    gram = factor * np.einsum('bcd,bce->cde', x, x)
    hs = np.einsum('cde,cehj->cdhj', gram, s.reshape(*t.shape, DIRECTIONS))
    forcing = factor * np.einsum('bcd,bch,bj->cdhj', x, residual, direction_basis()[groups])
    return gradient.reshape(-1), hs.reshape(-1, DIRECTIONS), forcing.reshape(-1, DIRECTIONS)
