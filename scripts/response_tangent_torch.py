"""GPU/CPU float64 implementation of the locked four-column N1 recurrence.

The real parent still runs the frozen float32 PyTorch Adam. This module reads
its gradient and updated moments; it never updates model or optimizer state.
The NumPy implementation in response_tangent.py is a separate reference.
"""
from dataclasses import dataclass
import torch

BETA1 = .9
BETA2 = .999
LR = .001
EPS = 1e-8
DIRECTIONS = 4


def basis(device):
    result = torch.tensor([[1., 1., 1., 1.], [-1., 1., 1., 1.],
                           [0., -2., 1., 1.], [0., 0., -3., 1.]],
                          device=device, dtype=torch.float64)
    return result / torch.tensor([2., 6., 12., 4.], device=device, dtype=torch.float64).sqrt()


def check_tensors(values, device=None):
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float64 for value in values):
        raise ValueError('Float64 tensors required')
    if not values or any(value.device != (device or values[0].device) for value in values):
        raise ValueError('Tensor devices must agree')
    if not bool(torch.stack([value.isfinite().all() for value in values]).all()):
        raise FloatingPointError('Nonfinite input')


@dataclass(frozen=True)
class TangentState:
    S: torch.Tensor
    M: torch.Tensor
    V: torch.Tensor

    @classmethod
    def zeros(cls, parameter_count, device):
        if type(parameter_count) is not int or parameter_count <= 0:
            raise ValueError('Positive integer parameter count required')
        return cls(*(torch.zeros((parameter_count, DIRECTIONS), dtype=torch.float64,
                                 device=device) for _ in range(3)))


@torch.no_grad()
def batch_terms(phi, targets, theta, group_ids, S):
    """Fixed-denominator quadratic gradient, H@S, and weight forcing.

    Shapes: phi [batch,channel,66], targets [batch,horizon,channel],
    theta [channel,66,horizon], S [theta.numel(),4]. The two bias columns stay.
    """
    check_tensors((phi, targets, theta, S))
    if phi.ndim != 3 or min(phi.shape[:2]) <= 0 or phi.shape[2] != 66:
        raise ValueError('Invalid feature shape')
    n, channels, _ = phi.shape
    if theta.ndim != 3 or theta.shape[:2] != (channels, 66) or theta.shape[2] <= 0:
        raise ValueError('Invalid original-coordinate shape')
    horizon = theta.shape[2]
    if targets.shape != (n, horizon, channels) or S.shape != (theta.numel(), 4):
        raise ValueError('Invalid target or tangent shape')
    if (group_ids.device != phi.device or group_ids.shape != (n,) or
            group_ids.dtype != torch.int64 or bool(((group_ids < 0) | (group_ids >= 4)).any())):
        raise ValueError('Invalid four-group assignments')
    if not bool(((phi[:, :, 32] == 1) & (phi[:, :, 65] == 1)).all()):
        raise ValueError('Both original bias features must equal one')
    factor = 2. / (n * channels * horizon)
    error = torch.einsum('bcd,cdh->bch', phi, theta) - targets.transpose(1, 2)
    gradient = factor * torch.einsum('bcd,bch->cdh', phi, error)
    gram = factor * torch.einsum('bcd,bce->cde', phi, phi)
    hs = torch.einsum('cde,cehj->cdhj', gram, S.reshape(*theta.shape, 4))
    forcing = factor * torch.einsum('bcd,bch,bj->cdhj', phi, error, basis(phi.device)[group_ids])
    return gradient.reshape(-1), hs.reshape(-1, 4), forcing.reshape(-1, 4)


@torch.no_grad()
def tangent_step(parent_theta, parent_m, parent_v, parent_step, gradient,
                 hvp_s, forcing, tangent, *, history_all_zero,
                 updated_parent_m=None, updated_parent_v=None):
    values = (parent_theta, parent_m, parent_v, gradient, hvp_s, forcing,
              tangent.S, tangent.M, tangent.V)
    check_tensors(values)
    if parent_theta.ndim != 1 or not parent_theta.numel() or type(parent_step) is not int or parent_step < 0:
        raise ValueError('Invalid parent shape or step')
    p = parent_theta.numel()
    if any(x.shape != (p,) for x in (parent_m, parent_v, gradient)):
        raise ValueError('Parent vectors must agree')
    if any(x.shape != (p, 4) for x in (hvp_s, forcing, tangent.S, tangent.M, tangent.V)):
        raise ValueError('Four tangent columns required')
    history = history_all_zero
    if history.shape != (p,) or history.dtype != torch.bool or history.device != parent_theta.device:
        raise ValueError('Boolean zero-history certificate required')
    invalid = ((parent_v < 0) | ((parent_v == 0) & (parent_m != 0)) |
               (history & ((parent_m != 0) | (parent_v != 0))))
    if bool(invalid.any()) or bool((tangent.V[history] != 0).any()):
        raise ValueError('Parent moments contradict zero-gradient history')
    if parent_step == 0 and (not bool(history.all()) or
            any(bool((x != 0).any()) for x in (parent_m, parent_v, tangent.S, tangent.M, tangent.V))):
        raise ValueError('Initial optimizer and tangents must be zero')
    if (updated_parent_m is None) != (updated_parent_v is None):
        raise ValueError('Updated moments must be supplied as a pair')
    g = gradient
    if updated_parent_m is None:
        m_new = BETA1 * parent_m + (1 - BETA1) * g
        v_new = BETA2 * parent_v + (1 - BETA2) * g.square()
    else:
        check_tensors((updated_parent_m, updated_parent_v), parent_theta.device)
        m_new, v_new = updated_parent_m, updated_parent_v
        if m_new.shape != (p,) or v_new.shape != (p,):
            raise ValueError('Updated moment shape mismatch')
        for value, left, right in ((m_new, BETA1 * parent_m, (1 - BETA1) * g),
                                   (v_new, BETA2 * parent_v, (1 - BETA2) * g.square())):
            bound = 8. * torch.finfo(torch.float32).eps * (left.abs() + right.abs())
            if bool(((value - left - right).abs() > bound).any()):
                raise ValueError('Updated moments disagree with Adam recurrence')
    zero = v_new == 0
    if bool(((v_new < 0) | (zero & (~history | (g != 0) | (m_new != 0)))).any()):
        raise ValueError('Invalid updated second moment')
    gd = hvp_s + forcing
    new_M = BETA1 * tangent.M + (1 - BETA1) * gd
    new_V = BETA2 * tangent.V + 2 * (1 - BETA2) * g[:, None] * gd
    step = parent_step + 1
    mbar, vbar = m_new / (1 - BETA1 ** step), v_new / (1 - BETA2 ** step)
    Mbar, Vbar = new_M / (1 - BETA1 ** step), new_V / (1 - BETA2 ** step)
    derivative = Mbar / EPS
    positive = ~zero
    root = vbar[positive].sqrt()
    denom = root + EPS
    derivative[positive] = (Mbar[positive] / denom[:, None] -
        (mbar[positive] / (2 * root * denom.square()))[:, None] * Vbar[positive])
    result = TangentState(tangent.S - LR * derivative, new_M, new_V)
    check_tensors((result.S, result.M, result.V))
    return result
