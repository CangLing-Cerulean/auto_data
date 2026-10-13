"""Original-coordinate quadratic gradients and paired DLinear response prediction.

This module never loads a dataset, fits a transform, chooses a budget, or trains
a real parent implicitly. Full moments approximate stochastic Adam dynamics.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


def features(model, x):
    z = x.transpose(1, 2)
    trend = F.avg_pool1d(F.pad(z, (model.kernel // 2,) * 2, mode='replicate'),
                        model.kernel, stride=1)
    one = torch.ones_like(z[..., :1])
    return torch.cat((z - trend, one, trend, one), dim=-1)


def parameters(model):
    return torch.stack([torch.cat((s.weight.T, s.bias[None], t.weight.T, t.bias[None]))
                        for s, t in zip(model.seasonal, model.trend)])


def _parts(value, length):
    return (value[:length].T, value[length], value[length + 1:2 * length + 1].T, value[-1])


@torch.no_grad()
def load_parameters(model, value):
    if value.shape != parameters(model).shape or not bool(torch.isfinite(value).all()):
        raise ValueError('Invalid original-coordinate parameters')
    for c, (s, t) in enumerate(zip(model.seasonal, model.trend)):
        for p, v in zip((s.weight, s.bias, t.weight, t.bias), _parts(value[c], s.in_features)):
            p.copy_(v)


@dataclass
class Moments:
    gram: torch.Tensor
    cross: torch.Tensor
    target_square: torch.Tensor

    def gradient(self, value):
        channels, _, horizon = value.shape
        return (2. / (channels * horizon)) * (self.gram @ value.double() - self.cross)

    def loss(self, value):
        v = value.double()
        return ((v * (self.gram @ v)).sum(1) - 2 * (v * self.cross).sum(1)
                + self.target_square).mean()

    def add_(self, other):
        self.gram.add_(other.gram)
        self.cross.add_(other.cross)
        self.target_square.add_(other.target_square)


@torch.no_grad()
def batch_moments(phi, y, mass):
    """mass = sampling probability * loss weight, without implicit renormalization."""
    if phi.ndim != 3 or y.ndim != 3 or mass.shape != (len(phi),):
        raise ValueError('Invalid moment shapes')
    if phi.shape[:2] != (len(y), y.shape[2]):
        raise ValueError('Channel/window mismatch')
    if not all(bool(torch.isfinite(t).all()) for t in (phi, y, mass)) or bool((mass < 0).any()):
        raise ValueError('Nonfinite or negative moment input')
    x = phi.double().permute(1, 0, 2)
    target = y.double().permute(2, 0, 1)
    weighted = x * mass.double()[None, :, None]
    return Moments(x.transpose(1, 2) @ weighted,
                   weighted.transpose(1, 2) @ target,
                   (target.square() * mass.double()[None, :, None]).sum(1))


@torch.no_grad()
def assign_gradient(model, moments):
    grad = moments.gradient(parameters(model))
    if not bool(torch.isfinite(grad).all()):
        raise FloatingPointError('Nonfinite moment gradient')
    for c, (s, t) in enumerate(zip(model.seasonal, model.trend)):
        for p, g in zip((s.weight, s.bias, t.weight, t.bias), _parts(grad[c], s.in_features)):
            p.grad = g.to(dtype=p.dtype).contiguous()


def paired_parameters(real_parent, proxy_child, proxy_parent):
    if real_parent.shape != proxy_child.shape or proxy_child.shape != proxy_parent.shape:
        raise ValueError('Paired coordinates differ')
    result = real_parent + (proxy_child - proxy_parent)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Nonfinite paired prediction')
    return result


def select_checkpoint(selection_losses, grid):
    """Only B losses enter this selector; C losses are not an argument."""
    if list(selection_losses) != list(grid) or list(grid) != sorted(set(grid)) or not grid:
        raise ValueError('Incomplete or unordered selection grid')
    best_step, best = None, math.inf
    for step in grid:
        loss = float(selection_losses[step])
        if not math.isfinite(loss) or loss < 0:
            raise ValueError('Invalid checkpoint loss')
        if loss < best:
            best_step, best = step, loss
    return best_step
