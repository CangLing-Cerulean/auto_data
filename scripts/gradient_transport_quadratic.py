"""Algebraic response prototype for fixed-index group loss weights.

This is a float64 NumPy research reference, not a production DLinear runner.
No data loading, sampling-policy changes, optimizer tuning, or training dispatch.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class AdamState:
    theta: np.ndarray
    first: np.ndarray
    second: np.ndarray
    step: int = 0


def adam_step(state, gradient, *, learning_rate, beta1=.9, beta2=.999, epsilon=1e-8):
    if gradient.shape != state.theta.shape or not np.isfinite(gradient).all():
        raise ValueError('Invalid gradient')
    if not (learning_rate > 0 and 0 <= beta1 < 1 and 0 <= beta2 < 1 and epsilon > 0):
        raise ValueError('Invalid optimizer constants')
    step = state.step + 1
    first = beta1 * state.first + (1 - beta1) * gradient
    second = beta2 * state.second + (1 - beta2) * gradient ** 2
    theta = state.theta - learning_rate * (first / (1 - beta1 ** step)) / (np.sqrt(second / (1 - beta2 ** step)) + epsilon)
    return AdamState(theta, first, second, step)


def group_statistics(features, targets, groups, n_groups, theta):
    """Per-group sums divided by the full batch size (including absent groups)."""
    x, y, group, point = map(np.asarray, (features, targets, groups, theta))
    if x.ndim != 2 or y.shape != (len(x),) or point.shape != (x.shape[1],) or not len(x):
        raise ValueError('Invalid quadratic array shapes')
    if group.shape != y.shape or not np.issubdtype(group.dtype, np.integer):
        raise ValueError('Invalid group vector')
    if n_groups < 1 or np.any(group < 0) or np.any(group >= n_groups):
        raise ValueError('Group outside declared range')
    if not all(np.isfinite(a).all() for a in (x, y, point)):
        raise ValueError('Nonfinite quadratic inputs')
    residual = x @ point - y
    q = np.zeros((n_groups, x.shape[1]), dtype=np.float64)
    h = np.zeros((n_groups, x.shape[1], x.shape[1]), dtype=np.float64)
    for group_id in range(n_groups):
        mask = group == group_id
        q[group_id] = x[mask].T @ residual[mask] / len(x)
        h[group_id] = x[mask].T @ x[mask] / len(x)
    return q, h


def transported_gradient(group_gradients, group_curvatures, weights, candidate_theta, parent_theta):
    q, h, w, child, parent = map(np.asarray, (group_gradients, group_curvatures, weights, candidate_theta, parent_theta))
    if q.ndim != 2 or h.shape != (q.shape[0], q.shape[1], q.shape[1]):
        raise ValueError('Invalid transport statistics')
    if w.shape != (q.shape[0],) or np.any(w < 0):
        raise ValueError('Weights must be nonnegative and match groups')
    if child.shape != (q.shape[1],) or parent.shape != child.shape:
        raise ValueError('Invalid transport parameters')
    if not all(np.isfinite(a).all() for a in (q, h, w, child, parent)):
        raise ValueError('Nonfinite transport inputs')
    return w @ q + np.einsum('g,gij,j->i', w, h, child - parent)
