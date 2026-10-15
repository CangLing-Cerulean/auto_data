"""Conditional one-step Adam risk allocation; no training or dataset I/O.

The optimized objective is a second-order approximation, not actual Adam risk.
Coordinates use the original optimizer parameterization, including both biases.
"""
from __future__ import annotations

import numpy as np


class NonsmoothAdam(ValueError):
    """The second derivative at the requested expansion point is undefined."""


def adam_derivatives(theta, m, v, gradient, *, next_step, lr=0.001,
                     beta1=0.9, beta2=0.999, eps=1e-8):
    """Return U(g), diagonal U'(g), U''(g) for unregularized standard Adam.

    m and v are the moments BEFORE next_step. No clipping, weight decay,
    AMSGrad, gradient accumulation, or smoothing is performed.
    """
    theta, m, v, x = [np.asarray(a, dtype=np.float64)
                       for a in (theta, m, v, gradient)]
    if not (theta.shape == m.shape == v.shape == x.shape):
        raise ValueError("parameter, moment, and gradient shapes must match")
    if not all(np.isfinite(a).all() for a in (theta, m, v, x)):
        raise ValueError("nonfinite state")
    if (not isinstance(next_step, (int, np.integer)) or next_step < 1
            or not 0 < beta1 < 1 or not 0 < beta2 < 1
            or not np.isfinite([lr, eps]).all() or lr <= 0 or eps <= 0
            or np.any(v < 0)):
        raise ValueError("invalid Adam configuration or second moment")
    a = beta1 * m / (1 - beta1 ** next_step)
    b = (1 - beta1) / (1 - beta1 ** next_step)
    c = beta2 * v / (1 - beta2 ** next_step)
    d = (1 - beta2) / (1 - beta2 ** next_step)
    n = a + b * x
    s = np.sqrt(c + d * x * x)
    if np.any(s == 0):
        raise NonsmoothAdam("zero second moment and zero mean gradient; no C2 expansion")
    r = s + eps
    sp = d * x / s
    spp = d * c / s ** 3
    updated = theta - lr * n / r
    first = -lr * (b / r - n * sp / r ** 2)
    second = lr * (2 * b * sp / r ** 2 + n * spp / r ** 2
                   - 2 * n * sp ** 2 / r ** 3)
    if not all(np.isfinite(a).all() for a in (updated, first, second)):
        raise FloatingPointError("nonfinite Adam derivatives")
    return updated, first, second


def allocate_signed(coefficients, masses, floor_fraction=0.5):
    """Globally minimize sum(a/P) over sum(P)=1, P>=alpha*q.

    Positive coefficients use the convex square-root allocation with active
    floors. If all coefficients are nonpositive, enumerate simplex vertices.
    Exact-zero objective returns q. Ties otherwise use the first group index.
    This guarantee applies only to the supplied scalar surrogate.
    """
    a, q = [np.asarray(x, dtype=np.float64) for x in (coefficients, masses)]
    if (a.ndim != 1 or a.shape != q.shape or len(a) < 1
            or not np.isfinite(a).all() or not np.isfinite(q).all()
            or np.any(q <= 0) or abs(q.sum() - 1) > 1e-12
            or not 0 < floor_fraction < 1):
        raise ValueError("invalid coefficients, probability masses, or floor")
    q = q / q.sum()  # only the already validated floating-point sum
    lower = floor_fraction * q
    if np.all(a == 0):
        return q.copy()
    scale = np.max(np.abs(a))
    scaled = a / scale
    positive = scaled > 0
    if not positive.any():
        vertices = np.tile(lower, (len(q), 1))
        vertices[np.arange(len(q)), np.arange(len(q))] += 1 - lower.sum()
        return vertices[np.argmin((scaled / vertices).sum(axis=1))]
    result = lower.copy()
    free = positive.copy()
    # Once a coordinate is below its floor it remains bound at the solution.
    for _ in range(len(q) + 1):
        available = 1 - result[~free].sum()
        roots = np.sqrt(scaled[free])
        proposed = available * roots / roots.sum()
        bound = proposed < lower[free]
        if not bound.any():
            result[free] = proposed
            break
        indices = np.flatnonzero(free)
        free[indices[bound]] = False
    if abs(result.sum() - 1) > 1e-12 or np.any(result < lower - 1e-15):
        raise ArithmeticError("probability solver failed feasibility")
    return result


def group_coefficients(scores, group_ids, group_count):
    """q_g^2 E_g[score], with empirical q from all supplied examples."""
    scores = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(group_ids)
    if (scores.ndim != 1 or ids.shape != scores.shape
            or ids.dtype.kind not in "iu" or group_count < 1
            or len(scores) == 0 or np.any(ids < 0) or np.any(ids >= group_count)
            or not np.isfinite(scores).all()):
        raise ValueError("invalid group scores")
    counts = np.bincount(ids, minlength=group_count)
    if np.any(counts == 0):
        raise ValueError("empty group")
    masses = counts / counts.sum()
    means = np.bincount(ids, weights=scores, minlength=group_count) / counts
    return masses * masses * means, masses


def quadratic_scores(gradients, first, second, risk_gradient, risk_hessian):
    """Dense reference for small algebra checks, never a full-model backend."""
    g = np.asarray(gradients, dtype=np.float64)
    jg = g * first
    geometry = np.einsum("ni,ij,nj->n", jg, risk_hessian, jg)
    optimizer = np.sum(g * g * (risk_gradient * second), axis=1)
    return {"gradient_norm": np.sum(g * g, axis=1),
            "jacobian_norm": np.sum(jg * jg, axis=1),
            "geometry": geometry, "optimizer": optimizer,
            "full": geometry + optimizer}


def dlinear_scores(gradients, first, second, risk_gradient, feature_second_moment):
    """Block contraction for a chunk of original DLinear sample gradients.

    gradients: (n, channels, coordinates, horizon); other parameter arrays:
    (channels, coordinates, horizon). feature_second_moment is E_C[phi phi^T],
    shape (channels, coordinates, coordinates), without loss normalization.
    The risk is mean squared error over C examples, channels, and horizon.
    """
    g = np.asarray(gradients, dtype=np.float64)
    if g.ndim != 4:
        raise ValueError("expected n, channels, coordinates, horizon")
    channels, dimension, horizon = g.shape[1:]
    shape = (channels, dimension, horizon)
    first, second, risk_gradient, moment = [np.asarray(a, dtype=np.float64)
        for a in (first, second, risk_gradient, feature_second_moment)]
    if (first.shape != shape or second.shape != shape or risk_gradient.shape != shape
            or moment.shape != (channels, dimension, dimension)):
        raise ValueError("incompatible DLinear risk arrays")
    if not all(np.isfinite(a).all() for a in (g, first, second, risk_gradient, moment)):
        raise ValueError("nonfinite DLinear risk arrays")
    jg = g * first
    geometry = (2 / (channels * horizon)
                * np.einsum("ncdh,cde,nceh->n", jg, moment, jg, optimize=True))
    optimizer = np.sum(g * g * (risk_gradient * second), axis=(1, 2, 3))
    return {"gradient_norm": np.sum(g * g, axis=(1, 2, 3)),
            "jacobian_norm": np.sum(jg * jg, axis=(1, 2, 3)),
            "geometry": geometry, "optimizer": optimizer,
            "full": geometry + optimizer}
