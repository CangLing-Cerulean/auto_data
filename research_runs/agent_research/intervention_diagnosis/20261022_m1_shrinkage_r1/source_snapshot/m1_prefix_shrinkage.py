"""Closed-form prefix calibration. Pure arrays; no labels, files, or training."""
import numpy as np


def fit_coefficients(parent_discrepancy, child_discrepancy, per_channel):
    u, v = np.asarray(parent_discrepancy, dtype=np.float64), np.asarray(child_discrepancy, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 3 or not all(u.shape):
        raise ValueError('Expected matching nonempty [windows,horizon,channels] discrepancies')
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError('Nonfinite prefix discrepancy')
    axes = (0, 1) if per_channel else (0, 1, 2)
    numerator = np.sum(u * v, axis=axes)
    denominator = np.sum(u * u, axis=axes)
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)
    alpha = np.clip(ratio, 0., 1.)
    if not per_channel:
        alpha = np.repeat(alpha, u.shape[-1])
    return alpha, dict(numerator=np.asarray(numerator).tolist(), denominator=np.asarray(denominator).tolist())


def blend_parameters(raw, paired, alpha):
    raw, paired, alpha = np.asarray(raw), np.asarray(paired), np.asarray(alpha, dtype=np.float64)
    if raw.shape != paired.shape or raw.ndim != 3 or alpha.shape != (raw.shape[0],):
        raise ValueError('Expected [channels,coordinates,horizon] with one coefficient per channel')
    if not all(np.isfinite(x).all() for x in (raw, paired, alpha)) or np.any((alpha < 0) | (alpha > 1)):
        raise ValueError('Nonfinite values or coefficients outside [0,1]')
    result = (raw.astype(np.float64) + alpha[:, None, None] * (paired.astype(np.float64) - raw)).astype(raw.dtype)
    # Endpoint preservation is a numerical property, not a model-selection rule.
    result[alpha == 0] = raw[alpha == 0]
    result[alpha == 1] = paired[alpha == 1]
    return result
