import numpy as np
import torch


CANDIDATE = "P1"
INPUT_LENGTH = 32
FIT_BATCH = 512


def _median3_numpy(a, b, c):
    return np.maximum(
        np.minimum(a, b),
        np.minimum(np.maximum(a, b), c),
    )


def _median3_torch(a, b, c):
    return torch.maximum(
        torch.minimum(a, b),
        torch.minimum(torch.maximum(a, b), c),
    )


def _check_candidate(candidate):
    if candidate != CANDIDATE:
        raise ValueError("Unknown candidate")


def _check_x(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 3 or x.shape[1] != INPUT_LENGTH:
        raise ValueError("Expected [batch, 32, channels]")
    if x.shape[2] < 1 or not x.is_floating_point():
        raise ValueError("Expected at least one floating-point channel")


def fit(train, starts, strata, seed):
    train = np.asarray(train)
    starts = np.asarray(starts)
    strata = np.asarray(strata)

    if train.ndim != 2 or train.shape[1] < 1:
        raise ValueError("train must be [rows, channels]")
    if train.dtype != np.float32:
        raise TypeError("train must have dtype float32")
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("starts must be a one-dimensional integer array")
    if strata.ndim != 1 or strata.shape != starts.shape:
        raise ValueError("strata must have the same shape as starts")
    if not np.issubdtype(strata.dtype, np.integer):
        raise ValueError("strata must be integer")
    if starts.size == 0:
        raise ValueError("No training windows")
    if np.any(starts < 0) or np.any(starts > train.shape[0] - 40):
        raise ValueError("Illegal 40-row window start")
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be integer")

    # No random control is needed for this structural representation pair.
    starts = starts.astype(np.int64, copy=False)
    n = starts.size
    channels = train.shape[1]
    offsets = np.arange(INPUT_LENGTH, dtype=np.int64)

    squared_step_sum = np.zeros(channels, dtype=np.float64)

    # Only offsets 0..31 are used. Future targets are never accessed.
    for lo in range(0, n, FIT_BATCH):
        hi = min(lo + FIT_BATCH, n)
        windows = train[starts[lo:hi, None] + offsets[None, :]]
        if not np.all(np.isfinite(windows)):
            raise ValueError("Non-finite T input")
        difference = np.diff(windows.astype(np.float64), axis=1)
        squared_step_sum += np.sum(
            difference * difference, axis=(0, 1), dtype=np.float64
        )

    # Store float32 thresholds so NumPy and Torch use identical thresholds.
    rms_step = np.sqrt(
        squared_step_sum / float(n * (INPUT_LENGTH - 1))
    ).astype(np.float32)

    if not np.all(np.isfinite(rms_step)):
        raise ValueError("Non-finite fitted threshold")

    active = rms_step > np.float32(0.0)
    target = np.zeros(n, dtype=np.bool_)

    for lo in range(0, n, FIT_BATCH):
        hi = min(lo + FIT_BATCH, n)
        windows = train[starts[lo:hi, None] + offsets[None, :]]
        left = windows[:, :-2, :]
        center = windows[:, 1:-1, :]
        right = windows[:, 2:, :]
        median = _median3_numpy(left, center, right)
        correction = np.abs(center - median)
        exceeds = (
            (correction > rms_step[None, None, :])
            & active[None, None, :]
        )
        target[lo:hi] = np.any(exceeds, axis=(1, 2))

    probability = np.full(n, 1.0 / float(n), dtype=np.float64)
    probability /= np.sum(probability, dtype=np.float64)
    weight = np.ones(n, dtype=np.float64)

    params = {
        "version": "median3_endpoint_preserved_v1",
        "input_length": INPUT_LENGTH,
        "rms_step": rms_step.tolist(),
        "active_channels": active.tolist(),
        "target_rule": "any_internal_median_correction_gt_T_step_RMS",
        "target_scope": "whole_window_all_channels_all_forecast_steps",
        "target_count_T": int(np.count_nonzero(target)),
        "complement_count_T": int(n - np.count_nonzero(target)),
        "control_coefficients": [0.25, 0.5, 0.25],
        "control_seed": int(seed),
        "control_seed_used": False,
        "mask_semantics": "diagnostic_evaluation_group_not_transform_support"
    }

    return {
        "params": params,
        "candidates": {
            CANDIDATE: {
                "target": target,
                "control": target.copy(),
                "weight": weight,
                "control_weight": weight.copy(),
                "probability": probability,
                "control_probability": probability.copy(),
            }
        },
    }


def transform(x, params, candidate, control=False):
    _check_candidate(candidate)
    _check_x(x)

    if params.get("input_length") != INPUT_LENGTH:
        raise ValueError("Unexpected fitted input length")
    if len(params["rms_step"]) != x.shape[2]:
        raise ValueError("Fitted channel count does not match x")
    if not isinstance(control, (bool, np.bool_)):
        raise TypeError("control must be boolean")

    left = x[:, :-2, :]
    center = x[:, 1:-1, :]
    right = x[:, 2:, :]

    if control:
        interior = 0.25 * left + 0.5 * center + 0.25 * right
    else:
        interior = _median3_torch(left, center, right)

    # Concatenation creates a new tensor and preserves both endpoints.
    return torch.cat(
        (x[:, :1, :], interior, x[:, -1:, :]),
        dim=1,
    )


def target_mask(raw_x, params, candidate):
    _check_candidate(candidate)
    _check_x(raw_x)

    if params.get("input_length") != INPUT_LENGTH:
        raise ValueError("Unexpected fitted input length")
    if len(params["rms_step"]) != raw_x.shape[2]:
        raise ValueError("Fitted channel count does not match raw_x")

    # Target comparisons use float32, matching fit's float32 input contract.
    raw = raw_x.to(dtype=torch.float32)
    threshold = torch.tensor(
        params["rms_step"],
        dtype=torch.float32,
        device=raw_x.device,
    )
    active = torch.tensor(
        params["active_channels"],
        dtype=torch.bool,
        device=raw_x.device,
    )

    left = raw[:, :-2, :]
    center = raw[:, 1:-1, :]
    right = raw[:, 2:, :]
    median = _median3_torch(left, center, right)
    correction = torch.abs(center - median)

    exceeds = (
        (correction > threshold[None, None, :])
        & active[None, None, :]
    )
    return exceeds.any(dim=2).any(dim=1)
