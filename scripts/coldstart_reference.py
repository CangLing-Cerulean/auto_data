"""Conjugate cold-start reference; all responses are in [Y, Z] order.

Only observations passed by the caller are fitted. ``evidence`` does not
update the reference or maintain diagnostic version credit.
"""

import math

import numpy as np


def _array(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _scalar(value, name):
    return float(_array(value, (), name))


def _finite(array, name):
    if not np.all(np.isfinite(array)):
        raise ValueError(f"nonfinite {name}")
    return array


def predict(history, x):
    """Return JSON-compatible posterior and bivariate Student-t prediction.

    ``history`` is a list of {"x": [six features], "y": [Y, Z]}.
    The returned ``scale`` is the Student-t scale matrix, not covariance.
    """
    x = _array(x, (6,), "x")
    if not isinstance(history, (list, tuple)):
        raise ValueError("history must be a list or tuple")
    features, responses = [], []
    for observation in history:
        if not isinstance(observation, dict) or not {"x", "y"} <= observation.keys():
            raise ValueError("each observation requires x and y")
        features.append(_array(observation["x"], (6,), "history x"))
        responses.append(_array(observation["y"], (2,), "history y"))
    design = np.array(features, dtype=float).reshape(-1, 6)
    targets = np.array(responses, dtype=float).reshape(-1, 2)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            precision = _finite(np.eye(6) + design.T @ design, "K")
            mean = _finite(np.linalg.solve(precision, design.T @ targets), "M")
            psi = _finite(np.eye(2) + targets.T @ targets - mean.T @ precision @ mean, "Psi")
            np.linalg.cholesky(psi)
            df = len(history) + 2
            location = _finite(x @ mean, "location")
            scale = _finite((1.0 + x @ np.linalg.solve(precision, x)) * psi / df, "scale")
            np.linalg.cholesky(scale)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise ValueError("invalid posterior numerical calculation") from error
    return {
        "n": len(history), "df": df, "loc": location.tolist(), "scale": scale.tolist(),
        "posterior": {"K": precision.tolist(), "M": mean.tolist(), "Psi": psi.tolist(), "nu": len(history) + 3},
    }


def conditional_z(prediction, observed_y):
    """Return exact scalar Student-t Z | Y parameters (scale is squared)."""
    df = _scalar(prediction["df"], "df")
    loc = _array(prediction["loc"], (2,), "loc")
    scale = _array(prediction["scale"], (2, 2), "scale")
    y = _scalar(observed_y, "observed_y")
    if df <= 0 or not np.allclose(scale, scale.T, rtol=1e-12, atol=0):
        raise ValueError("df must be positive and scale symmetric")
    try:
        np.linalg.cholesky(scale)
    except np.linalg.LinAlgError as error:
        raise ValueError("scale must be positive definite") from error
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            residual = y - loc[0]
            location = loc[1] + scale[1, 0] / scale[0, 0] * residual
            schur = scale[1, 1] - scale[1, 0] * scale[0, 1] / scale[0, 0]
            conditional_scale = (df + residual * residual / scale[0, 0]) / (df + 1) * schur
        except FloatingPointError as error:
            raise ValueError("invalid conditional numerical calculation") from error
    if not math.isfinite(location) or not math.isfinite(conditional_scale) or conditional_scale <= 0:
        raise ValueError("invalid conditional parameters")
    return {"df": df + 1, "loc": float(location), "scale": float(conditional_scale)}


def student_t_logpdf(value, df, loc, scale):
    """Scalar Student-t log density using a squared scale parameter."""
    value, df, loc, scale = [_scalar(v, name) for v, name in
                             zip((value, df, loc, scale), ("value", "df", "loc", "scale"))]
    if df <= 0 or scale <= 0:
        raise ValueError("df and scale must be positive")
    try:
        result = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                  - 0.5 * (math.log(df) + math.log(math.pi) + math.log(scale))
                  - (df + 1) / 2 * math.log1p(((value - loc) / math.sqrt(scale)) ** 2 / df))
    except (OverflowError, ValueError) as error:
        raise ValueError("invalid density numerical calculation") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite density")
    return result


def evidence(prediction, hypothesis, observed_y, observed_z):
    """Compute log p_h(Z|Y) - log p_0(Z|Y), without fitting this response.

    Hypothesis keys: mean_y, mean_z, std_y, std_z, rho. Standard deviations
    must be positive and abs(rho) < 1.
    """
    values = {key: _scalar(hypothesis[key], key) for key in
              ("mean_y", "mean_z", "std_y", "std_z", "rho")}
    y, z = _scalar(observed_y, "observed_y"), _scalar(observed_z, "observed_z")
    if values["std_y"] <= 0 or values["std_z"] <= 0 or abs(values["rho"]) >= 1:
        raise ValueError("invalid Gaussian standard deviations or correlation")
    reference = conditional_z(prediction, y)
    try:
        location = values["mean_z"] + values["rho"] * values["std_z"] / values["std_y"] * (y - values["mean_y"])
        std = values["std_z"] * math.sqrt(1 - values["rho"] ** 2)
        if not math.isfinite(location) or not math.isfinite(std) or std <= 0:
            raise ValueError("invalid Gaussian conditional parameters")
        gaussian_logpdf = -0.5 * math.log(2 * math.pi) - math.log(std) - 0.5 * ((z - location) / std) ** 2
        reference_logpdf = student_t_logpdf(z, **reference)
        ratio = gaussian_logpdf - reference_logpdf
        if not all(math.isfinite(v) for v in (gaussian_logpdf, reference_logpdf, ratio)):
            raise ValueError("nonfinite evidence")
    except OverflowError as error:
        raise ValueError("invalid evidence numerical calculation") from error
    return {"log_ratio": ratio, "hypothesis_logpdf": gaussian_logpdf,
            "reference_logpdf": reference_logpdf, "reference_conditional": reference,
            "hypothesis_conditional": {"loc": location, "std": std}}
