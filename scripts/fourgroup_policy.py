"""Pure algebra for the frozen Hydraulic four-group M1 policy."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

TASKS = ("G1", "G2", "G3")
SEEDS = (2026101701, 2026101702, 2026101703)
A_INTERVALS = {"G1": (0, 308), "G2": (514, 822), "G3": (1028, 1336)}
ACTIONS = tuple(f"G{group}_{direction}" for group in range(4) for direction in ("UP", "DOWN"))
METHODS = ("M1", "RAW", "SOURCE", "SHORT", "PARENT")
SHORT_STEPS = (500, 1000, 1500, 2000, 2500)
RADIUS = 0.20
REPRESENTATION_TOLERANCE = 1e-7
ADOPTION_GAIN_PERCENT = 1.0
WINDOWS_PER_CYCLE = 21
CYCLES_PER_GROUP = 77
WINDOWS_PER_GROUP = 1617
WINDOW_COUNT = 6468


def _keys(mapping: Mapping, expected: tuple, name: str) -> None:
    if not isinstance(mapping, Mapping) or tuple(mapping) != expected:
        raise ValueError(f"{name} must contain exactly these ordered keys: {expected}")


def _number(value: Real, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (result <= 0 if positive else False):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def assign_groups(task: str, window_cycles: ArrayLike, a_interval: Sequence[int]) -> NDArray[np.int64]:
    """Assign each supplied A window to its fixed consecutive 77-cycle group."""
    if task not in TASKS:
        raise ValueError("unknown task")
    interval = np.asarray(a_interval)
    if interval.shape != (2,) or interval.dtype.kind not in "iu":
        raise ValueError("A interval must contain two integers")
    if tuple(interval.tolist()) != A_INTERVALS[task]:
        raise ValueError("A interval differs from the frozen task")
    cycles = np.asarray(window_cycles)
    if cycles.shape != (WINDOW_COUNT,) or cycles.dtype.kind not in "iu":
        raise ValueError("A window cycles must be a one-dimensional integer array of length 6468")
    start, end = A_INTERVALS[task]
    unique, counts = np.unique(cycles, return_counts=True)
    if not np.array_equal(unique, np.arange(start, end)) or not np.all(counts == WINDOWS_PER_CYCLE):
        raise ValueError("A must cover every frozen cycle exactly 21 times")
    groups = ((cycles - start) // CYCLES_PER_GROUP).astype(np.int64)
    if not np.array_equal(np.bincount(groups, minlength=4), np.full(4, WINDOWS_PER_GROUP)):
        raise ValueError("each group must have 1617 windows")
    return groups


@dataclass(frozen=True)
class WeightPlan:
    """Actual float32 training weights and unnormalized float64 moment masses."""

    action: str
    group_weights: NDArray[np.float32]
    window_weights: NDArray[np.float32]
    sample_probability: NDArray[np.float64]
    moment_mass: NDArray[np.float64]
    mean_weight: float
    mass_deviation: float
    effective_radius: float
    ess: float
    ess_fraction: float


def actual_weights(groups: ArrayLike, group_weights: ArrayLike, *, action: str = "PARENT") -> WeightPlan:
    """Convert once to float32, validate, and preserve actual mass without repair."""
    if action not in (*ACTIONS, "PARENT"):
        raise ValueError("unknown action")
    labels = np.asarray(groups)
    if labels.shape != (WINDOW_COUNT,) or labels.dtype.kind not in "iu":
        raise ValueError("groups must be 6468 integer labels")
    if np.any(labels > 3) or np.any(labels < 0):
        raise ValueError("group label outside [0, 3]")
    if not np.array_equal(np.bincount(labels.astype(np.int64), minlength=4), np.full(4, WINDOWS_PER_GROUP)):
        raise ValueError("each group must contain 1617 windows")
    raw = np.asarray(group_weights)
    if raw.shape != (4,) or raw.dtype.kind not in "fiu" or not np.all(np.isfinite(raw)):
        raise ValueError("group weights must be four finite real values")
    with np.errstate(over="ignore"):
        weights = raw.astype(np.float32)
    effective = weights.astype(np.float64)
    mean_weight = float(np.mean(effective))
    deviation = mean_weight - 1.0
    radius = float(np.max(np.abs(effective - 1.0)))
    if not np.all(np.isfinite(effective)) or abs(deviation) > REPRESENTATION_TOLERANCE:
        raise ValueError("actual float32 weights violate the mass tolerance")
    if radius > RADIUS + REPRESENTATION_TOLERANCE:
        raise ValueError("actual float32 weights violate the radius tolerance")
    windows = weights[labels].copy()
    probability = np.full(WINDOW_COUNT, 1.0 / WINDOW_COUNT, dtype=np.float64)
    mass = probability * windows.astype(np.float64)
    ess_fraction = mean_weight**2 / float(np.mean(effective**2))
    return WeightPlan(action, weights.copy(), windows, probability, mass, mean_weight,
                      deviation, radius, WINDOW_COUNT * ess_fraction, ess_fraction)


def action_weights(groups: ArrayLike, action: str) -> WeightPlan:
    """Build one frozen menu action, or the zero-intervention PARENT."""
    if action == "PARENT":
        return actual_weights(groups, np.ones(4), action=action)
    if action not in ACTIONS:
        raise ValueError("unknown action")
    group = ACTIONS.index(action) // 2
    sign = 1.0 if action.endswith("UP") else -1.0
    direction = np.full(4, -1.0 / 3.0)
    direction[group] = 1.0
    return actual_weights(groups, 1.0 + sign * RADIUS * direction, action=action)


@dataclass(frozen=True)
class Recommendation:
    action: str
    best_action: str
    best_mean_gain: float
    mean_gains: dict[str, float]


def recommend(gains: Mapping[int, Mapping[str, Real]]) -> Recommendation:
    """Choose one common action from three-seed mean percentage gains."""
    _keys(gains, SEEDS, "gains seeds")
    matrix = []
    for seed in SEEDS:
        _keys(gains[seed], ACTIONS, f"gains[{seed}]")
        matrix.append([_number(gains[seed][action], "gain") for action in ACTIONS])
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.mean(np.asarray(matrix, dtype=np.float64), axis=0)
    if not np.all(np.isfinite(means)):
        raise ValueError("mean gains must be finite")
    best_index = int(np.argmax(means))
    best_action = ACTIONS[best_index]
    best_gain = float(means[best_index])
    return Recommendation(best_action if best_gain >= ADOPTION_GAIN_PERCENT else "PARENT",
                          best_action, best_gain, dict(zip(ACTIONS, means.tolist())))


def recommend_from_scores(parent_scores: Mapping[int, Real],
                          candidate_scores: Mapping[int, Mapping[str, Real]]) -> Recommendation:
    """Compute each seed's relative NMSE gain before averaging across seeds."""
    _keys(parent_scores, SEEDS, "parent scores")
    _keys(candidate_scores, SEEDS, "candidate score seeds")
    gains = {}
    for seed in SEEDS:
        parent = _number(parent_scores[seed], "parent NMSE", positive=True)
        _keys(candidate_scores[seed], ACTIONS, f"candidate scores[{seed}]")
        row = {}
        for action in ACTIONS:
            score = _number(candidate_scores[seed][action], "candidate NMSE")
            if score < 0:
                raise ValueError("candidate NMSE must be nonnegative")
            row[action] = 100.0 * (1.0 - score / parent)
        gains[seed] = row
    return recommend(gains)


class BudgetCalibrationError(ValueError):
    """No registered SHORT prefix meets the frozen calibration threshold."""


@dataclass(frozen=True)
class ShortCalibration:
    proxy_max_seconds: float
    identity_max_seconds: float
    candidate_seconds: float
    required_prefix_seconds: float
    prefix_min_seconds: dict[int, float]
    short_steps: int


def calibrate_short(proxy_seconds: Mapping[int, Real], identity_seconds: Mapping[int, Real],
                    prefix_seconds: Mapping[int, Mapping[int, Real]]) -> ShortCalibration:
    """Choose the earliest common prefix using seed maxima and prefix minima."""
    for name, values in (("proxy", proxy_seconds), ("identity", identity_seconds), ("prefix", prefix_seconds)):
        _keys(values, SEEDS, name)
    proxy = [_number(proxy_seconds[s], "proxy seconds", positive=True) for s in SEEDS]
    identity = [_number(identity_seconds[s], "identity seconds", positive=True) for s in SEEDS]
    prefixes = []
    for seed in SEEDS:
        _keys(prefix_seconds[seed], SHORT_STEPS, f"prefix[{seed}]")
        row = [_number(prefix_seconds[seed][k], "prefix seconds", positive=True) for k in SHORT_STEPS]
        if any(right < left for left, right in zip(row, row[1:])):
            raise ValueError("cumulative prefix timings must be nondecreasing")
        prefixes.append(row)
    cost = 1.125 * max(proxy) + max(identity)
    required = 1.25 * cost
    if not np.isfinite(required):
        raise ValueError("calibrated cost must be finite")
    minima = dict(zip(SHORT_STEPS, np.min(prefixes, axis=0).tolist()))
    for steps, seconds in minima.items():
        if seconds >= required:
            return ShortCalibration(max(proxy), max(identity), cost, required, minima, steps)
    raise BudgetCalibrationError("no SHORT prefix through 2500 steps satisfies the budget; stop the batch")


@dataclass(frozen=True)
class RecommendationStop:
    stop: bool
    matching_baselines: tuple[str, ...]


def recommendation_stop(recommendations: Mapping[str, Mapping[str, str]]) -> RecommendationStop:
    """Stop if one baseline matches M1 on all three fixed tasks."""
    _keys(recommendations, METHODS, "methods")
    for method in METHODS:
        _keys(recommendations[method], TASKS, f"{method} tasks")
        if any(action not in (*ACTIONS, "PARENT") for action in recommendations[method].values()):
            raise ValueError("unknown recommended action")
    if any(action != "PARENT" for action in recommendations["PARENT"].values()):
        raise ValueError("PARENT baseline must recommend PARENT on all tasks")
    matches = tuple(method for method in METHODS[1:]
                    if all(recommendations[method][task] == recommendations["M1"][task] for task in TASKS))
    return RecommendationStop(bool(matches), matches)
