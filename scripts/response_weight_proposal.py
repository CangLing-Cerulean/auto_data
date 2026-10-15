"""Frozen N2 proposal engine; no data/model loading or scientific activation.

Input arrays are parent-only N1 outputs at every B/C checkpoint. ``r`` and
``K_all`` already include the NMSE square-root Q. ``e``, ``J_all`` and
``mae_weights`` instead encode the original NMAE scale and weighted absolute
sum; NMSE Q must NOT be reused as NMAE weights. Channel NMSE is the squared
norm of each channel's rows times its explicit ``channel_nmse_multiplier``.
The caller must bind these arrays to the frozen input/environment locks.
"""

from dataclasses import dataclass
from importlib import metadata
from time import perf_counter

import numpy as np


CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000)
SEEDS = (2026101701, 2026101702, 2026101703)
Q = np.full(4, 0.25)
Z = np.array([[1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
              [-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(12)],
              [0, -2 / np.sqrt(6), 1 / np.sqrt(12)],
              [0, 0, -3 / np.sqrt(12)]])
W = np.column_stack((Z, np.full(4, 0.5)))
RHO, LAMBDA, ESS_MIN = 0.2, 0.01, 0.98
TOL, FLOOR = 1e-7, 1e-12


@dataclass(frozen=True)
class Response:
    r: np.ndarray
    K_all: np.ndarray
    e: np.ndarray
    J_all: np.ndarray
    mae_weights: np.ndarray
    channel_index: np.ndarray
    channel_nmse_multiplier: np.ndarray


@dataclass(frozen=True)
class SeedResponse:
    seed: int
    B: dict[int, Response]
    C: dict[int, Response]


class ProposalError(RuntimeError):
    """An unmodified failure is chained; ``report`` retains partial evidence."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def dependency_versions():
    result = {}
    for package in ("numpy", "cvxpy", "clarabel"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _validate_response(x):
    n = np.asarray(x.r).size
    for name, shape in (("r", (n,)), ("K_all", (n, 4)), ("e", (n,)),
                        ("J_all", (n, 4)), ("mae_weights", (n,)),
                        ("channel_index", (n,)),
                        ("channel_nmse_multiplier", (8,))):
        value = np.asarray(getattr(x, name))
        if not isinstance(getattr(x, name), np.ndarray):
            raise ValueError(f"{name} must be a numpy array")
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid {name}: expected finite {shape}")
    if n == 0 or set(np.asarray(x.channel_index).tolist()) != set(range(8)):
        raise ValueError("channel_index must contain exactly the eight channels")
    if np.any(x.mae_weights < 0) or not np.any(x.mae_weights > 0):
        raise ValueError("NMAE weights must be nonnegative with positive mass")
    if np.any(x.channel_nmse_multiplier <= 0):
        raise ValueError("channel NMSE multipliers must be positive")


def _metrics(x, a):
    residual = np.asarray(x.r, dtype=np.float64) + x.K_all @ a
    mae = float(np.dot(x.mae_weights, np.abs(x.e + x.J_all @ a)))
    channels = [float(np.dot(residual[x.channel_index == c],
                              residual[x.channel_index == c])
                      * x.channel_nmse_multiplier[c]) for c in range(8)]
    return {"nmse": float(residual @ residual), "nmae": mae,
            "channel_nmse": channels}


def _select(curve):
    # Python min is stable: CHECKPOINTS is ascending and ties pick the earliest.
    return min(CHECKPOINTS, key=lambda k: curve[k]["nmse"])


def _parents(seeds):
    if tuple(s.seed for s in seeds) != SEEDS:
        raise ValueError(f"requires ordered frozen seeds {SEEDS}")
    parents = []
    for s in seeds:
        for partition in (s.B, s.C):
            if set(partition) != set(CHECKPOINTS):
                raise ValueError("B and C must each contain all six checkpoints")
            for x in partition.values():
                _validate_response(x)
        b = {k: _metrics(s.B[k], np.zeros(4)) for k in CHECKPOINTS}
        k = _select(b)
        c = _metrics(s.C[k], np.zeros(4))
        if c["nmse"] <= 0 or not np.isfinite(c["nmse"]):
            raise ValueError("parent C NMSE must be finite and positive")
        gaps = {j: b[j]["nmse"] - b[k]["nmse"] for j in CHECKPOINTS if j != k}
        parents.append({"k": k, "B": b, "C": c, "gaps": gaps})
    return parents


def _audit(seeds, parents, u, *, actual):
    a = W.T @ np.asarray(u, dtype=np.float64)
    weights = 1 + np.asarray(u, dtype=np.float64)
    mass = float(Q @ weights)
    ess = float(mass**2 / (Q @ (weights**2)))
    residuals = {"mass": abs(float(Q @ u)),
                 "radius": float(np.max(np.abs(u)) - RHO),
                 "ess": ESS_MIN - ess}
    predictions, gains, selector_changes = [], [], []
    for s, p in zip(seeds, parents):
        curves = {d: {k: _metrics(getattr(s, d)[k], a) for k in CHECKPOINTS}
                  for d in ("B", "C")}
        chosen = _select(curves["B"])
        k = p["k"]
        if chosen != k:
            selector_changes.append({"seed": s.seed, "parent": k, "selected": chosen})
        for j, gap in p["gaps"].items():
            x, y = s.B[k], s.B[j]
            linear = (x.K_all.astype(np.float64).T @ x.r.astype(np.float64)
                      - y.K_all.astype(np.float64).T @ y.r.astype(np.float64))
            lhs = 2 * linear @ a + np.sum((x.K_all @ a)**2)
            residuals[f"{s.seed}/B/{j}"] = float(
                (lhs - gap / 2) / max(p["B"][k]["nmse"], FLOOR))
        pred = curves["C"][k]
        residuals[f"{s.seed}/C/nmae"] = float(
            (pred["nmae"] - 1.01 * p["C"]["nmae"])
            / max(p["C"]["nmae"], FLOOR))
        for c in range(8):
            baseline = p["C"]["channel_nmse"][c]
            residuals[f"{s.seed}/C/channel/{c}"] = float(
                (pred["channel_nmse"][c] - 1.05 * baseline) / max(baseline, FLOOR))
        gain = 100 * (1 - pred["nmse"] / p["C"]["nmse"])
        gains.append(float(gain))
        predictions.append({"seed": s.seed, "parent_checkpoint": k,
                            "selected_checkpoint": chosen, "gain_percent": float(gain),
                            "curves": curves})
    if not all(np.isfinite(v) for v in residuals.values()):
        raise ValueError("nonfinite constraint residual")
    max_violation = max(0.0, *residuals.values())
    # Actual mass/radius tolerances are representation allowances only.
    passed = max_violation <= TOL and not selector_changes
    return {"coordinates": a.tolist(), "mean_weight": mass, "ess_fraction": ess,
            "normalized_residuals": residuals, "max_positive_violation": max_violation,
            "passed": passed, "actual_float32": actual,
            "selector_changes": selector_changes,
            "predictions": predictions, "mean_gain_percent": float(np.mean(gains))}


def propose(seeds, method, *, n1_cost_seconds):
    """Return one FULL/LINEAR proposal, or branch_unsupported/PARENT.

    All three seeds pay the supplied full N1 cost in each method's ledger.
    Function cost includes validation/solve/audit, excludes caller serialization,
    hashing and writing (reported as unknown rather than zero). Failures raise
    ProposalError with the partial report and original exception chained.
    No training, solver fallback, clipping or renormalization is performed.
    """
    start = perf_counter()
    report = {"method": method, "status": "engineering_failure", "errors": [],
              "versions": dependency_versions(), "proposal": None,
              "recommended": None, "solver": None,
              "cost": {"n1_seconds": n1_cost_seconds,
                       "proposal_seconds": None, "necessary_write_hash_seconds": None}}
    try:
        if method not in ("FULL", "LINEAR"):
            raise ValueError("method must be FULL or LINEAR")
        if not np.isfinite(n1_cost_seconds) or n1_cost_seconds < 0:
            raise ValueError("full N1 cost must be a finite nonnegative measurement")
        parents = _parents(seeds)
        report["parent_audit"] = _audit(seeds, parents, np.zeros(4), actual=False)
        unsupported = [{"seed": s.seed, "checkpoint": j, "gap": gap}
                       for s, p in zip(seeds, parents) for j, gap in p["gaps"].items()
                       if gap / max(p["B"][p["k"]]["nmse"], FLOOR) <= 1e-8]
        if unsupported:
            report.update(status="branch_unsupported", recommended="PARENT",
                          unsupported_branches=unsupported)
            return report
        import cvxpy as cp

        if "CLARABEL" not in cp.installed_solvers():
            raise RuntimeError("frozen CLARABEL solver is unavailable")
        v = cp.Variable(3)
        u = Z @ v
        terms, constraints = [], [cp.abs(u) <= RHO,
                                  cp.sum(cp.multiply(Q, cp.square(1 + u))) <= 1 / ESS_MIN]
        for s, p in zip(seeds, parents):
            k, base = p["k"], p["C"]
            x = s.C[k]
            K = x.K_all[:, :3].astype(np.float64)
            if method == "FULL":
                terms.append(cp.sum_squares(x.r + K @ v) / base["nmse"])
            else:
                terms.append(2 * (x.r @ K) @ v / base["nmse"])
            constraints.append((cp.sum(cp.multiply(x.mae_weights,
                                cp.abs(x.e + x.J_all[:, :3] @ v)))
                                - 1.01 * base["nmae"]) / max(base["nmae"], FLOOR) <= 0)
            for c in range(8):
                rows = x.channel_index == c
                channel_base = base["channel_nmse"][c]
                constraints.append((x.channel_nmse_multiplier[c]
                                    * cp.sum_squares(x.r[rows] + K[rows] @ v)
                                    - 1.05 * channel_base) / max(channel_base, FLOOR) <= 0)
            xb = s.B[k]
            for j, gap in p["gaps"].items():
                yb = s.B[j]
                linear = (xb.K_all[:, :3].astype(np.float64).T @ xb.r.astype(np.float64)
                          - yb.K_all[:, :3].astype(np.float64).T @ yb.r.astype(np.float64))
                constraints.append((2 * linear @ v + cp.sum_squares(xb.K_all[:, :3] @ v)
                                    - gap / 2) / max(p["B"][k]["nmse"], FLOOR) <= 0)
        penalty = LAMBDA * cp.sum(cp.multiply(Q, cp.square(u)))
        problem = cp.Problem(cp.Minimize(sum(terms) / 3 + penalty), constraints)
        if not problem.is_dcp():
            raise ValueError("frozen proposal problem is not DCP")
        solve_start = perf_counter()
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-8, tol_gap_rel=1e-8,
                      tol_feas=1e-8)
        stats = problem.solver_stats
        report["solver"] = {"name": stats.solver_name, "status": problem.status,
                            "objective": problem.value, "iterations": stats.num_iters,
                            "solve_time": stats.solve_time,
                            "wall_seconds": perf_counter() - solve_start,
                            "gap_feasibility_target": 1e-8, "is_dcp": True}
        if problem.status != "optimal":
            raise RuntimeError(f"unaccepted Clarabel status: {problem.status}")
        if v.value is None or not np.all(np.isfinite(v.value)):
            raise ValueError("solver did not return a finite proposal")
        report["solver"]["constraint_duals"] = [
            np.asarray(c.dual_value).tolist() for c in constraints]
        report["solver"]["max_cvxpy_constraint_violation"] = max(
            float(np.max(c.violation())) for c in constraints)
        raw_u = Z @ np.asarray(v.value, dtype=np.float64)
        report["raw"] = {"v": v.value.tolist(), "u": raw_u.tolist()}
        report["raw_audit"] = _audit(seeds, parents, raw_u, actual=False)
        actual_weights = np.asarray(1 + raw_u, dtype=np.float32)
        actual_u = actual_weights.astype(np.float64) - 1
        report["proposal"] = {"weights_float32": actual_weights.tolist(),
                              "weights_hex": actual_weights.tobytes().hex(),
                              "u_effective": actual_u.tolist()}
        report["actual_audit"] = _audit(seeds, parents, actual_u, actual=True)
        if not report["raw_audit"]["passed"] or not report["actual_audit"]["passed"]:
            raise ValueError("proposal violates frozen normalized constraint tolerance")
        audit = report["actual_audit"]
        nmae_ok = all(v <= TOL for key, v in audit["normalized_residuals"].items()
                     if key.endswith("/nmae"))
        adopted = audit["mean_gain_percent"] >= 1.0 and nmae_ok
        report.update(status="complete" if adopted else "no_qualified_recommendation",
                      recommended=method if adopted else "PARENT")
        return report
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc),
                                 "repr": repr(exc)})
        raise ProposalError(f"N2 {method} failed: {exc}", report) from exc
    finally:
        report["cost"]["proposal_seconds"] = perf_counter() - start
        if isinstance(n1_cost_seconds, (int, float, np.number)) and np.isfinite(n1_cost_seconds) and n1_cost_seconds >= 0:
            report["cost"]["method_seconds_before_write_hash"] = (
                n1_cost_seconds + report["cost"]["proposal_seconds"])
