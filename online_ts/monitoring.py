from __future__ import annotations

import numpy as np

from .config import TaskConfig, digest
from .models import SCALE_FLOOR
from .labels import complete_labels, expected_keys


def row_key(row: dict) -> tuple:
    return (row["issue_ns"], row["target_ns"], row["variable"], row["horizon"])


def summarize(rows: list[dict], config: TaskConfig, scale: np.ndarray, *, expected_origins=None) -> dict:
    counts = np.zeros((config.horizon_steps, len(config.variables)), dtype=int)
    mae = np.zeros_like(counts, dtype=float)
    mse = np.zeros_like(counts, dtype=float)
    bias = np.zeros_like(counts, dtype=float)
    valid_rows = [r for r in rows if r["actual"] is not None and np.isfinite(r["actual"])]
    for r in valid_rows:
        h, v = r["horizon"] - 1, config.variables.index(r["variable"])
        error = r["value"] - r["actual"]
        counts[h, v] += 1
        mae[h, v] += abs(error)
        mse[h, v] += error ** 2
        bias[h, v] += error
    denominator = np.maximum(counts, 1)
    mae /= denominator
    mse /= denominator
    bias /= denominator
    complete = bool((counts > 0).all())
    scale = np.maximum(np.asarray(scale), SCALE_FLOOR)
    origins = sorted({r["issue_ns"] for r in valid_rows})
    return {**complete_labels(rows, config, expected_origins),
            "prediction_count": len(rows), "labeled_count": len(valid_rows), "origins": len(origins),
            "first_issue": origins[0] if origins else None, "last_issue": origins[-1] if origins else None,
            "complete_cells": complete, "counts_by_horizon_variable": counts.tolist(),
            "mae_by_horizon_variable": mae.tolist(), "mse_by_horizon_variable": mse.tolist(),
            "mae_by_variable": mae.mean(axis=0).tolist(), "mse_by_variable": mse.mean(axis=0).tolist(),
            "normalized_mae_by_variable": (mae.mean(axis=0) / scale).tolist(),
            "normalized_mse_by_variable": (mse.mean(axis=0) / scale ** 2).tolist(),
            "bias_by_variable": bias.mean(axis=0).tolist(),
            "business_groups": {name: {"variables": variables,
                "normalized_mae": float(np.mean(mae[:, [config.variables.index(v) for v in variables]] /
                    scale[[config.variables.index(v) for v in variables]])),
                "normalized_mse": float(np.mean(mse[:, [config.variables.index(v) for v in variables]] /
                    scale[[config.variables.index(v) for v in variables]] ** 2))}
                for name, variables in config.release.business_groups.items()},
            "normalized_mae": float(np.dot(mae.mean(axis=0) / scale, config.variable_weights)) if complete else None,
            "normalized_mse": float(np.dot(mse.mean(axis=0) / scale ** 2, config.variable_weights)) if complete else None,
            "maximum_inference_ms": max((r["inference_ms"] for r in rows), default=0.0),
            "clean_inputs": all(r["quality"] == "OK" for r in rows)}


def enough(evidence: dict, config: TaskConfig) -> bool:
    return (evidence.get("labels_complete", False) and evidence["complete_cells"] and evidence["label_coverage"] >= config.monitoring.minimum_label_coverage
            and evidence["origins"] >= config.release.minimum_origins)


def same_window_baselines(rows, source, config, scale, *, expected_origins):
    """Reconstruct cheap predictions with each issue's actual source visibility."""
    from .data import aligned, causal_fill
    from .config import GateError
    history = {}
    period = config.search.seasonal_period
    for issue in sorted({r["issue_ns"] for r in rows}):
        try:
            raw = source.read_until(issue, start=issue - (period + config.max_fill_steps) * config.step_ns)
            grid = aligned(raw, config, end=issue)
            values = causal_fill(grid, config).tail(period).to_numpy()
            if len(values) != period or not np.isfinite(values).all() or grid.tail(period).isna().any().any():
                return {"eligible": False, "issues": ["BASELINE_INPUT_UNAVAILABLE"]}
            history[issue] = values
        except GateError:
            return {"eligible": False, "issues": ["BASELINE_INPUT_UNAVAILABLE"]}
    reports = {}
    for name in ("last_value", "seasonal_naive"):
        predicted = []
        for row in rows:
            values = history[row["issue_ns"]]
            index = -1 if name == "last_value" else (row["horizon"] - 1) % period
            predicted.append({**row, "value": float(values[index, config.variables.index(row["variable"])])})
        reports[name] = summarize(predicted, config, scale, expected_origins=expected_origins)
    return {"eligible": all(enough(r, config) and r["clean_inputs"] for r in reports.values()),
            "reports": reports, "paired_to": "same_origins_targets_labels_and_model_scale"}


def paired_comparison(champion: list[dict], challenger: list[dict], config: TaskConfig, scale: np.ndarray, *, expected_origins=None) -> dict:
    a = {row_key(r): r for r in champion}
    b = {row_key(r): r for r in challenger}
    union = expected_keys(champion + challenger, config, expected_origins)
    keys = sorted(k for k in set(a) & set(b) if a[k]["actual"] is not None and b[k]["actual"] is not None)
    left = [a[k] for k in keys]
    right = [b[k] for k in keys]
    ca = summarize(champion, config, scale, expected_origins=expected_origins)
    cb = summarize(challenger, config, scale, expected_origins=expected_origins)
    coverage = len(keys) / len(union) if union else 0.0
    result = {"passed": False, "coverage": coverage, "paired_count": len(keys), "champion": ca, "challenger": cb, "issues": []}
    if not enough(ca, config) or not enough(cb, config) or set(a) != union or set(b) != union or coverage != 1.0:
        result["issues"].append("PAIRED_LABEL_COVERAGE")
        return result
    if any(a[k]["actual"] != b[k]["actual"] for k in keys):
        result["issues"].append("PAIRED_LABEL_MISMATCH")
        return result
    before, after = ca["normalized_mae"], cb["normalized_mae"]
    improvement = (before - after) / max(before, SCALE_FLOOR)
    result["improvement"] = improvement
    for name in config.release.business_groups:
        before_group = ca["business_groups"][name]["normalized_mae"]
        after_group = cb["business_groups"][name]["normalized_mae"]
        if after_group > before_group * (1 + config.release.maximum_critical_regression) + 1e-12:
            result["issues"].append("WORST_BUSINESS_GROUP_REGRESSION")
    if improvement < config.release.minimum_improvement:
        result["issues"].append("NO_ONLINE_GAIN")
    av = np.asarray(ca["mae_by_variable"])
    bv = np.asarray(cb["mae_by_variable"])
    for variable in config.release.critical_variables:
        index = config.variables.index(variable)
        if bv[index] > av[index] * (1 + config.release.maximum_critical_regression) + 1e-12:
            result["issues"].append("CRITICAL_VARIABLE_REGRESSION")
    ah = np.asarray(ca["mae_by_horizon_variable"]).mean(axis=1)
    bh = np.asarray(cb["mae_by_horizon_variable"]).mean(axis=1)
    if float(np.mean(bv <= av + 1e-12)) < 0.5 or float(np.mean(bh <= ah + 1e-12)) < 0.5:
        result["issues"].append("MAJORITY_REGRESSION")
    # Independent time subwindows discourage promotion driven by a few favorable timestamps.
    origins = sorted({k[0] for k in keys})
    subwindows = []
    for part in np.array_split(origins, 3):
        origin_set = set(int(x) for x in part)
        pair_keys = [k for k in keys if k[0] in origin_set]
        if not pair_keys:
            continue
        errors_a = [abs(a[k]["value"] - a[k]["actual"]) / scale[config.variables.index(k[2])] * config.variable_weights[config.variables.index(k[2])] for k in pair_keys]
        errors_b = [abs(b[k]["value"] - b[k]["actual"]) / scale[config.variables.index(k[2])] * config.variable_weights[config.variables.index(k[2])] for k in pair_keys]
        subwindows.append(float(np.mean(errors_b)) <= float(np.mean(errors_a)) + 1e-12)
    result["subwindow_non_regression"] = subwindows
    if len(subwindows) < 3 or not all(subwindows):
        result["issues"].append("UNSTABLE_ONLINE_GAIN")
    if cb["maximum_inference_ms"] > min(config.release.maximum_inference_ms, config.frequency_seconds * 1000):
        result["issues"].append("INFERENCE_BUDGET")
    if not ca["clean_inputs"] or not cb["clean_inputs"]:
        result["issues"].append("INPUT_QUALITY")
    result["passed"] = not result["issues"]
    return result


def degradation(evidence: dict, history: dict, config: TaskConfig, *, offline_reference: float) -> tuple[dict, bool]:
    from copy import deepcopy
    from .qualification import evaluation_qualification
    result = deepcopy(history)
    qualification = evaluation_qualification(evidence, config)
    result.update(qualification=qualification, assessments=[], count=0, triggered=False, optimization_opportunity=False,
                  offline_reference={"metric": "normalized_mse", "value": offline_reference})
    regimes = result.setdefault("regimes", {})
    if evidence.get("window_start") is not None and result.get("last_window_end") is not None:
        if evidence["window_start"] != result["last_window_end"]:
            for state in regimes.values():
                state["counts"] = {}
    if evidence.get("window_end") is not None:
        result["last_window_end"] = evidence["window_end"]
    context = evidence.get("regime", {"status": "unobserved", "regime_id": "unlabelled"})
    result["regime"] = context
    if not qualification["evaluation_eligible"] or context["status"] in {"unknown", "transition"}:
        for state in regimes.values():
            state["counts"] = {}
        result["status"] = "data_ineligible" if not qualification["evaluation_eligible"] else "regime_" + context["status"]
        return result, False
    regime_id = context["regime_id"]
    changed = result.get("active_regime") not in {None, regime_id}
    if changed:
        for state in regimes.values():
            state["counts"] = {}
        result.update(alert_id=None, recovery_count=0)
    result["active_regime"] = regime_id
    state = regimes.setdefault(regime_id, {"calibration": [], "counts": {}})
    values = {"overall": {m: evidence[m] for m in ("normalized_mae", "normalized_mse")}}
    for variable in config.release.critical_variables:
        i = config.variables.index(variable)
        values["variable:" + variable] = {m: evidence[m + "_by_variable"][i] for m in values["overall"]}
    for group in config.release.business_groups:
        values["group:" + group] = {m: evidence["business_groups"][group][m] for m in values["overall"]}
    if "reference" not in state:
        state["calibration"] = (state["calibration"] + [{"window_start": evidence.get("window_start"), "values": values}])[-config.monitoring.reference_windows:]
        result["status"] = "new_regime_collecting_labels" if changed else "reference_calibrating"
        if len(state["calibration"]) < config.monitoring.reference_windows:
            return result, False
        reference, stable = {}, True
        for scope, metrics in values.items():
            reference[scope] = {}
            for metric in metrics:
                samples = [v["values"][scope][metric] for v in state["calibration"]]
                floor = SCALE_FLOOR if metric == "normalized_mae" else SCALE_FLOOR ** 2
                center, spread = float(np.median(samples)), float(np.ptp(samples))
                stable &= spread <= max(center, floor) * config.monitoring.reference_max_relative_spread
                reference[scope][metric] = {"value": center, "minimum_delta": max(spread, floor)}
        if stable:
            state.update(reference=reference, reference_id=digest({"model": evidence.get("model_id"), "regime": regime_id, "windows": state["calibration"]}),
                         source_windows=[v["window_start"] for v in state["calibration"]])
            result["status"] = "reference_established"
        return result, False
    result["status"] = "known_regime_return" if changed else "monitoring"
    for scope, metrics in values.items():
        for metric, current in metrics.items():
            reference = state["reference"][scope][metric]
            floor = SCALE_FLOOR if metric == "normalized_mae" else SCALE_FLOOR ** 2
            ratio, delta = current / max(reference["value"], floor), current - reference["value"]
            threshold = config.monitoring.mae_ratio_trigger if metric == "normalized_mae" else config.monitoring.mse_ratio_trigger
            key = scope + ":" + metric
            count = state["counts"].get(key, 0) + 1 if ratio > threshold and delta > reference["minimum_delta"] else 0
            state["counts"][key] = count
            result["assessments"].append({"metric": metric, "scope": scope, "current": current,
                "reference": reference, "reference_id": state["reference_id"], "ratio": ratio, "delta": delta,
                "threshold": threshold, "count": count, "triggered": count >= config.monitoring.consecutive_checks})
    result["count"] = max(state["counts"].values(), default=0)
    result["triggered"] = any(a["triggered"] for a in result["assessments"])
    if result["triggered"]:
        result["alert_id"] = result.get("alert_id") or digest({"reference": state["reference_id"],
            "first_trigger_window": evidence.get("window_start"), "regime": regime_id})[:24]
        result["recovery_count"] = 0
    elif all(a["count"] == 0 for a in result["assessments"]):
        result["recovery_count"] = result.get("recovery_count", 0) + 1
        if result["recovery_count"] >= config.monitoring.consecutive_checks:
            result["alert_id"] = None
    baselines = evidence.get("same_window_baselines") or {}
    if baselines.get("eligible"):
        best = min(v["normalized_mae"] for v in baselines["reports"].values())
        weak = evidence["normalized_mae"] > max(best, SCALE_FLOOR) * (1 + config.release.minimum_improvement)
        state["baseline_weak_count"] = state.get("baseline_weak_count", 0) + 1 if weak else 0
    else:
        state["baseline_weak_count"] = 0
    result["optimization_opportunity"] = state["baseline_weak_count"] >= config.monitoring.consecutive_checks
    return result, result["triggered"]
