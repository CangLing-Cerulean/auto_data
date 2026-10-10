"""Versioned, data-minimizing context shared by initial planning and feedback."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from importlib.resources import files

import numpy as np

from .config import GateError, canonical, digest


PROFILE_VERSION = "training-prefix-profile-v1"
CONTEXT_VERSION = "planner-context-v1"
MAX_PROFILE_VARIABLES = 16
MAX_ACF_LAG = 64
POLICY = (
    "Only authorized historical inputs and development evidence are available. "
    "Propose experiments; never change data, budgets, Gate, final Test or deployment. "
    "Program facts are authoritative. Correlation is not causation."
)


def _number(value):
    return round(float(value), 6) if np.isfinite(value) else None


def training_profile(raw, config) -> dict:
    """Only the first fold's training prefix contributes values to this profile."""
    from .training import split_plan, training_as_of
    end = split_plan(len(raw), config)["folds"][0]["train_end"]
    prefix, _ = training_as_of(raw, 0, end, config)
    values = prefix.to_numpy(dtype=float)
    columns = []
    for i in range(min(len(config.variables), MAX_PROFILE_VARIABLES)):
        y = values[:, i]
        valid = np.isfinite(y)
        positions = np.flatnonzero(valid)
        run = longest = 0
        for absent in ~valid:
            run = run + 1 if absent else 0
            longest = max(run, longest)
        item = {"field": f"target:{i}", "n": int(valid.sum()),
                "missing_fraction": _number((~valid).mean()), "longest_gap": longest,
                "constant": bool(valid.sum() < 2 or np.ptp(y[valid]) == 0),
                "normalized_trend": None, "acf_peaks": []}
        if valid.sum() >= 8 and not item["constant"]:
            scale = float(np.std(y[valid]))
            x = positions.astype(float) - positions.mean()
            slope = float(np.dot(x, y[valid] - y[valid].mean()) / np.dot(x, x))
            item["normalized_trend"] = _number(slope * (len(y) - 1) / scale)
            # Pairwise observed samples, not imputed labels. Bound the work and
            # disclose each pair count; small peaks are not a seasonality claim.
            correlations = []
            for lag in range(1, min(MAX_ACF_LAG, len(y) // 4) + 1):
                paired = valid[lag:] & valid[:-lag]
                left, right = y[:-lag][paired], y[lag:][paired]
                if len(left) >= 8 and np.std(left) > 0 and np.std(right) > 0:
                    correlations.append({"lag": lag, "r": _number(np.corrcoef(left, right)[0, 1]),
                                         "n": len(left)})
            peaks = [v for j, v in enumerate(correlations)
                     if (j == 0 or v["r"] >= correlations[j - 1]["r"])
                     and (j == len(correlations) - 1 or v["r"] >= correlations[j + 1]["r"])]
            item["acf_peaks"] = sorted(peaks, key=lambda v: (-v["r"], v["lag"]))[:3]
        columns.append(item)
    relations = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            left, right = values[valid, i], values[valid, j]
            if len(left) >= 8 and np.std(left) > 0 and np.std(right) > 0:
                relations.append({"fields": [f"target:{i}", f"target:{j}"],
                                  "r": _number(np.corrcoef(left, right)[0, 1]), "n": len(left)})
    # Group names can reveal field/site identities; the membership is sufficient.
    groups = [[f"target:{config.variables.index(v)}" for v in members]
              for _, members in sorted(config.release.business_groups.items())]
    return {"version": PROFILE_VERSION, "scope": "first_fold_training",
            "interval_ns": [int(prefix.index[0]), int(prefix.index[-1])], "rows": len(prefix),
            "variable_count": len(config.variables), "profiled_variables": len(columns),
            "frequency_seconds": config.frequency_seconds, "horizon_steps": config.horizon_steps,
            "method": "observed_pairs_pearson;ols_slope_times_span_over_std;no_imputed_statistics",
            "availability": "retrospective_mature_history;historical_inputs_as_of_issue",
            "columns": columns, "relations": sorted(relations, key=lambda v: -abs(v["r"]))[:5],
            "groups": groups, "missing_fraction": _number((~np.isfinite(values)).mean())}


def approved_profile(value) -> dict:
    """An explicit projection, not permission to forward arbitrary summary keys."""
    if isinstance(value, dict) and value.get("version") == "panel-training-v1":
        if (set(value) != {"version", "scope", "rows", "variable_count", "entity_count",
                           "missing_fraction", "columns"} or value["scope"] != "agent_training"
                or type(value["rows"]) is not int or value["rows"] < 1
                or type(value["variable_count"]) is not int or value["variable_count"] < 1
                or type(value["entity_count"]) is not int or value["entity_count"] < 1
                or type(value["missing_fraction"]) not in {int, float}
                or not 0 <= value["missing_fraction"] <= 1
                or not isinstance(value["columns"], list) or len(value["columns"]) > 32):
            raise GateError("PLANNER_PROFILE_SCOPE")
        for column in value["columns"]:
            if (not isinstance(column, dict) or set(column) != {"field", "constant", "lag1_correlation"}
                    or not isinstance(column["field"], str)
                    or not re.fullmatch(r"target:\d+", column["field"])
                    or int(column["field"][7:]) >= value["variable_count"]
                    or type(column["constant"]) is not bool
                    or (column["lag1_correlation"] is not None and
                        (type(column["lag1_correlation"]) not in {int, float}
                         or not -1 <= column["lag1_correlation"] <= 1))):
                raise GateError("PLANNER_PROFILE_FIELD")
        return deepcopy(value)
    if not isinstance(value, dict) or value.get("version") != PROFILE_VERSION:
        return {"version": PROFILE_VERSION, "scope": "unavailable", "reason": "NO_TRAINING_PREFIX_PROFILE"}
    keys = {"version", "scope", "interval_ns", "rows", "variable_count", "profiled_variables",
            "frequency_seconds", "horizon_steps", "method", "availability", "columns",
            "relations", "groups", "missing_fraction"}
    if set(value) != keys or value["scope"] != "first_fold_training":
        raise GateError("PLANNER_PROFILE_SCOPE")
    if (value["method"] != "observed_pairs_pearson;ols_slope_times_span_over_std;no_imputed_statistics"
            or value["availability"] != "retrospective_mature_history;historical_inputs_as_of_issue"):
        raise GateError("PLANNER_PROFILE_METHOD")
    def check(item, key=""):
        if isinstance(item, dict):
            for k, v in item.items():
                check(v, k)
        elif isinstance(item, list):
            for v in item:
                check(v, key)
        elif isinstance(item, str):
            if key not in {"version", "scope", "method", "availability"} and not re.fullmatch(r"target:\d+", item):
                raise GateError("PLANNER_PROFILE_FIELD")
        elif item is not None and (type(item) not in {int, float, bool} or
                                   (type(item) is float and not math.isfinite(item))):
            raise GateError("PLANNER_PROFILE_VALUE")
    check(value)
    for column in value["columns"]:
        if set(column) != {"field", "n", "missing_fraction", "longest_gap", "constant", "normalized_trend", "acf_peaks"}:
            raise GateError("PLANNER_PROFILE_FIELD")
        if any(set(p) != {"lag", "r", "n"} for p in column["acf_peaks"]):
            raise GateError("PLANNER_PROFILE_FIELD")
    if any(set(r) != {"fields", "r", "n"} for r in value["relations"]):
        raise GateError("PLANNER_PROFILE_FIELD")
    return deepcopy(value)


def knowledge_cards(config, *, issues=(), candidate=None) -> list[dict]:
    cards = json.loads(files("online_ts").joinpath("prompts/knowledge_cards_v1.json").read_text(encoding="utf-8"))
    identifiers = []
    codes = " ".join(str(i.get("code", "")) for i in issues)
    if any(term in codes for term in ("MEMORY", "OOM", "VRAM")):
        identifiers.append("batch")
    if "LABEL" in codes:
        identifiers.append("labels")
    if candidate and candidate.get("family") == "chronos2":
        identifiers.append("adaptation")
    else:
        identifiers.append("ridge_alpha")
    identifiers += ["windows", "correlation", "adaptation", "batch", "labels"]
    return [cards[name] for name in dict.fromkeys(identifiers)][:config.planner.knowledge_limit]


def build_context(config, *, profile=None, history=(), issues=(), candidate=None) -> dict:
    """The stable shared layers; callers add a bounded phase-specific protocol."""
    from .memory import outbound_observation
    return {"context_version": CONTEXT_VERSION, "policy": POLICY,
            "knowledge": knowledge_cards(config, issues=issues, candidate=candidate),
            "task_profile": approved_profile(profile),
            "trial_history": [outbound_observation(v, f"m{i}")
                              for i, v in enumerate(list(history)[:config.planner.history_limit])]}


def parameter_facts(before: dict, after: dict) -> dict:
    result = {}
    for name in ("alpha", "input_steps", "history_steps", "batch_size", "num_steps", "lora_rank"):
        a, b = before.get(name), after.get(name)
        if type(a) in {int, float} and type(b) in {int, float} and a != b:
            result[name] = {"from": a, "to": b, "delta": b - a,
                            "direction": "increase" if b > a else "decrease"}
    if "alpha" in result and after.get("family") == before.get("family") == "ridge":
        result["regularization_strength"] = result["alpha"]["direction"]
    return result


def validate_parameter_language(text: str, before: dict, after: dict) -> dict:
    """Reject explicit known contradictions, not claim general language verification."""
    facts = parameter_facts(before, after)
    direction = facts.get("regularization_strength")
    if direction:
        opposite = (r"(?:reduc\w*|decreas\w*|lower\w*|weaken\w*)\s+(?:the\s+)?regulari[sz]\w*|(?:降低|减小|减弱)正则"
                    if direction == "increase" else
                    r"(?:increas\w*|rais\w*|stronger|strengthen\w*)\s+(?:the\s+)?regulari[sz]\w*|(?:提高|增加|增强)正则")
        if re.search(opposite, text, re.I):
            raise GateError("EXPERIMENT_PARAMETER_DIRECTION_CONTRADICTION")
    return facts


def fit_messages(prompt: str, observation: dict, available_tokens: int, output_tokens: int):
    """Reserve UTF-8 bytes plus framing; this is an estimate, never tokenizer usage."""
    value = deepcopy(observation)
    omitted = []
    while True:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": canonical(value)}]
        reserve = len(canonical(messages).encode("utf-8")) + 256
        if reserve + output_tokens <= available_tokens:
            return messages, reserve, output_tokens, omitted
        for key in ("examples", "trial_history", "knowledge"):
            if value.get(key):
                value[key].pop()
                omitted.append(key)
                break
        else:
            raise GateError("PLANNER_PROMPT_BUDGET")


def training_seconds_spent(job):
    """Slow LLM wall time is measured separately from the training allowance."""
    planning = job["checkpoint"].get("seconds_by_role", {}).get("planning_and_repair", 0.0)
    reserved = 0.0
    if job.get("status") == "RUNNING" and (job["plan"].get("needs_proposal") or job["checkpoint"].get("feedback_pending")):
        reserved = job.get("reservation", 0.0)
    return max(0.0, float(job.get("spent", 0.0)) - planning - reserved)
