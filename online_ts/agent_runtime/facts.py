"""Program-owned task, metric and experiment facts shared by both agent gateways."""
from dataclasses import asdict
import math

import numpy as np

from ..config import GateError, canonical, digest
from ..models import Candidate, SCALE_FLOOR


OBJECTIVE = {"name": "normalized_mae", "direction": "minimize", "tie_tolerance": 1e-12,
             "selection_scope": "measured_comparable_admissible", "multi_objective": False}
OBJECTIVE_VERSION = digest(OBJECTIVE)[:16]


def selection_eligible(row):
    return row["eligible"] and (row["protocol"] != "rolling-origin-v1"
        or bool((row.get("development_admission") or {}).get("worthy_of_final")))


def effective_candidate(value, config):
    c = Candidate.from_dict(value) if isinstance(value, dict) else value
    result = {"family": c.family, "input_observations": c.input_steps,
              "history_observations": c.history_steps}
    if c.family == "ridge":
        result.update(alpha=c.alpha, features=list(c.features), training="supervised_ridge",
                      channel_interaction="flattened_multivariate_linear_inputs")
    elif c.family == "seasonal_naive":
        result.update(period=c.period, period_unit="observations", training="no_weight_optimization",
                      input_effect="last_period_only")
    elif c.family == "last_value":
        result.update(training="no_weight_optimization", input_effect="last_observation_only")
    elif c.family == "patchtst":
        result.update(training="supervised_random_initialization", pretrained=False,
                      requested_optimizer_steps=c.num_steps, batch_size=c.batch_size, learning_rate=c.learning_rate,
                      architecture=asdict(config.patchtst), channel_interaction="not_established_by_this_adapter")
    elif c.family == "chronos2":
        result.update(model_id=c.model_id, revision=c.revision, training=c.adaptation_mode,
                      quantile_levels=c.quantile_levels)
        if c.adaptation_mode == "lora":
            result.update(requested_optimizer_steps=c.num_steps, batch_size=c.batch_size,
                          learning_rate=c.learning_rate, lora_rank=c.lora_rank)
    return result


def configuration_delta(left, right):
    if left is None:
        return {"changed_factors": None, "single_factor_contrast": None, "reason": "no_declared_parent"}
    changed = [k for k in sorted(set(left) | set(right)) if left.get(k) != right.get(k)]
    return {"changed_factors": {k: {"from": left.get(k), "to": right.get(k)} for k in changed},
            "single_factor_contrast": len(changed) == 1,
            "interpretation": "configuration_difference_only; causal mechanism not established"}


def metric_spec(config, *, protocol, scale=None):
    return {"version": "metric-spec-v1", **OBJECTIVE, "objective_version": OBJECTIVE_VERSION,
            "protocol": protocol, "formula": "sum_j(weight_j * mean_windows,horizon(abs(pred-truth)) / scale_j)",
            "fold_aggregation": "mean_of_fold_objectives" if protocol == "rolling-origin-v1" else "single_fixed_validation_panel",
            "normalization_source": "first_fold_training_population_std" if protocol == "rolling-origin-v1" else "all_finite_agent_training_population_std",
            "scale_floor": SCALE_FLOOR, "normalization_scales": scale,
            "variable_weights": list(config.variable_weights), "normalized_unit": "dimensionless",
            "missing_policy": "finite_complete_evaluation_windows; do_not_impute_truth",
            "comparison_requires": ["task", "snapshot", "protocol", "evaluation_origins", "scale", "weights"]}


def variable_metrics(raw_mae, scale, config, report_ref, units=None):
    raw, scales = np.asarray(raw_mae, dtype=float), np.asarray(scale, dtype=float)
    if (raw.shape != (len(config.variables),) or scales.shape != raw.shape
            or not np.isfinite(raw).all() or not np.isfinite(scales).all()
            or np.any(scales < SCALE_FLOOR) or np.any(raw < 0)):
        raise GateError("AGENT_METRIC_EVIDENCE")
    units = units or {}
    return [{"variable_id": f"target:{i}", "name": name, "index_zero_based": i, "unit": units.get(name),
             "raw_mae": float(raw[i]), "normalization_scale": float(scales[i]),
             "normalized_mae": float(raw[i] / scales[i]), "aggregation_weight": config.variable_weights[i],
             "contribution_to_objective": float(raw[i] / scales[i] * config.variable_weights[i]),
             "evidence_ref": report_ref} for i, name in enumerate(config.variables)]


def task_card(config, *, protocol="rolling-origin-v1", panel=None, metadata=None):
    metadata = metadata or {}
    card = {"version": "task-card-v1", "task_type": "multivariate_future_observation_forecast",
            "domain": metadata.get("domain", "unknown"), "source_evidence": metadata.get("source_evidence", []),
            "target": "future configured sensor variables", "non_targets": metadata.get("non_targets", ["RUL", "anomaly_classification"]),
            "variables": [{"variable_id": f"target:{i}", "name": name,
                           "unit": metadata.get("units", {}).get(name)} for i, name in enumerate(config.variables)],
            "horizon": {"value": config.horizon_steps, "unit": "observations"},
            "input_L": list(config.search.inputs), "training_history_D": list(config.search.histories),
            "evaluation_protocol": protocol, "input_columns": list(config.variables),
            "maximum_train_windows": config.search.max_train_samples,
            "actual_train_windows": None, "actual_evaluation_windows": None,
            "unmeasured_fields": "unknown_not_inferred", "objective_version": OBJECTIVE_VERSION}
    if panel:
        train, val = panel["provenance"]["agent_training"], panel["provenance"]["validation"]
        card.update(snapshot=panel["fingerprint"], entity_training="shared_parameters_across_entity_safe_windows",
                    entity_count=train["entity_count"], raw_train_rows=train["source_rows"], raw_validation_rows=val["source_rows"],
                    legal_train_windows=train["possible_windows"], legal_validation_windows=val["possible_windows"],
                    sampled_train_windows=train["sampled_windows"], sampled_validation_windows=val["sampled_windows"],
                    uses={"agent_training": "fit_and_normalize", "validation": "development_selection", "arena_reference": "inaccessible"},
                    time_interpretation="next observations within entity/segment; not necessarily seconds or calendar periods")
    else:
        card.update(entity_training="single_entity", entity_id=config.contract.entity_id,
                    frequency_seconds=config.frequency_seconds,
                    horizon_seconds=config.horizon_steps * config.frequency_seconds,
                    uses={"train": "fit", "development_validation": "selection", "final_test": "sealed_admission_only"})
    card["card_version"] = digest(card)[:16]
    return card


def score_delta(current, reference):
    if reference is None:
        return {"delta": None, "improved": None, "relative_improvement": None, "reason": "reference_unavailable"}
    if current["comparison_group"] != reference["comparison_group"]:
        return {"delta": None, "improved": None, "relative_improvement": None, "reason": "incomparable_protocol_or_snapshot"}
    a, b = current.get("score"), reference.get("score")
    if a is None or b is None:
        return {"delta": None, "improved": None, "relative_improvement": None, "reason": "metric_unavailable"}
    return {"delta": a - b, "improved": a < b - OBJECTIVE["tie_tolerance"],
            "relative_improvement": (b - a) / abs(b) if b != 0 else None, "reference_result_id": reference["result_id"]}


class ExperimentLedger:
    def __init__(self, repo):
        self.repo = repo

    def rows(self, now):
        return sorted([r for r in list(self.repo.iter_records("experiment_fact"))
                       if r["visible_at"] <= now and r["epoch"] == self.repo.binding()["epoch"]],
                      key=lambda r: (r["recorded_at"], r["result_id"]))

    def record(self, report, config, artifacts, *, now, experiment_id=None, parent_result_id=None, protocol="rolling-origin-v1"):
        reference = report.get("validation_artifact")
        if not reference:
            raise GateError("AGENT_REPORT_REFERENCE_REQUIRED")
        stored = artifacts.read_json(reference)
        if canonical(stored) != canonical({k: v for k, v in report.items() if k != "validation_artifact"}):
            raise GateError("VALIDATION_ARTIFACT_MISMATCH")
        candidate = Candidate.from_dict(report["candidate"])
        candidate.validate(config)
        if candidate.candidate_id != report["candidate_id"]:
            raise GateError("AGENT_CANDIDATE_BINDING")
        score = report.get("score")
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise GateError("AGENT_METRIC_EVIDENCE")
        rid = "result-" + digest({"task": self.repo.key, "epoch": self.repo.binding()["epoch"], "report": reference})[:20]
        existing = next((r for r in self.rows(now) if r["result_id"] == rid), None)
        if existing:
            return existing
        if protocol == "rolling-origin-v1":
            from ..training import choose
            if report.get("eligible"):
                choose([report], config, artifacts=artifacts, snapshot_sha=report["snapshot_sha"])
            folds = report["folds"]
            scale = report["normalization"]["scale"]
            raw = np.mean([f["mae_by_variable"] for f in folds], axis=0)
            origin_id = digest([f["issue_times"] for f in folds])
            windows = {"training_by_fold": [f.get("training_samples") for f in folds],
                       "evaluation_by_fold": [f["origins"] for f in folds]}
            baseline_scores = {name: float(np.mean([f["baselines"][name]["normalized_mae"] for f in folds]))
                               for name in ("last_value", "seasonal_naive")}
        else:
            raw, scale = report["metrics"]["mae_by_variable"], report["metrics"]["scale"]
            origin_id = report["validation_origin_sha256"]
            windows = {"training": report.get("actual_training_windows"), "evaluation": report["metrics"]["origins"]}
            baseline_scores = {name: value["normalized_mae"] for name, value in report["baselines"].items()}
        contributions = variable_metrics(raw, scale, config, reference)
        if not math.isclose(sum(v["contribution_to_objective"] for v in contributions), score, rel_tol=1e-10, abs_tol=1e-12):
            raise GateError("AGENT_METRIC_AGGREGATION_MISMATCH")
        group = digest({"task": self.repo.key, "snapshot": report["snapshot_sha"], "protocol": protocol,
                        "origins": origin_id, "scale": scale, "weights": config.variable_weights})[:20]
        before = self.rows(now)
        comparable = [r for r in before if r["comparison_group"] == group]
        best_before = min(comparable, key=lambda r: (r["score"], r["candidate_id"]), default=None)
        parent = next((r for r in before if r["result_id"] == parent_result_id), None)
        if parent_result_id is not None and parent is None:
            raise GateError("AGENT_PARENT_RESULT_SCOPE")
        row = {"record_id": rid, "result_id": rid, "task_key": self.repo.key, "epoch": self.repo.binding()["epoch"],
               "candidate_id": candidate.candidate_id, "proposed_spec": asdict(candidate),
               "compiled_config": asdict(candidate), "effective_config": effective_candidate(candidate, config),
               "actual_execution": report.get("execution"), "sample_counts": windows,
               "development_admission": report.get("development_admission"),
               "training_progress": [{k: f.get("adaptation_evidence", {}).get(k) for k in
                   ("optimizer_steps", "requested_steps", "best_step", "best_internal_validation_loss",
                    "termination_reason", "checkpoint_selection", "convergence_claimed")}
                    for f in report.get("folds", []) if f.get("adaptation_evidence")],
               "score": score, "eligible": report["eligible"], "gate_issues": report.get("issues", []),
               "comparison_group": group, "protocol": protocol, "snapshot_sha": report["snapshot_sha"],
               "report_ref": reference, "model_ref": report.get("model"), "experiment_id": experiment_id,
               "parent_result_id": parent_result_id, "variable_metrics": contributions,
               "baseline_results": [{"name": name, "score": value, "auxiliary": True, "model_ref": None,
                                     "deliverable": False} for name, value in baseline_scores.items()],
               "visible_at": now, "recorded_at": now, "evidence_scope": "development_validation"}
        row.update(best_before_result_id=best_before["result_id"] if best_before else None,
                   versus_parent=score_delta(row, parent), versus_best_before=score_delta(row, best_before),
                   configuration_delta=configuration_delta(parent["effective_config"] if parent else None, row["effective_config"]))
        after = min([*comparable, row], key=lambda r: (r["score"], r["candidate_id"]))
        row["best_after_result_id"] = after["result_id"]
        row["delta_vs_baselines"] = {name: score - value for name, value in baseline_scores.items()}
        self.repo.put(rid, "experiment_fact", row, now)
        return row

    def view(self, now):
        rows = self.rows(now)
        groups = {}
        for row in rows:
            groups.setdefault(row["comparison_group"], []).append(row)
        best = {g: min(rs, key=lambda r: (r["score"], r["candidate_id"]))["result_id"] for g, rs in groups.items()}
        eligible = {g: min([r for r in rs if selection_eligible(r)], key=lambda r: (r["score"], r["candidate_id"]), default={}).get("result_id")
                    for g, rs in groups.items()}
        config, _ = self.repo.store.task(self.repo.key)
        progress = {}
        for group, measured in groups.items():
            seen, best_score, stagnant = set(), None, 0
            for row in measured:
                if not row["eligible"] or row["candidate_id"] in seen:
                    continue
                seen.add(row["candidate_id"])
                if best_score is not None:
                    gain = (best_score - row["score"]) / max(abs(best_score), SCALE_FLOOR)
                    stagnant = 0 if gain >= config.budget.minimum_candidate_improvement else stagnant + 1
                best_score = min(best_score, row["score"]) if best_score is not None else row["score"]
            ready = any((r.get("development_admission") or {}).get("worthy_of_final") for r in measured)
            progress[group] = {"distinct_valid_candidates": len(seen), "consecutive_without_material_gain": stagnant,
                "minimum_gain": config.budget.minimum_candidate_improvement,
                "stagnation_limit": config.budget.max_no_improvement, "worthy_of_final": ready,
                "recommendation": "seal_best_qualified_candidate" if ready else "retain_current_wait_for_new_evidence"
                    if stagnant >= config.budget.max_no_improvement else "investigate_distinct_hypothesis",
                "continued_search_requires": "a distinct testable hypothesis with new evidence; repeated configurations are not progress"}
        return {"version": "experiment-ledger-v1", "rows": rows, "count": len(rows),
                "latest_result": rows[-1]["result_id"] if rows else None,
                "best_observed_by_objective": best, "best_eligible_candidate": eligible,
                "search_progress": progress,
                "agent_proposed_selection": None, "accepted_selection": None,
                "current_champion": self.repo.binding()["champion"]}
