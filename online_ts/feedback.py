"""Deterministic policy around LLM feedback; this module never calls an LLM."""
from __future__ import annotations

from dataclasses import asdict

from .config import GateError, TaskConfig
from .models import Candidate
from .planner import candidate_pool, proposal_fingerprint
from .remediation import AgentContext, FeedbackProposal, GateIssue, Outcome, ProgressDigest


EVALUATION_PROTOCOL = "rolling-origin-v1"
FORBIDDEN_ACTIONS = (
    "change_budget", "change_gate", "change_evaluation_protocol", "change_final_test",
    "use_unapproved_data", "execute_code", "deploy", "change_champion",
)


def _catalog(config: TaskConfig) -> list[dict]:
    return [{"candidate_ref": f"c{index}", "candidate": _compact_candidate(candidate)}
            for index, candidate in enumerate(candidate_pool(config))]


def _compact_candidate(candidate):
    value = asdict(candidate)
    defaults = asdict(Candidate(candidate.family, candidate.input_steps, candidate.history_steps))
    return {key: item for key, item in value.items()
            if key in {"family", "input_steps", "history_steps"} or item != defaults[key]}


def validation_outcome(job: dict, report: dict, config: TaskConfig, *, attempt: int, created_at: int) -> Outcome:
    candidate = Candidate.from_dict(report["candidate"])
    candidate.validate(config)
    metrics = {name: report[name] for name in ("score", "score_std") if name in report}
    reference = report.get("validation_artifact", {}).get("sha256")
    return Outcome.create(
        run_id=job["run_id"], candidate_id=candidate.candidate_id,
        attempt_id=f"{job['run_id']}:validation:{candidate.candidate_id}:{attempt}",
        plan_id=job["checkpoint"]["plan_id"], plan_version=job["checkpoint"]["plan_version"],
        created_at=created_at, node="validation", status="passed" if report.get("eligible") else "repairable",
        effective_config=asdict(candidate), artifact_refs=(["artifact:" + reference] if reference else ()),
        resource_usage=dict(report.get("execution") or {}), metrics=metrics or None,
        issue_codes=tuple(report.get("issues", ())), snapshot_sha=report["snapshot_sha"],
        evaluation_protocol=EVALUATION_PROTOCOL, evidence_scope="development_validation",
        gate_passed=bool(report.get("eligible")),
    )


def failure_outcome(job: dict, config: TaskConfig, *, code: str, stage: str,
                    attempt: int, created_at: int, retryable: bool) -> Outcome:
    candidate = None
    if stage.isdigit() and int(stage) < len(job["plan"].get("candidates", [])):
        candidate = Candidate.from_dict(job["plan"]["candidates"][int(stage)])
    status = "retryable_error" if retryable else "repairable"
    return Outcome.create(
        run_id=job["run_id"], candidate_id=candidate.candidate_id if candidate else None,
        attempt_id=f"{job['run_id']}:{stage}:{attempt}", plan_id=job["checkpoint"]["plan_id"],
        plan_version=job["checkpoint"]["plan_version"], created_at=created_at,
        node="training" if stage.isdigit() else stage, status=status,
        effective_config=asdict(candidate) if candidate else {}, artifact_refs=(), log_refs=(),
        resource_usage={}, metrics=None, issue_codes=(code,),
        snapshot_sha=job["snapshot"].get("sha256", "unknown"),
        evaluation_protocol=EVALUATION_PROTOCOL, evidence_scope="resource" if "RESOURCE" in code or "VRAM" in code else "execution",
        gate_passed=False,
    )


def gate_issue_for_outcome(outcome: Outcome, *, code: str, observed_value=None,
                           threshold=None, direction=None, repairable: bool = True,
                           severity: str = "error", waiting_reason=None,
                           human_reason=None) -> GateIssue:
    evidence = (outcome.message_id,) + tuple(outcome.artifact_refs)
    return GateIssue.create(
        code=code, stage=outcome.node, severity=severity, candidate_id=outcome.candidate_id,
        affected={"candidate_id": outcome.candidate_id, "evaluation_protocol": outcome.evaluation_protocol},
        observed_value=observed_value, threshold=threshold, direction=direction,
        evidence_refs=evidence, auto_repairable=repairable,
        allowed_changes=("candidate_ref", "experiment_spec", "training_target") if repairable else (),
        waiting_reason=waiting_reason, human_reason=human_reason,
    )


def compare_outcomes(parent: Outcome | None, current: Outcome) -> dict:
    if parent is None:
        return {"comparable": False, "reason": "NO_PARENT_RESULT", "metric_deltas": {},
                "quality_improved": None}
    execution_repaired = (parent.candidate_id == current.candidate_id
                          and parent.snapshot_sha == current.snapshot_sha
                          and parent.evidence_scope in {"execution", "resource"}
                          and current.evidence_scope == "development_validation"
                          and current.metrics is not None)
    comparable = (parent.snapshot_sha == current.snapshot_sha
                  and parent.evaluation_protocol == current.evaluation_protocol
                  and parent.evidence_scope == current.evidence_scope
                  and parent.metrics is not None and current.metrics is not None)
    if not comparable:
        return {"comparable": False, "reason": "EVALUATION_SCOPE_MISMATCH",
                "metric_deltas": {}, "quality_improved": None, "execution_repaired": execution_repaired}
    shared = sorted(set(parent.metrics) & set(current.metrics))
    deltas = {name: current.metrics[name] - parent.metrics[name] for name in shared}
    improved = "score" in deltas and deltas["score"] < -1e-12 and current.gate_passed is True
    return {"comparable": True, "reason": None, "parent_outcome_id": parent.message_id,
            "metric_deltas": deltas, "quality_improved": improved}


def build_agent_context(config: TaskConfig, job: dict, outcome: Outcome, issues: list[GateIssue],
                        *, comparison: dict, available_targets: set[str], trial_history=None) -> AgentContext:
    cp = job["checkpoint"]
    from .experiments import capabilities
    development = [item for item in cp.get("outcomes", [])
                   if item.get("evidence_scope") == "development_validation"
                   and item.get("snapshot_sha") == outcome.snapshot_sha
                   and item.get("evaluation_protocol") == outcome.evaluation_protocol]
    parent = next((item for item in reversed(development)
                   if item["message_id"] != outcome.message_id), None)
    report = next((item for item in reversed(cp.get("reports", []))
                   if item.get("candidate_id") == outcome.candidate_id
                   and item.get("snapshot_sha") == outcome.snapshot_sha), {})
    folds = report.get("folds", [])
    baselines = {name: {"normalized_mae_by_fold": [fold["baselines"][name]["normalized_mae"]
                                                 for fold in folds],
                        "same_origins_and_scale": True}
                 for name in ("last_value", "seasonal_naive")
                 if folds and all(name in fold.get("baselines", {}) for fold in folds)}
    if not folds and report.get("evaluation_protocol") == "entity-panel-validation-v1":
        baselines = {name: {"normalized_mae": item["normalized_mae"], "same_origins_and_scale": True}
                     for name, item in report.get("baselines", {}).items()
                     if name in {"last_value", "seasonal_naive"}}
    evidence_refs = tuple(dict.fromkeys(
        [outcome.message_id, *outcome.artifact_refs,
         *([parent["message_id"]] if parent else []),
         *(ref for issue in issues for ref in issue.evidence_refs)]))
    catalog = _catalog(config)
    tried = [report.get("candidate_id") for report in cp.get("reports", []) if report.get("candidate_id")]
    history = [{"outcome_id": item["message_id"], "candidate_id": item.get("candidate_id"),
                "status": item["status"], "metrics": item.get("metrics"),
                "issue_codes": item.get("issue_codes", [])}
               for item in cp.get("outcomes", []) if item.get("evidence_scope") != "final_test"][-3:]
    completed_attempts = int(cp.get("index", 0))
    from .context import training_seconds_spent
    remaining = {
        "repair_rounds": max(0, config.budget.max_repairs - cp.get("repair_rounds", 0)),
        "candidate_slots": max(0, config.budget.max_candidates - completed_attempts),
        "training_seconds": max(0.0, config.budget.max_training_seconds - training_seconds_spent(job)),
        "llm_tokens": max(0, config.budget.max_llm_tokens - cp.get("llm_tokens_used", 0)),
        "format_attempts": max(0, config.planner.max_format_repairs - cp.get("format_repairs", 0)),
    }
    return AgentContext.create(
        run_id=job["run_id"], plan_id=cp["plan_id"], plan_version=cp["plan_version"],
        created_at=job["cutoff"], current_goal="Resolve the listed issues using one bounded, testable plan change.",
        current_plan={"plan_id": cp["plan_id"], "plan_version": cp["plan_version"],
                      "snapshot_sha": job["snapshot"].get("sha256"),
                      "candidate_ids": tried,
                      "candidate_count": len(job["plan"].get("candidates", ())),
                      "completed_attempts": completed_attempts,
                      "actual_config": outcome.effective_config,
                      "executed_candidates": [{"candidate_id": Candidate.from_dict(c).candidate_id,
                                               "candidate": _compact_candidate(Candidate.from_dict(c))}
                                              for c in job["plan"].get("candidates", [])[:completed_attempts]],
                      "task_profile": cp.get("planner_profile"),
                      "trial_history": list(cp.get("trial_history", []) if trial_history is None else trial_history),
                      "evaluation_protocol": outcome.evaluation_protocol},
        outcome=outcome, gate_issues=issues,
        comparisons={**comparison, "best_so_far": cp.get("best_so_far"),
                     "parent": ({key: parent[key] for key in
                                 ("message_id", "candidate_id", "metrics", "gate_passed")}
                                if parent else None),
                     "baselines": baselines}, history=history,
        allowed_changes={"mutable_fields": ["candidate_ref", "experiment_spec", "training_target"],
                         "experiment_capabilities": capabilities(config),
                         "candidate_catalog": catalog,
                         "training_targets": sorted(available_targets)},
        remaining_budget=remaining,
        output_contract={"schema_version": 1,
                         "actions": ["experiment_patch", "request_evidence", "wait", "stop", "request_human"]},
        evidence_refs=evidence_refs, forbidden_actions=FORBIDDEN_ACTIONS,
    )


def apply_feedback_proposal(proposal_value: dict, context_value: dict, config: TaskConfig,
                            current_plan: dict, seen: set[str], available_targets: set[str]) -> tuple[dict | None, dict]:
    proposal = FeedbackProposal.from_dict(proposal_value)
    context = AgentContext.from_dict(context_value)
    if (proposal.run_id != context.run_id or proposal.base_plan_id != context.plan_id
            or proposal.base_plan_version != context.plan_version
            or proposal.feedback_id != context.feedback_id):
        raise GateError("STALE_FEEDBACK_PROPOSAL")
    issue_ids = {issue["issue_id"] for issue in context.gate_issues}
    if not set(proposal.issue_ids).issubset(issue_ids) or not set(proposal.evidence_refs).issubset(context.evidence_refs):
        raise GateError("FEEDBACK_PROPOSAL_EVIDENCE")
    if proposal.action != "experiment_patch":
        decision = {"action": proposal.action, "execute": False,
                    "terminal_reason": {"wait": "WAITING_FOR_EVIDENCE",
                                        "request_evidence": "EVIDENCE_REQUESTED",
                                        "request_human": "MANUAL_REVIEW_REQUIRED",
                                        "stop": "AGENT_REQUESTED_STOP"}[proposal.action]}
        return None, decision
    fields = set(proposal.proposed_changes)
    if fields not in ({"candidate_ref", "training_target"}, {"experiment_spec", "training_target"}):
        raise GateError("FEEDBACK_CHANGE_NOT_ALLOWED")
    target = proposal.proposed_changes["training_target"]
    if target not in config.execution.allowed_targets or target not in available_targets:
        raise GateError("TRAINING_TARGET_NOT_ALLOWED")
    catalog = {item["candidate_ref"]: item["candidate"]
               for item in context.allowed_changes["candidate_catalog"]}
    compiled = None
    if "experiment_spec" in proposal.proposed_changes:
        from .experiments import compile_experiment
        candidate, spec_target, compiled = compile_experiment(
            proposal.proposed_changes["experiment_spec"], config,
            evidence_refs=context.evidence_refs, available_targets=available_targets,
            parent_ids=[Candidate.from_dict(c).candidate_id for c in current_plan["candidates"]],
            feedback_id=context.feedback_id)
        if spec_target != target:
            raise GateError("EXPERIMENT_EXECUTION_MISMATCH")
    else:
        ref = proposal.proposed_changes["candidate_ref"]
        if not isinstance(ref, str) or ref not in catalog:
            raise GateError("CANDIDATE_NOT_OFFERED")
        candidate = Candidate.from_dict(catalog[ref])
    candidate.validate(config)
    from .context import validate_parameter_language
    before = context.outcome["effective_config"]
    if compiled is not None and compiled.get("parent_candidate_id"):
        before = next((c for c in current_plan["candidates"]
                       if Candidate.from_dict(c).candidate_id == compiled["parent_candidate_id"]), before)
    parameter_changes = validate_parameter_language(
        (proposal.hypothesis or "") + " " + proposal.diagnosis_summary, before, asdict(candidate))
    completed = int(context.current_plan.get("completed_attempts", 0))
    prefix = current_plan.get("candidates", [])[:completed]
    existing = [Candidate.from_dict(item).candidate_id for item in prefix]
    if candidate.candidate_id in existing:
        raise GateError("NO_PROGRESS")
    if completed >= config.budget.max_candidates:
        raise GateError("CANDIDATE_BUDGET")
    revised = {**current_plan, "candidates": [*prefix, asdict(candidate)],
               "training_target": target,
               "routing_reason": proposal.diagnosis_summary,
               "parent_plan_id": context.plan_id,
               "parent_plan_version": context.plan_version,
               "plan_version": context.plan_version + 1,
               "feedback_id": context.feedback_id,
               "proposal_id": proposal.message_id}
    fingerprint = proposal_fingerprint(revised)
    if fingerprint in seen:
        raise GateError("NO_PROGRESS")
    revised["proposal_fingerprint"] = fingerprint
    revised["program_parameter_changes"] = parameter_changes
    if compiled is not None:
        revised["experiment_history"] = [*current_plan.get("experiment_history", []), compiled]
    return revised, {"action": "experiment_patch", "execute": True,
                     "candidate_id": candidate.candidate_id, "proposal_fingerprint": fingerprint}


def progress_record(*, round_index: int, current_fingerprint: str | None,
                    previous_fingerprint: str | None, best_so_far: dict | None,
                    comparison: dict, issue_ids=(), resource_usage=None,
                    terminal_reason=None) -> dict:
    return ProgressDigest(
        round=round_index, semantic_change=current_fingerprint != previous_fingerprint,
        current_fingerprint=current_fingerprint, previous_fingerprint=previous_fingerprint,
        best_so_far=best_so_far, terminal_reason=terminal_reason,
        execution_progress=comparison.get("execution_repaired") is True, quality_improved=comparison.get("quality_improved"),
        comparable=comparison.get("comparable"), unresolved_issue_ids=tuple(issue_ids),
        metric_deltas=dict(comparison.get("metric_deltas", {})),
        actual_resource_usage=dict(resource_usage or {}),
    ).to_dict()
