"""Versioned messages for the evidence-driven, bounded repair workflow."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import GateError, digest


FEEDBACK_SCHEMA_VERSION = 1
OUTCOME_STATUSES = {
    "passed", "repairable", "retryable_error", "data_insufficient",
    "contract_changed", "no_qualified_candidate", "budget_exhausted",
    "no_progress", "approval_required", "rejected", "promoted",
    "rollback_required", "cancelled", "fatal", "evidence_insufficient",
}
PROPOSAL_ACTIONS = {"experiment_patch", "request_evidence", "wait", "stop", "request_human"}
SEVERITIES = {"info", "warning", "error", "critical"}
EVIDENCE_SCOPES = {"development_validation", "execution", "resource", "online_mature", "final_test"}


def _text(value: Any, code: str, *, maximum: int = 240, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or len(value) > maximum
            or any(ord(char) < 32 for char in value)):
        raise GateError(code)
    return value


def _refs(values: Any, code: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or len(values) > 32:
        raise GateError(code)
    result = tuple(_text(value, code, maximum=200) for value in values)
    if len(set(result)) != len(result):
        raise GateError(code)
    return result


def _mapping(value: Any, code: str) -> dict:
    if not isinstance(value, dict):
        raise GateError(code)
    return value


def _metrics(value: Any) -> dict | None:
    if value is None:
        return None
    data = _mapping(value, "OUTCOME_METRICS")
    for metric in data.values():
        if (isinstance(metric, bool) or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))):
            raise GateError("OUTCOME_METRICS")
    return {str(name): float(metric) for name, metric in data.items()}


@dataclass(frozen=True)
class Outcome:
    schema_version: int
    message_id: str
    run_id: str
    candidate_id: str | None
    attempt_id: str
    plan_id: str
    plan_version: int
    created_at: int
    node: str
    status: str
    effective_config: dict
    artifact_refs: tuple[str, ...]
    log_refs: tuple[str, ...]
    resource_usage: dict
    metrics: dict | None
    issue_codes: tuple[str, ...]
    snapshot_sha: str
    evaluation_protocol: str
    evidence_scope: str
    gate_passed: bool | None

    def __post_init__(self):
        if self.schema_version != FEEDBACK_SCHEMA_VERSION:
            raise GateError("FEEDBACK_SCHEMA_VERSION")
        for name in ("message_id", "run_id", "attempt_id", "plan_id", "node",
                     "snapshot_sha", "evaluation_protocol"):
            _text(getattr(self, name), "OUTCOME_IDENTITY", maximum=200)
        if self.candidate_id is not None:
            _text(self.candidate_id, "OUTCOME_IDENTITY", maximum=200)
        if (isinstance(self.plan_version, bool) or not isinstance(self.plan_version, int)
                or self.plan_version < 1 or isinstance(self.created_at, bool)
                or not isinstance(self.created_at, int) or self.created_at < 0):
            raise GateError("OUTCOME_IDENTITY")
        if self.status not in OUTCOME_STATUSES or self.evidence_scope not in EVIDENCE_SCOPES:
            raise GateError("OUTCOME_STATUS")
        _mapping(self.effective_config, "OUTCOME_EFFECTIVE_CONFIG")
        _mapping(self.resource_usage, "OUTCOME_RESOURCE_USAGE")
        object.__setattr__(self, "artifact_refs", _refs(self.artifact_refs, "OUTCOME_ARTIFACT_REFS"))
        object.__setattr__(self, "log_refs", _refs(self.log_refs, "OUTCOME_LOG_REFS"))
        object.__setattr__(self, "issue_codes", _refs(self.issue_codes, "OUTCOME_ISSUES"))
        object.__setattr__(self, "metrics", _metrics(self.metrics))
        if self.gate_passed not in {True, False, None}:
            raise GateError("OUTCOME_GATE")
        if self.status not in {"passed", "promoted"} and self.metrics == {}:
            raise GateError("OUTCOME_METRICS")

    @classmethod
    def create(cls, *, run_id: str, candidate_id: str | None, attempt_id: str,
               plan_id: str, plan_version: int, created_at: int, node: str,
               status: str, effective_config: dict, artifact_refs=(), log_refs=(),
               resource_usage=None, metrics=None, issue_codes=(), snapshot_sha: str,
               evaluation_protocol: str, evidence_scope: str,
               gate_passed: bool | None = None) -> "Outcome":
        payload = {"run_id": run_id, "candidate_id": candidate_id, "attempt_id": attempt_id,
                   "plan_id": plan_id, "plan_version": plan_version, "node": node,
                   "status": status, "snapshot_sha": snapshot_sha,
                   "evaluation_protocol": evaluation_protocol, "evidence_scope": evidence_scope,
                   "effective_config": effective_config, "metrics": metrics,
                   "issue_codes": list(issue_codes), "created_at": created_at}
        return cls(FEEDBACK_SCHEMA_VERSION, "outcome-" + digest(payload)[:24], run_id,
                   candidate_id, attempt_id, plan_id, plan_version, created_at, node,
                   status, dict(effective_config), tuple(artifact_refs), tuple(log_refs),
                   dict(resource_usage or {}), metrics, tuple(issue_codes), snapshot_sha,
                   evaluation_protocol, evidence_scope, gate_passed)

    @classmethod
    def from_dict(cls, value: dict) -> "Outcome":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != expected:
            raise GateError("OUTCOME_SCHEMA")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GateIssue:
    schema_version: int
    issue_id: str
    code: str
    stage: str
    severity: str
    candidate_id: str | None
    affected: dict
    observed_value: float | None
    threshold: float | None
    direction: str | None
    evidence_refs: tuple[str, ...]
    auto_repairable: bool
    allowed_changes: tuple[str, ...]
    waiting_reason: str | None = None
    human_reason: str | None = None

    def __post_init__(self):
        if self.schema_version != FEEDBACK_SCHEMA_VERSION:
            raise GateError("FEEDBACK_SCHEMA_VERSION")
        for name in ("issue_id", "code", "stage"):
            _text(getattr(self, name), "GATE_ISSUE_IDENTITY", maximum=200)
        if self.severity not in SEVERITIES or not isinstance(self.auto_repairable, bool):
            raise GateError("GATE_ISSUE_POLICY")
        if self.candidate_id is not None:
            _text(self.candidate_id, "GATE_ISSUE_IDENTITY", maximum=200)
        _mapping(self.affected, "GATE_ISSUE_AFFECTED")
        for value in (self.observed_value, self.threshold):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(float(value))):
                raise GateError("GATE_ISSUE_NUMERIC")
        if self.direction not in {None, "lt", "lte", "gt", "gte", "eq"}:
            raise GateError("GATE_ISSUE_DIRECTION")
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "GATE_ISSUE_EVIDENCE"))
        object.__setattr__(self, "allowed_changes", _refs(self.allowed_changes, "GATE_ISSUE_CHANGES"))
        if self.auto_repairable and not self.allowed_changes:
            raise GateError("GATE_ISSUE_POLICY")
        _text(self.waiting_reason, "GATE_ISSUE_REASON", optional=True)
        _text(self.human_reason, "GATE_ISSUE_REASON", optional=True)

    @classmethod
    def create(cls, *, code: str, stage: str, severity: str, candidate_id: str | None,
               affected: dict, evidence_refs, auto_repairable: bool, allowed_changes=(),
               observed_value=None, threshold=None, direction=None, waiting_reason=None,
               human_reason=None) -> "GateIssue":
        payload = {"code": code, "stage": stage, "candidate_id": candidate_id,
                   "affected": affected, "evidence_refs": list(evidence_refs)}
        return cls(FEEDBACK_SCHEMA_VERSION, "issue-" + digest(payload)[:24], code, stage,
                   severity, candidate_id, dict(affected), observed_value, threshold,
                   direction, tuple(evidence_refs), auto_repairable, tuple(allowed_changes),
                   waiting_reason, human_reason)

    @classmethod
    def from_dict(cls, value: dict) -> "GateIssue":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise GateError("GATE_ISSUE_SCHEMA")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AgentContext:
    schema_version: int
    message_id: str
    run_id: str
    plan_id: str
    plan_version: int
    feedback_id: str
    created_at: int
    current_goal: str
    current_plan: dict
    outcome: dict
    gate_issues: tuple[dict, ...]
    comparisons: dict
    history: tuple[dict, ...]
    allowed_changes: dict
    remaining_budget: dict
    output_contract: dict
    evidence_refs: tuple[str, ...]
    forbidden_actions: tuple[str, ...]

    def __post_init__(self):
        if self.schema_version != FEEDBACK_SCHEMA_VERSION:
            raise GateError("FEEDBACK_SCHEMA_VERSION")
        for name in ("message_id", "run_id", "plan_id", "feedback_id", "current_goal"):
            _text(getattr(self, name), "AGENT_CONTEXT_IDENTITY", maximum=300)
        if (not isinstance(self.plan_version, int) or isinstance(self.plan_version, bool)
                or self.plan_version < 1 or not isinstance(self.created_at, int)
                or isinstance(self.created_at, bool) or self.created_at < 0):
            raise GateError("AGENT_CONTEXT_IDENTITY")
        outcome = Outcome.from_dict(self.outcome)
        if outcome.run_id != self.run_id or outcome.plan_id != self.plan_id or outcome.plan_version != self.plan_version:
            raise GateError("AGENT_CONTEXT_SCOPE")
        if outcome.evidence_scope == "final_test":
            raise GateError("FINAL_TEST_FEEDBACK_FORBIDDEN")
        issues = tuple(GateIssue.from_dict(issue).to_dict() for issue in self.gate_issues)
        if any(issue["evidence_refs"] and not set(issue["evidence_refs"]).issubset(self.evidence_refs)
               for issue in issues):
            raise GateError("AGENT_CONTEXT_EVIDENCE")
        object.__setattr__(self, "gate_issues", issues)
        object.__setattr__(self, "history", tuple(self.history)[-8:])
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "AGENT_CONTEXT_EVIDENCE"))
        object.__setattr__(self, "forbidden_actions", _refs(self.forbidden_actions, "AGENT_CONTEXT_POLICY"))
        for value in (self.current_plan, self.comparisons, self.allowed_changes,
                      self.remaining_budget, self.output_contract):
            _mapping(value, "AGENT_CONTEXT_SCHEMA")

    @classmethod
    def create(cls, *, run_id: str, plan_id: str, plan_version: int, created_at: int,
               current_goal: str, current_plan: dict, outcome: Outcome, gate_issues,
               comparisons: dict, history, allowed_changes: dict,
               remaining_budget: dict, output_contract: dict, evidence_refs,
               forbidden_actions) -> "AgentContext":
        feedback_id = "feedback-" + digest({"outcome": outcome.message_id,
                                             "issues": [issue.issue_id for issue in gate_issues]})[:24]
        message_id = "context-" + digest({"feedback": feedback_id, "plan_version": plan_version})[:24]
        return cls(FEEDBACK_SCHEMA_VERSION, message_id, run_id, plan_id, plan_version,
                   feedback_id, created_at, current_goal, dict(current_plan),
                   outcome.to_dict(), tuple(issue.to_dict() for issue in gate_issues),
                   dict(comparisons), tuple(history), dict(allowed_changes),
                   dict(remaining_budget), dict(output_contract), tuple(evidence_refs),
                   tuple(forbidden_actions))

    @classmethod
    def from_dict(cls, value: dict) -> "AgentContext":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise GateError("AGENT_CONTEXT_SCHEMA")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeedbackProposal:
    schema_version: int
    message_id: str
    run_id: str
    action: str
    base_plan_id: str
    base_plan_version: int
    feedback_id: str
    issue_ids: tuple[str, ...]
    diagnosis_summary: str
    evidence_refs: tuple[str, ...]
    hypothesis: str | None
    proposed_changes: dict
    expected_direction: dict
    risks: tuple[str, ...]
    focus_metrics: tuple[str, ...]
    created_at: int
    prompt_version: str
    model_id: str

    def __post_init__(self):
        if self.schema_version != FEEDBACK_SCHEMA_VERSION:
            raise GateError("FEEDBACK_SCHEMA_VERSION")
        for name in ("message_id", "run_id", "base_plan_id", "feedback_id",
                     "diagnosis_summary", "prompt_version", "model_id"):
            _text(getattr(self, name), "FEEDBACK_PROPOSAL_IDENTITY", maximum=400)
        if (self.action not in PROPOSAL_ACTIONS
                or not isinstance(self.base_plan_version, int)
                or isinstance(self.base_plan_version, bool) or self.base_plan_version < 1
                or not isinstance(self.created_at, int) or isinstance(self.created_at, bool)
                or self.created_at < 0):
            raise GateError("FEEDBACK_PROPOSAL_ACTION")
        object.__setattr__(self, "issue_ids", _refs(self.issue_ids, "FEEDBACK_PROPOSAL_ISSUES"))
        object.__setattr__(self, "evidence_refs", _refs(self.evidence_refs, "FEEDBACK_PROPOSAL_EVIDENCE"))
        object.__setattr__(self, "risks", _refs(self.risks, "FEEDBACK_PROPOSAL_RISKS"))
        object.__setattr__(self, "focus_metrics", _refs(self.focus_metrics, "FEEDBACK_PROPOSAL_METRICS"))
        _text(self.hypothesis, "FEEDBACK_PROPOSAL_HYPOTHESIS", maximum=500, optional=True)
        _mapping(self.proposed_changes, "FEEDBACK_PROPOSAL_CHANGES")
        _mapping(self.expected_direction, "FEEDBACK_PROPOSAL_DIRECTION")
        if not self.issue_ids or not self.evidence_refs:
            raise GateError("FEEDBACK_PROPOSAL_EVIDENCE")
        if self.action == "experiment_patch" and (not self.issue_ids or not self.hypothesis or not self.proposed_changes):
            raise GateError("FEEDBACK_PROPOSAL_PATCH")
        if self.action != "experiment_patch" and self.proposed_changes:
            raise GateError("FEEDBACK_PROPOSAL_PATCH")

    @classmethod
    def from_dict(cls, value: dict) -> "FeedbackProposal":
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise GateError("FEEDBACK_PROPOSAL_SCHEMA")
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RepairProposal:
    round: int
    addressed_issue: str
    parent_fingerprint: str | None
    proposal_fingerprint: str
    proposal_id: str | None = None
    plan_version: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProgressDigest:
    round: int
    semantic_change: bool
    current_fingerprint: str | None
    previous_fingerprint: str | None
    best_so_far: dict | None
    terminal_reason: str | None = None
    execution_progress: bool | None = None
    quality_improved: bool | None = None
    comparable: bool | None = None
    resolved_issue_ids: tuple[str, ...] = field(default_factory=tuple)
    unresolved_issue_ids: tuple[str, ...] = field(default_factory=tuple)
    metric_deltas: dict = field(default_factory=dict)
    actual_resource_usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
