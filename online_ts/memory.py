"""Immutable experiment observations in the existing SQLite message store."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict

from .config import GateError, canonical, digest


VERSION = "trial-observation-v1"
DEVELOPMENT_SCOPES = {"development_validation", "execution", "resource"}


def _failure_class(outcome):
    if outcome.get("metrics") is not None:
        return None if outcome.get("gate_passed") else "development_quality"
    codes = " ".join(outcome.get("issue_codes", []))
    for terms, label in ((('AUTH', 'CREDENTIAL', 'CONFIGURATION'), 'authentication_or_configuration'),
                         (('OOM', 'MEMORY', 'VRAM'), 'resource'),
                         (('DEADLINE', 'TIMEOUT'), 'timeout'),
                         (('SCHEMA', 'SPEC', 'EVIDENCE', 'DIRECTION'), 'invalid_proposal'),
                         (('CANCEL',), 'cancelled')):
        if any(term in codes for term in terms):
            return label
    return "execution_without_quality"


def persist_observations(db, job, checkpoint):
    """Called in the same transaction as the measured Outcome, never from Test."""
    config_row = db.execute("SELECT config FROM tasks WHERE task_key=?", (job["task_key"],)).fetchone()
    config = json.loads(config_row[0])
    config_version = digest(config)
    for outcome in checkpoint.get("outcomes", []):
        if outcome.get("evidence_scope") not in DEVELOPMENT_SCOPES:
            continue
        report = next((r for r in reversed(checkpoint.get("reports", []))
                       if r.get("candidate_id") == outcome.get("candidate_id")
                       and r.get("snapshot_sha") == outcome.get("snapshot_sha")), {})
        specs = job["plan"].get("experiment_specs", []) + job["plan"].get("experiment_history", [])
        from .models import Candidate
        spec = next((s for s in specs if Candidate.from_dict(s["candidate"]).candidate_id == outcome.get("candidate_id")), None)
        normalization = report.get("normalization")
        samples = [f.get("issue_times") for f in report.get("folds", [])]
        metrics = outcome.get("metrics")
        if metrics is not None:
            metrics = {k: v for k, v in metrics.items() if type(v) in {int, float} and math.isfinite(v)}
            metrics.update({"folds": [{k: f[k] for k in ("normalized_mae", "normalized_mse", "mae_by_variable", "mae_by_horizon")
                                      if k in f} for f in report.get("folds", [])]})
        value = {
            "version": VERSION, "task_key": job["task_key"], "run_id": job["run_id"],
            "outcome_id": outcome["message_id"], "snapshot_sha": outcome["snapshot_sha"],
            "config_version": config_version, "plan_version": outcome["plan_version"],
            "evaluation_protocol": outcome["evaluation_protocol"], "evidence_scope": outcome["evidence_scope"],
            "sample_fingerprint": digest(samples) if samples and all(samples) else None,
            "normalization_fingerprint": digest(normalization) if normalization else None,
            "candidate_id": outcome.get("candidate_id"), "effective_config": outcome["effective_config"],
            "experiment_spec": spec, "feature_version": digest(outcome["effective_config"].get("features", [])),
            "profile_ref": checkpoint.get("planner_profile_ref"),
            "status": outcome["status"], "metrics": metrics,
            "failure_class": _failure_class(outcome), "issue_codes": outcome["issue_codes"],
            "gate_passed": outcome.get("gate_passed"), "resource_usage": outcome.get("resource_usage", {}),
            "event_cutoff_ns": job["cutoff"], "evidence_refs": [outcome["message_id"], *outcome.get("artifact_refs", [])],
        }
        value = json.loads(canonical(value))
        identity = "trial-" + digest(value)
        existing = db.execute("SELECT payload FROM feedback_messages WHERE message_id=?", (identity,)).fetchone()
        if existing:
            if json.loads(existing[0])["observation"] != value:
                raise GateError("TRIAL_OBSERVATION_CONFLICT")
            continue
        now = time.time_ns()
        envelope = {"observation_id": identity, "recorded_at": now, "observation": value}
        db.execute("INSERT INTO feedback_messages VALUES(?,?,?,?,?,?,?)",
                   (identity, job["task_key"], job["run_id"], "trial_observation", outcome["plan_version"], now, canonical(envelope)))


def comparable(left, right):
    fields = ("task_key", "snapshot_sha", "config_version", "evaluation_protocol",
              "sample_fingerprint", "normalization_fingerprint")
    return (left.get("evidence_scope") == right.get("evidence_scope") == "development_validation"
            and left.get("metrics") is not None and right.get("metrics") is not None
            and all(left.get(k) is not None and left[k] == right.get(k) for k in fields))


def query_observations(store, task_key, *, decision_ns, snapshot_sha=None, reference=None, limit=3):
    # task_key is resolved/authorized by the Worker or OS-authenticated management
    # caller. There is deliberately no wildcard or cross-task search API.
    if type(decision_ns) is not int or decision_ns < 0 or not 0 <= limit <= 500:
        raise GateError("TRIAL_QUERY_SCOPE")
    store.task(task_key)
    with store.connection() as db:
        rows = db.execute("""SELECT payload FROM feedback_messages
            WHERE task_key=? AND kind='trial_observation' AND created_ns<=?
            ORDER BY CASE WHEN json_extract(payload,'$.observation.snapshot_sha')=? THEN 0 ELSE 1 END,
            created_ns DESC,message_id LIMIT ?""", (task_key, decision_ns, snapshot_sha, limit)).fetchall()
    result = []
    for row in rows:
        envelope = json.loads(row[0])
        value = envelope["observation"]
        if (envelope["observation_id"] != "trial-" + digest(value)
                or value["task_key"] != task_key or envelope["recorded_at"] > decision_ns
                or value["version"] != VERSION or value["evidence_scope"] not in DEVELOPMENT_SCOPES):
            raise GateError("TRIAL_OBSERVATION_SCOPE")
        numerical = reference is not None and comparable(reference, value)
        result.append({**envelope, "comparable": numerical,
                       "comparison_reason": None if numerical else "BACKGROUND_ONLY_NO_MATCHED_REFERENCE"})
    return result


def outbound_observation(envelope, ref):
    """Exclude text interpretations, physical identities and raw evidence values."""
    from .models import Candidate
    value = envelope["observation"]
    if value["evidence_scope"] not in DEVELOPMENT_SCOPES:
        raise GateError("FINAL_TEST_FEEDBACK_FORBIDDEN")
    candidate = value["effective_config"]
    if candidate:
        c = Candidate.from_dict(candidate)
        defaults = asdict(Candidate(c.family, c.input_steps, c.history_steps))
        candidate = {k: v for k, v in asdict(c).items()
                     if k in {"family", "input_steps", "history_steps"} or v != defaults[k]}
    metrics = value["metrics"]
    return {"ref": ref, "candidate": candidate, "status": value["status"],
            "score": metrics.get("score") if metrics else None,
            "score_std": metrics.get("score_std") if metrics else None,
            "failure_class": value["failure_class"], "issue_codes": value["issue_codes"],
            "comparable": envelope["comparable"], "comparison_reason": envelope["comparison_reason"]}


def export_training_candidates(store, task_key, *, actor, decision_ns, limit=500):
    """Read-only, task-scoped review candidates; never automatically training data."""
    config, _ = store.task(task_key)
    from .copilot import current_actor
    if actor != current_actor() or "experiment" not in config.collaboration.permissions.get(actor, ()):
        raise GateError("HUMAN_PERMISSION_DENIED")
    observations = query_observations(store, task_key, decision_ns=decision_ns, limit=limit)
    records = []
    for envelope in observations:
        value = envelope["observation"]
        group = digest({"task": task_key, "snapshot": value["snapshot_sha"],
                        "protocol": value["evaluation_protocol"]})
        records.append({"schema_version": 1, "review_status": "pending_human_review",
                        "eligible_for_training": False, "series_group": group,
                        "suggested_split": "holdout" if int(group[:8], 16) % 5 == 0 else "train",
                        "observation_ref": envelope["observation_id"],
                        "evidence_refs": value["evidence_refs"],
                        "recorded_at": envelope["recorded_at"],
                        "facts": outbound_observation(envelope, "trial"),
                        "entry_conditions": ["human_permission_and_content_review", "group_disjoint_holdout",
                                             "no_test_or_future_evidence", "separate_training_authorization"]})
    # Include authorized bounded decision examples such as wait/stop and invalid
    # responses. Free text and arbitrary model completions are not exported.
    with store.connection() as db:
        rows = db.execute("SELECT payload,created_ns,run_id FROM feedback_messages WHERE task_key=? AND kind='agent_proposal' AND created_ns<=? ORDER BY created_ns,message_id LIMIT ?",
                          (task_key, decision_ns, limit)).fetchall()
    for row in rows:
        proposal = json.loads(row[0])
        job = store.job(row[2])
        if proposal.get("created_at", decision_ns + 1) > decision_ns:
            continue
        group = digest({"task": task_key, "snapshot": job["snapshot"]["sha256"], "protocol": "rolling-origin-v1"})
        records.append({"schema_version": 1, "review_status": "pending_human_review", "eligible_for_training": False,
                        "series_group": group, "suggested_split": "holdout" if int(group[:8], 16) % 5 == 0 else "train",
                        "proposal_ref": proposal["message_id"],
                        "facts": {"action": proposal["action"]},
                        "entry_conditions": ["human_permission_and_content_review", "group_disjoint_holdout",
                                             "no_test_or_future_evidence", "separate_training_authorization"]})
    return records
