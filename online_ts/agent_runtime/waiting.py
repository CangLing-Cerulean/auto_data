"""Object-specific durable waits. A deadline is a recheck, never evidence of maturity."""
from dataclasses import asdict
import math
import time

from ..config import GateError, SECOND, digest
from .state import WaitCondition


WAIT_SCHEMA = {"condition_type": "labels|job|time|event|human|input|release", "object_id": "window_id, run_id, request_id, task_key, model_id for release, or null for time",
               "predicate": 'JSON object, e.g. {"state":"complete"}. state: complete=labels; development_complete|final_complete=job; approved=human; due=time; shadow_eligible=Shadow passed; canary_eligible=Canary passed; post_observation_complete=release. max_age_seconds=input; event_kind=event.',
               "clock_domain": "event_time|wall_time", "deadline": "future integer ns in clock_domain; timeout causes recheck, not satisfaction",
               "reason": "brief reason"}


def window_id(window):
    return "window-" + digest({k: window[k] for k in ("task_key", "model_id", "window_start", "window_end")})[:24]


def label_windows(repo):
    return [{"window_id": window_id(w), **{k: w[k] for k in ("model_id", "window_start", "window_end", "label_state", "evaluated_ns", "label_fingerprint")},
             "expected_targets": w["evidence"].get("expected_count", 0),
             "label_coverage": w["evidence"].get("label_coverage", 0),
             "fully_evaluable": w["label_state"] in {"complete", "revised"} and w["evidence"].get("labels_complete", False)}
            for w in repo.store.monitoring_windows(repo.key)]


def evaluate(repo, wait, now, *, source=None, wall_ns=None):
    wall_ns = time.time_ns() if wall_ns is None else wall_ns
    binding = repo.binding()
    if wait["task_key"] != repo.key or wait["epoch"] != binding["epoch"] or binding["owner"] != "agent" or binding["stop_requested"]:
        return "cancelled", {"reason": "authority_changed"}
    clock = now if wait["clock_domain"] == "event_time" else wall_ns
    kind, obj, predicate = wait["condition_type"], wait["object_id"], wait["predicate"]
    satisfied, evidence = False, {}
    if kind == "labels":
        window = next((w for w in label_windows(repo) if w["window_id"] == obj), None)
        satisfied = bool(window and window["fully_evaluable"])
        evidence = window or {"window_id": obj, "available": False}
    elif kind == "job":
        job = repo.store.job(obj)
        if job and job["task_key"] == repo.key:
            complete = job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}
            development = job["checkpoint"]["index"] == len(job["plan"]["candidates"]) and not job["plan"].get("needs_proposal")
            satisfied = complete or predicate["state"] == "development_complete" and development
            evidence = {"run_id": obj, "status": job["status"], "development_complete": development}
    elif kind == "human":
        _, task = repo.store.task(repo.key)
        grant = task.get("agent_release_grant") or {}
        satisfied = (grant.get("request_id") == obj and grant.get("epoch") == binding["epoch"]
                     and grant.get("champion") == binding["champion"] and now <= grant.get("expires_at", -1))
        evidence = {"request_id": obj, "approved": satisfied}
    elif kind == "release":
        _, task = repo.store.task(repo.key)
        desired = predicate["state"]
        if desired == "post_observation_complete":
            if task["champion"] != obj:
                return "cancelled", {"reason": "champion_changed"}
            satisfied = task.get("observation_completed_for") == obj
        else:
            expected_stage = "SHADOW" if desired == "shadow_eligible" else "CANARY"
            if task["challenger"] != obj or task["stage"] != expected_stage:
                return "cancelled", {"reason": "release_stage_changed"}
            gates = [e for e in repo.store.events(repo.key) if e["kind"] == "PAIRED_GATE"
                     and task["stage_since"] <= e["event_time"] <= now and e["payload"]["stage"] == expected_stage]
            satisfied = bool(gates and gates[-1]["payload"]["evidence"]["passed"])
        evidence = {"model_id": obj, "state": desired, "qualified": satisfied, "metrics": "audit_only"}
    elif kind == "input" and source is not None:
        frame = source.read_until(now)
        age = (now - int(frame.index[-1])) / SECOND if len(frame) else None
        satisfied = age is not None and 0 <= age <= predicate["max_age_seconds"]
        evidence = {"as_of": now, "age_seconds": age}
    elif kind == "event":
        matches = [e for e in repo.store.events(repo.key) if e["event_time"] <= now and e["id"] > wait.get("event_cursor", 0)
                   and e["kind"] == predicate["event_kind"] and (obj is None or obj in
                   [e["payload"].get(k) for k in ("run_id", "model_id", "record_id", "intent_id")])]
        satisfied = bool(matches)
        evidence = {"event_id": matches[0]["id"]} if matches else {}
    elif kind == "time":
        satisfied = clock >= wait["deadline"]
        evidence = {"clock_domain": wait["clock_domain"], "due_at": wait["deadline"]}
    if satisfied:
        return "satisfied", evidence
    return ("timed_out" if clock >= wait["deadline"] else "waiting"), evidence


def register(gateway, record):
    args, repo = record["arguments"], gateway.repo
    identity = "wait-" + record["decision_id"]
    try:
        old = repo.get(identity)
        return {"condition_id": identity, "wait_condition": old, "recovered": True}
    except GateError as error:
        if str(error) != "AGENT_RECORD_NOT_FOUND":
            raise
    now = record.get("effect_at", record["created_at"])
    kind, obj, predicate = args["condition_type"], args["object_id"], args["predicate"]
    if args["clock_domain"] not in {"event_time", "wall_time"} or type(args["deadline"]) is not int:
        raise GateError("AGENT_WAIT_ARGUMENTS")
    clock = now if args["clock_domain"] == "event_time" else time.time_ns()
    if args["deadline"] <= clock or not isinstance(args["reason"], str) or not 1 <= len(args["reason"]) <= 1000:
        raise GateError("AGENT_WAIT_ARGUMENTS")
    if obj is not None and not isinstance(obj, str) or not isinstance(predicate, dict):
        raise GateError("AGENT_WAIT_ARGUMENTS")
    valid = False
    if kind == "labels":
        valid = predicate == {"state": "complete"} and any(w["window_id"] == obj for w in label_windows(repo))
    elif kind == "job":
        job = repo.store.job(obj) if isinstance(obj, str) else None
        valid = predicate in ({"state": "development_complete"}, {"state": "final_complete"}) and job and job["task_key"] == repo.key
    elif kind == "human":
        request = repo.get(obj) if isinstance(obj, str) else {}
        valid = predicate == {"state": "approved"} and request.get("kind") == "release_request" and request.get("epoch") == repo.binding()["epoch"]
    elif kind == "release":
        _, task = repo.store.task(repo.key)
        valid = set(predicate) == {"state"} and (
            predicate["state"] == "post_observation_complete" and obj is not None and obj == task["champion"]
            or predicate["state"] in {"shadow_eligible", "canary_eligible"} and obj is not None and obj == task["challenger"])
    elif kind == "input":
        threshold = predicate.get("max_age_seconds")
        valid = obj == repo.key and set(predicate) == {"max_age_seconds"} and type(threshold) in {int, float} and math.isfinite(threshold) and threshold >= 0
    elif kind == "event":
        valid = set(predicate) == {"event_kind"} and isinstance(predicate["event_kind"], str) and 0 < len(predicate["event_kind"]) <= 100
    elif kind == "time":
        valid = obj is None and predicate == {"state": "due"}
    if not valid:
        raise GateError("AGENT_WAIT_OBJECT_OR_PREDICATE")
    wait = asdict(WaitCondition(condition_id=identity, task_key=repo.key, epoch=repo.binding()["epoch"],
        created_at=now, created_wall_ns=time.time_ns(), agent_state_version=repo.state().decision_version + 1, **args))
    if kind == "event":
        wait["event_cursor"] = repo.state().last_event_id
    initial, evidence = evaluate(repo, wait, now, source=gateway.engine.source)
    wait.update(initial_evaluation=initial, initial_evidence=evidence)
    repo.put(identity, "wait_condition", wait, now)
    return {"condition_id": identity, "wait_condition": wait}


def resolve(repo, wait, status, evidence, now):
    old = repo.get(wait["condition_id"])
    if old["status"] != "waiting":
        return old
    value = {**wait, "status": status, "resolved_at": now, "resolved_wall_ns": time.time_ns(), "resolution_evidence": evidence}
    with repo.store.transaction() as db:
        repo.put(wait["condition_id"], "wait_condition", value, now, replace=True, db=db)
        repo.put("wake-" + wait["condition_id"], "wake", {"condition_id": wait["condition_id"], "status": status,
                 "created_at": now, "evidence": evidence}, now, db=db)
    return value
