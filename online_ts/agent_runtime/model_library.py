"""Reuse a previously admitted artifact for a confirmed operating regime."""
from ..config import GateError
from ..qualification import regime_context
from .policies import require_release


def list_versions(gateway, offset, limit):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 10:
        raise GateError("MODEL_LIBRARY_PAGE")
    with gateway.repo.store.connection() as db:
        ids = [r[0] for r in db.execute("SELECT model_id FROM models WHERE task_key=? ORDER BY created_ns DESC,model_id",
                                      (gateway.repo.key,))]
    rows = []
    for identity in ids[offset:offset + limit]:
        entry = gateway.repo.store.model(identity)
        candidate = entry["report"].get("candidate", {})
        rows.append({"model_id": identity, "model_ref": entry["artifact"],
                     "candidate": {k: candidate.get(k) for k in ("family", "input_steps", "history_steps", "adaptation_mode")},
                     "training_regime": entry["report"].get("training_regime"),
                     "final_admitted": entry["report"].get("passed") is True,
                     "metrics_scope": "sealed; current paired observation is still required"})
    return {"models": rows, "total_count": len(ids), "offset": offset}


def create_shadow(gateway, record):
    engine = gateway.engine
    state, now = engine._state(), engine._state()["now"]
    target = record["arguments"]["model_id"]
    entry = engine.store.model(target)
    if entry["task_key"] != engine.key or entry["report"].get("passed") is not True:
        raise GateError("EXISTING_MODEL_NOT_ADMITTED")
    policy = require_release(gateway, record, entry["report"]["candidate"])
    if policy.get("mode") != "controlled_release":
        raise GateError("EXISTING_MODEL_REQUIRES_CONTROLLED_RELEASE")
    if not state["champion"] or target == state["champion"] or state["challenger"] or engine.store.active_job(engine.key):
        raise GateError("EXISTING_MODEL_RELEASE_CONFLICT")
    if state.get("observation_until") is not None and state.get("observation_completed_for") != state["champion"]:
        raise GateError("POST_PROMOTION_OBSERVATION_REQUIRED")
    loaded = engine._load_model(target)
    if loaded.payload["scaler"]["fit_end"] >= now:
        raise GateError("EXISTING_MODEL_TRAINING_CUTOFF")
    frame = engine.source.read_until(now).tail(max(loaded.candidate.input_steps, engine.config.horizon_steps))
    current = regime_context(frame, int(frame.index[0]) if len(frame) else now, now + 1,
                             declared=bool(engine.config.source.regime_column))
    trained = entry["report"].get("training_regime") or {}
    if current["status"] != "confirmed" or trained.get("status") != "confirmed" or current["regime_id"] != trained["regime_id"]:
        raise GateError("EXISTING_MODEL_REGIME_EVIDENCE_REQUIRED")
    operation = "agent-release:" + record["decision_id"]
    attempt = {"kind": "existing_model", "model_id": target, "model_ref": entry["artifact"],
               "champion": state["champion"], "rollback_target": state["champion"], "epoch": state["decision_epoch"],
               "policy_version": policy["version"], "environment": engine.config.mode, "created_at": now,
               "selection_decision": record["decision_id"], "candidate": entry["report"]["candidate"], "regime": current}
    def apply(current_state):
        if current_state["champion"] != state["champion"] or current_state["challenger"] or current_state["decision_epoch"] != state["decision_epoch"]:
            raise GateError("AGENT_RELEASE_ATTEMPT_STALE")
        current_state.update(challenger=target, stage="SHADOW", stage_since=now, approved=False,
                             status="SHADOWING", agent_release_attempt=attempt)
    engine.store.change(engine.key, apply, "SHADOW_STARTED", {"model_id": target, "champion": state["champion"],
        "agent_operation": operation, "reuse_existing_artifact": True}, now)
    return {"stage": "SHADOW", "challenger": target, "model_ref": entry["artifact"], "retrained": False,
            "next_condition": "shadow_eligible", "paired_observation_required": True}
