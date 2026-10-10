from ...config import GateError
from ..policies import require_release


def request_release(gateway, record, experiment, job):
    """Prepare an immutable request for a real, passed final artifact; no service mutation."""
    from ...config import digest
    from ..completion import accepted_for_experiment
    from ...models import load_forecast_model
    repo, engine = gateway.repo, gateway.engine
    result = job.get("result") or {}
    accepted = accepted_for_experiment(repo, experiment["record_id"])
    if job["status"] != "COMPLETED" or not result.get("passed"):
        raise GateError("AGENT_FINAL_GATE_NOT_PASSED")
    if (accepted is None or accepted["snapshot_sha"] != job["snapshot"]["sha256"]
            or experiment["binding"]["epoch"] != repo.binding()["epoch"]
            or experiment["binding"]["champion"] != repo.binding()["champion"]):
        raise GateError("AGENT_SELECTION_STALE")
    loaded = load_forecast_model(engine.artifacts, result["artifact"], engine.config)
    if loaded.candidate.candidate_id != accepted["candidate_id"]:
        raise GateError("AGENT_SELECTION_STALE")
    identity = "release-request-" + digest({"experiment": experiment["record_id"], "model": result["artifact"], "epoch": repo.binding()["epoch"]})[:24]
    try:
        return repo.get(identity)
    except GateError as error:
        if str(error) != "AGENT_RECORD_NOT_FOUND":
            raise
    now = record.get("effect_at", record["created_at"])
    policy = repo.binding()["policy"] or {}
    value = {"record_id": identity, "request_id": identity, "kind": "release_request", "task_key": repo.key,
             "experiment_id": experiment["record_id"], "run_id": job["run_id"], "candidate_id": accepted["candidate_id"],
             "model_ref": result["artifact"], "epoch": repo.binding()["epoch"], "champion": repo.binding()["champion"],
             "policy_version": policy.get("version"), "environment": engine.config.mode,
             "rollback_target": repo.binding()["champion"], "created_at": now,
             "status": "approval_required" if policy.get("mode") != "controlled_release" else "preauthorized",
             "approval_required": policy.get("mode") != "controlled_release", "actual_published": False}
    with repo.store.transaction() as db:
        repo.put(identity, "release_request", value, now, db=db)
        repo.store._event(db, repo.key, "AGENT_RELEASE_APPROVAL_REQUIRED" if value["approval_required"] else "AGENT_RELEASE_PREAUTHORIZED",
                         {"run_id": job["run_id"], "request_id": identity}, now, identity)
    return value


RELEASE_TOOLS = {
    "list_model_versions": {"offset": "nonnegative integer", "limit": "integer 1..10"},
    "create_shadow_from_model": {"model_id": "previously admitted model for the confirmed current regime"},
    "create_shadow": {"experiment_id": "string"},
    "start_canary": {"model_id": "string"},
    "request_promotion": {"model_id": "string"},
    "rollback": {"model_id": "string"},
    "get_serving_state": {},
}


def recover_release(gateway, record):
    operation = "agent-release:" + record["decision_id"]
    for event in gateway.engine.store.events(gateway.engine.key):
        if event["event_key"] == operation or event["payload"].get("agent_operation") == operation:
            return {"recovered": True, "event_id": event["id"], "outcome": event["kind"]}
    if record["selected_action"] == "create_shadow":
        from ...config import digest
        experiment_id = record["arguments"].get("experiment_id")
        run_id = "train-" + digest({"task": gateway.engine.key, "experiment_id": experiment_id})[:24]
        for value in gateway.engine.store.job_dispositions(gateway.engine.key):
            if value["run_id"] == run_id:
                return {"recovered": True, "outcome": value["outcome"], "run_id": run_id}
    return None


def execute_release(gateway, record):
    engine, repo = gateway.engine, gateway.repo
    tool, args, now = record["selected_action"], record["arguments"], engine._state()["now"]
    if tool == "list_model_versions":
        from ..model_library import list_versions
        return list_versions(gateway, args["offset"], args["limit"])
    if tool == "create_shadow_from_model":
        from ..model_library import create_shadow
        return create_shadow(gateway, record)
    if tool == "get_serving_state":
        state = engine._state()
        return {k: state.get(k) for k in ("champion", "challenger", "previous", "stage", "stage_since", "status",
            "observation_until", "observation_completed_for")} | {"release_contract": policy_contract(engine),
            "release_wait": {"object_id": state["challenger"] or state["champion"],
                "state": "shadow_eligible" if state["stage"] == "SHADOW" else "canary_eligible"
                    if state["stage"] == "CANARY" else "post_observation_complete"}}
    policy = require_release(gateway, record)
    operation = "agent-release:" + record["decision_id"]
    if tool == "create_shadow":
        from ...config import digest
        experiment = repo.get(args["experiment_id"])
        if experiment.get("kind") != "experiment" or experiment["task_key"] != engine.key:
            raise GateError("AGENT_EXPERIMENT_SCOPE")
        run_id = "train-" + digest({"task": engine.key, "experiment_id": experiment["record_id"]})[:24]
        job = engine.store.job(run_id)
        grant = engine._state().get("agent_release_grant")
        if policy.get("mode") == "train_auto_release_manual" and (not grant or grant["run_id"] != run_id or grant["model_ref"] != (job.get("result") or {}).get("artifact")):
            raise GateError("AGENT_MANUAL_GRANT_SCOPE")
        if job["status"] != "COMPLETED" or not (job["result"] or {}).get("passed"):
            raise GateError("AGENT_FINAL_GATE_NOT_PASSED")
        if experiment["binding"]["champion"] != engine._state()["champion"]:
            raise GateError("AGENT_CHAMPION_CHANGED")
        require_release(gateway, record, job["result"]["candidate"])
        state = engine._state()
        if (state["champion"] and state.get("observation_until") is not None
                and state.get("observation_completed_for") != state["champion"]):
            raise GateError("POST_PROMOTION_OBSERVATION_REQUIRED")
        request = request_release(gateway, record, experiment, job)
        def bind_attempt(current):
            if current["champion"] != request["champion"] or current["decision_epoch"] != request["epoch"]:
                raise GateError("AGENT_RELEASE_ATTEMPT_STALE")
            current["agent_release_attempt"] = request
        engine.store.change(engine.key, bind_attempt)
        engine._consume_training(now, agent_run_id=run_id)
        return {"run_id": run_id, "stage": engine._state()["stage"], "champion": engine._state()["champion"],
                "challenger": engine._state()["challenger"]}
    state = engine._state()
    if tool == "rollback":
        deployment = state.get("last_deployment") or {}
        if (args["model_id"] != state["previous"] or deployment.get("to") != state["champion"]
                or deployment.get("from") != state["previous"]):
            raise GateError("AGENT_ROLLBACK_TARGET_CHANGED")
        engine.rollback("AGENT_REQUESTED_ROLLBACK", operation=operation)
        return {"champion": engine._state()["champion"], "rolled_back": True}
    if args["model_id"] != state["challenger"]:
        raise GateError("AGENT_CHALLENGER_CHANGED")
    require_release(gateway, record, engine.store.model(state["challenger"])["report"]["candidate"])
    if tool == "request_promotion" and state["stage"] == "INITIAL_APPROVAL":
        engine.store.approve_release(engine.key, policy["authorized_by"], expected_binding=engine.store.release_binding(engine.key), operation=operation + ":approval")
        engine.store.switch(engine.key, None, state["challenger"], now,
                            {"passed": True, "kind": "INITIAL_OFFLINE_GATE"}, operation)
        return {"champion": engine._state()["champion"], "advanced": True}
    expected = "SHADOW" if tool == "start_canary" else "CANARY"
    if state["stage"] != expected:
        raise GateError("AGENT_RELEASE_STAGE")
    if tool == "request_promotion" and engine.config.mode == "live":
        engine.store.approve_release(engine.key, policy["authorized_by"],
                                     expected_binding=engine.store.release_binding(engine.key), operation=operation + ":approval")
    engine._release_gate(now, now, agent_operation=operation)
    current = engine._state()
    return {"stage": current["stage"], "champion": current["champion"], "challenger": current["challenger"],
            "advanced": current["stage"] != state["stage"] or current["champion"] != state["champion"]}


def policy_contract(engine):
    from dataclasses import asdict
    return {"environment": engine.config.mode, "mode": (engine._state().get("autonomy_policy") or {}).get("mode"),
            "quality": asdict(engine.config.release), "evidence": "complete_mature_paired_windows"}
