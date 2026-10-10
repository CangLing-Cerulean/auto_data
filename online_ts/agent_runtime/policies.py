"""Explicit OS-authorized ownership transfer and bounded autonomy policy."""
from dataclasses import asdict, dataclass
import json
import time
import uuid

from ..config import GateError, canonical, strict
from ..copilot import current_actor


@dataclass(frozen=True)
class AutonomyPolicy:
    mode: str = "advisory"
    execution_profile: str = "bounded"
    model_whitelist: tuple = ("last_value", "seasonal_naive", "ridge")
    allow_online_development: bool = False
    failure_policy: str = "hold_current_state"
    max_calls: int | None = 128
    max_experiments: int | None = 8
    max_llm_tokens: int | None = 1_000_000
    max_call_tokens: int = 32768
    timeout_seconds: int = 120
    inspection_seconds: int = 3600
    allow_initial_release: bool = False

    def __post_init__(self):
        if self.execution_profile not in {"bounded", "research_open"}:
            raise GateError("AGENT_EXECUTION_PROFILE")
        cumulative = ("max_calls", "max_experiments", "max_llm_tokens")
        if self.execution_profile == "research_open":
            if any(getattr(self, name) is not None for name in cumulative):
                raise GateError("OPEN_REQUIRES_NULL_QUOTAS")
        if self.mode not in {"advisory", "train_auto_release_manual", "controlled_release"}:
            raise GateError("AUTONOMY_MODE")
        if self.failure_policy not in {"hold_current_state", "request_human", "fallback_workflow"}:
            raise GateError("AGENT_FAILURE_POLICY")
        if (not isinstance(self.model_whitelist, (list, tuple)) or not self.model_whitelist
                or set(self.model_whitelist) - {"last_value", "seasonal_naive", "ridge", "chronos2", "patchtst"}):
            raise GateError("AUTONOMY_MODEL_WHITELIST")
        for name, maximum in (("max_calls", 10000), ("max_experiments", 1000), ("max_llm_tokens", 100_000_000),
                              ("max_call_tokens", 131072), ("timeout_seconds", 300), ("inspection_seconds", 604800)):
            if self.execution_profile == "research_open" and name in cumulative:
                continue
            if type(getattr(self, name)) is not int or not 0 < getattr(self, name) <= maximum:
                raise GateError("AUTONOMY_BUDGET")
        if type(self.allow_online_development) is not bool or type(self.allow_initial_release) is not bool:
            raise GateError("AUTONOMY_BOOLEAN")


def configure(repo, controller_mode, *, policy=None, goal=None):
    if controller_mode not in {"workflow", "agent"}:
        raise GateError("CONTROLLER_MODE")
    actor = current_actor()
    config, _ = repo.store.task(repo.key)
    permissions = config.collaboration.permissions.get(actor, ())
    if "experiment" not in permissions:
        raise GateError("HUMAN_PERMISSION_DENIED")
    policy = policy or AutonomyPolicy()
    if not isinstance(policy, AutonomyPolicy):
        policy = strict(AutonomyPolicy, policy)
    if policy.mode == "controlled_release" and "release" not in permissions:
        raise GateError("HUMAN_PERMISSION_DENIED")
    if policy.mode == "controlled_release" and config.mode != "replay":
        raise GateError("AUTONOMOUS_RELEASE_REPLAY_ONLY")
    if controller_mode == "agent":
        repo.initialize(goal or "Maintain reliable predictions, investigate evidence, and train only when justified.")
    owner = "mode-transfer-" + uuid.uuid4().hex
    if not repo.store.acquire(repo.key, owner, 30):
        raise GateError("TASK_ALREADY_RUNNING")
    try:
        with repo.store.transaction() as db:
            active = db.execute("SELECT 1 FROM leases WHERE task_key=? AND expires>?", ("agent-decision:" + repo.key, time.time())).fetchone()
            pending = db.execute("SELECT 1 FROM agent_runtime_records WHERE task_key=? AND kind='decision' AND json_extract(payload,'$.decision_outcome')='prepared'", (repo.key,)).fetchone()
            jobs = db.execute("SELECT 1 FROM jobs j LEFT JOIN job_dispositions d ON j.run_id=d.run_id WHERE j.task_key=? AND (j.status IN ('PENDING','RUNNING') OR d.run_id IS NULL)", (repo.key,)).fetchone()
            row = db.execute("SELECT state FROM tasks WHERE task_key=?", (repo.key,)).fetchone()
            state = json.loads(row[0])
            if active or pending or jobs or state["challenger"]:
                raise GateError("CONTROLLER_TRANSFER_REQUIRES_QUIESCENT_TASK")
            previous = state.get("decision_owner", "workflow")
            epoch = state.get("decision_epoch", 0) + 1
            state.update(decision_owner=controller_mode, decision_epoch=epoch,
                         autonomy_policy={**asdict(policy), "version": epoch, "authorized_by": actor,
                           "task_key": repo.key, "config_version": config.version, "environment": config.mode,
                           "release_contract": asdict(config.release)})
            state.pop("agent_release_grant", None)
            state.pop("agent_release_attempt", None)
            if goal is not None and controller_mode == "agent":
                agent_state = json.loads(db.execute("SELECT payload FROM agent_runtime_state WHERE task_key=?", (repo.key,)).fetchone()[0])
                agent_state["task_goal"] = goal
                db.execute("UPDATE agent_runtime_state SET payload=? WHERE task_key=?", (canonical(agent_state), repo.key))
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), repo.key))
            repo.store._event(db, repo.key, "CONTROLLER_MODE_CHANGED", {"from": previous, "to": controller_mode,
                              "epoch": epoch, "actor": actor, "policy": state["autonomy_policy"]}, state["now"] or 0)
    finally:
        repo.store.release(repo.key, owner)


def require_release(gateway, record, candidate=None):
    engine = gateway.engine
    state = engine._state()
    policy = state.get("autonomy_policy") or {}
    grant = state.get("agent_release_grant")
    manual = policy.get("mode") == "train_auto_release_manual" and grant is not None
    if manual:
        job = engine.store.job(grant["run_id"])
        if (grant["epoch"] != state["decision_epoch"] or grant["champion"] != state["champion"]
                or grant["plan_version"] != job["checkpoint"]["plan_version"]
                or grant["model_ref"] != (job.get("result") or {}).get("artifact")
                or grant["environment"] != engine.config.mode or grant["policy_version"] != policy.get("version")
                or grant["expires_at"] < state["now"]
                or "release" not in engine.config.collaboration.permissions.get(grant["authorized_by"], ())):
            raise GateError("AGENT_MANUAL_GRANT_STALE")
        if candidate:
            from ..models import Candidate
            if Candidate.from_dict(candidate).candidate_id != grant["candidate_id"]:
                raise GateError("AGENT_MANUAL_GRANT_CANDIDATE")
        policy = {**policy, "authorized_by": grant["authorized_by"], "allow_initial_release": True}
    if (not manual and policy.get("mode") != "controlled_release" or policy.get("task_key") != engine.key
            or policy.get("config_version") != engine.config.version or policy.get("environment") != engine.config.mode
            or policy.get("release_contract") != json.loads(canonical(asdict(engine.config.release)))
            or "release" not in engine.config.collaboration.permissions.get(policy.get("authorized_by"), ())):
        raise GateError("AGENT_RELEASE_NOT_AUTHORIZED")
    if not manual and engine.config.mode != "replay":
        raise GateError("AUTONOMOUS_RELEASE_REPLAY_ONLY")
    if record["selected_action"] != "rollback":
        if state["champion"]:
            engine._load_model(state["champion"])
        elif not policy.get("allow_initial_release"):
            raise GateError("AGENT_INITIAL_RELEASE_NOT_AUTHORIZED")
    if candidate and candidate["family"] not in policy["model_whitelist"]:
        raise GateError("AGENT_RELEASE_MODEL_NOT_AUTHORIZED")
    if record["selected_action"] in {"start_canary", "request_promotion"}:
        attempt = state.get("agent_release_attempt") or {}
        challenger = engine.store.model(state["challenger"]) if state["challenger"] else None
        if (attempt.get("epoch") != state["decision_epoch"] or attempt.get("champion") != state["champion"]
                or attempt.get("policy_version") != policy.get("version") or not challenger
                or attempt.get("model_ref") != challenger["artifact"]):
            raise GateError("AGENT_RELEASE_ATTEMPT_STALE")
    return policy


def grant_release(repo, experiment_id):
    """Human approval binds the final model artifact; it never selects or starts final."""
    from ..config import digest, SECOND
    from ..data import ArtifactStore
    from ..models import Candidate
    from ..models import load_forecast_model
    config, state = repo.store.task(repo.key)
    actor = current_actor()
    if "release" not in config.collaboration.permissions.get(actor, ()):
        raise GateError("HUMAN_PERMISSION_DENIED")
    experiment = repo.get(experiment_id)
    if experiment.get("kind") != "experiment" or experiment["task_key"] != repo.key:
        raise GateError("AGENT_EXPERIMENT_SCOPE")
    run_id = "train-" + digest({"task": repo.key, "experiment_id": experiment_id})[:24]
    job = repo.store.job(run_id)
    result = job.get("result") or {}
    if job["status"] != "COMPLETED" or not result.get("passed"):
        raise GateError("AGENT_FINAL_GATE_NOT_PASSED")
    from .completion import accepted_for_experiment
    accepted = accepted_for_experiment(repo, experiment_id)
    if accepted is None:
        raise GateError("AGENT_SELECTION_REQUIRED")
    reports = [r for r in job["checkpoint"]["reports"] if r["candidate_id"] == accepted["candidate_id"]]
    if not reports or reports[0]["validation_artifact"] != accepted["report_ref"]:
        raise GateError("AGENT_SELECTION_STALE")
    loaded = load_forecast_model(ArtifactStore(repo.store.path.parent / "artifacts"), result["artifact"], config)
    if loaded.candidate.candidate_id != accepted["candidate_id"]:
        raise GateError("AGENT_SELECTION_STALE")
    binding = repo.binding()
    request = next((r for r in repo.iter_records("release_request") if r["experiment_id"] == experiment_id
                    and r["epoch"] == binding["epoch"] and r["model_ref"] == result["artifact"]), None)
    if request is None:
        raise GateError("AGENT_RELEASE_REQUEST_REQUIRED")
    with repo.store.transaction() as db:
        current = json.loads(db.execute("SELECT state FROM tasks WHERE task_key=?", (repo.key,)).fetchone()[0])
        current_job = repo.store._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
        if (current.get("decision_owner") != "agent" or current.get("decision_epoch") != binding["epoch"]
                or current["champion"] != experiment["binding"]["champion"]
                or current_job["checkpoint"] != job["checkpoint"] or current_job["status"] != "COMPLETED"
                or current_job["result"] != result or current["autonomy_policy"]["version"] != request["policy_version"]):
            raise GateError("STALE_RELEASE_CONFIRMATION")
        old = current.get("agent_release_grant") or {}
        if old.get("request_id") == request["request_id"] and old.get("expires_at", -1) >= current["now"]:
            return old
        current["agent_release_grant"] = {"run_id": run_id, "candidate_id": accepted["candidate_id"],
            "request_id": request["request_id"], "model_ref": result["artifact"], "environment": config.mode,
            "policy_version": request["policy_version"], "rollback_target": current["champion"],
            "plan_version": job["checkpoint"]["plan_version"], "epoch": binding["epoch"], "champion": current["champion"],
            "authorized_by": actor, "expires_at": current["now"] + int(config.release.maximum_observation_seconds * SECOND)}
        db.execute("UPDATE tasks SET state=? WHERE task_key=?", (canonical(current), repo.key))
        repo.put("approval-" + request["request_id"] + "-" + str(current["now"]), "release_approval",
                 {**current["agent_release_grant"], "wall_ns": time.time_ns()}, current["now"], db=db)
        repo.store._event(db, repo.key, "AGENT_RELEASE_HUMAN_GRANTED", current["agent_release_grant"], current["now"])
    return current["agent_release_grant"]


def fallback(repo, now, code):
    """Only an explicitly preauthorized, quiescent fallback transfers authority."""
    with repo.store.transaction() as db:
        state = json.loads(db.execute("SELECT state FROM tasks WHERE task_key=?", (repo.key,)).fetchone()[0])
        if state.get("decision_owner") != "agent" or state["autonomy_policy"]["failure_policy"] != "fallback_workflow":
            raise GateError("FALLBACK_NOT_AUTHORIZED")
        busy = db.execute("SELECT 1 FROM jobs WHERE task_key=? AND status IN ('PENDING','RUNNING')", (repo.key,)).fetchone()
        prepared = db.execute("SELECT 1 FROM agent_runtime_records WHERE task_key=? AND kind='decision' AND json_extract(payload,'$.decision_outcome')='prepared'", (repo.key,)).fetchone()
        if busy or prepared or state["challenger"]:
            repo.store._event(db, repo.key, "AGENT_FALLBACK_BLOCKED", {"reason": "active_work_requires_human", "code": code}, now)
            return
        state.update(decision_owner="workflow", decision_epoch=state["decision_epoch"] + 1)
        db.execute("UPDATE tasks SET state=? WHERE task_key=?", (canonical(state), repo.key))
        repo.store._event(db, repo.key, "AGENT_CONTROL_FALLBACK_TO_WORKFLOW", {"reason": code, "at": now}, now)
