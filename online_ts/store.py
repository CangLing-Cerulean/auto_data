from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager, closing
from pathlib import Path
from typing import Callable

from .config import GateError, TaskConfig, canonical, digest


class Store:
    """All lifecycle state and externally visible predictions share one transaction domain."""

    SCHEMA_VERSION = 9

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if self.path.exists() and self.path.stat().st_size:
            # Refuse unsupported versions before WAL setup, DDL, or migration.
            # Old runtimes remain intact and can be reopened by their old package.
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as existing:
                tables = {row[0] for row in existing.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if tables:
                    version = (existing.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0]
                               if "schema_migrations" in tables else 0)
                    if version != self.SCHEMA_VERSION:
                        raise GateError("UNSUPPORTED_UPGRADE_REQUIRES_OLD_RUNTIME")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY, applied_ns INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks (
                    task_key TEXT PRIMARY KEY, task_id TEXT NOT NULL, config TEXT NOT NULL,
                    state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, task_key TEXT NOT NULL, event_key TEXT NOT NULL,
                    event_time INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(task_key,event_key));
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, artifact TEXT NOT NULL,
                    report TEXT NOT NULL, created_ns INTEGER NOT NULL, run_id TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    run_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, request_key TEXT UNIQUE NOT NULL,
                    cutoff INTEGER NOT NULL, snapshot TEXT NOT NULL, plan TEXT NOT NULL,
                    status TEXT NOT NULL, checkpoint TEXT NOT NULL, spent REAL NOT NULL DEFAULT 0,
                    owner TEXT, lease_until REAL, reservation REAL NOT NULL DEFAULT 0,
                    started_wall REAL, result TEXT, reason TEXT NOT NULL);
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_training ON jobs(task_key)
                    WHERE status IN ('PENDING','RUNNING');
                CREATE TABLE IF NOT EXISTS predictions (
                    task_key TEXT NOT NULL, model_id TEXT NOT NULL, issue_ns INTEGER NOT NULL,
                    target_ns INTEGER NOT NULL, variable TEXT NOT NULL, horizon INTEGER NOT NULL,
                    value REAL NOT NULL, role TEXT NOT NULL, published INTEGER NOT NULL,
                    inference_ms REAL NOT NULL, quality TEXT NOT NULL,
                    PRIMARY KEY(task_key,model_id,issue_ns,target_ns,variable,horizon));
                CREATE TABLE IF NOT EXISTS labels (
                    task_key TEXT NOT NULL, target_ns INTEGER NOT NULL, variable TEXT NOT NULL,
                    value REAL NOT NULL, observed_ns INTEGER NOT NULL,
                    PRIMARY KEY(task_key,target_ns,variable));
                CREATE TABLE IF NOT EXISTS monitoring (
                    task_key TEXT NOT NULL, model_id TEXT NOT NULL, evaluated_ns INTEGER NOT NULL,
                    evidence TEXT NOT NULL, PRIMARY KEY(task_key,model_id,evaluated_ns));
                CREATE TABLE IF NOT EXISTS monitoring_windows (
                    task_key TEXT NOT NULL, model_id TEXT NOT NULL,
                    window_start INTEGER NOT NULL, window_end INTEGER NOT NULL,
                    label_state TEXT NOT NULL, label_fingerprint TEXT NOT NULL,
                    evaluated_ns INTEGER NOT NULL, evidence TEXT NOT NULL,
                    triggered INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(task_key,model_id,window_start,window_end));
                CREATE TABLE IF NOT EXISTS leases (
                    task_key TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS job_dispositions (
                    run_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, outcome TEXT NOT NULL,
                    payload TEXT NOT NULL, disposed_ns INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS label_revisions (
                    task_key TEXT NOT NULL, target_ns INTEGER NOT NULL, variable TEXT NOT NULL,
                    revision INTEGER NOT NULL, value REAL, observed_ns INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    PRIMARY KEY(task_key,target_ns,variable,revision));
                CREATE INDEX IF NOT EXISTS label_revisions_asof
                    ON label_revisions(task_key,target_ns,variable,observed_ns,revision);
                CREATE TABLE IF NOT EXISTS proposal_history (
                    task_key TEXT NOT NULL, fingerprint TEXT NOT NULL, run_id TEXT NOT NULL,
                    created_ns INTEGER NOT NULL, outcome TEXT NOT NULL,
                    PRIMARY KEY(task_key,fingerprint,run_id));
                CREATE TABLE IF NOT EXISTS resource_leases (
                    resource TEXT PRIMARY KEY, owner TEXT NOT NULL, task_key TEXT NOT NULL,
                    run_id TEXT NOT NULL, role TEXT NOT NULL, expires REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS feedback_messages (
                    message_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, run_id TEXT NOT NULL,
                    kind TEXT NOT NULL, plan_version INTEGER NOT NULL,
                    created_ns INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS feedback_messages_run
                    ON feedback_messages(task_key,run_id,created_ns,message_id);
                CREATE TABLE IF NOT EXISTS model_reviews (
                    review_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL, stage TEXT NOT NULL, evidence_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL, created_ns INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS reviews_scope ON model_reviews(task_key,run_id,candidate_id);
                CREATE TABLE IF NOT EXISTS agent_calls (
                    operation_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, run_id TEXT NOT NULL,
                    role TEXT NOT NULL, status TEXT NOT NULL, reserved_tokens INTEGER NOT NULL,
                    actual_tokens INTEGER, created_ns INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS human_sessions (
                    session_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, actor TEXT NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS human_intents (
                    intent_id TEXT PRIMARY KEY, task_key TEXT NOT NULL, session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, status TEXT NOT NULL, created_ns INTEGER NOT NULL,
                    payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS test_exposures (
                    source_id TEXT NOT NULL, test_start INTEGER NOT NULL, test_end INTEGER NOT NULL,
                    actor TEXT NOT NULL, review_id TEXT NOT NULL, created_ns INTEGER NOT NULL,
                    PRIMARY KEY(source_id,review_id,actor));
            """)
            # v9 supports explicit label withdrawal using a null revision. Old
            # databases remain untouched and require their frozen old package.
            db.execute("INSERT OR IGNORE INTO schema_migrations VALUES(?,?)",
                       (self.SCHEMA_VERSION, time.time_ns()))

    def schema_version(self) -> int:
        with self.connection() as db:
            return int(db.execute("SELECT COALESCE(MAX(version),0) FROM schema_migrations").fetchone()[0])

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    @staticmethod
    def task_key(config: TaskConfig) -> str:
        return f"{config.task_id}@{config.version[:16]}"

    @staticmethod
    def _event(db, key: str, kind: str, payload: dict, now: int, event_key: str | None = None):
        db.execute("INSERT OR IGNORE INTO events(task_key,event_key,event_time,kind,payload) VALUES(?,?,?,?,?)",
                   (key, event_key or uuid.uuid4().hex, now, kind, canonical(payload)))

    def create_task(self, config: TaskConfig, source_ref: dict | None = None) -> str:
        key = self.task_key(config)
        state = {"status": "CREATED", "champion": None, "previous": None, "challenger": None,
                 "stage": None, "stage_since": None, "approved": False, "now": None,
                 "last_issue": None, "last_check": None, "last_train_cutoff": None, "cooldown_until": 0,
                 "monitor": {}, "source_ref": source_ref, "stop_requested": False, "model_since": None,
                 "observation_until": None, "observation_completed_for": None,
                 "last_consumed_job": None}
        with self.transaction() as db:
            existing = db.execute("SELECT config FROM tasks WHERE task_key=?", (key,)).fetchone()
            if existing:
                if existing["config"] != canonical(config.to_dict()):
                    raise GateError("TASK_VERSION_COLLISION")
                return key
            db.execute("INSERT INTO tasks(task_key,task_id,config,state) VALUES(?,?,?,?)", (key, config.task_id, canonical(config.to_dict()), canonical(state)))
            self._event(db, key, "CREATED", {"config_version": config.version}, 0)
        return key

    def resolve(self, identifier: str) -> str:
        with self.connection() as db:
            rows = db.execute("SELECT task_key FROM tasks WHERE task_key=? OR task_id=?", (identifier, identifier)).fetchall()
        if len(rows) != 1:
            raise GateError("TASK_NOT_FOUND_OR_VERSION_REQUIRED")
        return str(rows[0]["task_key"])

    def task(self, key: str) -> tuple[TaskConfig, dict]:
        with self.connection() as db:
            row = db.execute("SELECT config,state FROM tasks WHERE task_key=?", (key,)).fetchone()
        if row is None:
            raise GateError("TASK_NOT_FOUND")
        return TaskConfig.from_dict(json.loads(row["config"])), json.loads(row["state"])

    def change(self, key: str, update: Callable[[dict], None], kind: str | None = None, payload: dict | None = None, now: int = 0):
        with self.transaction() as db:
            row = db.execute("SELECT state FROM tasks WHERE task_key=?", (key,)).fetchone()
            if row is None:
                raise GateError("TASK_NOT_FOUND")
            state = json.loads(row["state"])
            previous_roles = (state.get("champion"), state.get("challenger"), state.get("previous"), state.get("stage"))
            update(state)
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), key))
            if kind:
                self._event(db, key, kind, payload or {}, now)
            if previous_roles != (state.get("champion"), state.get("challenger"), state.get("previous"), state.get("stage")):
                self._persist_deployment_cards(db, key, state)
            return state

    def event(self, key: str, kind: str, payload: dict, now: int, event_key: str | None = None):
        with self.transaction() as db:
            self._event(db, key, kind, payload, now, event_key)

    def events(self, key: str) -> list[dict]:
        with self.connection() as db:
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in db.execute("SELECT * FROM events WHERE task_key=? ORDER BY id", (key,))]

    def acquire(self, key: str, owner: str, seconds: float = 30) -> bool:
        now = time.time()
        with self.transaction() as db:
            row = db.execute("SELECT * FROM leases WHERE task_key=?", (key,)).fetchone()
            if row and row["owner"] != owner and row["expires"] > now:
                return False
            db.execute("INSERT INTO leases VALUES(?,?,?) ON CONFLICT(task_key) DO UPDATE SET owner=excluded.owner,expires=excluded.expires", (key, owner, now + seconds))
            return True

    def release(self, key: str, owner: str):
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE task_key=? AND owner=?", (key, owner))

    def acquire_resource(self, resource: str, owner: str, key: str, run_id: str,
                         role: str, seconds: float) -> bool:
        allowed = ((resource == "gpu-training" and role == "training")
                   or (resource == "model-review:" + key and role == "model-review")
                   or (resource == "llm-call" and role in {"planning", "feedback", "review", "question", "intent",
                                                         "data_analyst", "feature_designer", "release_reporter"}))
        if not allowed:
            raise GateError("RESOURCE_LEASE_SCOPE")
        now = time.time()
        with self.transaction() as db:
            row = db.execute("SELECT owner,expires FROM resource_leases WHERE resource=?", (resource,)).fetchone()
            if row and row["owner"] != owner and row["expires"] > now:
                return False
            db.execute("""INSERT INTO resource_leases VALUES(?,?,?,?,?,?)
                ON CONFLICT(resource) DO UPDATE SET owner=excluded.owner,task_key=excluded.task_key,
                run_id=excluded.run_id,role=excluded.role,expires=excluded.expires""",
                (resource, owner, key, run_id, role, now + seconds))
            self._event(db, key, "RESOURCE_LEASE_ACQUIRED",
                        {"resource": resource, "run_id": run_id, "role": role},
                        time.time_ns(), f"resource-acquired:{run_id}:{owner}")
            return True

    def release_resource(self, resource: str, owner: str) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM resource_leases WHERE resource=? AND owner=?", (resource, owner))

    def resource_status(self) -> list[dict]:
        now = time.time()
        with self.connection() as db:
            rows = db.execute("SELECT * FROM resource_leases ORDER BY resource").fetchall()
        return [{**dict(row), "active": row["expires"] > now,
                 "remaining_seconds": max(0.0, float(row["expires"]) - now)} for row in rows]

    def request_job(self, key: str, cutoff: int, snapshot: dict, proposal: dict, reason: str, now: int,
                    *, agent_epoch: int | None = None, experiment_id: str | None = None) -> str | None:
        request_key = digest({"task": key, "cutoff": cutoff} if experiment_id is None else
                             {"task": key, "experiment_id": experiment_id})
        run_id = "train-" + request_key[:24]
        with self.transaction() as db:
            row = db.execute("SELECT state FROM tasks WHERE task_key=?", (key,)).fetchone()
            state = json.loads(row["state"])
            if agent_epoch is None:
                if state.get("decision_owner", "workflow") != "workflow":
                    raise GateError("DECISION_OWNER_MISMATCH")
            elif (state.get("decision_owner") != "agent" or state.get("decision_epoch") != agent_epoch
                  or not experiment_id or not proposal.get("agent_managed")):
                raise GateError("DECISION_OWNER_MISMATCH")
            existing = db.execute("SELECT * FROM jobs WHERE request_key=?", (request_key,)).fetchone()
            if existing and experiment_id:
                if existing["plan"] != canonical(proposal) or existing["snapshot"] != canonical(snapshot):
                    raise GateError("AGENT_JOB_IDEMPOTENCY_CONFLICT")
                return existing["run_id"]
            # Both controllers honor exposure of online evidence for development.
            # Ordinary Workflow jobs take this branch only when such evidence exists.
            source_row = db.execute("SELECT config FROM tasks WHERE task_key=?", (key,)).fetchone()
            task_config = json.loads(source_row[0])
            exposures = db.execute("SELECT test_start,test_end FROM test_exposures WHERE source_id=? AND actor='agent-development'",
                                   (digest(task_config["source"]),)).fetchall()
            if exposures:
                from .data import ArtifactStore
                meta = ArtifactStore(self.path.parent / "artifacts").read_json(snapshot)
                test_start, test_end = meta["times"][-task_config["search"]["test_steps"]], meta["times"][-1]
                if any(e["test_start"] <= test_end and e["test_end"] >= test_start for e in exposures):
                    raise GateError("FINAL_TEST_INDEPENDENT_EVALUATION_REQUIRED")
            if state["stop_requested"] or state["challenger"] or now < state["cooldown_until"]:
                return None
            if db.execute("SELECT 1 FROM jobs WHERE task_key=? AND status IN ('PENDING','RUNNING')", (key,)).fetchone():
                return None
            if db.execute("SELECT 1 FROM jobs WHERE request_key=?", (request_key,)).fetchone():
                return None
            checkpoint = {"schema_version": 2, "plan_id": "plan-" + request_key[:24],
                          "plan_version": 1, "index": 0, "reports": [], "selected": None, "attempts": {},
                          "infrastructure_retries": 0, "repair_rounds": 0,
                          "gate_issues": [], "repair_proposals": [], "progress_digests": [],
                          "best_so_far": None, "terminal_reason": None,
                          "llm_tokens_used": 0, "feedback_calls": 0,
                          "non_improvement_count": 0, "outcomes": [],
                          "agent_contexts": [], "agent_proposals": [],
                          "feedback_pending": None, "pending_agent_response": None,
                          "attempt_history": []}
            db.execute("INSERT INTO jobs(run_id,task_key,request_key,cutoff,snapshot,plan,status,checkpoint,reason) VALUES(?,?,?,?,?,?,'PENDING',?,?)",
                       (run_id, key, request_key, cutoff, canonical(snapshot), canonical(proposal), canonical(checkpoint), reason))
            fingerprint = proposal.get("proposal_fingerprint")
            if fingerprint:
                db.execute("INSERT OR IGNORE INTO proposal_history VALUES(?,?,?,?,?)",
                           (key, fingerprint, run_id, now, "accepted"))
            state["status"] = "RETRAINING" if state["champion"] else "INITIAL_TRAINING"
            state["last_train_cutoff"] = cutoff
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), key))
            self._event(db, key, "TRAINING_REQUESTED", {"run_id": run_id, "reason": reason, "cutoff": cutoff, "snapshot": snapshot, "proposal": proposal}, now, run_id)
        return run_id

    def proposal_fingerprints(self, key: str) -> set[str]:
        with self.connection() as db:
            return {str(row[0]) for row in db.execute("SELECT fingerprint FROM proposal_history WHERE task_key=?", (key,))}

    def record_planner_attempt(self, run_id: str, owner: str, tokens: int, fingerprint: str,
                               repair: dict | None = None, proposal: dict | None = None,
                               initial_proposal: dict | None = None) -> None:
        """Charge an external planner response before its proposal is trusted."""
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "RUNNING" or job["owner"] != owner:
                raise GateError("STALE_WORKER")
            cp = job["checkpoint"]
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise GateError("PLANNER_TOKEN_BUDGET")
            parsed = None
            if proposal is not None:
                from .remediation import FeedbackProposal
                parsed = FeedbackProposal.from_dict(proposal).to_dict()
                if parsed["run_id"] != run_id:
                    raise GateError("FEEDBACK_PROPOSAL_SCOPE")
                if any(item.get("message_id") == parsed["message_id"]
                       for item in cp.get("agent_proposals", ())):
                    raise GateError("DUPLICATE_FEEDBACK_PROPOSAL")
            used = int(cp.get("llm_tokens_used", 0)) + tokens
            config_row = db.execute("SELECT config FROM tasks WHERE task_key=?", (job["task_key"],)).fetchone()
            config = TaskConfig.from_dict(json.loads(config_row["config"]))
            # Actual service usage is never discarded, including a service-side
            # overrun. Admission fails separately; the account remains truthful.
            if used > config.budget.max_llm_tokens:
                cp["llm_budget_overrun"] = True
            cp["llm_tokens_used"] = used
            cp["llm_reservation"] = 0
            if cp.get("llm_call_id"):
                row = db.execute("SELECT payload FROM agent_calls WHERE operation_id=?", (cp["llm_call_id"],)).fetchone()
                if row:
                    trace = json.loads(row[0])
                    db.execute("UPDATE agent_calls SET status=?,actual_tokens=? WHERE operation_id=?",
                               ("failed" if trace.get("error_code") else "completed", tokens, cp["llm_call_id"]))
            category = "repair" if self._stage(job) == "feedback" or cp.get("repair_rounds", 0) else "planning"
            usage = cp.setdefault("llm_usage_by_role", {})
            bucket = usage.setdefault(category, {"tokens": 0, "calls": 0, "uncertain_reserved_tokens": 0})
            bucket["tokens"] += tokens
            bucket["calls"] += 1
            cp["last_proposal_fingerprint"] = fingerprint
            if self._stage(job) == "feedback":
                cp["feedback_calls"] = int(cp.get("feedback_calls", 0)) + 1
            if repair is not None:
                cp.setdefault("repair_proposals", []).append(repair)
            if parsed is not None:
                cp.setdefault("agent_proposals", []).append(parsed)
                cp["pending_agent_response"] = parsed
                self._persist_feedback_messages(db, job, cp)
            if initial_proposal is not None:
                cp["pending_initial_proposal"] = initial_proposal
            db.execute("UPDATE jobs SET checkpoint=? WHERE run_id=?", (canonical(cp), run_id))
            kind = "REPAIR_PROPOSAL" if repair is not None else "PLANNER_ATTEMPT"
            self._event(db, job["task_key"], kind,
                        {"run_id": run_id, "tokens": tokens, "total_tokens": used,
                         "proposal_fingerprint": fingerprint, "repair": repair},
                        job["cutoff"])
        if used > config.budget.max_llm_tokens:
            raise GateError("PLANNER_TOKEN_BUDGET")

    def record_context_material(self, run_id, owner, profile, profile_ref, history):
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "RUNNING" or job["owner"] != owner:
                raise GateError("STALE_WORKER")
            cp = job["checkpoint"]
            cp.update(planner_profile=profile, planner_profile_ref=profile_ref, trial_history=history)
            db.execute("UPDATE jobs SET checkpoint=? WHERE run_id=?", (canonical(cp), run_id))

    def reserve_llm_call(self, run_id, owner, trace):
        value = trace.data
        amount = value["prompt_reserved_tokens"] + value["output_limit"]
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "RUNNING" or job["owner"] != owner:
                raise GateError("STALE_WORKER")
            cp = job["checkpoint"]
            config = TaskConfig.from_dict(json.loads(db.execute("SELECT config FROM tasks WHERE task_key=?", (job["task_key"],)).fetchone()[0]))
            if amount + cp.get("llm_tokens_used", 0) > config.budget.max_llm_tokens:
                raise GateError("PLANNER_TOKEN_BUDGET")
            if db.execute("SELECT 1 FROM agent_calls WHERE operation_id=?", (trace.call_id,)).fetchone():
                raise GateError("AGENT_CALL_ALREADY_RESERVED")
            cp.update(llm_reservation=amount, llm_call_id=trace.call_id)
            db.execute("INSERT INTO agent_calls VALUES(?,?,?,?,?,?,?,?,?)",
                       (trace.call_id, job["task_key"], run_id, trace.role, "reserved", amount, None,
                        value["started_ns"], canonical(value)))
            db.execute("UPDATE jobs SET checkpoint=? WHERE run_id=?", (canonical(cp), run_id))

    def record_llm_trace(self, run_id, owner, value):
        with self.transaction() as db:
            job = db.execute("SELECT status,owner FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if job is None or job["status"] != "RUNNING" or job["owner"] != owner:
                return
            db.execute("UPDATE agent_calls SET payload=? WHERE operation_id=? AND run_id=? AND status='reserved'",
                       (canonical(value), value["call_id"], run_id))

    def llm_traces(self, key, run_id=None):
        with self.connection() as db:
            rows = db.execute("SELECT * FROM agent_calls WHERE task_key=? AND role IN ('planning','feedback') AND (? IS NULL OR run_id=?) ORDER BY created_ns,operation_id",
                              (key, run_id, run_id)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    @staticmethod
    def _normalize_checkpoint(checkpoint: dict, run_id: str, request_key: str) -> dict:
        if not isinstance(checkpoint, dict):
            raise GateError("CHECKPOINT_SCHEMA")
        value = dict(checkpoint)
        value.setdefault("schema_version", 2)
        if value["schema_version"] not in {1, 2}:
            raise GateError("CHECKPOINT_SCHEMA")
        value["schema_version"] = 2
        value.setdefault("plan_id", "plan-" + request_key[:24])
        value.setdefault("plan_version", 1)
        defaults = {
            "index": 0, "reports": [], "selected": None, "attempts": {},
            "infrastructure_retries": int(value.pop("repairs", 0)), "repair_rounds": 0,
            "gate_issues": [], "repair_proposals": [], "progress_digests": [],
            "best_so_far": None, "terminal_reason": None, "llm_tokens_used": 0,
            "feedback_calls": 0, "non_improvement_count": 0, "outcomes": [],
            "agent_contexts": [], "agent_proposals": [], "feedback_pending": None,
            "pending_agent_response": None, "attempt_history": [],
        }
        for name, default in defaults.items():
            value.setdefault(name, default)
        if (not isinstance(value["plan_version"], int) or value["plan_version"] < 1
                or not isinstance(value["attempts"], dict)
                or any(not isinstance(value[name], list) for name in (
                    "reports", "gate_issues", "repair_proposals", "progress_digests",
                    "outcomes", "agent_contexts", "agent_proposals", "attempt_history"))):
            raise GateError("CHECKPOINT_SCHEMA")
        return value

    @classmethod
    def _job(cls, row) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        for name in ("snapshot", "plan", "checkpoint", "result"):
            if result[name] is not None:
                result[name] = json.loads(result[name])
        result["checkpoint"] = cls._normalize_checkpoint(
            result["checkpoint"], result["run_id"], result["request_key"])
        cp = result["checkpoint"]
        if cp["selected"] is not None and "selection_binding" not in cp and cp.get("feedback_pending") is None:
            from .models import Candidate
            expected = [Candidate.from_dict(c).candidate_id for c in result["plan"].get("candidates", [])]
            actual = [Candidate.from_dict(r["candidate"]).candidate_id for r in cp["reports"]]
            if expected == actual and cp["index"] == len(expected):
                cp["selection_binding"] = digest({"plan_version": cp["plan_version"],
                                                   "candidates": expected, "reports": cp["reports"]})
        return result

    def job(self, run_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone()
        result = self._job(row)
        if result is None:
            raise GateError("JOB_NOT_FOUND")
        return result

    def jobs(self, key: str) -> list[dict]:
        with self.connection() as db:
            return [self._job(row) for row in db.execute("SELECT * FROM jobs WHERE task_key=? ORDER BY cutoff,run_id", (key,))]

    def terminal_jobs_without_disposition(self, key: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT j.* FROM jobs j
                LEFT JOIN job_dispositions d ON d.run_id=j.run_id
                WHERE j.task_key=? AND j.status IN ('COMPLETED','FAILED','CANCELLED')
                  AND d.run_id IS NULL ORDER BY j.cutoff,j.run_id""", (key,)).fetchall()
        return [self._job(row) for row in rows]

    def job_dispositions(self, key: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT d.* FROM job_dispositions d JOIN jobs j ON j.run_id=d.run_id
                WHERE d.task_key=? ORDER BY j.cutoff,j.run_id""", (key,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def feedback_messages(self, run_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM feedback_messages WHERE run_id=? AND kind IN ('outcome','agent_context','agent_proposal') ORDER BY created_ns,message_id",
                (run_id,)).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    @staticmethod
    def _stage(job: dict) -> str:
        cp = job["checkpoint"]
        if job["plan"].get("needs_proposal"):
            return "proposal"
        if cp.get("feedback_pending") is not None:
            return "feedback"
        return "final" if cp["selected"] else str(cp["index"])

    @classmethod
    def _persist_feedback_messages(cls, db, job: dict, checkpoint: dict) -> None:
        collections = (("outcome", "outcomes"), ("agent_context", "agent_contexts"),
                       ("agent_proposal", "agent_proposals"))
        for kind, name in collections:
            for payload in checkpoint.get(name, []):
                message_id = payload.get("message_id")
                if not message_id:
                    raise GateError("FEEDBACK_MESSAGE_ID")
                expected = canonical(payload)
                existing = db.execute(
                    "SELECT kind,payload FROM feedback_messages WHERE message_id=?", (message_id,)).fetchone()
                if existing and (existing["kind"] != kind or existing["payload"] != expected):
                    raise GateError("FEEDBACK_MESSAGE_CONFLICT")
                db.execute("INSERT OR IGNORE INTO feedback_messages VALUES(?,?,?,?,?,?,?)",
                           (message_id, job["task_key"], job["run_id"], kind,
                            int(payload.get("plan_version", payload.get("base_plan_version", 1))),
                            time.time_ns(), expected))
        from .memory import persist_observations
        persist_observations(db, job, checkpoint)

    def dispose_job(self, run_id: str, outcome: str, payload: dict, now: int,
                    update: Callable[[dict], None]) -> None:
        if outcome not in {"registered", "rejected", "cancelled", "failed"}:
            raise GateError("JOB_DISPOSITION_OUTCOME")
        with self.transaction() as db:
            job = db.execute("SELECT task_key,status FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if job is None or job["status"] not in {"COMPLETED", "FAILED", "CANCELLED"}:
                raise GateError("JOB_NOT_TERMINAL")
            expected = canonical(payload)
            existing = db.execute("SELECT outcome,payload FROM job_dispositions WHERE run_id=?", (run_id,)).fetchone()
            if existing:
                if existing["outcome"] != outcome or existing["payload"] != expected:
                    raise GateError("JOB_DISPOSITION_CONFLICT")
                return
            row = db.execute("SELECT state FROM tasks WHERE task_key=?", (job["task_key"],)).fetchone()
            state = json.loads(row["state"])
            update(state)
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), job["task_key"]))
            self._persist_deployment_cards(db, job["task_key"], state)
            db.execute("INSERT INTO job_dispositions VALUES(?,?,?,?,?)", (run_id, job["task_key"], outcome, expected, now))
            kind = {"registered": "TRAINING_REGISTERED", "rejected": "TRAINING_REJECTED",
                    "cancelled": "TRAINING_CANCELLED", "failed": "TRAINING_FAILED"}[outcome]
            self._event(db, job["task_key"], kind, {"run_id": run_id, **payload}, now, "dispose:" + run_id)

    def active_job(self, key: str) -> dict | None:
        with self.connection() as db:
            return self._job(db.execute("SELECT * FROM jobs WHERE task_key=? AND status IN ('PENDING','RUNNING')", (key,)).fetchone())

    def reserve_stage(self, run_id: str, owner: str, config: TaskConfig) -> dict | None:
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "PENDING":
                return None
            cp = job["checkpoint"]
            stage = self._stage(job)
            if cp.get("human_pending"):
                return None
            attempts = cp["attempts"].get(stage, 0)
            from .context import training_seconds_spent
            is_planner = stage in {"proposal", "feedback"}
            remaining = (config.planner.timeout_seconds + 10 if is_planner else
                         config.budget.max_training_seconds - training_seconds_spent(job))
            attempt_limit = (config.planner.max_format_repairs + 1 if stage == "proposal" else
                             max(1, config.budget.max_repairs) + config.planner.max_format_repairs
                             if stage == "feedback" else config.budget.max_attempts)
            if remaining <= 0.01 or attempts >= attempt_limit:
                cp["terminal_reason"] = "TRAINING_BUDGET_EXHAUSTED"
                from .reviews import persist_job_cards
                persist_job_cards(db, job, cp, None, "FAILED")
                db.execute("UPDATE jobs SET status='FAILED',checkpoint=?,result=? WHERE run_id=?", (canonical(cp), canonical({"code": "TRAINING_BUDGET_EXHAUSTED"}), run_id))
                return None
            cp["attempts"][stage] = attempts + 1
            if stage in {"proposal", "feedback"} and cp.get("llm_tokens_used", 0) >= config.budget.max_llm_tokens:
                cp["terminal_reason"] = "PLANNER_TOKEN_BUDGET"
                from .reviews import persist_job_cards
                persist_job_cards(db, job, cp, None, "FAILED")
                db.execute("UPDATE jobs SET status='FAILED',checkpoint=?,result=? WHERE run_id=?",
                           (canonical(cp), canonical({"code": "PLANNER_TOKEN_BUDGET"}), run_id))
                return None
            reservation = remaining if is_planner else min(config.budget.max_candidate_seconds, remaining)
            if is_planner and not cp.get("pending_agent_response") and not cp.get("pending_initial_proposal"):
                busy = db.execute("SELECT owner,expires FROM resource_leases WHERE resource='llm-call'").fetchone()
                if busy and busy["owner"] != owner and busy["expires"] > time.time():
                    return None
                db.execute("INSERT INTO resource_leases VALUES(?,?,?,?,?,?) ON CONFLICT(resource) DO UPDATE SET owner=excluded.owner,task_key=excluded.task_key,run_id=excluded.run_id,role=excluded.role,expires=excluded.expires",
                           ("llm-call", owner, job["task_key"], run_id, "feedback" if stage == "feedback" else "planning", time.time() + reservation + 5))
            if stage.isdigit() and int(stage) < len(job["plan"].get("candidates", [])):
                from .models import Candidate
                current_id = Candidate.from_dict(job["plan"]["candidates"][int(stage)]).candidate_id
                specs = job["plan"].get("experiment_history", []) + job["plan"].get("experiment_specs", [])
                for spec in specs:
                    if Candidate.from_dict(spec["candidate"]).candidate_id == current_id:
                        reservation = min(reservation, spec["resources"]["seconds"])
            now = time.time()
            db.execute("UPDATE jobs SET status='RUNNING',checkpoint=?,spent=spent+?,reservation=?,owner=?,started_wall=?,lease_until=? WHERE run_id=?",
                       (canonical(cp), reservation, reservation, owner, now, now + reservation + 5, run_id))
            self._event(db, job["task_key"], "TRAINING_STAGE_STARTED", {"run_id": run_id, "stage": stage, "attempt": attempts + 1, "reserved_seconds": reservation,
                                                                                 "training_target": job["plan"].get("training_target")}, job["cutoff"])
        return self.job(run_id)

    def complete_stage(self, run_id: str, owner: str, checkpoint: dict, result: dict | None, elapsed: float, plan: dict | None = None):
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "RUNNING" or job["owner"] != owner:
                raise GateError("STALE_WORKER")
            if job["lease_until"] is not None and time.time() > job["lease_until"]:
                raise GateError("WORKER_LEASE_EXPIRED")
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed < 0 or elapsed > job["reservation"]):
                raise GateError("TRAINING_DEADLINE")
            refund = max(0.0, job["reservation"] - elapsed)
            role = "training" if self._stage(job) not in {"proposal", "feedback"} else "planning_and_repair"
            times = checkpoint.setdefault("seconds_by_role", {})
            times[role] = times.get(role, 0.0) + elapsed
            if plan is not None:
                checkpoint["selected"] = None
                checkpoint.pop("selection_binding", None)
            status = "COMPLETED" if result is not None else "PENDING"
            eligible = [report for report in checkpoint.get("reports", []) if report.get("eligible")]
            if eligible:
                best = min(eligible, key=lambda report: (report.get("score", float("inf")), report.get("candidate_id", "")))
                checkpoint["best_so_far"] = {"candidate_id": best.get("candidate_id"), "score": best.get("score")}
            self._persist_feedback_messages(db, {**job, "plan": plan or job["plan"]}, checkpoint)
            from .reviews import persist_job_cards
            persist_job_cards(db, {**job, "plan": plan or job["plan"]}, checkpoint, result, status)
            db.execute("UPDATE jobs SET status=?,checkpoint=?,result=?,spent=MAX(0,spent-?),owner=NULL,lease_until=NULL,reservation=0 WHERE run_id=?",
                       (status, canonical(checkpoint), canonical(result) if result is not None else None, refund, run_id))
            db.execute("DELETE FROM resource_leases WHERE resource='llm-call' AND owner=?", (owner,))
            if plan is not None:
                db.execute("UPDATE jobs SET plan=? WHERE run_id=?", (canonical(plan), run_id))
                fingerprint = plan.get("proposal_fingerprint")
                if fingerprint:
                    db.execute("INSERT OR IGNORE INTO proposal_history VALUES(?,?,?,?,?)",
                               (job["task_key"], fingerprint, run_id, job["cutoff"], "accepted"))
                decision = {"run_id": run_id, "candidates": plan.get("candidates"),
                            "training_target": plan.get("training_target"), "routing_reason": plan.get("routing_reason"),
                            "routing_evidence": plan.get("routing_evidence"), "planner_tokens": plan.get("planner_tokens")}
                plan_version = int(plan.get("plan_version", checkpoint.get("plan_version", 1)))
                self._event(db, job["task_key"], "PLANNER_DECISION", decision, job["cutoff"],
                            f"planner:{run_id}:{plan_version}")
                if plan.get("parent_plan_id"):
                    self._event(db, job["task_key"], "PLAN_REVISED",
                                {"run_id": run_id, "plan_id": checkpoint["plan_id"],
                                 "plan_version": plan_version,
                                 "parent_plan_version": plan.get("parent_plan_version"),
                                 "feedback_id": plan.get("feedback_id"),
                                 "proposal_id": plan.get("proposal_id"),
                                 "proposal_fingerprint": plan.get("proposal_fingerprint")},
                                job["cutoff"], f"plan-revised:{run_id}:{plan_version}")
            for kind, name in (("WORKFLOW_OUTCOME", "outcomes"),
                               ("AGENT_CONTEXT_CREATED", "agent_contexts"),
                               ("AGENT_PROPOSAL_RECEIVED", "agent_proposals")):
                for payload in checkpoint.get(name, []):
                    self._event(db, job["task_key"], kind,
                                {"run_id": run_id, "message_id": payload["message_id"],
                                 "plan_version": payload.get("plan_version", payload.get("base_plan_version"))},
                                job["cutoff"], kind.lower() + ":" + payload["message_id"])
            completed = {"run_id": run_id, "status": status, "elapsed": elapsed,
                         "training_target": (plan or job["plan"]).get("training_target")}
            if plan is not None:
                completed["planner_decision"] = decision
            self._event(db, job["task_key"], "TRAINING_STAGE_COMPLETED", completed, job["cutoff"])

    def complete_waiting_evaluation(self, run_id: str, result: dict, now: int):
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            previous = job["result"] or {}
            if not previous.get("waiting_for_labels"):
                return
            if (result.get("waiting_for_labels") or previous.get("frozen_predictions") != result.get("frozen_predictions")
                    or previous.get("artifact") != result.get("artifact")):
                raise GateError("FROZEN_EVALUATION_SCOPE")
            db.execute("UPDATE jobs SET result=? WHERE run_id=?", (canonical(result), run_id))
            self._event(db, job["task_key"], "FINAL_LABELS_EVALUATED",
                        {"run_id": run_id, "evaluation": result["evaluation"]}, now,
                        "final-labels:" + run_id)

    def fail_stage(self, run_id: str, owner: str, code: str, config: TaskConfig, *, retryable: bool, elapsed=None):
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "RUNNING" or job["owner"] != owner:
                return
            cp = job["checkpoint"]
            stage = self._stage(job)
            if code == "NO_PROGRESS":
                cp["previous_proposal_fingerprint"] = cp.get("last_proposal_fingerprint")
            self._settle_open_llm_reservation(db, job, "failed")
            if cp["llm_tokens_used"] > config.budget.max_llm_tokens:
                cp["llm_budget_overrun"] = True
            role = "training" if stage not in {"proposal", "feedback"} else "planning_and_repair"
            times = cp.setdefault("seconds_by_role", {})
            charged = job["reservation"] if elapsed is None else min(job["reservation"], max(0.0, elapsed))
            times[role] = times.get(role, 0.0) + charged
            refund = job["reservation"] - charged
            job["spent"] -= refund
            if "infrastructure_retries" not in cp:
                cp["infrastructure_retries"] = int(cp.pop("repairs", 0))
            cp.setdefault("repair_rounds", 0)
            cp.setdefault("gate_issues", [])
            cp.setdefault("repair_proposals", [])
            cp.setdefault("progress_digests", [])
            cp.setdefault("best_so_far", None)
            cp.setdefault("terminal_reason", None)
            repairable = stage in {"proposal", "feedback"} and code in {
                "PROPOSAL_EVIDENCE_OR_SCHEMA", "CANDIDATE_BUDGET",
                "CANDIDATE_NOT_OFFERED", "DUPLICATE_CANDIDATE", "ROUTING_REASON",
                "FEEDBACK_PROPOSAL_SCHEMA", "FEEDBACK_PROPOSAL_EVIDENCE",
                "FEEDBACK_CHANGE_NOT_ALLOWED", "STALE_FEEDBACK_PROPOSAL",
                "DUPLICATE_FEEDBACK_PROPOSAL",
                "INITIAL_CANDIDATE_LIMIT", "EXPERIMENT_PARAMETER_DIRECTION_CONTRADICTION",
                "EXPERIMENT_CONTRACT_VERSION", "EXPERIMENT_FEEDBACK_VERSION", "EXPERIMENT_PARENT",
                "EXPERIMENT_EVIDENCE", "EXPERIMENT_RESOURCE_SCHEMA", "FEATURE_SPEC_NOT_ALLOWED",
                "FEEDBACK_PROPOSAL_DIRECTION",
                "FEEDBACK_PROPOSAL_IDENTITY", "FEEDBACK_PROPOSAL_HYPOTHESIS",
                "FEEDBACK_PROPOSAL_RISKS", "FEEDBACK_PROPOSAL_METRICS",
            }
            from .context import training_seconds_spent
            can_repair = (repairable and cp.get("format_repairs", 0) < config.planner.max_format_repairs
                          and cp.get("llm_tokens_used", 0) < config.budget.max_llm_tokens)
            durable_response = cp.get("pending_agent_response") or cp.get("pending_initial_proposal")
            can_retry = (retryable and
                ((stage not in {"proposal", "feedback"} and cp["attempts"].get(stage, 0) < config.budget.max_attempts
                  and training_seconds_spent({**job, "status": "PENDING"}) < config.budget.max_training_seconds)
                 or (stage in {"proposal", "feedback"} and durable_response
                     and cp["attempts"].get(stage, 0) < config.budget.max_attempts)))
            candidate_feedback = (stage.isdigit() and code in {
                "FOUNDATION_VRAM_INSUFFICIENT", "MEMORY_BUDGET", "CHRONOS_RESOURCE_INSUFFICIENT", "GPU_OOM",
            } and config.planner.kind == "external" and not job["plan"].get("agent_managed") and cp["repair_rounds"] < config.budget.max_repairs
                and cp.get("llm_tokens_used", 0) < config.budget.max_llm_tokens)
            if candidate_feedback:
                from .feedback import build_agent_context, failure_outcome, gate_issue_for_outcome, progress_record
                outcome = failure_outcome(
                    job, config, code=code, stage=stage,
                    attempt=cp["attempts"].get(stage, 0), created_at=job["cutoff"], retryable=False)
                issue = gate_issue_for_outcome(outcome, code=code, repairable=True)
                cp["outcomes"].append(outcome.to_dict())
                cp["gate_issues"].append(issue.to_dict())
                cp["reports"].append({"candidate": outcome.effective_config,
                                      "candidate_id": outcome.candidate_id, "code": code,
                                      "eligible": False})
                cp["index"] += 1
                cp["repair_rounds"] += 1
                job["checkpoint"] = cp
                context = build_agent_context(
                    config, job, outcome, [issue], comparison={"comparable": False,
                    "reason": "EXECUTION_DID_NOT_COMPLETE", "metric_deltas": {},
                    "quality_improved": None}, available_targets=set(config.execution.allowed_targets))
                cp["agent_contexts"].append(context.to_dict())
                cp["feedback_pending"] = context.to_dict()
                cp["progress_digests"].append(progress_record(
                    round_index=cp["repair_rounds"], current_fingerprint=None,
                    previous_fingerprint=cp.get("last_proposal_fingerprint"),
                    best_so_far=cp.get("best_so_far"),
                    comparison={"comparable": False, "metric_deltas": {}, "quality_improved": None},
                    issue_ids=(issue.issue_id,)))
                status = "PENDING"
                self._event(db, job["task_key"], "AGENT_FEEDBACK_REQUESTED",
                            {"run_id": run_id, "outcome_id": outcome.message_id,
                             "feedback_id": context.feedback_id, "issue_id": issue.issue_id},
                            job["cutoff"], "feedback:" + context.feedback_id)
            elif can_repair:
                from .remediation import AgentContext, GateIssue, Outcome, ProgressDigest
                previous = cp.get("previous_proposal_fingerprint")
                current = cp.get("last_proposal_fingerprint")
                issue = GateIssue.create(
                    code=code, stage=stage, severity="error", candidate_id=None,
                    affected={"proposal_fingerprint": current,
                              "attempt": cp["attempts"].get(stage, 0)},
                    evidence_refs=("proposal:" + (current or "unavailable"),),
                    auto_repairable=True,
                    allowed_changes=("candidate_ref", "training_target")).to_dict()
                cp["gate_issues"].append(issue)
                cp["format_repairs"] = cp.get("format_repairs", 0) + 1
                digest_record = ProgressDigest(
                    cp["repair_rounds"], current != previous, current, previous,
                    cp.get("best_so_far"), None,
                ).to_dict()
                cp["progress_digests"].append(digest_record)
                cp["previous_proposal_fingerprint"] = current
                if stage == "feedback" and cp.get("feedback_pending") is not None:
                    prior = AgentContext.from_dict(cp["feedback_pending"])
                    evidence_refs = tuple(dict.fromkeys((*prior.evidence_refs, *issue["evidence_refs"])))
                    remaining_budget = dict(prior.remaining_budget)
                    remaining_budget["repair_rounds"] = max(
                        0, config.budget.max_repairs - cp["repair_rounds"])
                    remaining_budget["llm_tokens"] = max(
                        0, config.budget.max_llm_tokens - cp.get("llm_tokens_used", 0))
                    updated = AgentContext.create(
                        run_id=prior.run_id, plan_id=prior.plan_id,
                        plan_version=prior.plan_version, created_at=job["cutoff"],
                        current_goal=prior.current_goal,
                        current_plan=prior.current_plan,
                        outcome=Outcome.from_dict(prior.outcome),
                        gate_issues=[*(GateIssue.from_dict(item) for item in prior.gate_issues),
                                     GateIssue.from_dict(issue)],
                        comparisons=prior.comparisons, history=prior.history,
                        allowed_changes=prior.allowed_changes,
                        remaining_budget=remaining_budget,
                        output_contract=prior.output_contract,
                        evidence_refs=evidence_refs,
                        forbidden_actions=prior.forbidden_actions)
                    cp["agent_contexts"].append(updated.to_dict())
                    cp["feedback_pending"] = updated.to_dict()
                    cp["pending_agent_response"] = None
                status = "PENDING"
                self._event(db, job["task_key"], "AGENT_REPAIR_REQUESTED",
                            {"run_id": run_id, "gate_issue": issue, "progress_digest": digest_record},
                            job["cutoff"])
            elif can_retry:
                cp["infrastructure_retries"] += 1
                status = "PENDING"
            elif stage.isdigit() and cp["index"] < len(job["plan"]["candidates"]):
                from .feedback import failure_outcome
                cp["outcomes"].append(failure_outcome(
                    job, config, code=code, stage=stage,
                    attempt=cp["attempts"].get(stage, 0), created_at=job["cutoff"],
                    retryable=False).to_dict())
                cp["reports"].append({"candidate": job["plan"]["candidates"][cp["index"]], "code": code, "eligible": False})
                cp["index"] += 1
                status = "PENDING"
            else:
                if stage in {"proposal", "feedback"}:
                    from .remediation import GateIssue, ProgressDigest
                    current = cp.get("last_proposal_fingerprint")
                    previous = cp.get("previous_proposal_fingerprint")
                    cp["gate_issues"].append(GateIssue.create(
                        code=code, stage=stage, severity="error", candidate_id=None,
                        affected={"proposal_fingerprint": current,
                                  "attempt": cp["attempts"].get(stage, 0)},
                        evidence_refs=("proposal:" + (current or "unavailable"),),
                        auto_repairable=False, allowed_changes=()).to_dict())
                    cp["terminal_reason"] = code
                    terminal = ProgressDigest(
                        cp["repair_rounds"], current != previous, current, previous,
                        cp.get("best_so_far"), code,
                    ).to_dict()
                    cp["progress_digests"].append(terminal)
                    self._event(db, job["task_key"], "AGENT_REPAIR_TERMINATED",
                                {"run_id": run_id, "reason": code, "progress_digest": terminal},
                                job["cutoff"], "repair-terminal:" + run_id)
                status = "FAILED"
            if stage in {"proposal", "feedback"}:
                from .feedback import failure_outcome
                cp["outcomes"].append(failure_outcome(job, config, code=code, stage=stage,
                    attempt=cp["attempts"].get(stage, 0), created_at=job["cutoff"], retryable=False).to_dict())
            self._persist_feedback_messages(db, job, cp)
            from .reviews import persist_job_cards
            persist_job_cards(db, job, cp, {"code": code}, status)
            db.execute("UPDATE jobs SET status=?,checkpoint=?,result=?,spent=MAX(0,spent-?),owner=NULL,lease_until=NULL,reservation=0 WHERE run_id=?",
                       (status, canonical(cp), canonical({"code": code}) if status == "FAILED" else None, refund, run_id))
            db.execute("DELETE FROM resource_leases WHERE resource='llm-call' AND owner=?", (owner,))
            self._event(db, job["task_key"], "TRAINING_STAGE_FAILED", {"run_id": run_id, "stage": stage, "code": code,
                                                                                "retry": can_retry, "repair": can_repair,
                                                                                "training_target": job["plan"].get("training_target")}, job["cutoff"])

    def _settle_open_llm_reservation(self, db, job, status):
        """Preserve known usage even if cancellation wins the response-commit race."""
        cp = job["checkpoint"]
        reserved = cp.pop("llm_reservation", 0)
        if not reserved:
            return
        row = db.execute("SELECT payload FROM agent_calls WHERE operation_id=?",
                         (cp.get("llm_call_id"),)).fetchone()
        usage = json.loads(row[0]).get("usage") if row else None
        actual = usage.get("total_tokens") if isinstance(usage, dict) else None
        if type(actual) is not int or actual < 0:
            actual = None
        charge = reserved if actual is None else actual
        cp["llm_tokens_used"] += charge
        role = "repair" if self._stage(job) == "feedback" or cp.get("repair_rounds", 0) else "planning"
        bucket = cp.setdefault("llm_usage_by_role", {}).setdefault(role,
            {"tokens": 0, "calls": 0, "uncertain_reserved_tokens": 0})
        bucket["uncertain_reserved_tokens" if actual is None else "tokens"] += charge
        bucket["calls"] += 1
        db.execute("UPDATE agent_calls SET status=?,actual_tokens=? WHERE operation_id=?",
                   (status, actual, cp.get("llm_call_id")))

    def cancel_job(self, run_id: str):
        with self.transaction() as db:
            row = db.execute("SELECT task_key,cutoff,status FROM jobs WHERE run_id=?", (run_id,)).fetchone()
            if row and row["status"] in {"PENDING", "RUNNING"}:
                job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
                cp = job["checkpoint"]
                self._settle_open_llm_reservation(db, job, "cancelled")
                from .reviews import persist_job_cards
                persist_job_cards(db, job, cp, None, "CANCELLED")
                db.execute("UPDATE jobs SET status='CANCELLED',checkpoint=?,owner=NULL,lease_until=NULL WHERE run_id=?", (canonical(cp), run_id))
                db.execute("DELETE FROM resource_leases WHERE run_id=?", (run_id,))
                self._event(db, row["task_key"], "TRAINING_CANCEL_REQUESTED", {"run_id": run_id}, row["cutoff"], "cancel:" + run_id)

    def seal_selection(self, run_id: str, selected: dict | None, code: str | None = None):
        with self.transaction() as db:
            job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (run_id,)).fetchone())
            if job["status"] != "PENDING" or job["checkpoint"]["selected"] is not None:
                return
            if (job["checkpoint"].get("feedback_pending") is not None
                    or job["checkpoint"].get("human_pending") is not None
                    or job["plan"].get("needs_proposal")
                    or job["checkpoint"]["index"] != len(job["plan"]["candidates"])):
                raise GateError("SELECTION_BEFORE_VALIDATION")
            cp = job["checkpoint"]
            from .models import Candidate
            expected = [Candidate.from_dict(c).candidate_id for c in job["plan"]["candidates"]]
            actual = [Candidate.from_dict(r["candidate"]).candidate_id for r in cp["reports"]]
            if actual != expected:
                raise GateError("SELECTION_INCOMPLETE_RESULTS")
            if selected is not None and not any(
                    r.get("eligible") and Candidate.from_dict(r["candidate"]) == Candidate.from_dict(selected)
                    for r in cp["reports"]):
                raise GateError("SELECTION_INELIGIBLE")
            cp["selected"] = selected
            cp["selection_binding"] = digest({"plan_version": cp["plan_version"],
                                               "candidates": expected, "reports": cp["reports"]})
            db.execute("UPDATE jobs SET checkpoint=?,status=?,result=? WHERE run_id=?", (canonical(cp), "PENDING" if selected else "FAILED", canonical({"code": code}) if code else None, run_id))
            self._event(db, job["task_key"], "SELECTION_SEALED" if selected else "NO_QUALIFIED_CANDIDATE", {"run_id": run_id, "selected": selected, "code": code}, job["cutoff"])

    def register_model(self, key: str, run_id: str, result: dict, now: int) -> str:
        model_id = "model-" + digest({"task": key, "artifact": result["artifact"]["sha256"]})[:24]
        with self.transaction() as db:
            artifact, report = canonical(result["artifact"]), canonical(result)
            existing = db.execute("SELECT task_key,artifact,report FROM models WHERE model_id=?", (model_id,)).fetchone()
            if existing and (existing["task_key"] != key or existing["artifact"] != artifact or existing["report"] != report):
                raise GateError("MODEL_REGISTRATION_CONFLICT")
            db.execute("INSERT OR IGNORE INTO models VALUES(?,?,?,?,?,?)", (model_id, key, artifact, report, now, run_id))
            state = json.loads(db.execute("SELECT state FROM tasks WHERE task_key=?", (key,)).fetchone()[0])
            self._persist_deployment_cards(db, key, state)
            self._event(db, key, "MODEL_REGISTERED", {"model_id": model_id, "run_id": run_id}, now, "register:" + model_id)
        return model_id

    def model(self, model_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM models WHERE model_id=?", (model_id,)).fetchone()
        if not row:
            raise GateError("MODEL_NOT_FOUND")
        result = dict(row)
        result["artifact"] = json.loads(result["artifact"])
        result["report"] = json.loads(result["report"])
        return result

    @classmethod
    def _persist_deployment_cards(cls, db, key, state):
        from .reviews import build_card, insert_card
        for model in db.execute("SELECT * FROM models WHERE task_key=?", (key,)).fetchall():
            job = cls._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (model["run_id"],)).fetchone())
            report = json.loads(model["report"])
            if job is None or "candidate" not in report or "offline" not in report:
                continue
            deployment = ("champion" if state.get("champion") == model["model_id"]
                          else state.get("stage") if state.get("challenger") == model["model_id"]
                          else "retained_for_rollback" if state.get("previous") == model["model_id"]
                          else "registered_inactive")
            insert_card(db, key, build_card(job, report["candidate"], "final_test", report,
                                           model_release_id=model["model_id"], deployment=deployment))

    @staticmethod
    def _release_binding(db, key, state, config):
        model_id = state.get("challenger")
        row = db.execute("SELECT artifact,report,run_id FROM models WHERE model_id=? AND task_key=?", (model_id, key)).fetchone()
        if row is None:
            raise GateError("NO_PENDING_APPROVAL")
        report = json.loads(row["report"])
        if not report.get("passed"):
            raise GateError("PROMOTION_EVIDENCE")
        job = db.execute("SELECT checkpoint FROM jobs WHERE run_id=?", (row["run_id"],)).fetchone()
        windows = [tuple(r) for r in db.execute(
            "SELECT model_id,window_start,window_end,label_fingerprint FROM monitoring_windows WHERE task_key=? AND label_state IN ('complete','revised') ORDER BY model_id,window_start,window_end", (key,))]
        return {"action": "approve_release", "model_id": model_id,
                "model_package_hash": json.loads(row["artifact"])["sha256"],
                "plan_version": json.loads(job[0])["plan_version"] if job else None,
                "policy_version": config.version, "target_environment": config.mode,
                "scope": config.contract.entity_id, "stage": state.get("stage"),
                "expected_champion": state.get("champion"), "rollback_target": state.get("champion"),
                "evidence_version": digest({"report": report, "label_windows": windows})}

    def release_binding(self, key):
        config, state = self.task(key)
        with self.connection() as db:
            return self._release_binding(db, key, state, config)

    def approve_release(self, key, actor, *, expected_binding=None, operation=None):
        with self.transaction() as db:
            row = db.execute("SELECT state,config FROM tasks WHERE task_key=?", (key,)).fetchone()
            state, config = json.loads(row["state"]), TaskConfig.from_dict(json.loads(row["config"]))
            if state["stop_requested"] or state.get("stage") not in {"INITIAL_APPROVAL", "CANARY"}:
                raise GateError("NO_PENDING_APPROVAL")
            if config.collaboration.permissions and "release" not in config.collaboration.permissions.get(actor, ()):
                raise GateError("HUMAN_PERMISSION_DENIED")
            binding = self._release_binding(db, key, state, config)
            if expected_binding is not None and canonical(binding) != canonical(expected_binding):
                raise GateError("STALE_RELEASE_CONFIRMATION")
            state.update(approved=True, approval_binding=binding)
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), key))
            self._event(db, key, "APPROVED", {"actor": actor, "binding": binding}, state["now"] or 0, operation)

    def switch(self, key: str, expected: str | None, target: str, now: int, evidence: dict, operation: str, *, rollback: bool = False, inject: Callable | None = None):
        with self.transaction() as db:
            existing = db.execute("SELECT payload FROM events WHERE task_key=? AND event_key=?", (key, operation)).fetchone()
            payload = {"from": expected, "to": target, "evidence": evidence}
            if existing:
                if existing["payload"] != canonical(payload):
                    raise GateError("OPERATION_PAYLOAD_CONFLICT")
                return
            row = db.execute("SELECT state,config FROM tasks WHERE task_key=?", (key,)).fetchone()
            state = json.loads(row["state"])
            config = TaskConfig.from_dict(json.loads(row["config"]))
            model = db.execute("SELECT task_key FROM models WHERE model_id=?", (target,)).fetchone()
            if not model or model["task_key"] != key:
                raise GateError("MODEL_TASK_SCOPE")
            if state["champion"] != expected:
                raise GateError("CHAMPION_CONFLICT")
            if state["stop_requested"]:
                raise GateError("TASK_STOPPING")
            if rollback:
                if state["previous"] != target:
                    raise GateError("ROLLBACK_TARGET")
            elif not evidence.get("passed") or (expected is not None and (state["challenger"] != target or state["stage"] != "CANARY")):
                raise GateError("PROMOTION_EVIDENCE")
            if not rollback and config.mode == "live" and not state["approved"]:
                raise GateError("APPROVAL_REQUIRED")
            if not rollback and config.mode == "live":
                binding = self._release_binding(db, key, state, config)
                if canonical(state.get("approval_binding")) != canonical(binding):
                    raise GateError("STALE_RELEASE_CONFIRMATION")
            state["last_deployment"] = {"from": expected, "to": target, "at": now,
                "operation": operation, "rollback": rollback, "attempt": state.pop("agent_release_attempt", None)}
            references = state.setdefault("monitor_by_model", {})
            if expected:
                references[expected] = state["monitor"]
            state.update(champion=target, previous=expected if not rollback else None, challenger=None, stage=None, stage_since=None,
                         approved=False, status="SERVING", model_since=now,
                         observation_until=now + int(config.release.post_promotion_seconds * 1e9),
                         observation_completed_for=None, monitor=references.get(target, {}))
            state["cooldown_until"] = now + int(config.monitoring.cooldown_seconds * 1e9)
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), key))
            if inject:
                inject()
            self._persist_deployment_cards(db, key, state)
            self._event(db, key, "ROLLED_BACK" if rollback else "PROMOTED", payload, now, operation)

    def record_tick(self, key: str, now: int, issue: int | None, forecasts: list[dict], labels: list[tuple], *, inject: Callable | None = None, label_versions: dict | None = None):
        # None is an explicit withdrawal, never a numeric truth. The immutable
        # revision history keeps prior as-of evaluations reproducible.
        if any(value is not None and not math.isfinite(value) for _, _, value in labels):
            raise GateError("LABEL_VALUE")
        with self.transaction() as db:
            state = json.loads(db.execute("SELECT state FROM tasks WHERE task_key=?", (key,)).fetchone()["state"])
            if state["stop_requested"]:
                return
            latest_labels = {}
            if labels:
                latest_labels = {(row["target_ns"], row["variable"]): row for row in db.execute("""
                    SELECT r.target_ns,r.variable,r.revision,r.value,r.source FROM label_revisions r
                    WHERE r.task_key=? AND NOT EXISTS (
                        SELECT 1 FROM label_revisions newer WHERE newer.task_key=r.task_key
                        AND newer.target_ns=r.target_ns AND newer.variable=r.variable
                        AND newer.revision>r.revision)""", (key,))}
            for target, variable, value in labels:
                source_version = str((label_versions or {}).get((target, variable), "record_tick"))
                latest = latest_labels.get((target, variable))
                if latest is None or latest["value"] != value or latest["source"] != source_version:
                    revision = 1 if latest is None else int(latest["revision"]) + 1
                    db.execute("INSERT INTO label_revisions VALUES(?,?,?,?,?,?,?)",
                               (key, target, variable, revision, value, now, source_version))
                    latest_labels[(target, variable)] = {"revision": revision, "value": value, "source": source_version}
                    if value is None:
                        db.execute("DELETE FROM labels WHERE task_key=? AND target_ns=? AND variable=?",
                                   (key, target, variable))
                    else:
                        db.execute("INSERT INTO labels VALUES(?,?,?,?,?) ON CONFLICT(task_key,target_ns,variable) DO UPDATE SET value=excluded.value,observed_ns=CASE WHEN labels.value=excluded.value THEN labels.observed_ns ELSE excluded.observed_ns END",
                                   (key, target, variable, value, now))
            for f in forecasts:
                db.execute("INSERT OR IGNORE INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (key, f["model_id"], f["issue_ns"], f["target_ns"], f["variable"], f["horizon"], f["value"], f["role"], int(f["published"]), f["inference_ms"], f["quality"]))
            if inject:
                inject()
            state["now"] = now
            if issue is not None:
                state["last_issue"] = max(state["last_issue"] or issue, issue)
            db.execute("UPDATE tasks SET state=?,revision=revision+1 WHERE task_key=?", (canonical(state), key))

    def prediction_rows(self, key: str, model_id: str, start: int, end: int, observed_until: int | None = None) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT p.*,l.value AS actual,l.revision AS label_revision,
                l.observed_ns AS label_observed_ns FROM predictions p LEFT JOIN label_revisions l
                ON l.task_key=p.task_key AND l.target_ns=p.target_ns AND l.variable=p.variable
                AND l.revision=(SELECT lr.revision FROM label_revisions lr
                    WHERE lr.task_key=p.task_key AND lr.target_ns=p.target_ns
                      AND lr.variable=p.variable AND lr.observed_ns<=?
                    ORDER BY lr.observed_ns DESC,lr.revision DESC LIMIT 1)
                WHERE p.task_key=? AND p.model_id=? AND p.target_ns>? AND p.target_ns<=?
                ORDER BY p.issue_ns,p.horizon,p.variable""", (end if observed_until is None else observed_until, key, model_id, start, end)).fetchall()
            return [dict(r) for r in rows]

    def published_rows(self, key: str, issue_ns: int) -> list[dict]:
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM predictions WHERE task_key=? AND issue_ns=? AND published=1 ORDER BY target_ns,variable",
                (key, issue_ns))]

    def prediction_issue_bounds(self, key: str, model_id: str) -> tuple[int, int] | None:
        with self.connection() as db:
            row = db.execute("SELECT MIN(issue_ns),MAX(issue_ns) FROM predictions WHERE task_key=? AND model_id=?",
                             (key, model_id)).fetchone()
        return None if row[0] is None else (int(row[0]), int(row[1]))

    def model_ids_with_predictions(self, key: str) -> list[str]:
        with self.connection() as db:
            return [str(row[0]) for row in db.execute(
                "SELECT DISTINCT model_id FROM predictions WHERE task_key=? ORDER BY model_id", (key,))]

    def prediction_rows_by_issue(self, key: str, model_id: str, start: int, end: int,
                                 observed_until: int) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT p.*,l.value AS actual,l.revision AS label_revision,
                l.observed_ns AS label_observed_ns FROM predictions p LEFT JOIN label_revisions l
                ON l.task_key=p.task_key AND l.target_ns=p.target_ns AND l.variable=p.variable
                AND l.revision=(SELECT lr.revision FROM label_revisions lr
                    WHERE lr.task_key=p.task_key AND lr.target_ns=p.target_ns
                      AND lr.variable=p.variable AND lr.observed_ns<=?
                    ORDER BY lr.observed_ns DESC,lr.revision DESC LIMIT 1)
                WHERE p.task_key=? AND p.model_id=? AND p.issue_ns>=? AND p.issue_ns<?
                ORDER BY p.issue_ns,p.horizon,p.variable""",
                (observed_until, key, model_id, start, end)).fetchall()
        return [dict(row) for row in rows]

    def prediction_window_counts(self, key: str, model_id: str, first_issue: int,
                                 width: int, observed_until: int) -> dict[int, tuple[int, int]]:
        """Count append-only prediction/visible-revision inputs before loading full rows.

        Count every visible revision, including superseded ones. A first label
        changes the non-null count; another revision or prediction changes at
        least one count. This also detects revisions inserted at the same time.
        """
        with self.connection() as db:
            rows = db.execute("""SELECT ? + ((p.issue_ns - ?) / ?) * ? AS window_start,
                COUNT(*) AS joined_count, COUNT(l.revision) AS revision_count
                FROM predictions p LEFT JOIN label_revisions l
                ON l.task_key=p.task_key AND l.target_ns=p.target_ns AND l.variable=p.variable
                   AND l.observed_ns<=?
                WHERE p.task_key=? AND p.model_id=? GROUP BY window_start""",
                (first_issue, first_issue, width, width, observed_until, key, model_id)).fetchall()
        return {int(row["window_start"]): (row["joined_count"], row["revision_count"]) for row in rows}

    def monitoring_window(self, key: str, model_id: str, start: int, end: int) -> dict | None:
        with self.connection() as db:
            row = db.execute("""SELECT * FROM monitoring_windows
                WHERE task_key=? AND model_id=? AND window_start=? AND window_end=?""",
                (key, model_id, start, end)).fetchone()
        if row is None:
            return None
        return {**dict(row), "evidence": json.loads(row["evidence"])}

    def monitoring_windows(self, key: str, model_id: str | None = None) -> list[dict]:
        with self.connection() as db:
            if model_id is None:
                rows = db.execute("""SELECT * FROM monitoring_windows WHERE task_key=?
                    ORDER BY model_id,window_start,window_end""", (key,)).fetchall()
            else:
                rows = db.execute("""SELECT * FROM monitoring_windows WHERE task_key=? AND model_id=?
                    ORDER BY window_start,window_end""", (key, model_id)).fetchall()
        return [{**dict(row), "evidence": json.loads(row["evidence"])} for row in rows]

    def save_monitoring_window(self, key: str, model_id: str, start: int, end: int, now: int,
                               label_state: str, fingerprint: str, evidence: dict,
                               *, triggered: bool = False) -> bool:
        if label_state not in {"not_arrived", "partial", "complete", "revised"}:
            raise GateError("LABEL_WINDOW_STATE")
        with self.transaction() as db:
            row = db.execute("""SELECT label_state,label_fingerprint FROM monitoring_windows
                WHERE task_key=? AND model_id=? AND window_start=? AND window_end=?""",
                (key, model_id, start, end)).fetchone()
            if row and row["label_state"] == label_state and row["label_fingerprint"] == fingerprint:
                return False
            db.execute("""INSERT INTO monitoring_windows VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(task_key,model_id,window_start,window_end) DO UPDATE SET
                  label_state=excluded.label_state,label_fingerprint=excluded.label_fingerprint,
                  evaluated_ns=excluded.evaluated_ns,evidence=excluded.evidence,
                  triggered=MAX(monitoring_windows.triggered,excluded.triggered)""",
                (key, model_id, start, end, label_state, fingerprint, now, canonical(evidence), int(triggered)))
            model = db.execute("SELECT report,run_id FROM models WHERE model_id=? AND task_key=?", (model_id, key)).fetchone()
            if model:
                job = self._job(db.execute("SELECT * FROM jobs WHERE run_id=?", (model["run_id"],)).fetchone())
                if job:
                    from .reviews import build_card, insert_card
                    insert_card(db, key, build_card(job, json.loads(model["report"])["candidate"],
                                "online_mature", {**evidence, "label_fingerprint": fingerprint},
                                model_release_id=model_id, deployment="see_current_registry"))
        return True

    def fallback_issue(self, key: str, issue_ns: int, champion: str, challenger: str, now: int, reason: str) -> None:
        with self.transaction() as db:
            db.execute("UPDATE predictions SET published=0 WHERE task_key=? AND issue_ns=? AND model_id=?",
                       (key, issue_ns, challenger))
            db.execute("UPDATE predictions SET published=1 WHERE task_key=? AND issue_ns=? AND model_id=?",
                       (key, issue_ns, champion))
            self._event(db, key, "CANARY_OUTPUT_RESTORED",
                        {"issue_ns": issue_ns, "champion": champion, "challenger": challenger, "reason": reason},
                        now, f"canary-fallback:{challenger}:{issue_ns}")

    def save_monitor(self, key: str, model_id: str, now: int, evidence: dict):
        with self.transaction() as db:
            db.execute("INSERT INTO monitoring VALUES(?,?,?,?) ON CONFLICT(task_key,model_id,evaluated_ns) DO UPDATE SET evidence=excluded.evidence", (key, model_id, now, canonical(evidence)))

    def counts(self, key: str) -> dict:
        with self.connection() as db:
            return {name: db.execute(f"SELECT COUNT(*) FROM {name} WHERE task_key=?", (key,)).fetchone()[0] for name in ("predictions", "labels", "monitoring", "monitoring_windows", "events", "models", "jobs")}
