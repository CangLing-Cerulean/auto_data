"""Durable operational state and action journal, colocated with lifecycle transactions."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from ..config import GateError, canonical, digest


@dataclass
class Event:
    event_id: str
    kind: str
    created_at: int
    payload: dict = field(default_factory=dict)
    object_id: str | None = None


@dataclass
class WaitCondition:
    condition_id: str
    task_key: str
    epoch: int
    condition_type: str
    object_id: str | None
    predicate: dict
    clock_domain: str
    created_at: int
    created_wall_ns: int
    deadline: int
    reason: str
    agent_state_version: int
    status: str = "waiting"


@dataclass
class AgentState:
    task_goal: str
    decision_version: int = 0
    active_incident: str | None = None
    current_hypotheses: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    pending_tool_jobs: list = field(default_factory=list)
    waiting_conditions: list = field(default_factory=list)
    relevant_memory_refs: list = field(default_factory=list)
    human_constraints: list = field(default_factory=list)
    last_decisions: list = field(default_factory=list)
    last_event_id: int = 0
    next_inspection: int = 0
    last_result: dict | None = None
    pending_event: dict | None = None
    status: str = "monitoring"
    incident_id: str | None = None
    incident_finished: bool = False
    phase: str = "exploration"
    deferred_events: list = field(default_factory=list)


@dataclass
class DecisionRecord:
    decision_id: str
    trigger_event: dict
    context_version: str
    memory_refs: list
    knowledge_refs: list
    tool_results_used: list
    hypotheses: list
    selected_action: str
    arguments: dict
    reason_summary: str
    expected_information_gain: str
    created_at: int
    binding: dict
    actual_result: dict | None = None
    result_refs: list = field(default_factory=list)
    decision_outcome: str = "prepared"
    hypothesis_update: list = field(default_factory=list)
    cited_memory_refs: list = field(default_factory=list)
    cited_knowledge_refs: list = field(default_factory=list)
    online_refs: list = field(default_factory=list)
    visible_evidence_refs: list = field(default_factory=list)
    cited_evidence_refs: list = field(default_factory=list)
    arguments_linked_to_evidence: list = field(default_factory=list)
    actual_tool_effect: dict = field(default_factory=dict)


class Repository:
    def __init__(self, store, key):
        self.store, self.key = store, key
        # Additive tables owned exclusively by this optional package. No lifecycle
        # schema migration or reinterpretation of existing records is required.
        with store.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS agent_runtime_state (
                    task_key TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS agent_runtime_records (
                    task_key TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT NOT NULL,
                    created_ns INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(task_key, record_id));
            """)

    def initialize(self, goal):
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
            raise GateError("AGENT_GOAL")
        with self.store.transaction() as db:
            db.execute("INSERT OR IGNORE INTO agent_runtime_state VALUES (?,?)",
                       (self.key, canonical(asdict(AgentState(goal)))))

    def state(self):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM agent_runtime_state WHERE task_key=?", (self.key,)).fetchone()
        if row is None:
            raise GateError("AGENT_NOT_INITIALIZED")
        return AgentState(**json.loads(row[0]))

    def save_state(self, state, expected_version, *, db=None):
        if db is None:
            with self.store.transaction() as connection:
                return self.save_state(state, expected_version, db=connection)
        current = json.loads(db.execute("SELECT payload FROM agent_runtime_state WHERE task_key=?", (self.key,)).fetchone()[0])
        if current["decision_version"] != expected_version:
            raise GateError("STALE_AGENT_STATE")
        db.execute("UPDATE agent_runtime_state SET payload=? WHERE task_key=?", (canonical(asdict(state)), self.key))

    def put(self, record_id, kind, payload, now, *, replace=False, db=None):
        if db is None:
            with self.store.transaction() as connection:
                return self.put(record_id, kind, payload, now, replace=replace, db=connection)
        existing = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?", (self.key, record_id)).fetchone()
        if existing and not replace:
            if existing[0] != canonical(payload):
                raise GateError("AGENT_RECORD_COLLISION")
            return
        db.execute("INSERT INTO agent_runtime_records VALUES (?,?,?,?,?) ON CONFLICT(task_key,record_id) DO UPDATE SET payload=excluded.payload",
                   (self.key, record_id, kind, now, canonical(payload)))

    def get(self, record_id):
        with self.store.connection() as db:
            row = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?", (self.key, record_id)).fetchone()
        if row is None:
            raise GateError("AGENT_RECORD_NOT_FOUND")
        return json.loads(row[0])

    def records(self, kind, limit=20, *, offset=0):
        if type(limit) is not int or limit < 1 or type(offset) is not int or offset < 0:
            raise GateError("AGENT_RECORD_PAGE")
        with self.store.connection() as db:
            rows = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND kind=? ORDER BY created_ns DESC,record_id DESC LIMIT ? OFFSET ?", (self.key, kind, limit, offset)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def iter_records(self, kind):
        """Stream one consistent SQLite snapshot, without a silent history cap."""
        with self.store.connection() as db:
            cursor = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND kind=? ORDER BY created_ns DESC,record_id DESC", (self.key, kind))
            for row in cursor:
                yield json.loads(row[0])

    def count_records(self, kind):
        with self.store.connection() as db:
            return db.execute("SELECT COUNT(*) FROM agent_runtime_records WHERE task_key=? AND kind=?", (self.key, kind)).fetchone()[0]

    def binding(self):
        _, state = self.store.task(self.key)
        return {"task_key": self.key, "owner": state.get("decision_owner", "workflow"),
                "epoch": state.get("decision_epoch", 0), "policy": state.get("autonomy_policy"),
                "champion": state["champion"], "challenger": state["challenger"],
                "stage": state["stage"], "stage_since": state["stage_since"],
                "release_grant": state.get("agent_release_grant"),
                "stop_requested": state["stop_requested"]}

    def check_binding(self, binding):
        if binding != self.binding() or binding["owner"] != "agent" or binding["stop_requested"]:
            raise GateError("STALE_AGENT_AUTHORITY")

    def finish(self, record, result):
        """Commit receipt and state together; crash recovery replays only prepared actions."""
        with self.store.transaction() as db:
            row = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?", (self.key, record["decision_id"])).fetchone()
            old = json.loads(row[0])
            if old["decision_outcome"] != "prepared":
                return old
            state = AgentState(**json.loads(db.execute("SELECT payload FROM agent_runtime_state WHERE task_key=?", (self.key,)).fetchone()[0]))
            record.update(actual_result=result, decision_outcome="failed" if result.get("error") else "completed",
                          result_refs=result.get("result_refs", []), hypothesis_update=record["hypotheses"])
            from .evidence import EvidenceBoard
            evidence_id = EvidenceBoard(self).capture(record, result, db=db)
            record["actual_tool_effect"] = {"operation_id": record["decision_id"], "tool": record["selected_action"],
                                            "status": record["decision_outcome"], "evidence_id": evidence_id,
                                            "delivery_receipt": result.get("record_id") if result.get("agent_finalized") else None}
            state.decision_version += 1
            state.pending_event = None
            state.active_incident = record["trigger_event"]["kind"]
            state.current_hypotheses = record["hypotheses"]
            state.last_result = {"decision_id": record["decision_id"], "tool": record["selected_action"], "result": result}
            state.last_decisions = (state.last_decisions + [record["decision_id"]])[-8:]
            if result.get("run_id") and record["selected_action"] == "start_experiment":
                state.pending_tool_jobs = list(dict.fromkeys(state.pending_tool_jobs + [result["run_id"]]))
            if result.get("run_id") and (result.get("development_complete") or result.get("status") in {"CANCELLED", "COMPLETED", "FAILED"}):
                state.pending_tool_jobs = [r for r in state.pending_tool_jobs if r != result["run_id"]]
            hypothesis = {"record_id": "hypothesis:" + record["decision_id"], "hypotheses": record["hypotheses"],
                          "evidence": record["tool_results_used"], "status": "unverified", "confidence": "descriptive_only"}
            db.execute("INSERT OR IGNORE INTO agent_runtime_records VALUES(?,?,?,?,?)",
                       (self.key, hypothesis["record_id"], "hypothesis", record["created_at"], canonical(hypothesis)))
            if record["selected_action"] == "wait_until" and "wait_condition" in result:
                state.waiting_conditions = [result["wait_condition"]]
                state.status = "waiting"
            elif record["selected_action"] == "finalize_selection" and record.get("environment_kind") != "panel" and not result.get("error"):
                state.status = "investigating"
                state.incident_finished = False
            elif record["selected_action"] == "report_to_human":
                state.status = "monitoring"
                state.active_incident = None
                state.incident_finished = True
            elif result.get("agent_finalized"):
                state.status = "monitoring"
                state.active_incident = None
                state.incident_finished = True
                # An explicit close does not remove its registered follow-up.
                state.waiting_conditions = [w for w in state.waiting_conditions if w.get("status") == "waiting"]
            else:
                state.status = "investigating"
            db.execute("UPDATE agent_runtime_state SET payload=? WHERE task_key=?", (canonical(asdict(state)), self.key))
            db.execute("UPDATE agent_runtime_records SET payload=? WHERE task_key=? AND record_id=?", (canonical(record), self.key, record["decision_id"]))
        return record
