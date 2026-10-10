"""Cheap deterministic wakeups; only an awake investigation spends LLM calls."""
from dataclasses import asdict

from .state import Event
from ..config import SECOND
from ..config import GateError
from .waiting import evaluate, resolve
from .budget import is_open


IMPORTANT = {"DATA_OR_WORKFLOW_BLOCKED", "MONITORING_WINDOW_EVALUATED", "MONITORING_WINDOW_REVISED",
             "WAITING_FOR_LABELS", "TRAINING_COMPLETED", "AGENT_EXPERIMENT_READY", "CANARY_SAFETY_ROLLBACK",
             "HUMAN_KNOWLEDGE_UPDATED", "MODEL_DEGRADATION_DETECTED", "MODEL_AGE_THRESHOLD", "AGENT_OBSERVER_UPDATE", "ROLLED_BACK"}
SAFETY_EVENTS = {"CANARY_SAFETY_ROLLBACK", "ROLLED_BACK"}


class EventLoop:
    def __init__(self, controller, inspection_seconds=3600, *, observation_team=None):
        self.controller, self.repo = controller, controller.repo
        self.inspection_ns = int(inspection_seconds * SECOND)
        self.observation_team = observation_team

    def next_event(self, now):
        state = self.repo.state()
        if state.status == "faulted":
            return None
        if state.pending_event:
            return Event(**state.pending_event)
        events = [e for e in self.repo.store.events(self.repo.key) if e["id"] > state.last_event_id and e["event_time"] <= now]
        open_run = is_open(self.repo)
        if open_run:
            # Persist the inbox before advancing the ingestion cursor. Repeated
            # observations of the same kind/object retain their latest payload
            # and an explicit count; a wait never consumes an unseen alert.
            pending = {(e["kind"], e.get("object_id")): e for e in state.deferred_events}
            for event in events:
                if event["kind"] not in IMPORTANT:
                    continue
                if self.observation_team and self.observation_team.manages(event, now):
                    continue  # Raw windows remain durable; the observer notifies meaningful changes.
                payload = event["payload"]
                object_id = (payload.get("run_id") or payload.get("model_id")
                             or payload.get("record_id") or payload.get("intent_id"))
                if event["kind"] == "AGENT_OBSERVER_UPDATE":
                    object_id = "observer:" + payload["observer"] + ":" + str(payload.get("model_id"))
                key = event["kind"], object_id
                previous = pending.get(key)
                pending[key] = {**event, "object_id": object_id,
                                "first_event_id": previous["first_event_id"] if previous else event["id"],
                                "occurrences": previous["occurrences"] + 1 if previous else 1}
            state.deferred_events = list(pending.values())
            if events:
                state.last_event_id = events[-1]["id"]
            priority = {"CANARY_SAFETY_ROLLBACK": 0, "ROLLED_BACK": 0, "DATA_OR_WORKFLOW_BLOCKED": 1,
                        "MODEL_DEGRADATION_DETECTED": 1, "MODEL_AGE_THRESHOLD": 1, "AGENT_OBSERVER_UPDATE": 1,
                        "HUMAN_KNOWLEDGE_UPDATED": 2, "TRAINING_COMPLETED": 2,
                        "AGENT_EXPERIMENT_READY": 2, "WAITING_FOR_LABELS": 4}
            events = sorted(state.deferred_events, key=lambda e: (priority.get(e["kind"], 3), e["first_event_id"]))
        wait = state.waiting_conditions[0] if state.waiting_conditions else None
        selected = None
        safety_pending = open_run and any(e["kind"] in SAFETY_EVENTS for e in events)
        if wait and not safety_pending:
            persisted = self.repo.get(wait["condition_id"])
            if persisted["status"] != "waiting":
                status, evidence = persisted["status"], persisted.get("resolution_evidence", {})
            else:
                status, evidence = evaluate(self.repo, wait, now, source=self.controller.gateway.engine.source)
            if status != "waiting":
                resolved = resolve(self.repo, wait, status, evidence, now)
                selected = Event("wake:" + wait["condition_id"],
                                 {"satisfied": "WAIT_SATISFIED", "timed_out": "WAIT_RECHECK", "cancelled": "WAIT_CANCELLED"}[status],
                                 resolved["resolved_at"], {"condition_id": wait["condition_id"], "status": status, "evidence": evidence}, wait["object_id"])
        if open_run and not selected and not wait and not safety_pending:
            epoch = self.repo.binding()["epoch"]
            due = [schedule for schedule in self.repo.iter_records("inspection_schedule")
                   if schedule["epoch"] == epoch and schedule["status"] == "scheduled" and schedule["due_at"] <= now]
            if due:
                # Registered follow-ups must not starve behind recurring
                # observations or investigation continuation. Keep alerts in
                # the durable inbox; active waits and safety rollback win.
                selected = Event("review:" + str(now), "PERIODIC_REVIEW", now,
                                 {"inspection_ids": sorted(s["record_id"] for s in due),
                                  "due_at": min(s["due_at"] for s in due)})
        for event in events:
            object_id = (event.get("object_id") or event["payload"].get("run_id") or event["payload"].get("model_id")
                         or event["payload"].get("record_id") or event["payload"].get("intent_id"))
            if selected and not (open_run and event["kind"] in SAFETY_EVENTS):
                break
            if (not wait and event["kind"] in IMPORTANT or event["kind"] in SAFETY_EVENTS):
                payload = ({**event["payload"], "first_event_id": event["first_event_id"],
                            "occurrences": event["occurrences"],
                            "occurrence_scope": "same_event_kind_and_object_across_windows",
                            "payload_scope": "latest_event_only"} if open_run else {})
                selected = Event(str(event["id"]), event["kind"], now, payload, object_id)
                break
        if open_run:
            if selected and selected.event_id.isdigit():
                state.deferred_events = [e for e in state.deferred_events if str(e["id"]) != selected.event_id]
        elif selected and selected.event_id.isdigit():
            state.last_event_id = int(selected.event_id)
        elif events:
            state.last_event_id = events[-1]["id"]
        if not selected and not wait and state.status == "investigating":
            selected = Event("continue:" + str(state.decision_version), "TOOL_RESULT", now)
        if not selected and not wait and now >= state.next_inspection:
            selected = Event("review:" + str(now), "PERIODIC_REVIEW", now)
        if selected:
            if wait and selected.kind in SAFETY_EVENTS:
                resolve(self.repo, wait, "cancelled", {"reason": "safety_rollback"}, now)
            state.waiting_conditions = []
            state.status = "investigating"
            if selected.kind == "PERIODIC_REVIEW" or state.next_inspection <= 0:
                state.next_inspection = now + self.inspection_ns
            state.pending_event = asdict(selected)
        with self.repo.store.transaction() as db:
            if selected and selected.kind == "PERIODIC_REVIEW":
                for schedule in self.repo.iter_records("inspection_schedule"):
                    if schedule["epoch"] == self.repo.binding()["epoch"] and schedule["status"] == "scheduled" and schedule["due_at"] <= now:
                        self.repo.put(schedule["record_id"], "inspection_schedule", {**schedule, "status": "triggered", "triggered_at": now,
                                      "event_id": selected.event_id}, now, replace=True, db=db)
            self.repo.save_state(state, state.decision_version, db=db)
        return selected

    def tick(self, now):
        if self.repo.binding()["owner"] != "agent":
            return None
        event = self.next_event(now)
        if not event:
            return None
        binding = self.repo.binding()
        try:
            return self.controller.step(event)
        except GateError as error:
            if str(error) in {"AGENT_DECISION_BUSY", "LLM_BUSY"}:
                return None
            current = self.repo.binding()
            serving_fields = {"champion", "challenger", "stage", "stage_since"}
            if (str(error) in {"STALE_AGENT_AUTHORITY", "PLANNER_CANCELLED"} and current != binding
                    and current["owner"] == "agent" and not current["stop_requested"]
                    and {k: v for k, v in binding.items() if k not in serving_fields}
                    == {k: v for k, v in current.items() if k not in serving_fields}):
                # The supervisor can roll back while the model is answering.
                # Discard the stale answer; the durable safety event wakes a new
                # decision. A changed owner, epoch, grant or policy still fails.
                state = self.repo.state()
                state.pending_event = None
                state.status = "investigating"
                self.repo.save_state(state, state.decision_version)
                self.repo.store.event(self.repo.key, "AGENT_DECISION_SUPERSEDED",
                    {"event_id": event.event_id, "reason": "serving_state_changed",
                     "before": {k: binding[k] for k in serving_fields},
                     "after": {k: current[k] for k in serving_fields}}, now)
                return {"superseded": True, "reason": "serving_state_changed"}
            self.failure(now, str(error), reason_code=getattr(error, "reason_code", None), details=getattr(error, "details", None))
            return {"failure": str(error), "reason_code": getattr(error, "reason_code", str(error))}
        except Exception:
            self.failure(now, "AGENT_EXECUTION_FAILED")
            return {"failure": "AGENT_EXECUTION_FAILED"}

    def failure(self, now, code, *, reason_code=None, details=None):
        from .completion import abort
        abort(self.repo, now, reason_code or code, details)
        state = self.repo.state()
        state.pending_event = None
        state.status = "monitoring"
        state.next_inspection = now + self.inspection_ns
        # A durable wait also suppresses repeated anomaly wakeups during outage.
        from .budget import is_open
        if is_open(self.repo):
            state.status = "faulted"
            state.waiting_conditions = []
            self.repo.put("interruption-" + str(state.decision_version) + "-" + str(now), "research_interruption",
                          {"reason": reason_code or code, "event_time": now, "details": details,
                           "agent_completed": False}, now)
        else:
            from .state import WaitCondition
            import time
            identity = "wait-failure-" + str(state.decision_version) + "-" + str(now)
            wait = asdict(WaitCondition(identity, self.repo.key, self.repo.binding()["epoch"], "time", None,
                          {"state": "due"}, "event_time", now, time.time_ns(), now + self.inspection_ns,
                          code, state.decision_version))
            self.repo.put(identity, "wait_condition", wait, now)
            state.waiting_conditions = [wait]
        self.repo.save_state(state, state.decision_version)
        policy = (self.repo.binding()["policy"] or {}).get("failure_policy", "hold_current_state")
        self.repo.store.event(self.repo.key, "AGENT_CONTROLLER_FAILED", {"code": code, "failure_policy": policy}, now)
        if policy == "request_human":
            self.repo.store.event(self.repo.key, "AGENT_HUMAN_REPORT", {"message": "Agent unavailable; human review requested", "code": code}, now)
        if policy == "fallback_workflow":
            from .policies import fallback
            fallback(self.repo, now, code)
