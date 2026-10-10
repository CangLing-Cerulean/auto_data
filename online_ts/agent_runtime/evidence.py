"""Scoped, versioned tool evidence. Readback never accepts arbitrary artifact paths."""
import time

from ..config import GateError, SECOND, digest


FACT_TOOLS = {"get_data_health", "get_label_maturity", "analyze_group_performance", "get_current_model",
              "get_task_status", "get_experiment_report", "query_experiment_memory", "run_experiment",
              "inspect_results", "compare_candidates", "get_serving_state", "wait_until", "get_wait_status", "schedule_inspection", "get_final_status", "request_release", "submit_final"}
TRANSIENT = {"get_data_health", "get_label_maturity", "analyze_group_performance", "get_current_model",
             "get_task_status", "get_serving_state"}


class EvidenceBoard:
    def __init__(self, repo):
        self.repo = repo

    def capture(self, record, result, *, db=None):
        tool = record["selected_action"]
        if tool not in FACT_TOOLS or result.get("error"):
            return None
        config, _ = self.repo.store.task(self.repo.key)
        now = record.get("effect_at", record["created_at"])
        lifetime = max(30, config.monitoring.check_seconds * 2) * SECOND
        identity = "evidence-" + digest({"task": self.repo.key, "operation": record["decision_id"]})[:20]
        with self.repo.store.connection() as connection:
            if connection.execute("SELECT 1 FROM agent_runtime_records WHERE task_key=? AND record_id=?", (self.repo.key, identity)).fetchone():
                return identity
        scope = record["arguments"].get("experiment_id") or record["arguments"].get("candidate_id") or self.repo.key
        value = {"record_id": identity, "fact_id": identity, "kind": "tool_evidence", "task_key": self.repo.key,
                 "epoch": record["binding"]["epoch"], "policy_version": (record["binding"]["policy"] or {}).get("version"),
                 "source_tool": tool, "operation_id": record["decision_id"], "object_scope": scope,
                 "value": result, "event_time": now, "recorded_at": time.time_ns(), "visible_at": now,
                 "expires_at": int(now + lifetime) if tool in TRANSIENT else None,
                 "evidence_scope": "development", "version": digest(result),
                 "supports_hypotheses": [], "refutes_hypotheses": [],
                 "hypothesis_relation": "not_automatically_inferred"}
        self.repo.put(identity, "tool_evidence", value, now, db=db)
        return identity

    def _status(self, fact, now, latest):
        if fact["epoch"] != self.repo.binding()["epoch"]:
            return "previous_authority_epoch"
        if fact["policy_version"] != (self.repo.binding()["policy"] or {}).get("version"):
            return "previous_policy_version"
        if fact["source_tool"] == "analyze_group_performance":
            windows = self.repo.store.monitoring_windows(self.repo.key)
            for observation in fact["value"].get("observations", []):
                current = next((w for w in windows if w["model_id"] == observation.get("model_id")
                                and w["window_start"] == observation.get("window_start")
                                and w["window_end"] == observation.get("window_end")), None)
                if current is None or current["label_fingerprint"] != observation.get("label_fingerprint") or current["label_state"] not in {"complete", "revised"}:
                    return "source_revised"
        if fact["expires_at"] is not None and now > fact["expires_at"]:
            return "expired"
        if latest.get((fact["source_tool"], fact["object_scope"])) != fact["fact_id"]:
            return "superseded"
        return "current"

    def view(self, now):
        facts = [v for v in list(self.repo.iter_records("tool_evidence")) if v["visible_at"] <= now]
        facts.sort(key=lambda f: (f["recorded_at"], f["fact_id"]))
        latest = {(f["source_tool"], f["object_scope"]): f["fact_id"] for f in facts}
        return [{**f, "status": self._status(f, now, latest)} for f in facts]

    def read(self, fact_id, now):
        if not isinstance(fact_id, str):
            raise GateError("AGENT_EVIDENCE_ID")
        self.repo.check_binding(self.repo.binding())
        fact = self.repo.get(fact_id)
        if (fact.get("kind") != "tool_evidence" or fact.get("task_key") != self.repo.key
                or fact.get("evidence_scope") != "development" or fact["visible_at"] > now
                or fact["epoch"] != self.repo.binding()["epoch"]
                or fact.get("source_tool") not in FACT_TOOLS):
            raise GateError("AGENT_EVIDENCE_SCOPE")
        result = next(v for v in self.view(now) if v["fact_id"] == fact_id)
        return {"fact_id": fact_id, "version": fact["version"], "status": result["status"],
                "value": fact["value"], "event_time": fact["event_time"], "expires_at": fact["expires_at"],
                "usage": "historical_only_requery_for_current_state" if result["status"] != "current" else "current_development_evidence"}
