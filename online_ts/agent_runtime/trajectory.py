"""Reconstruct a scoped working conversation from committed decisions and tool receipts."""
from copy import deepcopy
import json

from ..config import GateError
from .evidence import EvidenceBoard


HISTORY_TOOL = "read_session_history"
PAGE_LIMIT = 4


def _observation(record, facts):
    result = deepcopy(record["actual_result"])
    if record["selected_action"] == HISTORY_TOOL and not result.get("error"):
        # Reading history must not recursively embed earlier history pages.
        result = {key: result[key] for key in ("offset", "total_count", "next_offset") if key in result} | {
            "returned_sequences": [entry["sequence"] for entry in record["actual_result"].get("entries", [])],
            "scope": "historical_page_receipt; original entries are available through read_session_history"}
    fact_id = record.get("actual_tool_effect", {}).get("evidence_id")
    if record["selected_action"] == "get_tool_result" and not result.get("error"):
        fact_id = result.get("fact_id")
    fact = facts.get(fact_id)
    evidence = [] if fact is None else [{key: fact[key] for key in
        ("fact_id", "source_tool", "status", "event_time", "expires_at", "version")} | {
            "usage": "current_development_evidence" if fact["status"] == "current" else "historical_only"}]
    return {"outcome": record["decision_outcome"], "result": result, "evidence": evidence,
            "observed_at": record.get("effect_at", record["created_at"]), "projection": "full"}


def build(repo, now):
    state, binding = repo.state(), repo.binding()
    history = {"incident_id": state.incident_id, "epoch": binding["epoch"], "as_of": now,
               "scope": "committed actions in this task, authority epoch and incident; assessments are model judgments",
               "total_count": 0, "entries": [], "omitted_count": 0, "omitted_range": None,
               "readback": {"tool": HISTORY_TOOL, "offset": 0, "limit": PAGE_LIMIT}}
    if state.incident_id is None:
        return history
    # The saved request already carries both the incident and decision version.
    # Joining it avoids adding another source of truth or relying on timestamp ties.
    with repo.store.connection() as db:
        rows = db.execute("""
            SELECT d.payload AS decision, c.payload AS context
            FROM agent_runtime_records d JOIN agent_runtime_records c
              ON c.task_key=d.task_key AND c.record_id='context-' || d.record_id AND c.kind='context'
            WHERE d.task_key=? AND d.kind='decision'
              AND json_extract(d.payload,'$.binding.epoch')=?
              AND json_extract(c.payload,'$.state.incident_id')=?
              AND json_extract(d.payload,'$.decision_outcome') IN ('completed','failed')
              AND json_extract(c.payload,'$.state.decision_version') < ?
            ORDER BY json_extract(c.payload,'$.state.decision_version'), d.record_id
            """, (repo.key, binding["epoch"], state.incident_id, state.decision_version)).fetchall()
    facts = {fact["fact_id"]: fact for fact in EvidenceBoard(repo).view(now)
             if fact["task_key"] == repo.key and fact["epoch"] == binding["epoch"]
             and fact["evidence_scope"] == "development"}
    for row in rows:
        record, context = json.loads(row["decision"]), json.loads(row["context"])
        if (record["created_at"] > now or record.get("effect_at", record["created_at"]) > now
                or not isinstance(record.get("actual_result"), dict)):
            continue
        history["entries"].append({
            "sequence": context["state"]["decision_version"] + 1, "decision_id": record["decision_id"],
            "created_at": record["created_at"],
            "action": {"tool": record["selected_action"], "arguments": deepcopy(record["arguments"])},
            "assessment": {"source": "model_judgment", "reason": record["reason_summary"],
                           "hypotheses": deepcopy(record["hypotheses"]),
                           "evidence_refs": list(record["cited_evidence_refs"])},
            "observation": _observation(record, facts)})
    history["total_count"] = len(history["entries"])
    return history


def read(repo, now, offset, limit):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= PAGE_LIMIT:
        raise GateError("AGENT_HISTORY_PAGE")
    history = build(repo, now)
    entries = history["entries"][offset:offset + limit]
    following = offset + len(entries)
    return {key: history[key] for key in ("incident_id", "epoch", "as_of", "scope", "total_count")} | {
        "offset": offset, "entries": entries,
        "next_offset": following if following < history["total_count"] else None}


def visible_refs(history):
    return [fact["fact_id"] for entry in history.get("entries", [])
            for fact in entry["observation"]["evidence"]]
