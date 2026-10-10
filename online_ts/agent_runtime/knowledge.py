"""OS-authenticated human knowledge lifecycle, separate from release policy."""
from ..config import GateError, digest
from ..copilot import current_actor


class HumanKnowledge:
    def __init__(self, repo):
        self.repo = repo

    def _require(self):
        actor = current_actor()
        config, _ = self.repo.store.task(self.repo.key)
        if "knowledge" not in config.collaboration.permissions.get(actor, ()):
            raise GateError("HUMAN_PERMISSION_DENIED")
        return actor

    def propose(self, *, category, text, source, entity_scope, time_scope, now):
        actor = self._require()
        if category not in {"preference", "hypothesis", "business_fact", "annotation", "opinion"}:
            raise GateError("KNOWLEDGE_NOT_POLICY_AUTHORITY")
        if (not all(isinstance(s, str) and 0 < len(s) <= 1200 for s in (text, source))
                or not isinstance(entity_scope, list) or any(not isinstance(s, str) for s in entity_scope)
                or not isinstance(time_scope, list) or len(time_scope) != 2
                or any(type(n) is not int for n in time_scope) or time_scope[0] > time_scope[1]):
            raise GateError("KNOWLEDGE_SCOPE")
        value = {"category": category, "text": text, "source": source, "provider": actor,
                 "scope": "task", "entity_scope": entity_scope, "time_scope": time_scope,
                 "task_scope": self.repo.key, "truth_status": "human_statement",
                 "confirmation_status": "proposed", "version": 1, "created_at": now}
        value["record_id"] = "knowledge-" + digest(value)[:24]
        self.repo.put(value["record_id"], "knowledge", value, now)
        return value

    def transition(self, record_id, status, *, now):
        actor = self._require()
        if status not in {"confirmed", "superseded", "revoked"}:
            raise GateError("KNOWLEDGE_STATUS")
        value = self.repo.get(record_id)
        if value.get("task_scope") != self.repo.key or "confirmation_status" not in value:
            raise GateError("KNOWLEDGE_SCOPE")
        if value["confirmation_status"] in {"superseded", "revoked"}:
            raise GateError("KNOWLEDGE_TERMINAL")
        self.repo.put(record_id + ":v" + str(value["version"]), "knowledge_history", value, now)
        value.update(confirmation_status=status, confirmed_by=actor, version=value["version"] + 1)
        self.repo.put(record_id, "knowledge", value, now, replace=True)
        self.repo.store.event(self.repo.key, "HUMAN_KNOWLEDGE_UPDATED", {"record_id": record_id, "status": status}, now,
                              record_id + ":v" + str(value["version"]))
        return value

    def relevant(self, now, entities=()):
        return [v for v in self.repo.records("knowledge", 100) if v["confirmation_status"] == "confirmed"
                and v["task_scope"] == self.repo.key and v["created_at"] <= now
                and v["time_scope"][0] <= now <= v["time_scope"][1]
                and (not entities or not v["entity_scope"] or set(entities) & set(v["entity_scope"]))][:6]

    def import_copilot(self, intent_id, *, entity_scope, time_scope, now):
        """An approved legacy note becomes proposed, requiring explicit truth confirmation."""
        self._require()
        import json
        with self.repo.store.connection() as db:
            row = db.execute("SELECT payload,status FROM human_intents WHERE task_key=? AND intent_id=?", (self.repo.key, intent_id)).fetchone()
        if not row or row["status"] != "approved":
            raise GateError("KNOWLEDGE_INTENT_NOT_APPROVED")
        knowledge = json.loads(row["payload"]).get("knowledge")
        if not knowledge:
            raise GateError("KNOWLEDGE_INTENT_REQUIRED")
        return self.propose(category=knowledge["category"], text=knowledge["text"], source="copilot:" + intent_id,
                            entity_scope=entity_scope, time_scope=time_scope, now=now)
