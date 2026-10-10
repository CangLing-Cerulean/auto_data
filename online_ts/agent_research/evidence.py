"""Immutable proposal/result/update records; statements are not correctness labels.

Trusted platform adapters must register evidence and execution receipts. This
component is not an execution sandbox and is not exposed as arbitrary DB access.
"""
import math
import re
import time

from ..agent_runtime.state import Repository
from ..config import GateError, canonical
from ..store import Store


def _text(value, name, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise GateError("RESEARCH_FIELD:" + name)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GateError("RESEARCH_ASSET_HASH")


class EvidenceJournal:
    def __init__(self, path, *, run_id, target_dataset, track, scope, harness_sha256, library_sha256):
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise GateError("RESEARCH_RUN_ID")
        _text(target_dataset, "target_dataset", 200)
        if track not in {"experience_combination", "new_task_discovery"} or scope not in {"engineering_only", "development_only"}:
            raise GateError("RESEARCH_RUN_SCOPE")
        _hash(harness_sha256)
        _hash(library_sha256)
        self.repo = Repository(Store(path), "agent-research:" + run_id)
        self.repo.initialize("Evidence bookkeeping for autonomous data-processing research")
        self.identity = {"run_id": run_id, "target_dataset": target_dataset, "track": track, "scope": scope,
                         "harness_sha256": harness_sha256, "library_sha256": library_sha256}
        self.repo.put("identity", "research_identity", self.identity, time.time_ns())

    def evidence(self, evidence_id, *, dataset_id, role, artifact_sha256, summary):
        _text(evidence_id, "evidence_id", 120)
        _text(dataset_id, "dataset_id", 200)
        _text(summary, "summary")
        _hash(artifact_sha256)
        if role not in {"training_diagnostics", "development_metrics", "historical_development"}:
            raise GateError("RESEARCH_EVIDENCE_ROLE")
        if role == "historical_development":
            if self.identity["track"] == "new_task_discovery" and dataset_id == self.identity["target_dataset"]:
                raise GateError("NEW_TASK_HISTORY_EXCLUDED")
        elif dataset_id != self.identity["target_dataset"]:
            raise GateError("RESEARCH_TASK_EVIDENCE")
        self.repo.put("evidence:" + evidence_id, "research_evidence", {"evidence_id": evidence_id,
            "dataset_id": dataset_id, "role": role, "artifact_sha256": artifact_sha256, "summary": summary}, time.time_ns())

    def _refs(self, refs):
        if not isinstance(refs, list) or not refs or len(refs) > 32:
            raise GateError("RESEARCH_EVIDENCE_REFS")
        for ref in refs:
            _text(ref, "evidence_ref", 120)
            self.repo.get("evidence:" + ref)
        if len(set(refs)) != len(refs):
            raise GateError("RESEARCH_EVIDENCE_REFS")

    def proposal(self, candidate_id, value):
        _text(candidate_id, "candidate_id", 120)
        required = {"observation", "hypothesis", "expected_effect", "falsification", "control", "evidence_refs", "code_sha256"}
        if not isinstance(value, dict) or set(value) != required:
            raise GateError("RESEARCH_PROPOSAL_SCHEMA")
        for field in ("observation", "hypothesis", "falsification", "control"):
            _text(value[field], field)
        effect = value["expected_effect"]
        if not isinstance(effect, dict) or set(effect) != {"metric", "target_scope", "direction", "criterion"}:
            raise GateError("RESEARCH_EXPECTATION_SCHEMA")
        for field in effect:
            _text(effect[field], field, 1000)
        if effect["direction"] not in {"decrease", "increase", "unchanged"}:
            raise GateError("RESEARCH_EXPECTATION_DIRECTION")
        self._refs(value["evidence_refs"])
        _hash(value["code_sha256"])
        record = {"candidate_id": candidate_id, "actor": "agent", "before_execution": value,
                  "diagnosis_correctness": "unassessed"}
        with self.repo.store.transaction() as db:
            # Idempotent repeat is allowed, but cannot insert or rewrite a
            # prediction once a receipt already exists.
            existing = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?",
                                  (self.repo.key, "proposal:" + candidate_id)).fetchone()
            if existing:
                if existing[0] != canonical(record):
                    raise GateError("RESEARCH_PROPOSAL_IMMUTABLE")
                return
            if db.execute("SELECT 1 FROM agent_runtime_records WHERE task_key=? AND record_id=?",
                          (self.repo.key, "receipt:" + candidate_id)).fetchone():
                raise GateError("RESEARCH_PROPOSAL_AFTER_EXECUTION")
            self.repo.put("proposal:" + candidate_id, "research_proposal", record, time.time_ns(), db=db)

    def receipt(self, candidate_id, *, status, metrics, artifact_sha256, reason):
        proposal = self.repo.get("proposal:" + candidate_id)
        _hash(artifact_sha256)
        if status not in {"completed", "failed"} or not isinstance(metrics, dict):
            raise GateError("RESEARCH_RECEIPT_SCHEMA")
        if any(not isinstance(k, str) or type(v) not in {int, float} or not math.isfinite(v) for k, v in metrics.items()):
            raise GateError("RESEARCH_NONFINITE_METRICS")
        _text(reason, "receipt_reason")
        if status == "failed" and metrics:
            raise GateError("FAILED_EXECUTION_HAS_METRICS")
        self.repo.put("receipt:" + candidate_id, "research_receipt", {"candidate_id": candidate_id,
            "proposal_code_sha256": proposal["before_execution"]["code_sha256"], "actor": "trusted_executor",
            "status": status, "metrics": metrics, "artifact_sha256": artifact_sha256,
            "reason": reason, "scope": self.identity["scope"]}, time.time_ns())

    def update(self, candidate_id, value):
        self.repo.get("proposal:" + candidate_id)
        receipt = self.repo.get("receipt:" + candidate_id)
        if not isinstance(value, dict) or set(value) != {"assessment", "reason", "revised_hypothesis", "next_action", "receipt_sha256"}:
            raise GateError("RESEARCH_UPDATE_SCHEMA")
        if value["assessment"] not in {"supported", "contradicted", "inconclusive"}:
            raise GateError("RESEARCH_ASSESSMENT")
        for field in ("reason", "revised_hypothesis", "next_action"):
            _text(value[field], field)
        if value["receipt_sha256"] != receipt["artifact_sha256"]:
            raise GateError("RESEARCH_UPDATE_WRONG_RECEIPT")
        if receipt["status"] == "failed" and value["assessment"] != "inconclusive":
            raise GateError("EXECUTION_FAILURE_IS_NOT_HYPOTHESIS_TEST")
        self.repo.put("update:" + candidate_id, "research_update", {"candidate_id": candidate_id,
            "actor": "agent", "after_execution": value, "diagnosis_correctness": "unassessed"}, time.time_ns())

    def intervention(self, intervention_id, *, reason, changed_assets):
        _text(intervention_id, "intervention_id", 120)
        _text(reason, "intervention_reason")
        if not isinstance(changed_assets, list) or not changed_assets or len(changed_assets) > 100:
            raise GateError("RESEARCH_INTERVENTION_ASSETS")
        for asset in changed_assets:
            _text(asset, "changed_asset", 1000)
        self.repo.put("intervention:" + intervention_id, "research_intervention", {"actor": "human",
            "reason": reason, "changed_assets": changed_assets}, time.time_ns())

    def summary(self):
        proposals = list(self.repo.iter_records("research_proposal"))
        receipts = {r["candidate_id"]: r for r in self.repo.iter_records("research_receipt")}
        updates = {r["candidate_id"]: r for r in self.repo.iter_records("research_update")}
        interventions = list(self.repo.iter_records("research_intervention"))
        return {"identity": self.identity, "proposals": len(proposals), "receipts": len(receipts),
                "updates": len(updates), "interventions": interventions,
                "autonomy_eligible": not interventions and self.identity["scope"] != "engineering_only",
                "completed_agent_experiment": False, "diagnosis_correctness": "unassessed",
                "pending_execution": [p["candidate_id"] for p in proposals if p["candidate_id"] not in receipts],
                "pending_update": [i for i in receipts if i not in updates]}
