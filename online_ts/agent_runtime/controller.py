from __future__ import annotations

import time
import uuid
import json
from dataclasses import asdict

from ..config import GateError, canonical, digest
from .context import build_context
from .state import DecisionRecord
from .budget import is_open


class DecisionSchemaError(GateError):
    def __init__(self, issues):
        super().__init__("AGENT_DECISION_SCHEMA")
        self.validation_errors = issues


class AgentController:
    """One model-selected action per turn. No predetermined investigation sequence."""
    def __init__(self, repository, gateway, model, *, context_builder=build_context):
        self.repo, self.gateway, self.model = repository, gateway, model
        self.context_builder = context_builder
        self.owner = "agent-controller-" + uuid.uuid4().hex

    def step(self, event):
        # Separate lease from Engine: slow LLM I/O must never block serving.
        lease_key = "agent-decision:" + self.repo.key
        timeout = (self.repo.binding()["policy"] or {}).get("timeout_seconds", 120)
        if not self.repo.store.acquire(lease_key, self.owner, 6 * timeout + 60):
            raise GateError("AGENT_DECISION_BUSY")
        try:
            pending = [r for r in self.repo.iter_records("decision") if r["decision_outcome"] == "prepared"]
            if pending:
                return self._execute(pending[-1])
            context = self.context_builder(self.repo, self.gateway, event)
            self.repo.check_binding(context["binding"])
            self.repo.put("context-" + context["context_version"], "context", context, event.created_at)
            for attempt in range(2):
                response = self.model(context)
                response_id = "response-" + context["context_version"]
                self.repo.put(response_id, "model_response", {"response": response}, event.created_at)
                try:
                    self._validate(response)
                    if any(ref not in context.get("visible_evidence_refs", []) for ref in response["evidence_refs"]):
                        raise GateError("AGENT_CITATION_SCOPE")
                except GateError as error:
                    self.repo.put("invalid-" + context["context_version"], "invalid_response",
                                  {"response_id": response_id, "error": str(error), "attempt": attempt}, event.created_at)
                    if attempt:
                        raise
                    self.repo.check_binding(context["binding"])
                    from copy import deepcopy
                    public = deepcopy({k: v for k, v in context.items() if k not in {"_audit", "binding", "context_version"}})
                    public["format_recovery"] = {"attempt": 1, "error": str(error),
                        "instruction": "Return a new legal response with action, reason, hypotheses, evidence_refs only. No tool ran for the rejected response."}
                    if isinstance(response, dict):
                        public["format_recovery"]["rejected_action"] = deepcopy(response.get("action"))
                    elif isinstance(response, str):
                        try:
                            json.loads(response)
                        except json.JSONDecodeError as syntax:
                            start = max(0, syntax.pos - 160)
                            public["format_recovery"]["json_syntax"] = {
                                "message": syntax.msg, "position": syntax.pos, "line": syntax.lineno,
                                "column": syntax.colno, "text_length": len(response),
                                "excerpt_start": start, "excerpt": response[start:syntax.pos + 80]}
                            public["format_recovery"]["instruction"] += (
                                " Correct JSON syntax and outer object boundaries. Close action after arguments; reason, hypotheses and evidence_refs belong at the top level.")
                    if hasattr(error, "validation_errors"):
                        public["format_recovery"]["validation_errors"] = error.validation_errors
                    if str(error) == "AGENT_CITATION_SCOPE":
                        public["format_recovery"].update(
                            rejected_action=deepcopy(response["action"]),
                            unsupported_refs=[ref for ref in response["evidence_refs"]
                                              if ref not in context.get("visible_evidence_refs", [])],
                            instruction="The rejected action did not run. Return a new response citing only visible_evidence_refs. Do not invent support; reconsider the action if its evidence is unavailable.")
                    if "session_history" in public:
                        from .work_context import fit_history
                        from .budget import CLOSING_RESERVATION
                        limit = (context["binding"]["policy"] or {}).get("max_call_tokens", 32768)
                        fit_history(public, min(limit, CLOSING_RESERVATION) if public.get("phase") == "closing" else limit)
                    context = {**public, "binding": context["binding"], "context_version": digest(public),
                               "_audit": {**context.get("_audit", {}), "outbound": deepcopy(public), "outbound_sha256": digest(public)}}
                    self.repo.put("context-" + context["context_version"], "context", context, event.created_at)
                    continue
                break
            self.repo.check_binding(context["binding"])
            decision_id = "decision-" + digest({"task": self.repo.key, "context": context["context_version"], "version": self.repo.state().decision_version})[:32]
            self.repo.put("context-" + decision_id, "context", context, event.created_at)
            record = asdict(DecisionRecord(
                decision_id, asdict(event), context["context_version"],
                [r["record_id"] for r in context["memory"]], [r["record_id"] for r in context["knowledge"]],
                [], response["hypotheses"],
                response["action"]["tool"], response["action"]["arguments"], response["reason"],
                "", event.created_at, context["binding"]))
            for field, available in (("cited_memory_refs", record["memory_refs"]),
                                     ("cited_knowledge_refs", record["knowledge_refs"])):
                refs = [ref for ref in response["evidence_refs"] if ref in available]
                record[field] = refs
            record["online_refs"] = [v["record_id"] for v in context.get("online_background", [])]
            visible = context.get("visible_evidence_refs", [])
            cited = response["evidence_refs"]
            if not isinstance(cited, list) or any(not isinstance(ref, str) or ref not in visible for ref in cited):
                raise GateError("AGENT_CITATION_SCOPE")
            record["visible_evidence_refs"] = visible
            record["cited_evidence_refs"] = cited
            record["tool_results_used"] = cited
            record["arguments_linked_to_evidence"] = [{"argument": k, "evidence_ref": v}
                for k, v in response["action"]["arguments"].items() if isinstance(v, str) and v in visible]
            record["environment_kind"] = getattr(self.gateway, "environment_kind", "production")
            self.repo.put(decision_id, "decision", record, event.created_at)
            return self._execute(record)
        finally:
            self.repo.store.release(lease_key, self.owner)

    def _execute(self, record):
        try:
            result = self.gateway.execute(record)
        except GateError as error:
            if str(error) == "TASK_ALREADY_RUNNING":
                return record  # Retry the prepared action locally, without another model call.
            result = {"error": str(error)}
            from .tools.feedback import ToolArgumentError
            if isinstance(error, ToolArgumentError):
                result["validation_errors"] = error.validation_errors
            if hasattr(error, "reason_code"):
                result.update(reason_code=error.reason_code, budget_details=error.details)
                if not is_open(self.repo):
                    state = self.repo.state()
                    state.phase = "closing"
                    self.repo.save_state(state, state.decision_version)
        except (TypeError, KeyError, ValueError, OverflowError):
            result = {"error": "AGENT_TOOL_ARGUMENTS"}
        return self.repo.finish(record, result)

    @staticmethod
    def _validate(value):
        issues = []
        for path, item, fields in (("$", value, {"action", "reason", "hypotheses", "evidence_refs"}),
                ("$.action", value.get("action") if isinstance(value, dict) else None, {"tool", "arguments"})):
            if not isinstance(item, dict):
                issues.append({"path": path, "expected": "JSON object", "required_fields": sorted(fields)})
            elif set(item) != fields:
                issues.append({"path": path, "missing_fields": sorted(fields - set(item)),
                               "unexpected_fields": sorted(set(item) - fields), "expected_fields": sorted(fields)})
        if issues:
            raise DecisionSchemaError(issues)
        for path, valid, expected in (
                ("$.action.tool", isinstance(value["action"]["tool"], str), "string"),
                ("$.action.arguments", isinstance(value["action"]["arguments"], dict), "JSON object"),
                ("$.reason", isinstance(value["reason"], str) and 1 <= len(value["reason"]) <= 3000, "string, 1..3000 characters"),
                ("$.hypotheses", isinstance(value["hypotheses"], list) and len(value["hypotheses"]) <= 8
                    and all(isinstance(s, str) and len(s) <= 1000 for s in value["hypotheses"]), "array, at most 8 strings of at most 1000 characters"),
                ("$.evidence_refs", isinstance(value["evidence_refs"], list) and len(value["evidence_refs"]) <= 32
                    and all(isinstance(ref, str) for ref in value["evidence_refs"]), "array, at most 32 strings")):
            if not valid:
                issues.append({"path": path, "expected": expected})
        if issues:
            raise DecisionSchemaError(issues)
        if len(canonical(value).encode()) > 16000:
            raise GateError("AGENT_DECISION_SIZE")
        if value["action"]["tool"] == "wait_until":
            from .tools.feedback import validate_wait_shape
            validate_wait_shape(value["action"]["arguments"])
