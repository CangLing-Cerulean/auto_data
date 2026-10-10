"""Reuse Controller/Repository/Store; replace forecasting operations with selection tools."""
from dataclasses import asdict
import re
import time
import math
from types import SimpleNamespace

import numpy as np

from ..config import GateError, canonical, digest
from ..store import Store
from ..agent_runtime.controller import AgentController
from ..agent_runtime.state import AgentState, Event, Repository
from ..agent_runtime.budget import AgentBudgetError, snapshot
from ..agent_runtime.llm import DeepSeekModel
from .common import STUDY, StudyConfig, read_json, write_json, verify_freeze
from .dataset import TrainingData
from .rules import compile_rule, check_mask, mask_hash, random_rule


class StudyRepository(Repository):
    def __init__(self, frozen, run_id, arm):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id) or arm not in {"agent", "random", "uniform"}:
            raise GateError("STUDY_RUN_ID")
        self.frozen, self.run_id, self.arm = frozen, run_id, arm
        self.config = StudyConfig(**frozen["config"])
        super().__init__(Store(STUDY / "runtime.sqlite"), run_id)
        self.folder = STUDY / "runs" / run_id
        self.folder.mkdir(parents=True, exist_ok=True)
        identity = {"harness_sha256": frozen["sha256"], "arm": arm, "run_id": run_id}
        meta = list(self.iter_records("run_metadata"))
        if meta:
            if any(meta[0][k] != v for k, v in identity.items()):
                raise GateError("STUDY_RESUME_IDENTITY")
        else:
            self.initialize("Autonomously select training windows for fixed DLinear; minimize development NMSE with NMAE constraint.")
            self.put("run-metadata", "run_metadata", {**identity, "created_wall": time.time()}, time.time_ns())
            state = self.state()
            state.incident_id = run_id
            state.status = "investigating"
            self.save_state(state, 0)

    def binding(self):
        return {"task_key": self.key, "owner": "agent", "epoch": 1, "stop_requested": False,
                "harness_sha256": self.frozen["sha256"], "policy": {
                    "execution_profile": "bounded", "max_calls": self.config.max_calls,
                    "max_experiments": self.config.max_candidates, "max_llm_tokens": self.config.max_llm_tokens,
                    "max_call_tokens": self.config.max_call_tokens, "timeout_seconds": self.config.timeout_seconds}}

    def finish(self, record, result):
        with self.store.transaction() as db:
            old = self.get(record["decision_id"])
            if old["decision_outcome"] != "prepared":
                return old
            state = self.state()
            record.update(actual_result=result, decision_outcome="failed" if result.get("error") else "completed",
                          hypothesis_update=record["hypotheses"], result_refs=[])
            evidence = "selection-evidence-" + record["decision_id"]
            self.put(evidence, "selection_evidence", {"record_id": evidence, "tool": record["selected_action"],
                     "result": result, "scope": "training_diagnostics_or_development_metrics"}, record["created_at"], db=db)
            record["actual_tool_effect"] = {"evidence_id": evidence, "operation_id": record["decision_id"]}
            state.decision_version += 1
            state.current_hypotheses = record["hypotheses"]
            state.last_result = {"decision_id": record["decision_id"], "tool": record["selected_action"], "result": result}
            if result.get("agent_finalized"):
                state.incident_finished, state.status = True, "completed"
            self.save_state(state, state.decision_version - 1, db=db)
            self.put(record["decision_id"], "decision", record, record["created_at"], replace=True, db=db)
        self.export()
        return record

    def export(self):
        # DB is authoritative. Export rebuilding cannot duplicate journal entries.
        rows = sorted(self.iter_records("decision"), key=lambda r: (r["created_at"], r["decision_id"]))
        trace = self.folder / "trace.jsonl"
        trace.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")
        with self.store.connection() as db:
            calls = [dict(r) for r in db.execute("SELECT * FROM agent_calls WHERE task_key=? ORDER BY created_ns", (self.key,))]
        for call in calls:
            call["payload"] = __import__("json").loads(call["payload"])
        (self.folder / "model_calls.jsonl").write_text("".join(canonical(r) + "\n" for r in calls), encoding="utf-8")
        write_json(self.folder / "status.json", {"run_id": self.run_id, "arm": self.arm,
                   "state": asdict(self.state()), "budget": snapshot(self),
                   "attempts": list(self.iter_records("candidate_attempt")),
                   "training_costs": list(self.iter_records("training_cost")),
                   "evaluation_scope": "development_only"})


TOOLS = {"describe_task": {}, "profile_training_data": {},
         "check_selection": {"rule": "structured expression"},
         "evaluate_candidate": {"rule": "structured expression"},
         "inspect_results": {}, "submit_policy": {}}


class SelectionGateway:
    environment_kind = "selection"

    def __init__(self, repo, data, evaluator):
        self.repo, self.data, self.evaluator = repo, data, evaluator
        self.config = repo.config

    def catalog(self):
        return TOOLS

    def results(self):
        return sorted(self.repo.iter_records("candidate_attempt"), key=lambda r: r["index"])

    def baseline(self):
        return self.repo.get("baseline")["metrics"]

    def execute(self, record):
        tool, args = record["selected_action"], record["arguments"]
        if tool not in TOOLS or set(args) != set(TOOLS[tool]):
            raise GateError("SELECTION_TOOL_ARGUMENTS")
        self.repo.check_binding(record["binding"])
        if self.repo.state().incident_finished:
            raise GateError("RUN_ALREADY_SUBMITTED")
        if tool == "describe_task":
            return {"task": {k: self.data.task[k] for k in ("task_id", "dataset_id", "legal_windows", "sensor_columns", "mapping", "roles")},
                    "trainer": trainer_spec(self.data.task["protocol"]), "constraints": asdict(self.config),
                    "metric": {"primary": "NMSE", "secondary": "NMAE"}, "rule_grammar": RULE_HELP}
        if tool == "profile_training_data":
            return self.data.profile()
        if tool in {"check_selection", "evaluate_candidate"}:
            mask = compile_rule(args["rule"], self.data, self.config)
            audit = check_mask(mask, self.data, self.config)
            if tool == "check_selection" or not audit["valid"]:
                return audit | ({"error": "SELECTION_COVERAGE_REJECTED"} if not audit["valid"] else {})
            return self.evaluate(args["rule"], mask, audit)
        if tool == "inspect_results":
            return {"baseline": self.baseline(), "candidates": self.results()}
        if tool == "submit_policy":
            return self.submit("agent_stop")
        raise GateError("SELECTION_TOOL")

    def evaluate(self, rule, mask, audit):
        identity = mask_hash(mask)
        previous = next((r for r in self.results() if r["mask_sha256"] == identity), None)
        if previous:
            return {**previous, "recovered": True}
        rows = self.results()
        if len(rows) >= self.config.max_candidates:
            raise GateError("CANDIDATE_BUDGET")
        if self.repo.state().phase == "closing":
            raise GateError("RUN_CLOSING")
        attempt_id = "candidate-" + identity
        now = time.time_ns()
        value = {"record_id": attempt_id, "index": len(rows), "rule": rule, "mask_sha256": identity,
                 "audit": audit, "status": "prepared", "seed": self.config.search_seed}
        self.repo.put(attempt_id, "candidate_attempt", value, now)
        try:
            metrics = self.evaluator(mask, self.config.search_seed)
            validate_metrics(metrics)
            value.update(status="completed", metrics=metrics)
        except Exception as error:
            # Keep a failed training in the denominator. The exact failure is in
            # the evaluator's job receipt; no hidden strategy repair is performed.
            value.update(status="failed", error=str(error) if isinstance(error, GateError) else "CANDIDATE_EXECUTION_FAILED")
        self.repo.put(attempt_id, "candidate_attempt", value, now, replace=True)
        self.repo.export()
        return value

    def recover_attempts(self):
        for row in self.results():
            if row["status"] != "prepared":
                continue
            mask = compile_rule(row["rule"], self.data, self.config)
            try:
                metrics = self.evaluator(mask, self.config.search_seed)
                validate_metrics(metrics)
                row.update(status="completed", metrics=metrics)
            except Exception as error:
                row.update(status="failed", error=str(error) if isinstance(error, GateError) else "CANDIDATE_EXECUTION_FAILED")
            self.repo.put(row["record_id"], "candidate_attempt", row, time.time_ns(), replace=True)

    def submit(self, reason):
        path = self.repo.folder / "final_policy.json"
        if path.exists():
            from .common import file_hash
            result = read_json(path)
            if (result["harness_sha256"] != self.repo.frozen["sha256"] or result["run_id"] != self.repo.run_id
                    or file_hash(self.repo.folder / "selected_windows.npz") != result["selected_windows_sha256"]):
                raise GateError("FINAL_POLICY_ASSET_CHANGED")
            self.repo.put("final-policy", "final_policy", result, time.time_ns())
            state = self.repo.state()
            state.incident_finished, state.status = True, "completed"
            self.repo.save_state(state, state.decision_version)
            self.repo.export()
            return {"agent_finalized": True, "final_policy": result, "recovered": True}
        baseline = self.baseline()
        candidates = [{"rule": {"kind": "all"}, "metrics": baseline, "index": -1}]
        candidates += [r for r in self.results() if r["status"] == "completed"
                       and r["metrics"]["NMAE"] <= baseline["NMAE"] * (1 + self.config.max_nmae_regression)]
        best = min(candidates, key=lambda r: (r["metrics"]["NMSE"], r["index"]))
        mask = compile_rule(best["rule"], self.data, self.config)
        target = self.repo.folder / "selected_windows.npz"
        np.savez_compressed(target, retained=mask, start_row=self.data.starts,
                            selected_start_row=self.data.starts[mask],
                            source_index_start=self.data.arrays["source_index_start"][mask],
                            source_index_end_inclusive=self.data.arrays["source_index_end_inclusive"][mask])
        from .common import file_hash
        result = {"run_id": self.repo.run_id, "arm": self.repo.arm, "harness_sha256": self.repo.frozen["sha256"],
                  "rule": best["rule"], "metrics": best["metrics"], "baseline": baseline,
                  "mask_sha256": mask_hash(mask), "selected_windows_sha256": file_hash(target),
                  "retained": int(mask.sum()), "termination": reason,
                  "nmse_gain": (baseline["NMSE"] - best["metrics"]["NMSE"]) / baseline["NMSE"] if baseline["NMSE"] else 0.,
                  "evaluation_scope": "development_only", "independent_confirmation": "not_performed"}
        write_json(path, result)
        self.repo.put("final-policy", "final_policy", result, time.time_ns())
        state = self.repo.state()
        state.incident_finished, state.status = True, "completed"
        self.repo.save_state(state, state.decision_version)
        self.repo.export()
        return {"agent_finalized": True, "final_policy": result}


RULE_HELP = {"all": {"kind": "all"}, "compare": {"kind": "compare", "feature": "one frozen feature",
             "op": "lt|le|gt|ge", "value": "finite number", "scale": "quantile|literal"},
             "combine": {"kind": "and|or", "children": "2..3 rules"}, "negate": {"kind": "not", "child": "rule"}}


def validate_metrics(metrics):
    if any(type(metrics.get(k)) not in {int, float} or not math.isfinite(metrics[k]) or metrics[k] < 0
           for k in ("NMSE", "NMAE")):
        raise GateError("TRAINING_METRIC_SCHEMA")


def result_summary(result):
    if "candidates" in result:
        result = {**result, "candidates": [result_summary(r) for r in result["candidates"]]}
    if "audit" in result:
        result = {k: v for k, v in result.items() if k != "audit"} | {
            "coverage": {k: result["audit"][k] for k in ("valid", "retained", "retention", "issues")}}
    if "groups" in result:
        result = {k: v for k, v in result.items() if k != "groups"} | {"group_count": len(result["groups"])}
    return result


def context_builder(repo, gateway, event):
    decisions = sorted(repo.iter_records("decision"), key=lambda r: r["created_at"])
    evidence = list(repo.iter_records("selection_evidence"))
    context = {"goal": repo.state().task_goal, "state": asdict(repo.state()), "binding": repo.binding(),
               "phase": repo.state().phase, "memory": [], "knowledge": [],
               "task": {"dataset_id": gateway.data.task["dataset_id"], "trainer": trainer_spec(gateway.data.task["protocol"]), "rules": RULE_HELP},
               "tools": gateway.catalog(), "features": list(gateway.data.features),
               "budget": snapshot(repo), "baseline": gateway.baseline(), "results": [result_summary(r) for r in gateway.results()],
               "constraints": asdict(repo.config),
               "visible_evidence_refs": [e["record_id"] for e in evidence],
               "recent_history": [{"action": {"tool": r["selected_action"], "arguments": r["arguments"]},
                   "reason": r["reason_summary"], "hypotheses": r["hypotheses"], "result": result_summary(r["actual_result"])}
                   for r in decisions[-6:]], "trigger": asdict(event)}
    if repo.state().phase == "closing":
        context["recent_history"] = []
        context["results"] = [{k: r[k] for k in ("index", "status", "metrics") if k in r} for r in gateway.results()]
        context["tools"] = {"submit_policy": {}}
    context["context_version"] = digest(context)
    return context


def trainer_spec(protocol):
    return {k: v for k, v in protocol.items() if k not in {"max_data_rounds", "session_gpu_wall_budget_seconds", "confirmation_seeds"}}


def run(freeze_path, run_id, arm, *, evaluator=None, model=None):
    frozen = verify_freeze(freeze_path)
    repo = StudyRepository(frozen, run_id, arm)
    data = TrainingData(frozen["task"])
    if evaluator is None:
        from .execution import CandidateEvaluator
        evaluator = CandidateEvaluator(repo, data, freeze_path)
    gateway = SelectionGateway(repo, data, evaluator)
    if (repo.folder / "final_policy.json").exists():
        return gateway.submit("resume_final")["final_policy"]
    if arm == "agent" and model is None:
        import os
        if not os.environ.get("DEEPSEEK_API_KEY"):
            raise GateError("DEEPSEEK_API_KEY_REQUIRED")
        config = SimpleNamespace(planner=SimpleNamespace(kind="external", allow_external=True,
            endpoint_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY", model=repo.config.model,
            timeout_seconds=repo.config.timeout_seconds))
        model = DeepSeekModel(repo, config, system_prompt=(STUDY / "prompts/selection.txt").read_text(encoding="utf-8"))
    if not list(repo.iter_records("baseline")):
        baseline = evaluator(np.ones(len(data.starts), bool), repo.config.search_seed)
        validate_metrics(baseline)
        repo.put("baseline", "baseline", {"metrics": baseline}, time.time_ns())
    gateway.recover_attempts()
    created = repo.get("run-metadata")["created_wall"]
    if arm == "uniform":
        return gateway.submit("uniform_baseline")["final_policy"]
    if arm == "random":
        rng = np.random.default_rng(repo.config.random_seed + int(digest(run_id)[:8], 16))
        # Replaying from the seed recreates proposal identities; completed masks
        # are read back without training again. Rejections consume proposal slots.
        for index in range(repo.config.max_calls):
            if time.time() - created >= repo.config.total_seconds or len(gateway.results()) >= repo.config.max_candidates:
                break
            rule = random_rule(rng)
            operation = "random-proposal-" + str(index)
            if list(r for r in repo.iter_records("random_proposal") if r["record_id"] == operation):
                continue
            mask = compile_rule(rule, data, repo.config)
            audit = check_mask(mask, data, repo.config)
            result = gateway.evaluate(rule, mask, audit) if audit["valid"] else audit
            repo.put(operation, "random_proposal", {"record_id": operation, "rule": rule, "result": result}, time.time_ns())
        return gateway.submit("random_budget_complete")["final_policy"]
    controller = AgentController(repo, gateway, model, context_builder=context_builder)
    try:
        while not repo.state().incident_finished:
            if time.time() - created >= repo.config.total_seconds:
                return gateway.submit("wall_time_limit")["final_policy"]
            remaining = snapshot(repo)
            if remaining["remaining_calls"] <= 2 or len(gateway.results()) >= repo.config.max_candidates:
                state = repo.state()
                state.phase = "closing"
                repo.save_state(state, state.decision_version)
            controller.step(Event("selection-step-" + str(repo.state().decision_version), "selection", time.time_ns()))
    except AgentBudgetError:
        return gateway.submit("llm_budget_exhausted")["final_policy"]
    except Exception:
        repo.put("runtime-failure", "failure", {"reason": "AGENT_EXECUTION_FAILED"}, time.time_ns(), replace=True)
        repo.export()
        return gateway.submit("agent_execution_failed")["final_policy"]
    return read_json(repo.folder / "final_policy.json")
