from __future__ import annotations

from dataclasses import asdict

from ...config import GateError, SECOND
from ..state import WaitCondition
from ..waiting import WAIT_SCHEMA
from ..trajectory import HISTORY_TOOL


READ_TOOLS = {
    "get_task_status": {},
    "get_current_model": {},
    "get_recent_events": {},
    "get_data_health": {},
    "get_label_maturity": {},
    "analyze_group_performance": {},
    "query_experiment_memory": {},
    "get_observer_report": {"record_id": "visible observer report ID"},
    HISTORY_TOOL: {"offset": "nonnegative position in the current incident history", "limit": "integer 1..4"},
    "wait_until": WAIT_SCHEMA,
    "get_wait_status": {"condition_id": "registered condition ID"},
    "schedule_inspection": {"due_at": "future event-time ns", "reason": "brief purpose"},
    "report_to_human": {"message": "string"},
}


class ToolGateway:
    def __init__(self, engine, repository):
        self.engine, self.repo = engine, repository
        self.config, self.artifacts = engine.config, engine.artifacts
        self.model_config = self.config
        self.environment_kind = "production"
        from ..observers import ObservationTeam
        self.observers = ObservationTeam(self)

    def describe_task(self):
        from ..facts import task_card, metric_spec
        metadata = self.repo.records("task_metadata", 1)
        return task_card(self.config, metadata=metadata[0] if metadata else None), metric_spec(self.config, protocol="rolling-origin-v1")

    def describe_candidates(self):
        from ...experiments import capabilities
        value = capabilities(self.config)
        from ...models import Candidate
        value["seasonal_period"] = {"required_candidate_value": self.config.search.seasonal_period,
            "unit": "observations", "must_be_explicit": True,
            "default_if_omitted": Candidate.__dataclass_fields__["period"].default}
        value["base_inputs"] = {"variables": list(self.config.variables), "automatically_included": True,
            "features": "Optional additional transforms only. features=[] still uses every original variable; ridge remains available."}
        if not self.config.search.feature_operations:
            value["feature_schema"] = None
        value["candidate_fields"] = {"all": ["family", "input_steps", "history_steps"],
                                     "ridge": ["alpha", "features"], "seasonal_naive": ["period"]}
        value["effective_training"] = {"last_value": "no_weight_optimization", "seasonal_naive": "no_weight_optimization", "ridge": "supervised_linear_regression"}
        if "chronos2" in self.config.search.model_families:
            value["candidate_fields"]["chronos2"] = ["model_id", "revision", "adaptation_mode", "quantile_levels",
                                                       "batch_size", "learning_rate", "num_steps", "lora_rank"]
            value["effective_training"]["chronos2"] = {"zero_shot": "pretrained_inference_no_weight_updates",
                                                        "lora": "supervised_adapter_training"}
        if "patchtst" in self.config.search.model_families:
            value["patchtst_required"] = {"model_id": "patchtst", "adaptation_mode": "from_scratch",
                                          "num_steps": self.config.patchtst.num_steps, "batch_size": self.config.patchtst.batch_size,
                                          "learning_rate": self.config.patchtst.learning_rate}
            value["effective_training"]["patchtst"] = "supervised_from_scratch_cpu"
        value["execution_backends"] = {"classical": "numpy-cpu", "patchtst": "torch-cpu",
                                        "chronos2": "torch-" + self.config.foundation.device,
                                        "local_target_meaning": "the host running this program"}
        value["experiment_purposes"] = {
            "optimization": "Compare valid candidate recipes; multiple fields may change. No single-factor causal claim.",
            "diagnosis": "At least one pair must differ in exactly one Candidate field after defaults are expanded. "
                         "Including the champion is not required and does not itself satisfy this constraint."}
        return value

    def refresh_facts(self, now):
        from ..facts import ExperimentLedger
        ledger = ExperimentLedger(self.repo)
        for experiment in list(self.repo.iter_records("experiment")):
            if experiment["binding"]["epoch"] != self.repo.binding()["epoch"]:
                continue
            from ...config import digest
            run_id = "train-" + digest({"task": self.engine.key, "experiment_id": experiment["record_id"]})[:24]
            try:
                job = self.repo.store.job(run_id)
            except GateError:
                continue
            for report in job["checkpoint"]["reports"]:
                if report.get("validation_artifact") and "score" in report:
                    ledger.record(report, self.config, self.artifacts, now=now, experiment_id=experiment["record_id"])

    def catalog(self):
        from .experiments import EXPERIMENT_TOOLS
        from .release import RELEASE_TOOLS
        from ..completion import COMPLETION_TOOLS
        return {**READ_TOOLS, **EXPERIMENT_TOOLS, **RELEASE_TOOLS, **COMPLETION_TOOLS,
                "get_tool_result": {"fact_id": "visible evidence ID"},
                "inspect_results": {"offset": "nonnegative integer", "limit": "integer 1..4"},
                "compare_candidates": {"result_id": "measured result ID", "reference_result_id": "measured reference ID"}}

    def execute(self, record):
        import uuid
        owner = "agent-tool-" + uuid.uuid4().hex
        duration = self.engine.config.source.retries * self.engine.config.source.timeout_seconds + 30
        if not self.engine.store.acquire(self.engine.key, owner, duration):
            raise GateError("TASK_ALREADY_RUNNING")
        try:
            return self._execute(record)
        finally:
            self.engine.store.release(self.engine.key, owner)

    def _execute(self, record):
        tool, args = record["selected_action"], record["arguments"]
        if tool not in self.catalog() or not isinstance(args, dict):
            raise GateError("AGENT_TOOL_NOT_ALLOWED")
        if set(args) != set(self.catalog()[tool]):
            from .feedback import ToolArgumentError
            schema = self.catalog()[tool]
            raise ToolArgumentError("AGENT_TOOL_ARGUMENTS", [{
                "path": "arguments", "missing_fields": sorted(set(schema) - set(args)),
                "unexpected_fields": sorted(set(args) - set(schema)), "expected_fields": schema}])
        if tool == "wait_until":
            from .feedback import validate_wait_shape
            validate_wait_shape(args)
        from .release import RELEASE_TOOLS, execute_release, recover_release
        if tool in RELEASE_TOOLS:
            recovered = recover_release(self, record)
            if recovered is not None:
                return recovered
        self.repo.check_binding(record["binding"])
        if "effect_at" not in record:
            record["effect_at"] = self.engine._state()["now"] or record["created_at"]
            self.repo.put(record["decision_id"], "decision", record, record["created_at"], replace=True)
        from ..budget import EXPLORATION_TOOLS, admit_experiment
        if tool in EXPLORATION_TOOLS:
            admit_experiment(self.repo)
        from ..completion import COMPLETION_TOOLS, execute
        if tool in COMPLETION_TOOLS:
            self.refresh_facts(record["created_at"])
            return execute(self, record)
        if tool == "get_tool_result":
            from ..evidence import EvidenceBoard
            return EvidenceBoard(self.repo).read(args["fact_id"], record["created_at"])
        if tool == "get_observer_report":
            return self.observers.read(args["record_id"], record["created_at"])
        if tool == HISTORY_TOOL:
            from ..trajectory import read
            return read(self.repo, record["effect_at"], args["offset"], args["limit"])
        if tool in {"inspect_results", "compare_candidates"}:
            from ..facts import ExperimentLedger, score_delta, configuration_delta
            rows = ExperimentLedger(self.repo).rows(record["created_at"])
            if tool == "inspect_results":
                if type(args["offset"]) is not int or args["offset"] < 0 or type(args["limit"]) is not int or not 1 <= args["limit"] <= 4:
                    raise GateError("AGENT_RESULT_PAGE")
                return {"rows": rows[args["offset"]:args["offset"] + args["limit"]], "total": len(rows)}
            row = next((r for r in rows if r["result_id"] == args["result_id"]), None)
            reference = next((r for r in rows if r["result_id"] == args["reference_result_id"]), None)
            if row is None or reference is None:
                raise GateError("AGENT_RESULT_SCOPE")
            return {"comparison": score_delta(row, reference),
                    "configuration_difference": configuration_delta(reference["effective_config"], row["effective_config"])}
        if tool in RELEASE_TOOLS:
            return execute_release(self, record)
        from .experiments import EXPERIMENT_TOOLS, Experiments
        if tool in EXPERIMENT_TOOLS:
            return Experiments(self).execute(record)
        now = record["effect_at"]
        if tool == "get_task_status":
            state = self.engine._state()
            return {name: state.get(name) for name in ("status", "champion", "challenger", "stage", "now", "last_issue", "stop_requested")}
        if tool == "get_current_model":
            state = self.engine._state()
            model = self.engine.store.model(state["champion"]) if state["champion"] else None
            return {"model_id": state["champion"], "candidate": model["report"]["candidate"] if model else None,
                    "model_since": state["model_since"], "final_test_metrics": "not_available_for_development"}
        if tool == "get_recent_events":
            # Lifecycle payloads can contain sealed final Test metrics. Project
            # only event identity; dedicated tools control access to evidence.
            return {"events": [{k: event[k] for k in ("id", "kind", "event_time")} for event in self.engine.store.events(self.engine.key) if event["event_time"] <= now][-15:]}
        if tool == "get_data_health":
            from ..observers import data_health
            return data_health(self.engine, self.repo, now)
        if tool == "get_label_maturity":
            from ..observers import label_status
            return label_status(self.engine, self.repo, now)
        if tool == "analyze_group_performance":
            from ..online_evidence import approved_online
            return {"observations": approved_online(self, now)}
        if tool == "query_experiment_memory":
            from ..memory import observations
            return {"observations": observations(self.repo, now), "experiences": self.repo.records("verified_experience", 4)}
        if tool == "report_to_human":
            if not isinstance(args["message"], str) or not 1 <= len(args["message"]) <= 3000:
                raise GateError("AGENT_REPORT_ARGUMENTS")
            self.engine.store.event(self.engine.key, "AGENT_HUMAN_REPORT", args, now, record["decision_id"])
            return {"reported": True}
        if tool == "wait_until":
            from ..waiting import register
            return register(self, record)
        if tool == "get_wait_status":
            wait = self.repo.get(args["condition_id"])
            if wait.get("task_key") != self.repo.key or wait.get("epoch") != self.repo.binding()["epoch"] or "condition_id" not in wait:
                raise GateError("AGENT_WAIT_SCOPE")
            return {"wait_condition": wait, "read_only": True}
        if tool == "schedule_inspection":
            identity = "inspection-" + record["decision_id"]
            try:
                return self.repo.get(identity)
            except GateError as error:
                if str(error) != "AGENT_RECORD_NOT_FOUND":
                    raise
            if type(args["due_at"]) is not int or args["due_at"] <= now or not isinstance(args["reason"], str) or not 1 <= len(args["reason"]) <= 1000:
                raise GateError("AGENT_INSPECTION_ARGUMENTS")
            value = {"record_id": identity, "epoch": self.repo.binding()["epoch"], "task_key": self.repo.key,
                     "status": "scheduled", "due_at": args["due_at"], "reason": args["reason"], "created_at": now}
            state = self.repo.state()
            state.next_inspection = args["due_at"]
            with self.repo.store.transaction() as db:
                self.repo.save_state(state, state.decision_version, db=db)
                self.repo.put(identity, "inspection_schedule", value, now, db=db)
            return value
        raise GateError("AGENT_TOOL_NOT_ALLOWED")
