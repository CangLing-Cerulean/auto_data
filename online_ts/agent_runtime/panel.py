"""Offline panel adapter using the same candidate worker as the frozen workflow."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..config import GateError, TaskConfig, digest
from ..data import ArtifactStore
from ..models import Candidate
from .budget import admit_experiment, is_open
from .completion import COMPLETION_TOOLS, execute as complete
from .evidence import EvidenceBoard
from .facts import ExperimentLedger, effective_candidate, task_card, metric_spec, score_delta, configuration_delta


class PanelGateway:
    def __init__(self, repo, config, folder, output, *, metadata=None):
        from scripts.benchmark_data_split import file_hash
        self.repo, self.config, self.folder, self.output = repo, config, Path(folder), Path(output)
        self.artifacts = ArtifactStore(self.output / "artifacts")
        self.environment_kind = "panel"
        self.model_config = TaskConfig.from_dict(json.loads((self.folder / "config.json").read_text(encoding="utf-8")))
        left = {k: v for k, v in config.to_dict().items() if k not in {"planner", "collaboration"}}
        right = {k: v for k, v in self.model_config.to_dict().items() if k not in {"planner", "collaboration"}}
        if left != right:
            raise GateError("PANEL_EXECUTOR_CONFIG_MISMATCH")
        self.protocol = json.loads((self.folder / "protocol.json").read_text(encoding="utf-8"))
        if file_hash(self.folder / "arrays.npz") != self.protocol["arrays_sha256"]:
            raise GateError("BENCHMARK_PREPARED_HASH")
        self.profile = json.loads((self.folder / "profile.json").read_text(encoding="utf-8"))
        self.pool = {Candidate.from_dict(c).candidate_id: c for c in self.protocol["candidate_pool"]}
        self.metadata = metadata

    def attempts(self):
        return sorted(list(self.repo.iter_records("panel_attempt")), key=lambda a: a["index"])

    def status(self):
        attempts = self.attempts()
        charged = sum(a.get("charged_seconds", a["reservation_seconds"]) for a in attempts)
        return {"candidate_pool_size": len(self.pool), "untested_candidates": len(self.pool) - len(attempts),
                "executed_attempts": len(attempts), "completed_attempts": sum(a["status"] == "completed" for a in attempts),
                "pending_attempts": sum(a["status"] == "prepared" for a in attempts),
                "remaining_execution_slots": None if is_open(self.repo) else max(0, self.config.budget.max_candidates - len(attempts)),
                "worker_seconds_charged_or_reserved": charged,
                "remaining_worker_seconds": None if is_open(self.repo) else max(0., self.config.budget.max_training_seconds - charged)}

    def describe_task(self):
        return (task_card(self.config, protocol=self.protocol["version"], panel=self.protocol, metadata=self.metadata),
                metric_spec(self.config, protocol=self.protocol["version"], scale=self.protocol["scale"]))

    def describe_candidates(self):
        candidates = {cid: effective_candidate(c, self.config) for cid, c in self.pool.items()}
        for c in candidates.values():
            c.pop("architecture", None)
        return {"candidate_pool": candidates, **self.status(), "selection": "choose a candidate_id; no automatic experiment order"}

    def catalog(self):
        return {"get_data_health": {}, "get_task_status": {},
                "run_experiment": {"candidate_id": "ID from candidate_pool", "parent_result_id": "measured parent result ID or null", "hypothesis": "brief testable rationale"},
                "get_experiment_report": {"result_id": "measured result ID"},
                "inspect_results": {"offset": "nonnegative integer", "limit": "integer 1..4"},
                "compare_candidates": {"result_id": "measured result ID", "reference_result_id": "measured reference ID"},
                "get_tool_result": {"fact_id": "visible evidence ID"}, **COMPLETION_TOOLS}

    def refresh_facts(self, now):
        # Recovery only reads completed immutable reports; it does not launch another worker.
        for attempt in self.attempts():
            if attempt["status"] == "completed":
                report = json.loads((self.output / attempt["result_file"]).read_text(encoding="utf-8"))
                ExperimentLedger(self.repo).record(report, self.config, self.artifacts, now=attempt["completed_at"],
                    experiment_id=attempt["record_id"], parent_result_id=attempt["parent_result_id"], protocol=self.protocol["version"])

    def execute(self, record):
        self.repo.check_binding(record["binding"])
        tool, args = record["selected_action"], record["arguments"]
        if tool not in self.catalog() or set(args) != set(self.catalog()[tool]):
            raise GateError("AGENT_TOOL_ARGUMENTS")
        now = record["created_at"]
        if tool in COMPLETION_TOOLS:
            self.refresh_facts(now)
            return complete(self, record)
        if tool == "get_data_health":
            return {"profile": self.profile, "source_hash_verified": True,
                    "missing_fraction": self.profile["missing_fraction"], "training_rows": self.profile["rows"],
                    "task_card": self.describe_task()[0]}
        if tool == "get_task_status":
            return self.status()
        if tool == "get_tool_result":
            return EvidenceBoard(self.repo).read(args["fact_id"], now)
        rows = ExperimentLedger(self.repo).rows(now)
        if tool == "inspect_results":
            if type(args["offset"]) is not int or args["offset"] < 0 or type(args["limit"]) is not int or not 1 <= args["limit"] <= 4:
                raise GateError("AGENT_RESULT_PAGE")
            return {"rows": rows[args["offset"]:args["offset"] + args["limit"]], "total": len(rows)}
        if tool in {"get_experiment_report", "compare_candidates"}:
            row = next((r for r in rows if r["result_id"] == args["result_id"]), None)
            if row is None:
                raise GateError("AGENT_RESULT_SCOPE")
            if tool == "get_experiment_report":
                return row
            reference = next((r for r in rows if r["result_id"] == args["reference_result_id"]), None)
            if reference is None:
                raise GateError("AGENT_RESULT_SCOPE")
            return {"comparison": score_delta(row, reference),
                    "configuration_difference": configuration_delta(reference["effective_config"], row["effective_config"])}
        return self._run(record, rows)

    def _run(self, record, rows):
        from scripts.benchmark_data_split import write_json
        args, now = record["arguments"], record["created_at"]
        rid = "panel-attempt-" + digest({"task": self.repo.key, "decision": record["decision_id"]})[:24]
        old = next((a for a in self.attempts() if a["record_id"] == rid), None)
        if old:
            if old["status"] == "completed":
                self.refresh_facts(now)
                row = next(r for r in ExperimentLedger(self.repo).rows(now) if r["experiment_id"] == rid)
                return {"result_id": row["result_id"], "score": row["score"], "recovered": True}
            return {"error": "PANEL_EXPERIMENT_UNCERTAIN" if old["status"] == "prepared" else old["error"],
                    "reservation_retained": True, "recovered": True}
        admit_experiment(self.repo)
        if (not isinstance(args["candidate_id"], str) or args["candidate_id"] not in self.pool
                or not isinstance(args["hypothesis"], str) or not 1 <= len(args["hypothesis"]) <= 1000):
            raise GateError("BENCHMARK_CANDIDATE_ARGUMENTS")
        if args["parent_result_id"] is not None and not any(r["result_id"] == args["parent_result_id"] for r in rows):
            raise GateError("AGENT_PARENT_RESULT_SCOPE")
        if any(a["candidate_id"] == args["candidate_id"] for a in self.attempts()):
            raise GateError("BENCHMARK_DUPLICATE_CANDIDATE")
        status = self.status()
        if not is_open(self.repo) and (status["remaining_execution_slots"] <= 0 or status["remaining_worker_seconds"] <= 0):
            raise GateError("BENCHMARK_TRAINING_BUDGET")
        index = len(self.attempts())
        attempt = {"record_id": rid, "index": index, "candidate_id": args["candidate_id"],
                   "parent_result_id": args["parent_result_id"], "hypothesis": args["hypothesis"],
                   "status": "prepared", "created_at": now, "result_file": f"result-{index}.json",
                   "reservation_seconds": self.config.budget.max_candidate_seconds if is_open(self.repo) else min(status["remaining_worker_seconds"], self.config.budget.max_candidate_seconds)}
        self.repo.put(rid, "panel_attempt", attempt, now)
        candidate_path, result_path = self.output / f"candidate-{index}.json", self.output / attempt["result_file"]
        write_json(candidate_path, self.pool[args["candidate_id"]])
        started = time.perf_counter()
        with (self.output / f"worker-{index}.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run([sys.executable, "-m", "scripts.benchmark_data_split", "worker",
                    "--root", str(self.folder), "--candidate", str(candidate_path), "--output", str(result_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=attempt["reservation_seconds"],
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUTF8": "1"},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                succeeded, error = process.returncode == 0 and result_path.exists(), "BENCHMARK_WORKER_FAILED"
            except subprocess.TimeoutExpired:
                succeeded, error = False, "BENCHMARK_CANDIDATE_TIMEOUT"
        attempt.update(status="completed" if succeeded else "failed", charged_seconds=time.perf_counter() - started,
                       completed_at=now, error=None if succeeded else error)
        self.repo.put(rid, "panel_attempt", attempt, now, replace=True)
        if not succeeded:
            return {"error": error, "budget": self.status()}
        self.refresh_facts(now)
        row = next(r for r in ExperimentLedger(self.repo).rows(now) if r["experiment_id"] == rid)
        return {"result_id": row["result_id"], "candidate_id": row["candidate_id"], "score": row["score"],
                "eligible": row["eligible"], "budget": self.status(), "report_ref": row["report_ref"]}


def build_panel_context(repo, gateway, event):
    from .work_context import compose
    return compose(repo, gateway, event, {"memory": [], "knowledge": [], "online_background": [],
        "serving": "offline_panel", "trigger_event": asdict(event), "experiments": [],
        "safety": ["Training uses frozen agent_training windows; validation is development; no final Test or arena access.",
                   "run_experiment is synchronous in this offline adapter. Selection does not register a live model."]})
