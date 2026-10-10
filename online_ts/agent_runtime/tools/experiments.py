"""Adapters to the existing snapshot/compiler/Worker, with development-only reports."""
from dataclasses import asdict

from ...config import GateError, SECOND, canonical, digest
from ...data import aligned, snapshot
from ...experiments import compile_experiment
from ...models import Candidate
from ...planner import validate_proposal, proposal_fingerprint
from ...training import choose


EXPERIMENT_TOOLS = {
    "create_experiment": {"purpose": "optimization|diagnosis", "hypothesis": "string",
                          "candidates": "list of Candidate objects", "training_target": "local|server"},
    "start_experiment": {"experiment_id": "string"},
    "get_experiment_report": {"experiment_id": "string"},
    "cancel_experiment": {"experiment_id": "string"},
    "submit_final": {"experiment_id": "string"},
    "get_final_status": {"experiment_id": "string"},
    "request_release": {"experiment_id": "string"},
}


class Experiments:
    def __init__(self, gateway):
        self.gateway, self.engine, self.repo = gateway, gateway.engine, gateway.repo

    def execute(self, record):
        args, tool = record["arguments"], record["selected_action"]
        if tool == "create_experiment":
            return self.create(record)
        experiment = self.repo.get(args["experiment_id"])
        if experiment.get("kind") != "experiment" or experiment["task_key"] != self.engine.key:
            raise GateError("AGENT_EXPERIMENT_SCOPE")
        if tool == "start_experiment":
            self._admit_training()
            self.repo.check_binding(experiment["binding"])
            plan = experiment["plan"]
            run_id = self.engine.store.request_job(self.engine.key, experiment["cutoff"], experiment["snapshot"],
                plan, "AGENT_EXPERIMENT", self.engine._state()["now"],
                agent_epoch=record["binding"]["epoch"], experiment_id=experiment["record_id"])
            if run_id is None:
                active = self.engine.store.active_job(self.engine.key)
                if active:
                    from .feedback import ToolArgumentError
                    raise ToolArgumentError("AGENT_TRAINING_BUSY_OR_COOLDOWN", [{
                        "path": "arguments.experiment_id", "blocking_run_id": active["run_id"],
                        "blocking_experiment_id": active["plan"].get("agent_experiment_id"), "status": active["status"],
                        "requirement": "The existing job still owns the training slot. Complete its selected final or cancel_experiment if discarded; get_experiment_report is read-only."}])
                raise GateError("AGENT_TRAINING_BUSY_OR_COOLDOWN")
            return {"run_id": run_id, "experiment_id": experiment["record_id"], "status": "submitted"}
        run_id = "train-" + digest({"task": self.engine.key, "experiment_id": experiment["record_id"]})[:24]
        job = self.engine.store.job(run_id)
        if job["task_key"] != self.engine.key or job["plan"].get("agent_experiment_id") != experiment["record_id"]:
            raise GateError("AGENT_JOB_SCOPE")
        if tool == "cancel_experiment":
            self.engine.store.cancel_job(run_id)
            self.engine.store.dispose_job(run_id, "cancelled", {"reason": "agent_abandoned"}, self.engine._state()["now"],
                lambda s: s.update(status="SERVING" if s["champion"] else "DATA_BLOCKED"))
            return {"run_id": run_id, "status": "CANCELLED"}
        if tool == "get_experiment_report":
            cp = job["checkpoint"]
            from ..memory import verify_contrasts
            experience_refs = verify_contrasts(self.repo, experiment, cp["reports"], experiment["cutoff"])
            # Never expose job.result, final outcomes, final reviews or raw split rows.
            reports = [{k: r[k] for k in ("candidate_id", "candidate", "eligible", "issues", "score", "score_std", "execution",
                        "data_qualification", "deployed_champion_comparison", "development_admission") if k in r}
                       for r in cp["reports"]]
            for projected, raw in zip(reports, cp["reports"]):
                projected["folds"] = [{k: fold[k] for k in ("normalized_mae", "normalized_mse", "mae_by_variable", "mae_by_horizon",
                    "deployed_champion", "training_samples", "weight_updates", "adaptation_evidence") if k in fold} for fold in raw.get("folds", [])]
            champion = self.engine._state()["champion"]
            incumbent = Candidate.from_dict(self.engine.store.model(champion)["report"]["candidate"]).candidate_id if champion else None
            reference = next((r for r in reports if r.get("candidate_id") == incumbent and "score" in r), None)
            for projected in reports:
                projected["champion_recipe_comparable"] = reference is not None and "score" in projected
                if projected["champion_recipe_comparable"]:
                    projected["score_delta_vs_champion_recipe"] = projected["score"] - reference["score"]
            return {"experiment_id": experiment["record_id"], "run_id": run_id, "status": job["status"],
                    "development_complete": cp["index"] == len(job["plan"]["candidates"]),
                    "awaiting_selection_or_cancellation": (job["status"] == "PENDING"
                        and cp["index"] == len(job["plan"]["candidates"]) and cp.get("selected") is None),
                    "purpose": experiment["purpose"], "reports": reports, "spent_seconds": job["spent"],
                    "evidence_scope": "development_validation", "experience_refs": experience_refs,
                    "result_refs": [r["validation_artifact"] for r in cp["reports"] if r.get("validation_artifact")],
                    "champion_comparison_note": "deployed_champion_comparison identifies the frozen deployed artifact and cutoff. Recipe comparisons remain separately labelled."}
        if tool == "get_final_status":
            result = job.get("result") or {}
            return {"experiment_id": experiment["record_id"], "run_id": run_id, "status": job["status"],
                    "final_sealed": job["checkpoint"].get("selected") is not None,
                    "final_passed": bool(result.get("passed")) if job["status"] == "COMPLETED" else None,
                    "model_ref": result.get("artifact") if job["status"] == "COMPLETED" and result.get("passed") else None,
                    "numeric_final_metrics": "sealed_not_available_for_development"}
        if tool == "request_release":
            from .release import request_release
            return request_release(self.gateway, record, experiment, job)
        if tool == "submit_final":
            from ..completion import accepted_for_experiment
            accepted = accepted_for_experiment(self.repo, experiment["record_id"])
            if accepted is None:
                raise GateError("AGENT_SELECTION_REQUIRED")
            selected = accepted["candidate"]
            matching = [r for r in job["checkpoint"]["reports"] if r["candidate_id"] == accepted["candidate_id"]]
            if (not matching or matching[0]["validation_artifact"] != accepted["report_ref"]
                    or job["snapshot"]["sha256"] != accepted["snapshot_sha"]):
                raise GateError("AGENT_SELECTION_STALE")
            choose(matching, self.engine.config, artifacts=self.engine.artifacts, snapshot_sha=job["snapshot"]["sha256"])
            admission = matching[0].get("development_admission")
            if self.engine._state()["champion"] and (not admission or not admission["worthy_of_final"]):
                raise GateError(admission["reason"] if admission else "DEVELOPMENT_CHAMPION_COMPARISON_REQUIRED")
            self._admit_training()
            if experiment["binding"]["champion"] != self.engine._state()["champion"]:
                raise GateError("AGENT_CHAMPION_CHANGED")
            if job["checkpoint"].get("selected") is not None:
                if Candidate.from_dict(job["checkpoint"]["selected"]).candidate_id != accepted["candidate_id"]:
                    raise GateError("AGENT_SELECTION_STALE")
                return {"run_id": run_id, "selected_candidate": accepted["candidate_id"], "status": "sealed_for_original_final_gate", "recovered": True}
            # The sealed Test may yield a release outcome, so prevent later
            # optimization against that same window even without numeric exposure.
            with self.engine.store.transaction() as db:
                meta = self.engine.artifacts.read_json(job["snapshot"])
                db.execute("INSERT OR IGNORE INTO test_exposures VALUES(?,?,?,?,?,?)",
                           (digest(asdict(self.engine.config.source)), meta["times"][-self.engine.config.search.test_steps],
                            meta["times"][-1], "agent-development", "sealed:" + run_id, self.engine._state()["now"]))
            self.engine.store.seal_selection(run_id, selected)
            return {"run_id": run_id, "selected_candidate": Candidate.from_dict(selected).candidate_id,
                    "status": "sealed_for_original_final_gate"}
        raise GateError("AGENT_TOOL_NOT_ALLOWED")

    def _admit_training(self):
        policy = self.repo.binding()["policy"] or {}
        if policy.get("mode") not in {"train_auto_release_manual", "controlled_release"}:
            raise GateError("AGENT_TRAINING_NOT_AUTHORIZED")

    def create(self, record):
        self._admit_training()
        args, config = record["arguments"], self.engine.config
        policy = self.repo.binding()["policy"] or {}
        prior = list(self.repo.iter_records("experiment"))
        if args["purpose"] not in {"optimization", "diagnosis"}:
            raise GateError("AGENT_EXPERIMENT_PURPOSE")
        if not isinstance(args["candidates"], list) or not 1 <= len(args["candidates"]) <= config.budget.max_candidates:
            raise GateError("CANDIDATE_BUDGET")
        from .feedback import parse_candidates
        candidates = parse_candidates(args["candidates"], config)
        if args["purpose"] == "diagnosis":
            # Require an identifiable contrast (one changed factor). Multiple
            # such edges permit economical factorial designs, without causal claims.
            from .feedback import validate_diagnosis
            validate_diagnosis(candidates)
        now = record.get("effect_at", record["created_at"])
        if now is None or not hasattr(self.engine.source, "labels_until"):
            raise GateError("TRAINING_LABELS_INSUFFICIENT")
        frame = self.engine.source.labels_until(now)
        cutoff = now - int(config.allowed_lateness_seconds * SECOND)
        frame = frame.loc[frame.index <= cutoff]
        if frame.empty:
            raise GateError("TRAINING_LABELS_INSUFFICIENT")
        cutoff = int(frame.index[-1])
        from ..completion import incident
        from ...data import load_snapshot
        from ...qualification import candidate_qualification
        incident_id = incident(self.repo)
        cohort = next((e for e in prior if e.get("incident_id") == incident_id
            and e["binding"]["epoch"] == record["binding"]["epoch"]
            and e["binding"]["champion"] == record["binding"]["champion"]), None)
        if cohort:
            ref, cutoff = cohort["snapshot"], cohort["cutoff"]
            history, _ = load_snapshot(ref, config, self.engine.artifacts)
        else:
            last = self.engine._state()["last_train_cutoff"]
            if last is not None and cutoff - last < config.monitoring.minimum_new_steps * config.step_ns:
                raise GateError("NEW_MATURE_DATA_REQUIRED")
            history = aligned(frame, config, end=cutoff).tail(config.search.required_history)
            ref = None
        qualification = [candidate_qualification(history, c, config) for c in candidates]
        for value in qualification:
            if not value["training_eligible"]:
                raise GateError(value["issues"][0])
        if ref is None:
            ref = snapshot(history, cutoff, config, self.engine.artifacts, as_of=now).to_dict()
        specs = []
        for candidate in candidates:
            spec = {"objective": args["purpose"], "hypothesis": args["hypothesis"], "candidate": asdict(candidate),
                    "contract_ref": config.version, "evaluation_protocol": "rolling-origin-v1",
                    "resources": {"training_target": args["training_target"], "seconds": config.budget.max_candidate_seconds,
                                  "memory_mb": config.budget.max_memory_mb}, "evidence_refs": [ref["sha256"][:24]]}
            compile_experiment(spec, config, evidence_refs={ref["sha256"][:24]}, available_targets=set(config.execution.allowed_targets))
            specs.append(spec)
        proposal = {"snapshot_sha": ref["sha256"], "candidates": [asdict(c) for c in candidates],
                    "training_target": args["training_target"], "routing_reason": "Agent selected authorized execution target",
                    "experiment_specs": specs}
        validate_proposal(proposal, config, ref["sha256"], self.engine.store.proposal_fingerprints(self.engine.key))
        experiment_id = "agent-exp-" + digest({"task": self.engine.key, "proposal": proposal_fingerprint(proposal)})[:24]
        from ..budget import is_open
        if (not is_open(self.repo) and not any(e["record_id"] == experiment_id for e in prior)
                and sum(e["binding"]["epoch"] == record["binding"]["epoch"] for e in prior) >= policy.get("max_experiments", 8)):
            raise GateError("AGENT_EXPERIMENT_BUDGET")
        champion = self.engine._state()["champion"]
        plan = {**proposal, "proposal_fingerprint": proposal_fingerprint(proposal), "planner_tokens": 0,
                "agent_managed": True, "agent_experiment_id": experiment_id,
                "champion_artifact": self.engine.store.model(champion)["artifact"] if champion else None}
        value = {"record_id": experiment_id, "kind": "experiment", "task_key": self.engine.key,
                 "incident_id": incident_id, "data_qualification": qualification,
                 "purpose": args["purpose"], "hypothesis": args["hypothesis"], "snapshot": ref, "cutoff": cutoff,
                 "plan": plan, "binding": record["binding"]}
        self.repo.put(experiment_id, "experiment", value, now)
        return {"experiment_id": experiment_id, "candidate_ids": [c.candidate_id for c in candidates],
                "purpose": args["purpose"], "result_refs": [ref]}
