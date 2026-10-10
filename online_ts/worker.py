from __future__ import annotations

import os
import time
from pathlib import Path


def _training_target(plan: dict, allowed_targets: tuple[str, ...]) -> str:
    from .config import GateError
    target = plan.get("training_target")
    if target not in {"local", "server"} or target not in allowed_targets:
        raise GateError("TRAINING_TARGET_NOT_ALLOWED")
    return target


def execute_stage(database: str, artifact_root: str, run_id: str, owner: str):
    # Set before NumPy import in the spawned interpreter. Only one CPU BLAS thread.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    from .config import GateError
    from .data import ArtifactStore, load_snapshot
    from .models import Candidate
    from .store import Store
    from .training import final_fit, validate_candidate
    from .planner import (PlannerResponseError, make_feedback_proposal, make_proposal,
                          planner_prompt_version, proposal_fingerprint, validate_proposal)
    from .feedback import (apply_feedback_proposal, build_agent_context, compare_outcomes,
                           gate_issue_for_outcome, progress_record, validation_outcome)
    from .remediation import Outcome, RepairProposal
    from .resources import admit_foundation_workload, memory_limit
    from .resources import local_execution_status
    from .execution import execution_identity, remote_execute_stage
    from .context import training_profile
    from .llm_transport import LLMCallTrace
    from .memory import query_observations
    import threading

    store = Store(database)
    job = store.job(run_id)
    config, state = store.task(job["task_key"])
    if job["status"] != "RUNNING" or job["owner"] != owner:
        return
    artifacts = ArtifactStore(artifact_root)
    cp = job["checkpoint"]
    def timed_out():
        try:
            store.fail_stage(run_id, owner, "TRAINING_DEADLINE", config, retryable=True)
        finally:
            os._exit(124)
    remaining = job["started_wall"] + job["reservation"] - time.time()
    timer = threading.Timer(max(0, remaining), timed_out)
    timer.daemon = True
    timer.start()
    resource_owned = False
    try:
        memory_mb = config.budget.max_memory_mb
        candidate_values = job["plan"].get("candidates", [])
        if cp["index"] < len(candidate_values):
            current_id = Candidate.from_dict(candidate_values[cp["index"]]).candidate_id
            specs = job["plan"].get("experiment_history", []) + job["plan"].get("experiment_specs", [])
            for spec in specs:
                if Candidate.from_dict(spec["candidate"]).candidate_id == current_id:
                    memory_mb = min(memory_mb, spec["resources"]["memory_mb"])
        memory_handle = memory_limit(memory_mb)
        plan = None
        stage = store._stage(job)
        if stage in {"proposal", "feedback"}:
            if "planner_profile" not in cp:
                raw, _ = load_snapshot(job["snapshot"], config, artifacts)
                profile = training_profile(raw, config)
                profile_ref = artifacts.put_json(profile).to_dict()
                history = query_observations(store, job["task_key"], decision_ns=time.time_ns(),
                                             snapshot_sha=job["snapshot"]["sha256"], limit=config.planner.history_limit)
                store.record_context_material(run_id, owner, profile, profile_ref, history)
                cp = store.job(run_id)["checkpoint"]
            trace = LLMCallTrace(f"{run_id}:llm:{stage}:{cp['attempts'][stage]}",
                                 "feedback" if stage == "feedback" else "planning",
                                 callback=lambda value: store.record_llm_trace(run_id, owner, value))
            on_request = lambda value: store.reserve_llm_call(run_id, owner, value)
            cancelled = lambda: (store.job(run_id)["status"] != "RUNNING" or
                                 store.task(job["task_key"])[1]["stop_requested"])
        if stage == "feedback":
            context = cp["feedback_pending"]
            pending = cp.get("pending_agent_response")
            if pending is None:
                remaining_tokens = config.budget.max_llm_tokens - int(cp.get("llm_tokens_used", 0))
                proposal, tokens, routing = make_feedback_proposal(
                    config, context, token_limit=remaining_tokens, trace=trace,
                    on_request=on_request, cancelled=cancelled)
                store.record_planner_attempt(
                    run_id, owner, tokens, proposal["message_id"],
                    RepairProposal(cp.get("repair_rounds", 0),
                                   context["gate_issues"][0]["code"],
                                   job["plan"].get("proposal_fingerprint"), proposal["message_id"],
                                   proposal["message_id"], context["plan_version"] + 1).to_dict(),
                    proposal=proposal)
                cp = store.job(run_id)["checkpoint"]
            else:
                proposal, routing = pending, []
            revised, decision = apply_feedback_proposal(
                proposal, context, config, job["plan"],
                store.proposal_fingerprints(job["task_key"]),
                {item["target"] for item in (routing or job["plan"].get("routing_evidence", []))
                 if item.get("available", True)} or set(config.execution.allowed_targets))
            cp["pending_agent_response"] = None
            cp["feedback_pending"] = None
            if revised is None:
                cp["terminal_reason"] = decision["terminal_reason"]
                result = {"passed": False, "issues": [decision["terminal_reason"]],
                          "feedback_decision": decision}
            else:
                cp["plan_version"] = revised["plan_version"]
                plan = revised
                result = None
            store.complete_stage(run_id, owner, cp, result, time.time() - job["started_wall"], plan)
            return
        if job["plan"].get("needs_proposal"):
            summary = cp["planner_profile"]
            repair_context = None
            if cp.get("gate_issues"):
                repair_context = {
                    "round": cp.get("format_repairs", 0),
                    "last_gate_issue": cp["gate_issues"][-1],
                    "last_progress_digest": cp.get("progress_digests", [None])[-1],
                    "best_so_far": cp.get("best_so_far"),
                }
            remaining_tokens = config.budget.max_llm_tokens - int(cp.get("llm_tokens_used", 0))
            pending = cp.get("pending_initial_proposal")
            if pending is None:
                proposal, tokens, routing = make_proposal(
                    config, job["snapshot"]["sha256"], summary,
                    repair_context=repair_context, token_limit=remaining_tokens,
                    trial_history=cp.get("trial_history", []), trace=trace,
                    on_request=on_request, cancelled=cancelled)
            else:
                proposal, tokens, routing = pending["proposal"], pending["tokens"], pending["routing"]
            fingerprint = proposal_fingerprint(proposal)
            repair = None
            if repair_context:
                from .remediation import RepairProposal
                repair = RepairProposal(
                    cp.get("repair_rounds", 0), repair_context["last_gate_issue"]["code"],
                    cp.get("previous_proposal_fingerprint"), fingerprint,
                ).to_dict()
            if pending is None:
                store.record_planner_attempt(run_id, owner, tokens, fingerprint, repair,
                    initial_proposal={"proposal": proposal, "tokens": tokens, "routing": routing})
            validate_proposal(proposal, config, job["snapshot"]["sha256"], store.proposal_fingerprints(job["task_key"]))
            plan = {**proposal, "planner_tokens": tokens, "routing_evidence": routing,
                    "prompt_version": planner_prompt_version(config), "llm_model": config.planner.model,
                    "proposal_fingerprint": fingerprint,
                    "champion_artifact": job["plan"].get("champion_artifact")}
            cp = store.job(run_id)["checkpoint"]
            cp.pop("pending_initial_proposal", None)
            result = None
        elif cp["selected"] is not None:
            from .config import digest
            binding = digest({"plan_version": cp["plan_version"],
                              "candidates": [Candidate.from_dict(c).candidate_id for c in job["plan"]["candidates"]],
                              "reports": cp["reports"]})
            if cp.get("selection_binding") != binding:
                raise GateError("STALE_SELECTION")
            target = _training_target(job["plan"], config.execution.allowed_targets)
            incumbent_ref = job["plan"].get("champion_artifact")
            candidate = Candidate.from_dict(cp["selected"])
            final_reports = cp["reports"]
            if job["plan"].get("agent_managed"):
                from .agent_runtime.completion import final_reports_for_selection
                from .agent_runtime.state import Repository
                final_reports = final_reports_for_selection(Repository(store, job["task_key"]), job)
            if target == "server":
                result = remote_execute_stage(config, artifacts, run_id, "final", job["snapshot"],
                                              candidate=cp["selected"], reports=final_reports, champion_ref=incumbent_ref,
                                              stage_timeout_seconds=job["reservation"])
            else:
                if candidate.family == "chronos2":
                    from .chronos_adapter import Chronos2Adapter, spec_for_candidate
                    estimate = Chronos2Adapter(spec_for_candidate(candidate, config),
                                               config.foundation.model_roots[candidate.model_id],
                                               device=config.foundation.device).resource_estimate()
                    local_status = admit_foundation_workload(config, candidate, estimate)
                    if config.foundation.device == "cuda":
                        if state["champion"]:
                            incumbent = store.model(state["champion"])["report"].get("candidate", {})
                            if incumbent.get("family") == "chronos2":
                                raise GateError("GPU_RESERVED_FOR_CHRONOS_SERVING")
                        resource_owned = store.acquire_resource(
                            "gpu-training", owner, job["task_key"], run_id, "training", job["reservation"] + 10)
                        if not resource_owned:
                            raise GateError("GPU_TRAINING_BUSY")
                else:
                    local_status = local_execution_status(config.execution.expected_local_gpu)
                result = final_fit(config, job["snapshot"], candidate, final_reports, artifacts,
                                   incumbent_ref, execution_identity(local_status))
            if result.get("waiting_for_labels"):
                store.complete_stage(run_id, owner, cp, result, time.time() - job["started_wall"], plan)
                return
            final_outcome = Outcome.create(
                run_id=run_id, candidate_id=candidate.candidate_id,
                attempt_id=f"{run_id}:final:{cp['attempts'].get('final', 1)}",
                plan_id=cp["plan_id"], plan_version=cp["plan_version"],
                created_at=job["cutoff"], node="final_evaluation", status="passed",
                effective_config=result["candidate"],
                artifact_refs=("artifact:" + result["artifact"]["sha256"],), log_refs=(),
                resource_usage=result.get("resource_observation", {}),
                metrics={"normalized_mae": result["offline"]["normalized_mae"],
                         "normalized_mse": result["offline"]["normalized_mse"]},
                issue_codes=tuple(result.get("issues", ())),
                snapshot_sha=job["snapshot"]["sha256"], evaluation_protocol="sealed-final-test-v1",
                evidence_scope="final_test", gate_passed=bool(result.get("passed")))
            cp["outcomes"].append(final_outcome.to_dict())
        else:
            target = _training_target(job["plan"], config.execution.allowed_targets)
            candidate = Candidate.from_dict(job["plan"]["candidates"][cp["index"]])
            if target == "server":
                report = remote_execute_stage(config, artifacts, run_id, "validate", job["snapshot"],
                                              candidate=job["plan"]["candidates"][cp["index"]], reports=cp["reports"],
                                              champion_ref=job["plan"].get("champion_artifact"),
                                              stage_timeout_seconds=job["reservation"])
            else:
                if candidate.family == "chronos2":
                    from .chronos_adapter import Chronos2Adapter, spec_for_candidate
                    estimate = Chronos2Adapter(spec_for_candidate(candidate, config),
                                               config.foundation.model_roots[candidate.model_id],
                                               device=config.foundation.device).resource_estimate()
                    local_status = admit_foundation_workload(config, candidate, estimate)
                    if config.foundation.device == "cuda":
                        if state["champion"]:
                            incumbent = store.model(state["champion"])["report"].get("candidate", {})
                            if incumbent.get("family") == "chronos2":
                                raise GateError("GPU_RESERVED_FOR_CHRONOS_SERVING")
                        resource_owned = store.acquire_resource(
                            "gpu-training", owner, job["task_key"], run_id, "training", job["reservation"] + 10)
                        if not resource_owned:
                            raise GateError("GPU_TRAINING_BUSY")
                else:
                    local_status = local_execution_status(config.execution.expected_local_gpu)
                report = validate_candidate(config, job["snapshot"], candidate, artifacts, execution_identity(local_status),
                                            champion_artifact=job["plan"].get("champion_artifact"))
            parent_values = [value for value in cp.get("outcomes", [])
                             if value.get("evidence_scope") in {"development_validation", "execution", "resource"}]
            parent = Outcome.from_dict(parent_values[-1]) if parent_values else None
            outcome = validation_outcome(
                job, report, config, attempt=cp["attempts"].get(stage, 1), created_at=job["cutoff"])
            comparison = compare_outcomes(parent, outcome)
            best = cp.get("best_so_far")
            if comparison.get("comparable") and best and outcome.metrics is not None:
                comparison["quality_improved"] = (outcome.gate_passed is True
                    and outcome.metrics["score"] < best["score"] - 1e-12)
                comparison["best_score_delta"] = outcome.metrics["score"] - best["score"]
            if comparison.get("comparable"):
                cp["non_improvement_count"] = (
                    0 if comparison.get("quality_improved") is True
                    else int(cp.get("non_improvement_count", 0)) + 1)
            cp["reports"].append(report)
            cp["index"] += 1
            cp["outcomes"].append(outcome.to_dict())
            cp["attempt_history"].append({"attempt_id": outcome.attempt_id,
                                          "candidate_id": outcome.candidate_id,
                                          "plan_version": outcome.plan_version,
                                          "outcome_id": outcome.message_id})
            cp["progress_digests"].append(progress_record(
                round_index=cp.get("repair_rounds", 0),
                current_fingerprint=job["plan"].get("proposal_fingerprint"),
                previous_fingerprint=cp.get("previous_proposal_fingerprint"),
                best_so_far=cp.get("best_so_far"), comparison=comparison,
                resource_usage=outcome.resource_usage))
            should_repair = (config.planner.kind == "external" and not job["plan"].get("agent_managed") and parent is not None
                             and comparison.get("comparable")
                             and comparison.get("quality_improved") is False
                             and cp.get("repair_rounds", 0) < config.budget.max_repairs
                             and cp["index"] < config.budget.max_candidates)
            if (config.planner.kind == "external" and not job["plan"].get("agent_managed")
                    and cp.get("non_improvement_count", 0) >= config.budget.max_no_improvement):
                cp["terminal_reason"] = "NO_QUALITY_PROGRESS"
                should_repair = False
                result = {"passed": False, "issues": ["NO_QUALITY_PROGRESS"]}
            else:
                result = None
            if should_repair:
                issue = gate_issue_for_outcome(
                    outcome, code="VALIDATION_NO_IMPROVEMENT",
                    observed_value=outcome.metrics["score"], threshold=parent.metrics["score"],
                    direction="lt", repairable=True)
                cp["gate_issues"].append(issue.to_dict())
                cp["repair_rounds"] += 1
                job["checkpoint"] = cp
                from .config import digest
                reference = {"task_key": job["task_key"], "snapshot_sha": report["snapshot_sha"],
                             "config_version": config.version, "evaluation_protocol": "rolling-origin-v1",
                             "evidence_scope": "development_validation", "metrics": outcome.metrics,
                             "sample_fingerprint": digest([f["issue_times"] for f in report["folds"]]),
                             "normalization_fingerprint": digest(report["normalization"])}
                history = query_observations(store, job["task_key"], decision_ns=time.time_ns(),
                    snapshot_sha=job["snapshot"]["sha256"], reference=reference, limit=config.planner.history_limit)
                context = build_agent_context(
                    config, job, outcome, [issue], comparison=comparison,
                    available_targets=set(config.execution.allowed_targets), trial_history=history)
                cp["agent_contexts"].append(context.to_dict())
                cp["feedback_pending"] = context.to_dict()
        store.complete_stage(run_id, owner, cp, result, time.time() - job["started_wall"], plan)
    except PlannerResponseError as error:
        try:
            store.record_planner_attempt(run_id, owner, error.tokens, error.response_fingerprint)
        except GateError as charge_error:
            if str(charge_error) != "PLANNER_TOKEN_BUDGET":
                raise
        store.fail_stage(run_id, owner, str(error), config, retryable=False, elapsed=time.time() - job["started_wall"])
    except GateError as error:
        retryable = str(error) in {"TRAINING_DEADLINE", "TRAINING_SERVER_UNAVAILABLE", "REMOTE_BUSY",
                                   "REMOTE_TRAINING_DEADLINE", "REMOTE_TRAINING_INTERRUPTED", "REMOTE_RESPONSE_MISSING"}
        retryable = retryable or str(error) in {"GPU_TRAINING_BUSY", "FOUNDATION_VRAM_INSUFFICIENT"}
        store.fail_stage(run_id, owner, str(error), config, retryable=retryable, elapsed=time.time() - job["started_wall"])
    except MemoryError:
        store.fail_stage(run_id, owner, "MEMORY_BUDGET", config, retryable=False, elapsed=time.time() - job["started_wall"])
    except Exception:
        # Neither source rows nor remote response/credentials are written to logs.
        store.fail_stage(run_id, owner, "TRAINING_EXECUTION_FAILED", config, retryable=True, elapsed=time.time() - job["started_wall"])
    finally:
        if resource_owned:
            store.release_resource("gpu-training", owner)
        timer.cancel()


if __name__ == "__main__":
    import sys
    execute_stage(*sys.argv[1:])
