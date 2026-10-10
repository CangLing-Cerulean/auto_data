"""Audited agent intent, accepted selection and delivery; never an implicit argmin."""
from dataclasses import asdict

from ..config import GateError, canonical, digest
from .facts import OBJECTIVE, OBJECTIVE_VERSION, ExperimentLedger, selection_eligible


COMPLETION_TOOLS = {
    "finalize_selection": {"result_id": "measured result ID", "candidate_id": "candidate ID of that result",
                           "objective_version": "current objective_version", "policy_version": "current policy version integer",
                           "evidence_refs": "visible evidence/result IDs", "reason": "brief reason", "limitations": "list of unresolved limitations"},
    "finish_without_new_model": {"outcome": "retain_current|insufficient_evidence|no_eligible_candidate|handoff",
                                 "reason": "brief evidence-based conclusion", "evidence_refs": "visible evidence IDs", "followup_id": "registered inspection/condition ID, or null for non-retain outcomes"},
    "complete_incident": {"outcome": "retain_current|handoff|awaiting_release_approval|published_with_monitoring",
                          "reason": "brief conclusion", "evidence_refs": "visible evidence IDs", "followup_id": "registered inspection/condition ID, or null for non-retain outcomes"},
}


def incident(repo):
    return repo.state().incident_id or "epoch-" + str(repo.binding()["epoch"])


def current_delivery(repo):
    return next((r for r in list(repo.iter_records("delivery")) if r["incident_id"] == incident(repo)
                 and r["epoch"] == repo.binding()["epoch"] and r["agent_finalized"]), None)


def accepted_for_experiment(repo, experiment_id):
    return next((r["accepted_selection"] for r in list(repo.iter_records("delivery"))
                 if r["epoch"] == repo.binding()["epoch"] and r.get("accepted_selection")
                 and r["accepted_selection"]["experiment_id"] == experiment_id), None)


def final_reports_for_selection(repo, job):
    """Pass the explicitly sealed report to the original final Gate, retaining the full checkpoint ledger."""
    accepted = accepted_for_experiment(repo, job["plan"]["agent_experiment_id"])
    if accepted is None:
        raise GateError("AGENT_SELECTION_REQUIRED")
    from ..models import Candidate
    if (Candidate.from_dict(job["checkpoint"]["selected"]).candidate_id != accepted["candidate_id"]
            or job["snapshot"]["sha256"] != accepted["snapshot_sha"]):
        raise GateError("AGENT_SELECTION_STALE")
    reports = [r for r in job["checkpoint"]["reports"] if r.get("candidate_id") == accepted["candidate_id"]
               and r.get("validation_artifact") == accepted["report_ref"]]
    if len(reports) != 1:
        raise GateError("AGENT_SELECTION_STALE")
    return reports


def proposals(repo):
    return sorted([r for r in list(repo.iter_records("selection_proposal"))
                   if r["incident_id"] == incident(repo) and r["epoch"] == repo.binding()["epoch"]],
                  key=lambda r: r["decision_version"])


def execute(gateway, record):
    repo, args, tool = gateway.repo, record["arguments"], record["selected_action"]
    now = record.get("effect_at", record["created_at"])
    operation = "delivery-" + record["decision_id"]
    with repo.store.transaction() as db:
        old = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?", (repo.key, operation)).fetchone()
        if old:
            import json
            return {**json.loads(old[0]), "recovered": True}
        repo.check_binding(record["binding"])
        refs = args["evidence_refs"]
        visible = set(record.get("visible_evidence_refs", []))
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in visible for r in refs):
            raise GateError("AGENT_CITATION_SCOPE")
        if not isinstance(args["reason"], str) or not 1 <= len(args["reason"]) <= 1200:
            raise GateError("AGENT_DELIVERY_REASON")
        rows = ExperimentLedger(repo).rows(now)
        bests = ExperimentLedger(repo).view(now)["best_observed_by_objective"]
        value = {"record_id": operation, "task_key": repo.key, "epoch": record["binding"]["epoch"],
                 "incident_id": incident(repo), "decision_id": record["decision_id"], "agent_finalized": False,
                 "program_known_best": bests, "original_selection": None, "accepted_selection": None,
                 "objective_version": OBJECTIVE_VERSION, "policy_version": (record["binding"]["policy"] or {}).get("version"),
                 "evidence_refs": refs, "reason": args["reason"], "actual_published": False,
                 "publication_status": "not_requested", "created_at": now}
        if tool == "finalize_selection":
            proposal = {"record_id": "proposal-" + record["decision_id"], "task_key": repo.key,
                        "epoch": record["binding"]["epoch"], "incident_id": incident(repo),
                        "decision_version": repo.state().decision_version, "raw_arguments": args, "accepted": False}
            row = next((r for r in rows if r["result_id"] == args["result_id"]), None)
            code = None
            if args["objective_version"] != OBJECTIVE_VERSION or args["policy_version"] != value["policy_version"]:
                code = "STALE_SELECTION_VERSION"
            elif row is None or row["candidate_id"] != args["candidate_id"]:
                code = "SELECTION_REQUIRES_MEASURED_RESULT"
            elif row["result_id"] not in visible or row["result_id"] not in refs:
                code = "SELECTION_REQUIRES_VISIBLE_EVIDENCE"
            elif not row["eligible"]:
                code = "SELECTION_NOT_ELIGIBLE"
            elif not selection_eligible(row):
                code = "SELECTION_NOT_FINAL_ADMISSIBLE"
            elif not isinstance(args["limitations"], list) or any(not isinstance(s, str) or len(s) > 500 for s in args["limitations"]):
                code = "AGENT_DELIVERY_LIMITATIONS"
            else:
                comparable = [r for r in rows if selection_eligible(r) and r["comparison_group"] == row["comparison_group"]]
                best = min(comparable, key=lambda r: (r["score"], r["candidate_id"]))
                if row["score"] > best["score"] + OBJECTIVE["tie_tolerance"]:
                    code = "SELECTION_INCONSISTENT_WITH_OBJECTIVE"
                proposal["program_best_result_id"] = best["result_id"]
                proposal["selection_regret"] = row["score"] - best["score"]
            previous_errors = sum(not p["accepted"] for p in proposals(repo))
            from .budget import is_open
            if not is_open(repo) and previous_errors >= 2:
                code = "SELECTION_CORRECTION_LIMIT"
            proposal.update(accepted=code is None, reason_code=code)
            if code is not None:
                repo.put(proposal["record_id"], "selection_proposal", proposal, now, db=db)
                return {"error": code, "original_selection": args, "accepted_selection": None,
                        "program_best": proposal.get("program_best_result_id"), "corrections_remaining": None if is_open(repo) else max(0, 1 - previous_errors)}
            # Verify actual content-addressed reports and packages again at delivery.
            try:
                report = gateway.artifacts.read_json(row["report_ref"])
                if report["candidate_id"] != row["candidate_id"] or report["snapshot_sha"] != row["snapshot_sha"]:
                    raise GateError("AGENT_SELECTION_ARTIFACT_SCOPE")
                if gateway.environment_kind == "panel":
                    if not row["model_ref"]:
                        raise GateError("SELECTION_MODEL_ARTIFACT_REQUIRED")
                    from ..models import load_forecast_model
                    loaded = load_forecast_model(gateway.artifacts, row["model_ref"], gateway.model_config)
                    if loaded.candidate.candidate_id != row["candidate_id"]:
                        raise GateError("AGENT_SELECTION_ARTIFACT_SCOPE")
            except GateError as error:
                proposal.update(accepted=False, reason_code=str(error))
                repo.put(proposal["record_id"], "selection_proposal", proposal, now, db=db)
                return {"error": str(error), "original_selection": args, "accepted_selection": None,
                        "corrections_remaining": None if is_open(repo) else max(0, 1 - previous_errors)}
            repo.put(proposal["record_id"], "selection_proposal", proposal, now, db=db)
            selection = {"result_id": row["result_id"], "candidate_id": row["candidate_id"], "score": row["score"],
                         "report_ref": row["report_ref"], "model_ref": row["model_ref"], "experiment_id": row["experiment_id"],
                         "comparison_group": row["comparison_group"], "snapshot_sha": row["snapshot_sha"],
                         "candidate": row["compiled_config"]}
            value.update(agent_finalized=True, original_selection=args, accepted_selection=selection,
                         delivery_status="offline_model_delivered" if gateway.environment_kind == "panel" else "selection_delivered_pending_final_admission",
                         limitations=args["limitations"])
        else:
            from .budget import is_open
            allowed = {"retain_current", "insufficient_evidence", "no_eligible_candidate", "handoff"} if tool == "finish_without_new_model" else {"retain_current", "handoff", "awaiting_release_approval", "published_with_monitoring"}
            if args["outcome"] not in allowed:
                raise GateError("AGENT_DELIVERY_OUTCOME")
            active = repo.store.active_job(repo.key)
            if args["outcome"] != "handoff" and active and active["plan"].get("agent_managed"):
                from .tools.feedback import ToolArgumentError
                raise ToolArgumentError("AGENT_EXPERIMENT_DISPOSITION_REQUIRED", [{
                    "path": "arguments.outcome", "run_id": active["run_id"],
                    "experiment_id": active["plan"]["agent_experiment_id"], "status": active["status"],
                    "requirement": "Resolve the active experiment before completing the incident: select and submit final, or explicitly cancel_experiment when discarding it. Reading a report does not release its training slot."}])
            existing = current_delivery(repo)
            if args["outcome"] == "published_with_monitoring":
                task = gateway.engine._state()
                deployment = task.get("last_deployment") or {}
                attempt = deployment.get("attempt") or {}
                existing = next((r for r in repo.iter_records("delivery") if r["epoch"] == repo.binding()["epoch"]
                    and r.get("accepted_selection") and r["accepted_selection"]["experiment_id"] == attempt.get("experiment_id")), None)
                if attempt.get("kind") == "existing_model" and attempt.get("epoch") == repo.binding()["epoch"]:
                    existing = {"accepted_selection": {"source": "previously_admitted_model", "experiment_id": None,
                        "model_id": attempt["model_id"], "model_ref": attempt["model_ref"], "candidate": attempt["candidate"]},
                        "original_selection": {"decision_id": attempt["selection_decision"], "model_id": attempt["model_id"]}}
                if (not existing or deployment.get("rollback") or deployment.get("to") != task["champion"]
                        or task.get("observation_completed_for") != task["champion"]):
                    raise GateError("PUBLICATION_OBSERVATION_REQUIRED")
                value.update(actual_published=True, publication_status="published_and_observed",
                             deployment=deployment, model_id=task["champion"])
            if args["outcome"] == "awaiting_release_approval":
                binding = repo.binding()
                grant = binding.get("release_grant") or {}
                requests = [r for r in repo.iter_records("release_request") if r["epoch"] == binding["epoch"]
                    and r["champion"] == binding["champion"] and r["policy_version"] == (binding["policy"] or {}).get("version")
                    and r["approval_required"] and not (grant.get("request_id") == r["request_id"] and now <= grant.get("expires_at", -1))]
                existing = next((r for r in repo.iter_records("delivery") if r["epoch"] == repo.binding()["epoch"]
                    and r.get("accepted_selection") and any(q["experiment_id"] == r["accepted_selection"]["experiment_id"] for q in requests)), None)
            if args["outcome"] == "no_eligible_candidate" and any(selection_eligible(r) for r in rows):
                raise GateError("DELIVERY_FACT_INCONSISTENT")
            if args["outcome"] == "retain_current" and repo.binding()["champion"] is None:
                raise GateError("RETAIN_REQUIRES_CHAMPION")
            followup = None
            if args["followup_id"] is not None:
                followup = repo.get(args["followup_id"])
                if (followup.get("task_key") != repo.key or followup.get("epoch") != repo.binding()["epoch"]
                        or followup.get("status") not in {"scheduled", "waiting"}
                        or not ("condition_id" in followup or "due_at" in followup)):
                    raise GateError("AGENT_FOLLOWUP_SCOPE")
            if is_open(repo) and args["outcome"] == "retain_current" and followup is None:
                raise GateError("RETAIN_REQUIRES_REGISTERED_MONITORING")
            if args["outcome"] == "published_with_monitoring" and followup is None:
                raise GateError("PUBLICATION_REQUIRES_REGISTERED_MONITORING")
            if args["outcome"] == "awaiting_release_approval" and not existing:
                raise GateError("DELIVERY_REQUIRES_SELECTION")
            value.update(agent_finalized=True, delivery_status=args["outcome"],
                         accepted_selection=existing["accepted_selection"] if existing else None,
                         original_selection=existing["original_selection"] if existing else None)
            if followup:
                value["followup_id"] = args["followup_id"]
                if args["outcome"] == "retain_current":
                    value["delivery_status"] = "retained_with_monitoring"
            from .evidence import EvidenceBoard
            value["unresolved_evidence"] = [{"fact_id": f["fact_id"], "status": f["status"], "value": f["value"]}
                for f in EvidenceBoard(repo).view(now) if f["source_tool"] in
                {"get_data_health", "get_label_maturity", "analyze_group_performance"}]
        repo.put(operation, "delivery", value, now, db=db)
        return value


def abort(repo, now, reason_code, details=None):
    if current_delivery(repo):
        return current_delivery(repo)
    ledger = ExperimentLedger(repo).view(now)
    from .budget import snapshot
    value = {"task_key": repo.key, "epoch": repo.binding()["epoch"], "incident_id": incident(repo),
             "agent_finalized": False, "agent_selected": None, "accepted_selection": None,
             "program_known_best": ledger["best_observed_by_objective"], "measured_results": ledger["rows"],
             "uncompleted_steps": ["agent_selection_or_explicit_handoff"],
             "pending_tool_jobs": repo.state().pending_tool_jobs, "reason_code": reason_code,
             "reason_details": details, "budget": snapshot(repo), "safety_state": repo.binding(),
             "delivery_status": "aborted", "actual_published": False, "human_handoff": "inspect durable decision/experiment records"}
    value["record_id"] = "abort-" + digest({"task": repo.key, "incident": incident(repo), "reason": reason_code})[:24]
    repo.put(value["record_id"], "aborted_delivery", value, now, replace=True)
    return value
