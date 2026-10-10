"""Bounded outbound facts with an exact audit of what was retained or omitted."""
from copy import deepcopy
from dataclasses import asdict

from ..config import digest, canonical
from .budget import AgentBudgetError, CLOSING_RESERVATION, check_request, record_boundary, snapshot, is_open
from .completion import current_delivery, proposals
from .evidence import EvidenceBoard
from .facts import ExperimentLedger, OBJECTIVE_VERSION, selection_eligible
from . import trajectory


def brief_result(value):
    if not isinstance(value, dict):
        return value
    if value.get("observer") in {"data", "experiments"} and "value" in value:
        facts = value["value"]
        return {k: value[k] for k in ("record_id", "observer", "as_of")} | {
            "value": {k: facts[k] for k in ("alert", "training_readiness", "collection_error", "jobs") if k in facts},
            "observations": brief_result({"observations": facts.get("online_measurements", [])})["observations"],
            "readback": "get_observer_report(record_id); complete report remains stored"}
    if "incident_id" in value and "entries" in value and "total_count" in value:
        return deepcopy(value)  # An explicitly requested history page is already scoped and bounded.
    kept = {k: value[k] for k in ("error", "validation_errors", "reason_code", "status", "development_complete", "awaiting_selection_or_cancellation", "experiment_id", "run_id",
                                 "result_id", "candidate_ids", "reported", "accepted_selection", "delivery_status",
                                 "approval_required", "wait_condition", "budget", "corrections_remaining",
                                 "champion", "challenger", "previous", "stage", "stage_since", "advanced", "rolled_back",
                                 "observation_until", "observation_completed_for", "release_wait", "release_contract",
                                 "final_sealed", "final_passed", "model_ref", "selected_candidate", "models", "total_count", "offset") if k in value}
    if "wait_condition" in kept:
        wait = kept["wait_condition"]
        kept["wait_condition"] = {key: deepcopy(wait[key]) for key in (
            "condition_id", "condition_type", "object_id", "predicate", "clock_domain", "deadline",
            "created_at", "status", "initial_evaluation", "initial_evidence", "resolved_at", "resolution_evidence") if key in wait}
        kept["wait_condition"]["readback"] = "get_wait_status(condition_id) for the complete current receipt"
    if not kept:
        kept = {k: v for k, v in value.items() if isinstance(v, (int, float, bool, str)) or v is None}
    if "observations" in value:
        observations = value["observations"]
        observations = [observations[0], *observations[-2:]] if len(observations) > 3 else observations
        kept["observations"] = [{"record_id": v.get("record_id"), "model_id": v.get("model_id"),
                                  **{k: v[k] for k in ("window_start", "window_end") if k in v},
                                  "evidence": {k: v.get("evidence", {}).get(k) for k in
                                               ("normalized_mae", "normalized_mse", "business_groups", "clean_inputs", "same_window_baselines") if k in v.get("evidence", {})}}
                                 for v in observations]
    for key in ("windows", "latest_mature_label", "mature_rows_available", "labels_adapter_available",
                "by_variable", "candidate", "delay_seconds", "missing_fraction", "latest_timestamp", "input_age_seconds",
                "freshness_threshold_seconds", "timeliness", "sampling_gap_count", "schema_valid", "arrival_lag_seconds",
                "evaluable_history_count", "latest_evaluable_cutoff", "recent_evaluable_windows", "latest_window", "next_maturity_condition"):
        if key in value:
            kept[key] = value[key][-3:] if key == "windows" else value[key]
    if "fact_id" in value and "value" in value:
        kept["value"] = brief_result(value["value"])
    if "variable_metrics" in value:
        kept["variable_metrics"] = value["variable_metrics"]
    if "comparison" in value:
        kept.update(comparison=value["comparison"], configuration_difference=value["configuration_difference"])
    if "events" in value:
        # Preserve the already restricted event receipt through compression.
        # Never include lifecycle payloads, which may contain sealed Test facts.
        kept["events"] = [{k: event[k] for k in ("id", "kind", "event_time")}
                          for event in value["events"]]
    if "rows" in value and isinstance(value["rows"], list):
        kept["rows"] = [{k: r[k] for k in ("result_id", "candidate_id", "score", "effective_config", "variable_metrics") if k in r}
                        for r in value["rows"]]
    return kept


def compact_row(row, *, closing=False):
    result = {k: row[k] for k in ("result_id", "candidate_id", "score", "eligible", "comparison_group")}
    result["selection_eligible"] = selection_eligible(row)
    result["config"] = {k: v for k, v in row["effective_config"].items() if k != "architecture"
                        and (not closing or k in {"family", "alpha", "period", "input_observations", "training"})}
    if not closing:
        result.update(versus_parent=row["versus_parent"], versus_best_before=row["versus_best_before"],
                      sample_counts=deepcopy(row["sample_counts"]), model_available=row["model_ref"] is not None,
                      development_admission=row.get("development_admission"), training_progress=row.get("training_progress"))
        counts = result["sample_counts"]
        for key in ("training_by_fold", "evaluation_by_fold"):
            values = counts.get(key)
            if values and all(value == values[0] for value in values):
                counts[key] = {"fold_count": len(values), "each_fold": values[0]}
    return result


def visible_refs(outbound):
    refs = [row["result_id"] for row in outbound["experiment_ledger"]["rows"]]
    refs += [fact["fact_id"] for fact in outbound["evidence_board"]]
    refs += [row["record_id"] for key in ("knowledge", "memory", "online_background") for row in outbound.get(key, [])]
    refs += trajectory.visible_refs(outbound.get("session_history", {}))
    refs += [report["record_id"] for report in outbound.get("observer_briefings", [])]
    refs += [row["record_id"] for report in outbound.get("observer_briefings", [])
             for row in report["value"].get("online_measurements", []) if row.get("record_id")]
    last = outbound["state"].get("last_result")
    if last and last["tool"] == trajectory.HISTORY_TOOL:
        refs += trajectory.visible_refs(last["result"])
    return list(dict.fromkeys(refs))


def fit_history(outbound, limit):
    """Preserve the newest complete turn and current facts; never truncate protocol JSON."""
    from .llm import messages_for, reservation_for
    history = outbound["session_history"]
    def fits():
        outbound["visible_evidence_refs"] = visible_refs(outbound)
        return reservation_for(messages_for(outbound), outbound["phase"]) <= limit
    selection = outbound["experiment_ledger"].get("accepted_selection")
    for entry in history["entries"]:
        if fits():
            return
        observation = entry["observation"]
        value = observation["result"]
        if selection and value.get("accepted_selection") == selection:
            value["accepted_selection"] = {"result_id": selection["result_id"],
                "readback": "experiment_ledger.accepted_selection contains this identical selection in full"}
            observation["projection"] = "selection duplicated in current ledger; remaining receipt unchanged"
        fact = next((fact for fact in outbound["evidence_board"]
                     if fact["tool"] == entry["action"]["tool"] and fact["facts"] == brief_result(value)), None)
        if fact and entry is not history["entries"][-1] and entry["action"]["tool"] != trajectory.HISTORY_TOOL:
            observation["result"] = {"evidence_ref": fact["fact_id"],
                                     "readback": "Identical tool facts are retained in evidence_board."}
            observation["projection"] = "brief; duplicate facts referenced; complete stored receipt in read_session_history"
    # A requested page is also context, even though its durable tool receipt
    # remains complete. It must not bypass the same request-size boundary.
    last = outbound["state"].get("last_result")
    page = last["result"] if last and last["tool"] == trajectory.HISTORY_TOOL else None
    if page and "entries" in page:
        for entry in page["entries"]:
            if fits():
                return
            observation = entry["observation"]
            reduced = brief_result(observation["result"])
            if len(canonical(reduced).encode()) < len(canonical(observation["result"]).encode()):
                observation.update(result=reduced, projection="brief; full receipt available through read_session_history")
    for entry in history["entries"][:-1]:
        if fits():
            return
        observation = entry["observation"]
        reduced = brief_result(observation["result"])
        if len(canonical(reduced).encode()) < len(canonical(observation["result"]).encode()):
            observation.update(result=reduced, projection="brief; full receipt available through read_session_history")
    while not fits() and len(history["entries"]) > 1:
        removed = history["entries"].pop(0)
        history["omitted_count"] += 1
        history["omitted_range"] = {
            "first_sequence": (history["omitted_range"] or {}).get("first_sequence", removed["sequence"]),
            "last_sequence": removed["sequence"]}
    while not fits() and page and len(page.get("entries", [])) > 1:
        page["entries"].pop()
        page["omitted_entries"] = page.get("omitted_entries", 0) + 1
        page["next_offset"] = page["offset"] + len(page["entries"])
        page["projection"] = "Partial requested page to fit context; continue at next_offset. Stored receipt is unchanged."
    if not fits() and history["entries"]:
        observation = history["entries"][-1]["observation"]
        reduced = brief_result(observation["result"])
        if len(canonical(reduced).encode()) < len(canonical(observation["result"]).encode()):
            observation.update(result=reduced,
                               projection="brief; complete latest turn retained, full stored receipt available through read_session_history")
    for report in outbound.get("observer_briefings", []):
        if fits():
            break
        rows = report["value"].get("online_measurements", [])
        if len(rows) > 1:
            report["value"]["online_measurements"] = rows[-1:]
            report["value"]["omitted_measurements"] = len(rows) - 1
    if not fits():
        ledger = outbound["experiment_ledger"]
        required = set(ledger["best_observed_by_objective"].values()) | set(ledger["best_eligible_candidate"].values())
        required.add(ledger["latest_result"])
        required.add((ledger.get("accepted_selection") or {}).get("result_id"))
        latest = next((row for row in ledger["rows"] if row["result_id"] == ledger["latest_result"]), None)
        if latest:
            required.add(latest.get("versus_parent", {}).get("reference_result_id"))
        kept = [row for row in ledger["rows"] if row["result_id"] in required]
        ledger["omitted_count"] += len(ledger["rows"]) - len(kept)
        ledger["rows"] = kept
    if not fits() and selection:
        outbound["experiment_ledger"]["accepted_selection"] = {
            key: value for key, value in selection.items() if key not in {"candidate", "snapshot_sha"}}
        outbound["experiment_ledger"]["accepted_selection"]["readback"] = (
            "Measured configuration is in the matching ledger row; full selection in read_session_history.")
    # Once measurements are known, their identity, score, effective config and
    # admission remain decision facts. Detailed comparisons and training traces
    # can be read by result ID; they must not prevent the next lifecycle action.
    for row in outbound["experiment_ledger"]["rows"]:
        if fits():
            break
        omitted = [key for key in ("versus_parent", "versus_best_before", "sample_counts", "training_progress")
                   if key in row]
        if omitted:
            for key in omitted:
                row.pop(key)
            row["omitted_details"] = omitted
            row["readback"] = "inspect_results for this result_id; full measurements remain stored"
    fits()


def compose(repo, gateway, event, base):
    state = repo.state()
    if state.incident_id is None or state.incident_finished:
        state.incident_id = "incident-" + digest({"task": repo.key, "epoch": repo.binding()["epoch"],
                                                "event": event.event_id, "decision": state.decision_version})[:16]
        state.incident_finished = False
        state.phase = "exploration"
        repo.save_state(state, state.decision_version)
    gateway.refresh_facts(event.created_at)
    ledger = ExperimentLedger(repo).view(event.created_at)
    delivery = current_delivery(repo)
    ledger["agent_proposed_selection"] = proposals(repo)[-1:] or None
    ledger["accepted_selection"] = delivery["accepted_selection"] if delivery else None
    board = EvidenceBoard(repo).view(event.created_at)
    consecutive = 0
    for row in reversed(ledger["rows"]):
        if row["versus_best_before"]["improved"] is False:
            consecutive += 1
        else:
            break
    if consecutive >= gateway.config.budget.max_no_improvement and not is_open(repo):
        state.phase = "closing"
        repo.save_state(state, state.decision_version)
    card, metric = gateway.describe_task()
    history = trajectory.build(repo, event.created_at)
    if ledger["rows"]:
        card = {**card, "actual_execution_samples": ledger["rows"][-1]["sample_counts"]}
    full = {**base, "task_card": card, "metric_spec": metric, "experiment_ledger": ledger,
            "evidence_board": board, "budget": snapshot(repo), "state": asdict(state), "session_history": history}
    if is_open(repo):
        state.phase = "exploration"
        base = {**base, "progress": {"consecutive_non_improvements": consecutive,
                "notice": "Reconsider unresolved hypotheses; this notice does not stop exploration." if consecutive else None}}

    def projection(phase):
        closing = phase == "closing"
        result = {k: v for k, v in base.items() if k not in {"state", "tools", "experiment_capabilities", "context_version", "binding", "goal"}}
        binding = repo.binding()
        result.update(protocol="agent-facts-v2", phase=phase, goal=state.task_goal if not closing else "Finalize measured selection or explicit handoff.",
                      authority={"owner": binding["owner"], "epoch": binding["epoch"],
                                 "policy_version": (binding["policy"] or {}).get("version"),
                                 "mode": (binding["policy"] or {}).get("mode"), "selection_is_not_release": True},
                      tools=gateway.catalog(), task_card=card, metric_spec=metric, budget=snapshot(repo))
        result["session_history"] = deepcopy(history)
        if closing:
            permitted = {"finalize_selection", "finish_without_new_model", "complete_incident", "get_tool_result",
                         "get_experiment_report", "inspect_results", "wait_until", "request_release", "cancel_experiment", trajectory.HISTORY_TOOL}
            result["tools"] = {k: v for k, v in result["tools"].items() if k in permitted}
            if not history["total_count"]:
                result["tools"].pop(trajectory.HISTORY_TOOL, None)
            # Compact descriptions, not argument keys or executable validation.
            result["tools"] = {tool: {key: (value if key in {"condition_type", "clock_domain", "predicate", "outcome"} else "required")
                               for key, value in schema.items()} for tool, schema in result["tools"].items()}
            if not state.pending_tool_jobs:
                result["tools"].pop("get_experiment_report", None)
                result["tools"].pop("cancel_experiment", None)
            if not delivery:
                result["tools"].pop("request_release", None)
            if not ledger["rows"]:
                for key in ("finalize_selection", "inspect_results", "request_release"):
                    result["tools"].pop(key, None)
                if not state.pending_tool_jobs:
                    result["tools"].pop("get_experiment_report", None)
            result["task_card"] = {k: card[k] for k in ("task_type", "target", "horizon", "evaluation_protocol", "objective_version")}
            result["metric_spec"] = {k: metric[k] for k in ("name", "direction", "formula", "normalized_unit", "tie_tolerance")}
            for key in ("resources", "online_background", "experiments", "safety", "serving", "observer_briefings"):
                result.pop(key, None)
            result["trigger_event"] = {"created_at": event.created_at}
        rows = ledger["rows"]
        mandatory = set(ledger["best_observed_by_objective"].values()) | set(ledger["best_eligible_candidate"].values())
        mandatory.add((ledger.get("accepted_selection") or {}).get("result_id"))
        if ledger["latest_result"]:
            mandatory.add(ledger["latest_result"])
            mandatory.add(rows[-1]["parent_result_id"])
        visible_rows = rows if len(rows) <= 8 and not closing else [r for r in rows if r["result_id"] in mandatory]
        result["experiment_ledger"] = {k: v for k, v in ledger.items() if k != "rows"}
        if ledger.get("agent_proposed_selection"):
            result["experiment_ledger"]["agent_proposed_selection"] = [
                {**{k: proposal.get(k) for k in ("record_id", "decision_version", "accepted", "reason_code",
                     "program_best_result_id", "selection_regret")},
                 "proposed_result_id": proposal["raw_arguments"].get("result_id"),
                 "proposed_candidate_id": proposal["raw_arguments"].get("candidate_id"),
                 "readback": "read_session_history for the complete decision and receipt"}
                for proposal in ledger["agent_proposed_selection"]]
        result["experiment_ledger"].update(rows=[compact_row(r, closing=closing) for r in visible_rows],
                                           omitted_count=len(rows) - len(visible_rows), readback="inspect_results(offset,limit)")
        if closing:
            result["experiment_ledger"].pop("agent_proposed_selection", None)
            result["experiment_ledger"].pop("search_progress", None)  # Exploration advice is already in the stored ledger.
        latest = [f for f in board if f["status"] == "current"][-6:]
        if closing:
            latest = [f for f in latest if f["source_tool"] not in {"run_experiment", "get_experiment_report", "inspect_results", "compare_candidates"}]
        result["evidence_board"] = [{"fact_id": f["fact_id"], "tool": f["source_tool"], "status": f["status"],
                                     "event_time": f["event_time"], "expires_at": f["expires_at"],
                                     "facts": brief_result(f["value"])} for f in latest]
        if closing:
            for fact in result["evidence_board"]:
                # This view already selects only facts current at event time.
                # The complete expiry binding remains in the stored evidence.
                fact.pop("expires_at", None)
                if "as_of" in fact["facts"]:
                    fact.pop("event_time", None)
                if "windows" in fact["facts"]:
                    fact["facts"]["windows"] = fact["facts"]["windows"][-1:]
                observations = fact["facts"].get("observations", [])
                if observations:
                    fact["facts"]["observations"] = [observations[0], observations[-1]] if len(observations) > 1 else observations
                    for observation in fact["facts"]["observations"]:
                        observation["evidence"].pop("normalized_mse", None)
                if fact["tool"] in {"run_experiment", "get_experiment_report", "inspect_results", "compare_candidates"}:
                    fact.pop("facts", None)  # Measurements are in the retained ledger; full history is addressable.
                    fact.pop("event_time", None)
                    fact.pop("expires_at", None)
        result["evidence_history"] = {"omitted_count": len(board) - len(latest), "readback": "get_tool_result(fact_id)"}
        last = state.last_result
        result["state"] = {"incident_id": state.incident_id, "decision_version": state.decision_version,
                           "status": state.status, "pending_tool_jobs": state.pending_tool_jobs,
                           "next_inspection": state.next_inspection,
                           "registered_inspections": [{k: v[k] for k in ("record_id", "due_at", "status")}
                              for v in repo.records("inspection_schedule", 3)],
                           "last_result": {**last, "result": brief_result(last["result"])} if last else None}
        if last:
            newest = history["entries"][-1] if history["entries"] else None
            captured = next((f for f in reversed(latest) if f["source_tool"] == last["tool"]
                             and f["value"] == last["result"]), None)
            if (last["tool"] != trajectory.HISTORY_TOOL and newest
                    and newest["decision_id"] == last["decision_id"]
                    and newest["observation"]["result"] == last["result"]):
                result["state"]["last_result"]["result"] = {
                    "session_sequence": newest["sequence"],
                    "readback": "Latest historical_tool_observation above; full receipt in read_session_history."}
            elif captured:
                result["state"]["last_result"]["result"] = {"evidence_ref": captured["fact_id"]}
        if closing and last and last["tool"] == "run_experiment":
            result["state"]["last_result"]["result"] = {k: v for k, v in result["state"]["last_result"]["result"].items()
                if k in {"result_id", "candidate_id", "score", "eligible", "error"}}
        if not closing:
            result["state"].update(current_hypotheses=state.current_hypotheses[-3:], waiting_conditions=state.waiting_conditions)
            result["experiment_capabilities"] = gateway.describe_candidates()
        result["memory"] = base.get("memory", [])[:2] if not closing else []
        result["knowledge"] = base.get("knowledge", [])[:2] if not closing else []
        result["visible_evidence_refs"] = visible_refs(result)
        return result

    from .llm import messages_for, reservation_for
    phase = state.phase
    outbound = projection(phase)
    omitted = []
    policy = repo.binding()["policy"] or {}
    limit = (min(policy.get("max_call_tokens", 32768), CLOSING_RESERVATION) if phase == "closing"
             else policy.get("max_call_tokens", 32768))
    # Background is removable; the known best, task and objective are mandatory.
    for key in ("online_background", "knowledge", "memory"):
        if reservation_for(messages_for(outbound), phase) > limit:
            omitted.extend(f"{key}:{v.get('record_id', i)}" for i, v in enumerate(outbound.get(key, [])))
            outbound[key] = []
    fit_history(outbound, limit)
    reservation = reservation_for(messages_for(outbound), phase)
    try:
        check_request(repo, reservation, phase)
    except AgentBudgetError as error:
        record_boundary(repo, event.created_at, phase, reservation, error.reason_code)
        if phase == "closing" or is_open(repo):
            raise
        phase = "closing"
        state.phase = phase
        repo.save_state(state, state.decision_version)
        outbound = projection(phase)
        outbound["exploration_stopped_reason"] = error.reason_code
        fit_history(outbound, min(policy.get("max_call_tokens", 32768), CLOSING_RESERVATION))
        reservation = reservation_for(messages_for(outbound), phase)
        check_request(repo, reservation, phase)
    outbound["visible_evidence_refs"] = visible_refs(outbound)
    # A reference list costs tokens too; reserve the exact final outbound request.
    reservation = reservation_for(messages_for(outbound), phase)
    check_request(repo, reservation, phase)
    record_boundary(repo, event.created_at, phase, reservation)
    result = {**outbound, "binding": repo.binding(), "context_version": digest(outbound),
              "_audit": {"precompression": full, "outbound": deepcopy(outbound), "omitted_background": omitted,
                         "precompression_sha256": digest(full), "outbound_sha256": digest(outbound),
                         "retained_result_ids": [r["result_id"] for r in outbound["experiment_ledger"]["rows"]]}}
    return result
