"""Deterministic assistants: observe and notify, never choose or start training."""
from copy import deepcopy

import numpy as np

from ..config import GateError, SECOND, digest


def data_health(engine, repo, now):
    frame = engine.source.read_until(now)
    tail = frame.tail(engine.config.search.required_history)
    latest = int(frame.index[-1]) if len(frame) else None
    metadata = repo.records("task_metadata", 1)
    threshold = metadata[0].get("input_freshness_threshold_seconds") if metadata else None
    age = (now - latest) / SECOND if latest is not None else None
    gaps = int(np.sum(np.diff(tail.index.to_numpy(dtype=np.int64)) != engine.config.step_ns)) if len(tail) > 1 else 0
    from ..qualification import input_qualification, regime_context
    qualification = input_qualification(tail, engine.config, now=now, freshness_seconds=threshold)
    return {"as_of": now, "rows": len(tail), "latest_timestamp": latest,
            "delay_seconds": age, "input_age_seconds": age, "freshness_threshold_seconds": threshold,
            "timeliness": "unknown" if threshold is None or age is None else "late" if age > threshold else "within_threshold",
            "sampling_gap_count": gaps, "schema_valid": list(frame.columns) == list(engine.config.variables),
            "qualification": qualification,
            "regime": regime_context(tail, int(tail.index[0]) if len(tail) else now, now + 1,
                                     declared=bool(engine.config.source.regime_column)),
            "revision_status": "as_of_source_contract", "arrival_lag_seconds": None,
            "age_definition": "event-time now minus newest available event timestamp; not per-row arrival latency",
            "missing_fraction": float(tail.isna().to_numpy().mean()) if len(tail) else None,
            "by_variable": {str(k): float(v) for k, v in tail.isna().mean().items()} if len(tail) else {}}


def label_status(engine, repo, now):
    from .waiting import label_windows
    windows = sorted(label_windows(repo), key=lambda w: (w["window_end"], w["model_id"]))
    labels = engine.source.labels_until(now) if hasattr(engine.source, "labels_until") else None
    complete = [w for w in windows if w["fully_evaluable"]]
    pending = [w for w in windows if not w["fully_evaluable"]]
    return {"as_of": now, "windows": windows[-12:], "total_windows": len(windows),
            "evaluable_history_count": len(complete), "latest_evaluable_cutoff": max((w["window_end"] for w in complete), default=None),
            "recent_evaluable_windows": complete[-3:], "latest_window": windows[-1] if windows else None,
            "next_maturity_condition": {"object_id": pending[0]["window_id"], "predicate": {"state": "complete"}} if pending else None,
            "latest_mature_label": int(labels.index[-1]) if labels is not None and len(labels) else None,
            "mature_rows_available": len(labels) if labels is not None else 0,
            "labels_adapter_available": hasattr(engine.source, "labels_until")}


def record_assessment(engine, model_id, fingerprint, evidence, history, reference, triggered, now):
    """Save the detector's actual result at the evaluated window, not a reconstruction."""
    if engine._state().get("decision_owner") != "agent":
        return
    config = engine.config.monitoring
    payload = {"model_id": model_id, "window_start": evidence["window_start"], "window_end": evidence["window_end"],
               "label_fingerprint": fingerprint, "label_state": evidence["label_state"],
               "metric": "normalized_mae", "risk_metric": "normalized_mse", "reference_source": "frozen_stable_online_windows",
               "assessments": history.get("assessments", []), "reference_status": history.get("status"),
               "optimization_opportunity": history.get("optimization_opportunity", False),
               "regime": history.get("regime"), "qualification": history.get("qualification"),
               "ratio_threshold": config.mae_ratio_trigger, "risk_ratio_threshold": config.mse_ratio_trigger,
               "consecutive_required": config.consecutive_checks, "consecutive_observed": history.get("count", 0),
               "triggered": triggered,
               "epoch": engine._state().get("decision_epoch"), "rule_version": digest({
                   "version": "regime-dual-metric-v1", "mae": config.mae_ratio_trigger,
                   "mse": config.mse_ratio_trigger, "consecutive": config.consecutive_checks,
                   "reference_windows": config.reference_windows, "spread": config.reference_max_relative_spread})}
    engine.store.event(engine.key, "AGENT_MONITOR_ASSESSMENT", payload, now,
                       "observer-assessment:" + digest({"model": model_id, "window": evidence["window_start"],
                                                       "labels": fingerprint, "assessment": history.get("assessments"),
                                                       "epoch": payload["epoch"]}))


class ObservationTeam:
    """Runs on serving pulses as well as on-demand; no LLM calls or model selection."""
    def __init__(self, gateway, *, notify=False):
        self.gateway, self.engine, self.repo = gateway, gateway.engine, gateway.repo
        self.notify = notify

    def latest(self, name, now):
        values = self.repo.iter_records("observer_" + name)
        binding = self.repo.binding()
        return max((v for v in values if v["epoch"] == binding["epoch"] and v["champion"] == binding["champion"]
                    and v["as_of"] <= now), key=lambda v: (v["as_of"], v["sequence"]), default=None)

    def _save(self, name, now, value, signal):
        binding = self.repo.binding()
        body = {"observer": name, "implementation": "deterministic", "task_key": self.repo.key,
                "sequence": self.repo.count_records("observer_" + name) + 1,
                "epoch": binding["epoch"], "champion": binding["champion"], "as_of": now,
                "evidence_scope": "development", "value": value, "signal": signal}
        identity = "observer-" + name + "-" + digest(body)[:24]
        report = {**body, "record_id": identity}
        self.repo.put(identity, "observer_" + name, report, now)
        return report

    def _publish(self, report):
        import json
        identity = "observer-notified-" + report["observer"]
        with self.repo.store.transaction() as db:
            row = db.execute("SELECT payload FROM agent_runtime_records WHERE task_key=? AND record_id=?",
                             (self.repo.key, identity)).fetchone()
            previous = json.loads(row[0]) if row else None
            unchanged = previous and all(previous[k] == report[k] for k in ("epoch", "champion", "signal"))
            if unchanged:
                return
            self.repo.store._event(db, self.repo.key, "AGENT_OBSERVER_UPDATE",
                {"observer": report["observer"], "record_id": report["record_id"], "model_id": report["champion"],
                 "changed": [key for key in report["signal"] if not previous or previous["signal"].get(key) != report["signal"][key]]},
                report["as_of"], "observer-notify:" + report["record_id"])
            self.repo.put(identity, "observer_notification", {k: report[k] for k in ("epoch", "champion", "signal")},
                          report["as_of"], replace=True, db=db)

    def manages(self, event, now):
        """Only replace observations covered by our current model's briefing."""
        report = self.latest("data", now)
        if not report or report["value"]["collection_error"] or not report["champion"]:
            return False
        if event["payload"].get("model_id") != report["champion"]:
            return False
        if event["kind"] == "MODEL_DEGRADATION_DETECTED":
            return report["value"]["alert"]["triggered"] is True
        return event["kind"] in {"MONITORING_WINDOW_EVALUATED", "MONITORING_WINDOW_REVISED", "WAITING_FOR_LABELS"}

    def refresh(self, now):
        binding = self.repo.binding()
        if binding["owner"] != "agent" or binding["stop_requested"]:
            return []
        previous = self.latest("data", now)
        # A window cadence bounds collection cost; task state can also request a fresh briefing.
        if previous is None or now - previous["as_of"] >= self.engine.config.evaluation_ns:
            self._data(now)
        self._experiments(now)
        reports = [r for name in ("data", "experiments") if (r := self.latest(name, now)) is not None]
        if self.notify:
            for report in reports:
                self._publish(report)
        return reports

    def _data(self, now):
        from .online_evidence import approved_online
        config = self.engine.config
        events = [e for e in self.repo.store.events(self.repo.key) if e["event_time"] <= now]
        binding = self.repo.binding()
        assessments = [e for e in events if e["kind"] == "AGENT_MONITOR_ASSESSMENT"
                       and e["payload"]["model_id"] == binding["champion"] and e["payload"]["epoch"] == binding["epoch"]]
        latest = assessments[-1]["payload"] if assessments else None
        if latest:
            window = self.engine.store.monitoring_window(self.repo.key, latest["model_id"],
                                                         latest["window_start"], latest["window_end"])
            if (not window or window["label_fingerprint"] != latest["label_fingerprint"]
                    or window["label_state"] not in {"complete", "revised"}):
                latest = None
        # Do not expose detector current/reference/ratio: their windows may be held out.
        alert = {k: latest[k] for k in ("model_id", "window_start", "window_end", "label_fingerprint", "label_state",
                 "metric", "risk_metric", "reference_source", "reference_status", "regime", "qualification", "optimization_opportunity",
                 "ratio_threshold", "risk_ratio_threshold", "consecutive_required", "consecutive_observed",
                 "triggered", "rule_version")} if latest else {"triggered": None}
        alert["numeric_comparison_scope"] = "audit_only; measurements below pass the online development exposure gate"
        alert["threshold_applies_to"] = "current_error / same_scope_frozen_reference_error; never absolute MAE/MSE or another group"
        health, labels, observations, error = {}, {}, [], None
        try:
            health = data_health(self.engine, self.repo, now)
            labels = label_status(self.engine, self.repo, now)
            observations = approved_online(self.gateway, now)
        except (GateError, OSError, ValueError, TypeError) as failure:
            error = str(failure) if isinstance(failure, GateError) else type(failure).__name__
        health_ok = not error and health.get("qualification", {}).get("input_eligible") is True
        minimum_history = min(config.search.histories) + config.search.folds * config.search.validation_steps + config.search.test_steps
        history_ready = labels.get("mature_rows_available", 0) >= minimum_history
        readiness = "ready_for_candidate_validation" if health_ok and history_ready else "needs_data_review"
        active = self.engine.store.active_job(self.repo.key)
        signal = {"input": "healthy" if health_ok else "needs_review", "history_ready": history_ready,
                  "degradation": alert["triggered"], "error": error,
                  "optimization_opportunity": alert.get("optimization_opportunity"),
                  "regime": alert.get("regime"), "reference_status": alert.get("reference_status"),
                  "latest_revision": next((e["id"] for e in reversed(events) if e["kind"] == "MONITORING_WINDOW_REVISED"), None)}
        value = {"input_health": health, "labels": {k: v for k, v in labels.items() if k != "windows"},
                 "alert": alert, "online_measurements": observations, "collection_error": error,
                 "training_readiness": {"status": readiness, "minimum_candidate_history": minimum_history,
                    "maximum_candidate_history": config.search.required_history,
                    "mature_rows": labels.get("mature_rows_available", 0), "active_run_id": active["run_id"] if active else None,
                    "scope": "Preliminary data check only; candidate, resource and permission gates still run."},
                 "interpretation": "Normalized MAE is primary; normalized MSE measures tail risk against frozen stable references. "
                    "Explicit regime transitions require investigation; input shifts alone do not establish a regime or degradation. "
                    "A partial latest window does not invalidate the identified mature alert window. "
                    "Collection is complete as of this report; investigate further only for missing, changed or disputed evidence."}
        return self._save("data", now, value, signal)

    def _experiments(self, now):
        jobs = []
        for experiment in self.repo.iter_records("experiment"):
            if experiment["binding"]["epoch"] != self.repo.binding()["epoch"]:
                continue
            run_id = "train-" + digest({"task": self.repo.key, "experiment_id": experiment["record_id"]})[:24]
            try:
                job = self.repo.store.job(run_id)
            except GateError as error:
                if str(error) != "JOB_NOT_FOUND":
                    raise
                job = None
            if job is None:
                status, completed, development = "not_started", 0, False
            else:
                status, completed = job["status"], job["checkpoint"]["index"]
                development = completed == len(job["plan"]["candidates"]) and not job["plan"].get("needs_proposal")
            jobs.append({"experiment_id": experiment["record_id"], "run_id": run_id if job else None, "status": status,
                         "next_action": "start_experiment" if job is None else "get_experiment_report",
                         "candidates_processed": completed, "development_complete": development,
                         "readback": "get_experiment_report", "selection_owner": "main_agent"})
        signal = {"jobs": sorted(jobs, key=lambda j: j["experiment_id"])}
        previous = self.latest("experiments", now)
        if previous and previous["signal"] == signal:
            return previous
        return self._save("experiments", now, {"jobs": jobs, "metrics_scope": "Status only; final Test is never included."}, signal)

    def read(self, identity, now):
        value = self.repo.get(identity)
        if (value.get("observer") not in {"data", "experiments"} or value.get("task_key") != self.repo.key
                or value.get("epoch") != self.repo.binding()["epoch"] or value.get("as_of", now + 1) > now):
            raise GateError("AGENT_OBSERVER_SCOPE")
        return deepcopy(value)

    def briefings(self, now):
        from .work_context import brief_result
        result = []
        for report in self.refresh(now):
            value = deepcopy(report["value"])
            if report["observer"] == "data":
                health, labels = value["input_health"], value["labels"]
                value = {"input_health": {k: health.get(k) for k in
                         ("rows", "timeliness", "schema_valid", "missing_fraction", "sampling_gap_count")},
                         "labels": {k: labels.get(k) for k in
                         ("latest_mature_label", "mature_rows_available", "evaluable_history_count", "next_maturity_condition")},
                         "alert": value["alert"], "training_readiness": value["training_readiness"],
                         "collection_error": value["collection_error"],
                         "online_measurements": brief_result({"observations": value["online_measurements"]})["observations"]}
            result.append({k: report[k] for k in ("record_id", "observer", "implementation", "as_of")}
                          | {"value": value, "readback": "get_observer_report(record_id)"})
        return result
