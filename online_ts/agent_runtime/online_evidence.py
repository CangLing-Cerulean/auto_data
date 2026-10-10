"""Mature online evidence may become development background, never a fresh Test."""
from dataclasses import asdict

from ..config import GateError, canonical, digest
from ..monitoring import enough, summarize


def approved_online(gateway, now):
    engine, repo = gateway.engine, gateway.repo
    policy = repo.binding()["policy"] or {}
    if not policy.get("allow_online_development", False):
        return []
    # Keep the newest mature suffix unexposed for the next independent holdout.
    # Otherwise continuously importing the latest online window would make every
    # immediately subsequent experiment's Test already seen.
    if not hasattr(engine.source, "labels_until"):
        return []
    mature = engine.source.labels_until(now)
    if len(mature) < engine.config.search.test_steps:
        return []
    independent_start = int(mature.index[-engine.config.search.test_steps])
    result = []
    for window in engine.store.monitoring_windows(engine.key)[-12:]:
        if window["label_state"] not in {"complete", "revised"} or window["evaluated_ns"] > now:
            continue
        model_id, start, end = window["model_id"], window["window_start"], window["window_end"]
        rows = engine.store.prediction_rows_by_issue(engine.key, model_id, start, end, now)
        if not rows or max(r["target_ns"] for r in rows) > now:
            continue
        fingerprint = digest([(r["issue_ns"], r["target_ns"], r["variable"], r.get("label_revision"), r["actual"]) for r in rows])
        if fingerprint != window["label_fingerprint"]:
            continue  # Revised labels require a fresh deterministic evaluation.
        try:
            model = engine._load_model(model_id)
        except GateError:
            continue  # A corrupt retired package does not invalidate the current model's measurements.
        evidence = summarize(rows, engine.config, model.scale, expected_origins=range(start, end, engine.config.step_ns))
        if not enough(evidence, engine.config) or not evidence["labels_complete"] or evidence["protocol_version"] != "mature-labels-v1":
            continue
        target_start, target_end = min(r["target_ns"] for r in rows), max(r["target_ns"] for r in rows)
        if target_end >= independent_start:
            continue
        source_id = digest(asdict(engine.config.source))
        record_id = "online-" + digest({"task": engine.key, "model": model_id, "window": [start, end], "labels": fingerprint})[:24]
        value = {"record_id": record_id, "model_id": model_id, "task_scope": engine.key,
                 "window_start": start, "window_end": end, "target_start": target_start, "target_end": target_end,
                 "label_fingerprint": fingerprint, "evaluation_protocol": "mature-labels-v1", "as_of": window["evaluated_ns"],
                 "source_id": source_id, "usage": "development_background", "used_for_development": True,
                 "evidence": {k: evidence[k] for k in ("normalized_mae", "normalized_mse", "mae_by_variable", "mae_by_horizon_variable", "business_groups", "bias_by_variable", "clean_inputs")}}
        baseline = window["evidence"].get("same_window_baselines")
        if baseline:
            value["evidence"]["same_window_baselines"] = {"eligible": baseline["eligible"],
                "reports": {name: {k: r[k] for k in ("normalized_mae", "normalized_mse")}
                            for name, r in baseline.get("reports", {}).items()}}
        with engine.store.transaction() as db:
            # Do not contaminate an already reserved independent Test of any
            # task sharing the source. This check and exposure are serialized.
            blocked = False
            jobs = db.execute("SELECT j.snapshot,t.config FROM jobs j JOIN tasks t ON t.task_key=j.task_key WHERE j.status IN ('PENDING','RUNNING')").fetchall()
            import json
            for job in jobs:
                config = json.loads(job["config"])
                if digest(config["source"]) != source_id:
                    continue
                meta = engine.artifacts.read_json(json.loads(job["snapshot"]))
                if meta["times"][-config["search"]["test_steps"]] <= target_end and meta["times"][-1] >= target_start:
                    blocked = True
                    break
            # Creating an experiment reserves its Test before start_experiment
            # creates a Job. Background observers must respect that interval too.
            experiments = db.execute("SELECT r.payload,t.config FROM agent_runtime_records r JOIN tasks t ON t.task_key=r.task_key WHERE r.kind='experiment'").fetchall()
            for row in experiments:
                config = json.loads(row["config"])
                if digest(config["source"]) != source_id:
                    continue
                experiment = json.loads(row["payload"])
                meta = engine.artifacts.read_json(experiment["snapshot"])
                if meta["times"][-config["search"]["test_steps"]] <= target_end and meta["times"][-1] >= target_start:
                    blocked = True
                    break
            if blocked:
                continue
            db.execute("INSERT OR IGNORE INTO test_exposures VALUES(?,?,?,?,?,?)",
                       (source_id, target_start, target_end, "agent-development", record_id, now))
            db.execute("INSERT OR IGNORE INTO agent_runtime_records VALUES(?,?,?,?,?)",
                       (engine.key, record_id, "approved_online", now, canonical(value)))
        result.append(value)
    return result[-6:]
