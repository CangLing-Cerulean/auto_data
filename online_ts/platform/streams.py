"""Operating interface for the existing rolling forecast/training lifecycle."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

from ..config import GateError, digest
from .common import file_lock, read_json, write_json


def _database(runtime):
    path = Path(runtime).resolve() / "lifecycle.db"
    if not path.is_file():
        raise GateError("STREAM_RUNTIME_NOT_FOUND")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    return db


def _task(db, name):
    rows = db.execute("SELECT * FROM tasks WHERE task_key=? OR task_id=?",
                      (name, name)).fetchall()
    if len(rows) != 1:
        raise GateError("STREAM_TASK_NOT_FOUND_OR_AMBIGUOUS")
    return rows[0]


def status(runtime, task):
    with closing(_database(runtime)) as db:
        row = _task(db, task)
        key, state = row["task_key"], json.loads(row["state"])
        config = json.loads(row["config"])
        heartbeat = Path(runtime).resolve() / "platform" / (digest(key) + ".json")
        health = read_json(heartbeat) if heartbeat.exists() else None
        counts = {table: db.execute(f"SELECT COUNT(*) FROM {table} WHERE task_key=?", (key,)).fetchone()[0]
                  for table in ("predictions", "labels", "models", "monitoring", "jobs")}
        jobs = [dict(r) for r in db.execute(
            "SELECT run_id,status,reason,spent,cutoff,lease_until FROM jobs WHERE task_key=? ORDER BY cutoff DESC LIMIT 10",
            (key,))]
        windows = [json.loads(r["evidence"]) for r in db.execute(
            "SELECT evidence FROM monitoring_windows WHERE task_key=? ORDER BY evaluated_ns DESC LIMIT 3", (key,))]
        events = [{"kind": r["kind"], "event_time": r["event_time"], "payload": json.loads(r["payload"])}
                  for r in db.execute("SELECT * FROM events WHERE task_key=? ORDER BY id DESC LIMIT 10", (key,))]
    age = None if not health else max(0., time.time() - health["updated_at"])
    alive = bool(health and health["runner_state"] == "running" and age <= health["stale_after_seconds"])
    return {"task_key": key, "mode": config["mode"], "state": state, "counts": counts,
            "runner": {**(health or {}), "heartbeat_age_seconds": age, "heartbeat_fresh": alive},
            "latest_jobs": jobs, "latest_monitoring_windows": windows, "latest_events": events,
            "source": {"kind": config["source"]["kind"], "variables": config["variables"],
                       "frequency_seconds": config["frequency_seconds"], "horizon_steps": config["horizon_steps"]},
            "monitoring_policy": config["monitoring"], "release_policy": config["release"]}


def predictions(runtime, task, *, after_issue=None, limit=10):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise GateError("STREAM_PREDICTION_LIMIT")
    with closing(_database(runtime)) as db:
        key = _task(db, task)["task_key"]
        minimum = -1 if after_issue is None else after_issue
        issues = [r[0] for r in db.execute(
            "SELECT DISTINCT issue_ns FROM predictions WHERE task_key=? AND published=1 AND issue_ns>? ORDER BY issue_ns LIMIT ?",
            (key, minimum, limit))]
        rows = [] if not issues else [dict(r) for r in db.execute(
            "SELECT model_id,issue_ns,target_ns,variable,horizon,value,role,inference_ms,quality FROM predictions "
            "WHERE task_key=? AND published=1 AND issue_ns>=? AND issue_ns<=? ORDER BY issue_ns,horizon,variable",
            (key, issues[0], issues[-1]))]
    return {"task_key": key, "predictions": rows, "origins": len(issues),
            "next_after_issue": issues[-1] if issues else after_issue,
            "cursor_semantics": "Only published forecasts; complete origins, exclusive nanosecond issue cursor."}


def run(config_path, runtime, *, poll_seconds=1., max_ticks=None):
    from ..cli import read_config
    from ..engine import Engine
    if type(poll_seconds) not in {int, float} or not .05 <= poll_seconds <= 60:
        raise GateError("STREAM_POLL_INTERVAL")
    if max_ticks is not None and (type(max_ticks) is not int or max_ticks < 1):
        raise GateError("STREAM_TICK_LIMIT")
    config = read_config(str(config_path))
    if config.mode != "live":
        raise GateError("STREAM_RUN_REQUIRES_LIVE_CONFIG")
    root = Path(runtime).resolve()
    marker = root / "platform" / (digest(f"{config.task_id}@{config.version[:16]}") + ".json")
    with file_lock(marker.with_suffix(".lock")):
        engine = Engine.create(root, config)
        started = time.time()
        ticks = 0
        runner_state, error = "running", None
        def heartbeat():
            state = engine._state()
            write_json(marker, {"task_key": engine.key, "started_at": started,
                "updated_at": time.time(), "ticks": ticks, "runner_state": runner_state,
                "last_event_ns": state["now"], "state": state["status"],
                "stale_after_seconds": max(30., config.source.timeout_seconds * config.source.retries + 30.),
                "error": error})
        try:
            engine.resume()
            heartbeat()
            while not engine._state()["stop_requested"]:
                engine.advance(time.time_ns())
                ticks += 1
                heartbeat()
                if max_ticks is not None and ticks >= max_ticks:
                    runner_state = "paused"
                    break
                time.sleep(poll_seconds)
            else:
                runner_state = "stopped"
        except KeyboardInterrupt:
            # Pause the service; explicit `stream stop` cancels the task.
            runner_state = "paused"
        except Exception as exc:
            runner_state, error = "failed", type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            engine.close()
            heartbeat()
    return status(root, engine.key)


def manage(runtime, task, action, *, actor=None):
    from ..engine import Engine
    # Refuse a missing runtime rather than creating an empty database.
    with closing(_database(runtime)) as db:
        key = _task(db, task)["task_key"]
    engine = Engine(runtime, key)
    try:
        if action == "stop":
            engine.stop()
        elif action == "retrain":
            if engine.retrain() is None:
                raise GateError("RETRAIN_NOT_ADMITTED")
        elif action == "approve":
            if not actor:
                raise GateError("APPROVER_REQUIRED")
            engine.approve(actor)
        elif action == "rollback":
            engine.rollback()
        else:
            raise GateError("STREAM_ACTION")
    finally:
        engine.close()
    return status(runtime, key)
