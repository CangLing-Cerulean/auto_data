from __future__ import annotations

import argparse
import json
import sys
import sqlite3
import time
from pathlib import Path

from .config import GateError, SECOND, TaskConfig
from .data import InfluxSource, ReplaySource, iso_time
from .engine import Engine
from .store import Store


def read_config(path: str) -> TaskConfig:
    filename = Path(path).resolve(strict=True)
    data = json.loads(filename.read_text(encoding="utf-8-sig"))
    if data.get("source", {}).get("kind", "file") == "file":
        source_path = Path(data["source"]["path"])
        if not source_path.is_absolute():
            data["source"]["path"] = str((filename.parent / source_path).resolve())
    roots = data.get("foundation", {}).get("model_roots", {})
    for model_id, value in list(roots.items()):
        model_path = Path(value)
        if not model_path.is_absolute():
            roots[model_id] = str((filename.parent / model_path).resolve())
    return TaskConfig.from_dict(data)


def status(engine: Engine) -> dict:
    config, state = engine.store.task(engine.key)
    return {"task_key": engine.key, "config_version": config.version, "mode": config.mode,
            "state": state, "counts": engine.store.counts(engine.key),
            "resources": engine.store.resource_status(),
            "llm_requests": [{"run_id": row["run_id"], "call_id": row["operation_id"], "status": row["status"],
                              "reserved_tokens": row["reserved_tokens"], "actual_tokens": row["actual_tokens"],
                              **{name: row["payload"].get(name) for name in
                                 ("role", "elapsed_ms", "total_ms", "received_bytes", "received_content_chars",
                                  "last_response_ns", "remaining_seconds", "error_code", "legal_proposal_ms")}}
                             for row in engine.store.llm_traces(engine.key)],
            "job_dispositions": engine.store.job_dispositions(engine.key),
            "jobs": [{"run_id": j["run_id"], "status": j["status"], "reason": j["reason"],
                      "spent_seconds": j["spent"], "candidate_index": j["checkpoint"]["index"],
                      "infrastructure_retries": j["checkpoint"].get("infrastructure_retries", j["checkpoint"].get("repairs", 0)),
                      "repair_rounds": j["checkpoint"].get("repair_rounds", 0),
                      "llm_tokens_used": j["checkpoint"].get("llm_tokens_used", 0),
                      "llm_usage_by_role": j["checkpoint"].get("llm_usage_by_role", {}),
                      "seconds_by_role": j["checkpoint"].get("seconds_by_role", {}),
                      "human_pending": j["checkpoint"].get("human_pending"),
                      "best_so_far": j["checkpoint"].get("best_so_far"),
                      "terminal_reason": j["checkpoint"].get("terminal_reason")}
                     for j in engine.store.jobs(engine.key)]}


def bootstrap_replay(engine: Engine, speed: float, max_steps: int | None = None):
    config = engine.config
    if config.mode != "replay" or not isinstance(engine.source, ReplaySource):
        raise GateError("REPLAY_REQUIRES_FILE_SOURCE")
    first, last = engine.source.bounds
    initial = first + (config.search.required_history - 1) * config.step_ns + int(config.allowed_lateness_seconds * SECOND)
    needed = (config.evaluation_ns + int(config.release.shadow_seconds * SECOND)
              + int(config.release.canary_seconds * SECOND)
              + int(config.release.post_promotion_seconds * SECOND)
              + config.horizon_steps * config.step_ns
              + int(config.allowed_lateness_seconds * SECOND))
    if last - initial < needed:
        raise GateError("REPLAY_ONLINE_DURATION_INSUFFICIENT")
    saved = engine._state()["now"]
    now = saved if saved is not None else initial
    engine.resume()
    engine.advance(now)
    # Bootstrap has no incumbent to serve. Keep event time fixed while first training runs.
    while engine._state()["champion"] is None and engine.store.active_job(engine.key):
        engine.advance(now)
        if engine._state()["stop_requested"]:
            return
        time.sleep(0.02)
    steps = 0
    while now + config.step_ns <= last and (max_steps is None or steps < max_steps):
        now += config.step_ns
        engine.advance(now)
        steps += 1
        if engine._state()["stop_requested"]:
            return
        if speed > 0:
            time.sleep(config.frequency_seconds / speed)
    # Finish pending compute at the final observable time; never invent more labels.
    while engine.store.active_job(engine.key):
        engine.advance(now)
        if engine._state()["stop_requested"]:
            return
        time.sleep(0.02)
    engine.advance(now)
    engine.store.event(engine.key, "REPLAY_PAUSED", {"reason": "step_limit" if max_steps is not None and steps >= max_steps else "source_exhausted", "event_time": iso_time(now)}, now)


def run_live(engine: Engine):
    if engine.config.mode != "live":
        raise GateError("START_LIVE_REQUIRES_LIVE_CONFIG")
    engine.resume()
    now = time.time_ns()
    if isinstance(engine.source, InfluxSource):
        report = engine.source.discover(now)
        engine.store.event(engine.key, "SOURCE_DISCOVERED", report, now)
    while not engine._state()["stop_requested"]:
        engine.advance(time.time_ns())
        time.sleep(min(1.0, engine.config.frequency_seconds / 2))
    engine.stop()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="agent", description="可恢复的多变量时序预测闭环")
    root.add_argument("--runtime", default="runtime/agent", help="独立运行目录")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("start", "replay"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "replay":
            command.add_argument("--speed", type=float, default=600.0, help="事件秒/实际秒；0表示尽快回放")
            command.add_argument("--max-steps", type=int)
    for name in ("status", "stop", "retrain", "approve", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("task", help="任务名，存在多个版本时使用完整 task_key")
        if name == "approve":
            command.add_argument("--actor", required=True)
        if name == "status":
            command.add_argument("--events", action="store_true")
    for name in ("models", "review", "compare", "copilot", "draft", "decide", "audit"):
        command = commands.add_parser(name)
        command.add_argument("task")
        command.add_argument("--session")
        if name == "models":
            command.add_argument("--history", action="store_true")
        if name == "review":
            command.add_argument("review_id")
            command.add_argument("--explain", action="store_true")
            command.add_argument("--details", action="store_true")
        if name == "compare":
            command.add_argument("left")
            command.add_argument("right")
        if name in {"copilot", "draft"}:
            command.add_argument("--run")
        if name == "draft":
            command.add_argument("--spec", help="受控意图 JSON 文件：kind/summary/experiments/knowledge")
            command.add_argument("--text", help="自然语言生成草案；需要已授权 LLM")
        if name == "decide":
            command.add_argument("intent_id")
            command.add_argument("--decision", choices=("approve", "reject"), required=True)
    command = commands.add_parser("memory", help="查询同任务开发实验观察；不调用 LLM 或启动训练")
    command.add_argument("task")
    command.add_argument("--index", action="store_true", help="将现有受支持的开发记录加入不可变索引")
    command.add_argument("--as-of-ns", type=int)
    command.add_argument("--limit", type=int, default=20)
    command = commands.add_parser("export-trials", help="导出待人工审核的开发实验样本，不自动用于训练")
    command.add_argument("task")
    command.add_argument("--output", required=True)
    command.add_argument("--as-of-ns", type=int)
    command.add_argument("--limit", type=int, default=500)
    return root


def _print_comparison(result):
    print("同口径可比：" + str(result["comparable"]))
    if result["reason"]:
        print(result["reason"])
    for name, delta in result["deltas_right_minus_left"].items():
        if name in {"metrics.score", "metrics.score_std", "metrics.offline.normalized_mae"}:
            relative = "不适用" if delta["relative_percent"] is None else f"{delta['relative_percent']:.3f}%"
            print(f"{name}：右−左 {delta['absolute']}；相对变化 {relative}；分母 {delta['denominator']}")
    print(result["recommendation"])


def human_command(args):
    from .copilot import Copilot, render_draft
    from .data import ArtifactStore
    from .reviews import render_card
    store = Store(Path(args.runtime) / "lifecycle.db")
    copilot = Copilot(store, ArtifactStore(Path(args.runtime) / "artifacts"), args.task, args.session)
    print("会话：" + copilot.session_id)
    if args.command == "models":
        for card in copilot.models(args.history):
            print(f"{card['candidate_id']} | {card['candidate']['family']} | {card['stage']} | {card['review_id']}")
    elif args.command == "review":
        print(render_card(copilot.view(args.review_id, explain=args.explain), details=args.details))
    elif args.command == "compare":
        _print_comparison(copilot.compare(args.left, args.right))
    elif args.command == "draft":
        if not args.run or bool(args.text) == bool(args.spec):
            raise GateError("DRAFT_REQUIRES_RUN_AND_TEXT_OR_SPEC")
        if args.text:
            value = copilot.interpret(args.run, args.text)
        else:
            value = copilot.draft(args.run, **json.loads(Path(args.spec).read_text(encoding="utf-8-sig")))
        print(render_draft(value))
    elif args.command == "decide":
        print(json.dumps(copilot.decide(args.intent_id, args.decision), ensure_ascii=False, indent=2))
    elif args.command == "audit":
        print(json.dumps(copilot.history(), ensure_ascii=False, indent=2))
    else:
        print(copilot.introduction())
        print("命令：/models，/review 报告ID，/ask 报告ID 问题，/compare 报告ID 报告ID，"
              "/draft 自然语言，/release-draft，/approve 草案ID，/reject 草案ID，/history，/status，/quit。")
        while True:
            try:
                text = input("你> ").strip()
            except EOFError:
                break
            if text == "/quit":
                break
            try:
                command, _, rest = text.partition(" ")
                if command == "/models":
                    for card in copilot.models():
                        print(f"{card['candidate']['family']} {card['candidate_id']} {card['stage']} {card['review_id']}")
                elif command == "/review":
                    print(render_card(copilot.view(rest, explain=True), details=True))
                elif command == "/ask":
                    review_id, _, question = rest.partition(" ")
                    answer = copilot.ask(question, review_id)
                    if answer["status"] != "answered":
                        print("本次未获得 LLM 解释：" + answer["reason"] + "\n" + answer["deterministic_card"])
                    else:
                        for kind, values in answer["explanation"].items():
                            print(kind + "：" + json.dumps(values, ensure_ascii=False))
                elif command == "/compare":
                    left, right = rest.split()
                    _print_comparison(copilot.compare(left, right))
                elif command == "/draft":
                    if not args.run:
                        raise GateError("DRAFT_REQUIRES_RUN")
                    print(render_draft(copilot.interpret(args.run, rest)))
                elif command in {"/approve", "/reject"}:
                    print(json.dumps(copilot.decide(rest, command[1:]), ensure_ascii=False))
                elif command == "/release-draft":
                    print(render_draft(copilot.release_draft()))
                elif command == "/history":
                    for value in copilot.history():
                        print(value["status"] + "\n" + render_draft(value))
                elif command == "/status":
                    print(copilot.introduction())
                else:
                    print("请使用 /ask 报告ID 问题 或 /draft 修改意图；模糊表达不会执行操作。")
            except (GateError, ValueError) as error:
                print("未执行：" + str(error))
    return 0


def main(argv: list[str] | None = None) -> int:
    # PowerShell 7 pipes use UTF-8; Windows Python otherwise falls back to the
    # legacy ANSI encoding for redirected streams and corrupts Chinese dialogs.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    engine = None
    try:
        if args.command in {"memory", "export-trials"}:
            from .copilot import current_actor
            from .memory import query_observations, export_training_candidates, persist_observations
            from .config import canonical, digest
            store = Store(Path(args.runtime) / "lifecycle.db")
            key = store.resolve(args.task)
            config, _ = store.task(key)
            actor = current_actor()
            if args.command == "memory" and args.index:
                if "experiment" not in config.collaboration.permissions.get(actor, ()):
                    raise GateError("HUMAN_PERMISSION_DENIED")
                with store.transaction() as db:
                    for row in db.execute("SELECT * FROM jobs WHERE task_key=?", (key,)).fetchall():
                        job = store._job(row)
                        persist_observations(db, job, job["checkpoint"])
            decision_ns = time.time_ns() if args.as_of_ns is None else args.as_of_ns
            if args.command == "memory":
                value = query_observations(store, key, decision_ns=decision_ns, limit=args.limit)
                print(json.dumps(value, ensure_ascii=False, indent=2))
            else:
                records = export_training_candidates(store, key, actor=actor, decision_ns=decision_ns, limit=args.limit)
                output = Path(args.output).resolve()
                with output.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.writelines(canonical(record) + "\n" for record in records)
                print(json.dumps({"output": str(output), "records": len(records), "review_status": "pending_human_review",
                                  "content_fingerprint": digest(records)}, ensure_ascii=False))
            return 0
        if args.command in {"models", "review", "compare", "copilot", "draft", "decide", "audit"}:
            return human_command(args)
        if args.command in {"start", "replay"}:
            config = read_config(args.config)
            engine = Engine.create(args.runtime, config)
            if args.command == "replay":
                if args.speed < 0 or (args.max_steps is not None and args.max_steps < 1):
                    raise GateError("REPLAY_OPTIONS")
                bootstrap_replay(engine, args.speed, args.max_steps)
            else:
                run_live(engine)
        else:
            engine = Engine(args.runtime, args.task)
            if args.command == "stop":
                engine.stop()
            elif args.command == "retrain":
                run_id = engine.retrain()
                if run_id is None:
                    raise GateError("RETRAIN_NOT_ADMITTED")
            elif args.command == "approve":
                engine.approve(args.actor)
            elif args.command == "rollback":
                engine.rollback()
        output = status(engine)
        if args.command == "status" and args.events:
            output["events"] = engine.store.events(engine.key)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        if engine is not None:
            engine.stop()
        return 130
    except GateError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except sqlite3.Error:
        print('{"error":"STATE_DATABASE_UNAVAILABLE"}', file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):
        print('{"error":"CONFIGURATION_OR_IO_ERROR"}', file=sys.stderr)
        return 2
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
