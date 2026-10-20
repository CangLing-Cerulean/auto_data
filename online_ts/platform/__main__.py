"""Unified command line: research workflows and continuous forecasting."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import urllib.parse
import urllib.request

from ..config import GateError
from .common import read_json


def parser():
    root = argparse.ArgumentParser(prog="python -m online_ts.platform",
        description="时序科研平台：冻结实验、恢复执行、持续预测与监视")
    commands = root.add_subparsers(dest="area", required=True)
    commands.add_parser("doctor", help="检查本地运行依赖；不连接数据源或调用模型")
    study = commands.add_parser("study", help="管理新实验工作流").add_subparsers(dest="action", required=True)
    validate = study.add_parser("validate")
    validate.add_argument("--config", required=True)
    create = study.add_parser("create")
    create.add_argument("--config", required=True)
    create.add_argument("--run", required=True, help="工作区outputs下的新目录")
    for name in ("run", "resume", "status", "report"):
        study.add_parser(name).add_argument("--run", required=True)
    resolve = study.add_parser("reconcile", help="为未知执行状态附上人工核对证据，不重跑旧尝试")
    resolve.add_argument("--run", required=True)
    resolve.add_argument("--step", required=True)
    resolve.add_argument("--attempt", type=int, required=True)
    resolve.add_argument("--outcome", choices=("completed", "failed"), required=True)
    resolve.add_argument("--returncode", type=int, required=True)
    resolve.add_argument("--evidence", required=True)
    resolve.add_argument("--actor", required=True)
    resolve.add_argument("--reason", required=True)
    stream = commands.add_parser("stream", help="接入、运行和管理滚动预测").add_subparsers(dest="action", required=True)
    serve = stream.add_parser("serve", help="启动本机持久到达日志；适配外部producer")
    serve.add_argument("--database", required=True)
    serve.add_argument("--variables", nargs="+", required=True)
    serve.add_argument("--port", type=int, default=18761)
    ingest = stream.add_parser("ingest", help="发送已有JSONL记录；幂等键必填")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--url", default="http://127.0.0.1:18761")
    worker = stream.add_parser("run")
    worker.add_argument("--config", required=True)
    worker.add_argument("--runtime", required=True)
    worker.add_argument("--poll-seconds", type=float, default=1.)
    worker.add_argument("--max-ticks", type=int)
    for name in ("status", "monitor", "predictions", "stop", "retrain", "approve", "rollback"):
        item = stream.add_parser(name)
        item.add_argument("--runtime", required=True)
        item.add_argument("--task", required=True)
        if name == "predictions":
            item.add_argument("--after-issue", type=int)
            item.add_argument("--limit", type=int, default=10)
        if name == "approve":
            item.add_argument("--actor", required=True)
    replay = stream.add_parser("replay", help="使用同一Engine做有限历史回放")
    replay.add_argument("--config", required=True)
    replay.add_argument("--runtime", required=True)
    replay.add_argument("--max-steps", type=int, required=True)
    return root


def ingest(path, url):
    # No arbitrary HTTP destination, redirects or credentials in the CLI.
    from ..data import request_bytes
    endpoint = urllib.parse.urlsplit(url)
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
            or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/")):
        raise GateError("STREAM_LOCAL_URL_REQUIRED")
    count = 0
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            request = urllib.request.Request(url.rstrip("/") + "/records", data=body,
                                            headers={"Content-Type": "application/json"}, method="POST")
            result = json.loads(request_bytes(request, 10))
            count += 1
            print(json.dumps({"line": number, "receipt": result}, ensure_ascii=False), flush=True)
    return {"records_submitted": count}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.area == "doctor":
            dependencies = {name: importlib.util.find_spec(name) is not None for name in ("numpy", "pandas", "torch")}
            result = {"python": sys.version, "executable": sys.executable, "sqlite": sqlite3.sqlite_version,
                      "dependencies": dependencies,
                      "ready": dependencies["numpy"] and dependencies["pandas"],
                      "torch_required_for": ["DLinear research", "PatchTST"], "network_access": False}
        elif args.area == "study":
            from . import studies
            if args.action == "validate":
                result = {"status": "valid", "spec": studies.validate_config(args.config)}
            elif args.action == "create":
                result = studies.create(args.config, args.run)
            elif args.action == "reconcile":
                result = studies.reconcile(args.run, args.step, args.attempt,
                    outcome=args.outcome, returncode=args.returncode, evidence=args.evidence,
                    actor=args.actor, reason=args.reason)
            else:
                operation = studies.run if args.action in {"run", "resume"} else getattr(studies, args.action)
                result = operation(args.run)
        elif args.action == "serve":
            from ..history_service import serve
            serve(args.database, args.port, variables=args.variables)
            return 0
        elif args.action == "ingest":
            result = ingest(args.input, args.url)
        elif args.action == "replay":
            from ..cli import main as engine_main
            return engine_main(["--runtime", args.runtime, "replay", "--config", args.config,
                                "--max-steps", str(args.max_steps), "--speed", "0"])
        else:
            from . import streams
            if args.action == "run":
                result = streams.run(args.config, args.runtime, poll_seconds=args.poll_seconds, max_ticks=args.max_ticks)
            elif args.action in {"status", "monitor"}:
                result = streams.status(args.runtime, args.task)
            elif args.action == "predictions":
                result = streams.predictions(args.runtime, args.task, after_issue=args.after_issue, limit=args.limit)
            else:
                result = streams.manage(args.runtime, args.task, args.action, actor=getattr(args, "actor", None))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if args.area == "study" and args.action in {"run", "resume", "report"} and result["state"] != "completed":
            return 3
        return 0 if args.area != "doctor" or result["ready"] else 2
    except (GateError, OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
