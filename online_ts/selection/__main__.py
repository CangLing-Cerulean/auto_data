"""Development-only study CLI. No remote upload or old-test evaluation capability."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from ..config import GateError
from .common import ROOT, STUDY, StudyConfig, freeze, verify_freeze, write_json


def main():
    parser = argparse.ArgumentParser(description="Fixed DLinear training-window selection study")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--task-name", default="metropt3_dev")
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--task", type=Path, default=STUDY / "tasks/metropt3_dev/task.json")
    freezing.add_argument("--config", type=Path, default=STUDY / "config.json")
    freezing.add_argument("--output", type=Path, default=STUDY / "harness_manifest.json")
    for command in ("run", "run-batch", "report", "freeze-batch", "confirm", "status", "preflight"):
        item = sub.add_parser(command)
        item.add_argument("--manifest", type=Path, default=STUDY / "harness_manifest.json")
        if command in {"run", "status"}:
            item.add_argument("--run-id", required=True)
            item.add_argument("--arm", choices=("agent", "random", "uniform"), default="agent")
        if command in {"freeze-batch", "confirm"}:
            item.add_argument("--batch", type=Path, default=STUDY / "batch_freeze.json")
    args = parser.parse_args()
    if args.command == "init":
        path = STUDY / "config.json"
        if path.exists():
            raise GateError("STUDY_CONFIG_EXISTS")
        write_json(path, asdict(StudyConfig()))
        value = {"config": str(path), "scope": "development_only"}
    elif args.command == "prepare":
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.task_name):
            raise GateError("TASK_NAME")
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify_dataset.py")], cwd=ROOT, check=True)
        from .dataset import prepare_task
        value = prepare_task(STUDY / "tasks" / args.task_name)
    elif args.command == "freeze":
        value = freeze(args.task, args.config, args.output)
    else:
        frozen = verify_freeze(args.manifest)
        from .runtime import run, StudyRepository
        from .execution import planned_runs, freeze_batch, confirm, preflight
        if args.command == "run":
            value = run(args.manifest, args.run_id, args.arm)
        elif args.command == "run-batch":
            value = [run(args.manifest, p["run_id"], p["arm"]) for p in planned_runs(StudyConfig(**frozen["config"]))]
        elif args.command == "freeze-batch":
            value = freeze_batch(frozen, [r["run_id"] for r in planned_runs(StudyConfig(**frozen["config"]))], args.batch)
        elif args.command == "confirm":
            value = confirm(args.manifest, args.batch)
        elif args.command == "report":
            from .report import report
            value = report(frozen, STUDY / "report.json")
        elif args.command == "preflight":
            value = preflight(frozen)
        else:
            repo = StudyRepository(frozen, args.run_id, args.arm)
            repo.export()
            value = asdict(repo.state())
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
