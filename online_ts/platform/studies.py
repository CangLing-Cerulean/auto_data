"""Frozen, sequential dependency workflows for trusted research commands.

Commands run with shell=False. This is orchestration, not a sandbox for generated
code. Existing scientific runners retain their own program/data access boundary.
"""
from __future__ import annotations

import os
import importlib.metadata
from pathlib import Path
import re
import subprocess
import sys
import time

from ..config import GateError, canonical, digest
from .common import file_hash, file_lock, read_json, regular_under, write_json

IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}\Z")
ROLES = {"training", "validation", "response", "protocol", "code"}


def _positive(value, name, *, integer=False):
    if type(value) not in ({int} if integer else {int, float}) or not 0 < value < 10**7:
        raise GateError("STUDY_INVALID_" + name)


def validate_config(path):
    source = Path(path).resolve(strict=True)
    spec = read_json(source)
    if not isinstance(spec, dict) or set(spec) != {"version", "name", "workspace", "inputs", "steps"}:
        raise GateError("STUDY_CONFIG_FIELDS")
    if spec["version"] != 1 or not isinstance(spec["name"], str) or not IDENTIFIER.fullmatch(spec["name"]):
        raise GateError("STUDY_IDENTITY")
    workspace = (source.parent / spec["workspace"]).resolve(strict=True)
    if not workspace.is_dir():
        raise GateError("STUDY_WORKSPACE")
    if not isinstance(spec["inputs"], dict) or not isinstance(spec["steps"], list) or not spec["steps"]:
        raise GateError("STUDY_INPUTS_OR_STEPS")
    assets = {}
    for key, item in spec["inputs"].items():
        if (not IDENTIFIER.fullmatch(key) or not isinstance(item, dict)
                or set(item) != {"path", "role"} or item["role"] not in ROLES):
            raise GateError("STUDY_INPUT_ROLE_OR_FIELDS")
        target = regular_under(workspace, item["path"])
        assets[key] = {**item, "bytes": target.stat().st_size, "sha256": file_hash(target)}
    seen = set()
    normalized = []
    for step in spec["steps"]:
        required = {"id", "depends_on", "command", "timeout_seconds", "outputs"}
        optional = {"max_attempts", "retry_exit_codes"}
        if not isinstance(step, dict) or set(step) - required - optional or required - set(step):
            raise GateError("STUDY_STEP_FIELDS")
        key = step["id"]
        if not isinstance(key, str) or not IDENTIFIER.fullmatch(key) or key in seen:
            raise GateError("STUDY_STEP_ID")
        dependencies = step["depends_on"]
        if (not isinstance(dependencies, list) or len(dependencies) != len(set(dependencies))
                or not set(dependencies) <= seen):
            raise GateError("STUDY_DEPENDENCIES_MUST_PRECEDE_STEP")
        command = step["command"]
        if not isinstance(command, list) or not command or any(not isinstance(s, str) or not s or "\0" in s for s in command):
            raise GateError("STUDY_ARGV_REQUIRED")
        # Tokens are expanded by the worker; no shell or environment interpolation.
        for argument in command:
            for token in re.findall(r"\{([^{}]+)\}", argument):
                if token not in {"python", "workspace", "attempt", "context"}:
                    raise GateError("STUDY_UNKNOWN_COMMAND_TOKEN")
        _positive(step["timeout_seconds"], "TIMEOUT")
        attempts = step.get("max_attempts", 1)
        _positive(attempts, "ATTEMPTS", integer=True)
        if attempts > 5:
            raise GateError("STUDY_MAX_ATTEMPTS")
        exits = step.get("retry_exit_codes", [])
        if not isinstance(exits, list) or any(type(v) is not int or not 1 <= v <= 255 for v in exits):
            raise GateError("STUDY_RETRY_CODES")
        outputs = step["outputs"]
        if not isinstance(outputs, list) or len(outputs) != len(set(outputs)):
            raise GateError("STUDY_OUTPUTS")
        for item in outputs:
            if (not isinstance(item, str) or not Path(item).parts or Path(item).is_absolute()
                    or ".." in Path(item).parts or Path(item).parts[0] in {
                        "receipt.json", "context.json", "stdout.log", "stderr.log", "worker.lock",
                        "started.json", "supervisor.log", "reconciliation.json", "reconciliation-evidence.bin"}):
                raise GateError("STUDY_OUTPUT_PATH")
        seen.add(key)
        normalized.append({**step, "max_attempts": attempts, "retry_exit_codes": exits})
    return {**spec, "workspace": str(workspace), "inputs": assets, "steps": normalized}


def create(config, run):
    spec = validate_config(config)
    root = Path(run).resolve()
    workspace = Path(spec["workspace"])
    # Protect datasets and historical evidence by keeping platform runs separate.
    output_root = workspace / "outputs"
    if not root.is_relative_to(output_root) or root == output_root or root.exists():
        raise GateError("STUDY_NEW_OUTPUTS_DIRECTORY_REQUIRED")
    code = {}
    package = Path(__file__).resolve().parent.parent
    for file in sorted(package.rglob("*.py")):
        code[str(file)] = file_hash(file)
    lock = {"version": 1, "created_at": time.time(), "spec": spec, "platform_code": code,
            "python": sys.version, "python_executable": sys.executable,
            "dependencies": _dependencies()}
    lock["sha256"] = digest(lock)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "study.json", lock, exclusive=True)
    write_json(root / "created.json", {"state": "ready", "lock_sha256": lock["sha256"]}, exclusive=True)
    return status(root)


def verify(root):
    root = Path(root).resolve(strict=True)
    lock = read_json(root / "study.json")
    if lock.get("sha256") != digest({k: v for k, v in lock.items() if k != "sha256"}):
        raise GateError("STUDY_LOCK_CHANGED")
    if lock["python"] != sys.version or lock["python_executable"] != sys.executable:
        raise GateError("STUDY_PYTHON_CHANGED")
    if lock["dependencies"] != _dependencies():
        raise GateError("STUDY_DEPENDENCIES_CHANGED")
    for path, expected in lock["platform_code"].items():
        if file_hash(path) != expected:
            raise GateError("STUDY_PLATFORM_CODE_CHANGED")
    for item in lock["spec"]["inputs"].values():
        if file_hash(regular_under(lock["spec"]["workspace"], item["path"])) != item["sha256"]:
            raise GateError("STUDY_INPUT_CHANGED")
    return lock


def _dependencies():
    result = {}
    for name in ("numpy", "pandas", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _attempts(root, step):
    folder = Path(root) / "steps" / step["id"]
    if not folder.exists():
        return []
    result = []
    for path in sorted(folder.iterdir()):
        if not path.is_dir() or path.is_symlink() or not re.fullmatch(r"[0-9]{3}", path.name):
            raise GateError("STUDY_ATTEMPT_LAYOUT")
        receipt = read_json(path / "receipt.json") if (path / "receipt.json").exists() else None
        reconciled = False
        if receipt is None and (path / "reconciliation.json").exists():
            receipt = read_json(path / "reconciliation.json")["receipt"]
            reconciled = True
        state = receipt["state"] if receipt else "uncertain"
        if receipt is None:
            try:
                with file_lock(path / "worker.lock"):
                    pass
            except GateError as exc:
                if str(exc) != "PLATFORM_BUSY":
                    raise
                state = "running"
        result.append({"attempt": int(path.name), "state": state, "directory": str(path),
                       "receipt": receipt, "reconciled": reconciled})
    return result


def status(root):
    root = Path(root).resolve(strict=True)
    lock = read_json(root / "study.json")
    steps = []
    for step in lock["spec"]["steps"]:
        attempts = _attempts(root, step)
        state = attempts[-1]["state"] if attempts else "pending"
        steps.append({"id": step["id"], "state": state, "attempts": attempts})
    states = {step["state"] for step in steps}
    state = ("completed" if states == {"completed"} else "running" if "running" in states
             else "needs_attention" if states - {"completed", "pending"} else "ready")
    receipts = [attempt["receipt"] for step in steps for attempt in step["attempts"] if attempt["receipt"]]
    return {"name": lock["spec"]["name"], "run": str(root), "state": state,
            "lock_sha256": lock["sha256"], "steps": steps,
            "attempts": sum(len(step["attempts"]) for step in steps),
            "known_command_seconds": sum(r["seconds"] for r in receipts if r["seconds"] is not None),
            "unknown_command_seconds": sum(r["seconds"] is None for r in receipts),
            "unknown_attempts": sum(a["receipt"] is None for s in steps for a in s["attempts"]),
            "cost_scope": "Every physical command attempt including failures; LLM tokens/GPU training counts belong to command outputs."}


def _check_outputs(attempt):
    for relative, record in attempt["receipt"]["outputs"].items():
        path = regular_under(attempt["directory"], relative)
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise GateError("STUDY_OUTPUT_CHANGED")


def run(root):
    """Resume reconciles durable receipts. Unknown outcomes are never retried."""
    root = Path(root).resolve(strict=True)
    with file_lock(root / "runner.lock"):
        lock = verify(root)
        for step in lock["spec"]["steps"]:
            attempts = _attempts(root, step)
            if attempts and attempts[-1]["state"] == "completed":
                _check_outputs(attempts[-1])
                continue
            if attempts:
                last = attempts[-1]
                if last["state"] in {"running", "uncertain"}:
                    return status(root)
                if (last["state"] != "failed" or last["receipt"]["returncode"] not in step["retry_exit_codes"]
                        or len(attempts) >= step["max_attempts"]):
                    return status(root)
            while len(attempts) < step["max_attempts"]:
                attempt = root / "steps" / step["id"] / f"{len(attempts) + 1:03d}"
                attempt.mkdir(parents=True, exist_ok=False)
                dependencies = {}
                for key in step["depends_on"]:
                    dependency = next(s for s in lock["spec"]["steps"] if s["id"] == key)
                    saved = _attempts(root, dependency)[-1]
                    if saved["state"] != "completed":
                        raise GateError("STUDY_DEPENDENCY_INCOMPLETE")
                    _check_outputs(saved)
                    dependencies[key] = saved["directory"]
                context = {"run": str(root), "attempt": str(attempt), "step_id": step["id"],
                           "workspace": lock["spec"]["workspace"], "dependencies": dependencies,
                           "inputs": {name: str(Path(lock["spec"]["workspace"]) / item["path"])
                                      for name, item in lock["spec"]["inputs"].items()},
                           "study_sha256": lock["sha256"]}
                write_json(attempt / "context.json", context, exclusive=True)
                command = [sys.executable, "-B", "-m", "online_ts.platform.study_worker",
                           str(root), step["id"], str(attempt)]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                with (attempt / "supervisor.log").open("xb") as log:
                    process = subprocess.Popen(command, env=env, cwd=lock["spec"]["workspace"],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=flags, start_new_session=os.name != "nt")
                    try:
                        process.wait()
                    except KeyboardInterrupt:
                        # The detached worker retains its own receipt; later resume
                        # reconciles it instead of submitting the command again.
                        return status(root)
                attempts = _attempts(root, step)
                last = attempts[-1]
                if last["state"] == "completed":
                    _check_outputs(last)
                    break
                if (last["state"] != "failed" or last["receipt"]["returncode"] not in step["retry_exit_codes"]):
                    return status(root)
            if attempts[-1]["state"] != "completed":
                return status(root)
        return status(root)


def report(root):
    result = status(root)
    result["verified"] = False
    verify(root)
    for step in result["steps"]:
        if step["state"] == "completed":
            _check_outputs(step["attempts"][-1])
    result["verified"] = True
    return result


def reconcile(root, step_id, number, *, outcome, returncode, evidence, actor, reason):
    """An operator may attach verified external evidence to an unknown outcome.

    The original attempt is never rerun or overwritten. Failure only permits a
    new attempt if the original frozen retry code/attempt limit already allows it.
    """
    root = Path(root).resolve(strict=True)
    if outcome not in {"completed", "failed"} or not actor.strip() or not reason.strip():
        raise GateError("STUDY_RECONCILIATION_REQUIRED")
    if type(returncode) is not int or (returncode == 0) != (outcome == "completed"):
        raise GateError("STUDY_RECONCILIATION_CODE")
    with file_lock(root / "runner.lock"):
        lock = verify(root)
        step = next((s for s in lock["spec"]["steps"] if s["id"] == step_id), None)
        if step is None or type(number) is not int or number < 1:
            raise GateError("STUDY_ATTEMPT_NOT_FOUND")
        attempts = _attempts(root, step)
        if not attempts or attempts[-1]["attempt"] != number or attempts[-1]["state"] != "uncertain":
            raise GateError("STUDY_ONLY_UNCERTAIN_MAY_BE_RECONCILED")
        folder = Path(attempts[-1]["directory"])
        with file_lock(folder / "worker.lock"):
            if (folder / "receipt.json").exists():
                raise GateError("STUDY_WORKER_ALREADY_FINISHED")
            proof = Path(evidence).resolve(strict=True)
            if not proof.is_file() or proof.is_symlink():
                raise GateError("STUDY_EVIDENCE_REQUIRED")
            payload = proof.read_bytes()
            destination = folder / "reconciliation-evidence.bin"
            with destination.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            outputs = {}
            if outcome == "completed":
                for relative in step["outputs"]:
                    path = regular_under(folder, relative)
                    outputs[relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            write_json(folder / "reconciliation.json", {
                "actor": actor, "reason": reason, "recorded_at": time.time(),
                "evidence": {"sha256": file_hash(destination), "bytes": len(payload)},
                "receipt": {"state": outcome, "returncode": returncode, "seconds": None,
                            "outputs": outputs, "error": "Externally reconciled; execution time unknown."}
            }, exclusive=True)
    return status(root)
