"""A detached command worker owns durable attempt receipts after CLI disconnect."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import file_hash, file_lock, read_json, regular_under, write_json
from .studies import verify


def work(root, step_id, folder):
    root, folder = Path(root).resolve(strict=True), Path(folder).resolve(strict=True)
    if folder.parent != root / "steps" / step_id:
        raise ValueError("Attempt scope")
    with file_lock(folder / "worker.lock"):
        if any((folder / name).exists() for name in ("started.json", "receipt.json", "reconciliation.json")):
            raise ValueError("Attempt already started")
        lock = verify(root)
        step = next(s for s in lock["spec"]["steps"] if s["id"] == step_id)
        substitutions = {"python": sys.executable, "workspace": lock["spec"]["workspace"],
                         "attempt": str(folder), "context": str(folder / "context.json")}
        command = step["command"]
        for token, value in substitutions.items():
            command = [part.replace("{" + token + "}", value) for part in command]
        start = time.perf_counter()
        write_json(folder / "started.json", {"pid": os.getpid(), "started_at": time.time(),
                    "command": command, "study_sha256": lock["sha256"]}, exclusive=True)
        receipt = {"state": "failed", "returncode": None, "outputs": {},
                   "started_at": time.time(), "error": None}
        env = os.environ.copy()
        env.update(PLATFORM_CONTEXT=str(folder / "context.json"), PLATFORM_OUTPUT_DIR=str(folder),
                   PYTHONPATH=str(Path(__file__).resolve().parents[2]), PYTHONIOENCODING="utf-8")
        process = None
        try:
            with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
                process = subprocess.Popen(command, cwd=folder, env=env, shell=False,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    receipt["returncode"] = process.wait(timeout=step["timeout_seconds"])
                    receipt["state"] = "completed" if receipt["returncode"] == 0 else "failed"
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        # Native taskkill is for process-tree termination only.
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    receipt.update(state="timed_out", returncode=process.returncode)
            if receipt["state"] == "completed":
                try:
                    for relative in step["outputs"]:
                        path = regular_under(folder, relative)
                        receipt["outputs"][relative] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                except (OSError, ValueError) as exc:
                    receipt.update(state="invalid_output", error=str(exc))
        except Exception as exc:
            receipt.update(state="failed", error=type(exc).__name__ + ": " + str(exc))
        finally:
            receipt.update(seconds=time.perf_counter() - start, finished_at=time.time())
            write_json(folder / "receipt.json", receipt, exclusive=True)


if __name__ == "__main__":
    work(*sys.argv[1:])
