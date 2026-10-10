from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import canonical
from .execution import MAX_BODY_BYTES, PROTOCOL_VERSION, SERVICE_NAME, sign
from .models import environment_fingerprint
from .resources import local_execution_status


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class TrainingServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address: tuple[str, int], token: str, state: Path, *, auth_token: str | None = None,
                 max_cached_jobs: int = 1024, cache_ttl_seconds: float = 7 * 86400, io_timeout_seconds: float = 15.0):
        if len(token) < 24:
            raise ValueError("TRAINING_SERVER_API_KEY must contain at least 24 characters")
        if max_cached_jobs < 1 or cache_ttl_seconds <= 0 or io_timeout_seconds <= 0:
            raise ValueError("invalid remote server limits")
        super().__init__(address, TrainingHandler)
        self.token = token
        self.auth_token = token if auth_token is None else auth_token
        if not self.auth_token:
            raise ValueError("TRAINING_SERVER_AUTH_TOKEN is required")
        self.state = state.resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.max_cached_jobs = max_cached_jobs
        self.cache_ttl_seconds = cache_ttl_seconds
        self.io_timeout_seconds = io_timeout_seconds
        self.execution_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.active_jobs = 0


class TrainingHandler(BaseHTTPRequestHandler):
    server: TrainingServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.io_timeout_seconds)

    def log_message(self, format: str, *args) -> None:
        return

    def _authorized(self) -> bool:
        value = self.headers.get("Authorization", "")
        return hmac.compare_digest(value, "Bearer " + self.server.auth_token)

    def _send(self, status: int, value: dict) -> None:
        payload = canonical(value).encode("utf-8")
        if len(payload) > MAX_BODY_BYTES:
            status = 422
            payload = canonical({"error": "REMOTE_RESPONSE_TOO_LARGE"}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Training-Signature", sign(payload, self.server.token))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        client_digest = query.get("client_digest", [])
        valid_cache_key = not query or (
            set(query) == {"client_digest"} and len(client_digest) == 1
            and len(client_digest[0]) == 64
            and not any(char not in "0123456789abcdef" for char in client_digest[0]))
        if parsed.path != "/health" or not valid_cache_key:
            self._send(404, {"error": "NOT_FOUND"})
            return
        if not self._authorized():
            self._send(401, {"error": "UNAUTHORIZED"})
            return
        with self.server.state_lock:
            active_jobs = self.server.active_jobs
        value = {"protocol_version": PROTOCOL_VERSION, "service": SERVICE_NAME,
                 "ready": not self.server.execution_lock.locked(),
                 "code_digest": environment_fingerprint()["code_digest"], "active_jobs": active_jobs,
                 "capacity": local_execution_status("")}
        self._send(200, value)

    def do_POST(self) -> None:
        if self.path != "/train-stage":
            self._send(404, {"error": "NOT_FOUND"})
            return
        if not self._authorized():
            self._send(401, {"error": "UNAUTHORIZED"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(400, {"error": "REQUEST_LENGTH"})
            return
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(413, {"error": "REQUEST_TOO_LARGE"})
            return
        try:
            payload = self.rfile.read(length)
        except (OSError, socket.timeout):
            self._send(408, {"error": "REQUEST_TIMEOUT"})
            return
        if len(payload) != length:
            self._send(400, {"error": "REQUEST_LENGTH"})
            return
        signature = self.headers.get("X-Training-Signature", "")
        if not signature or not hmac.compare_digest(signature, sign(payload, self.server.token)):
            self._send(401, {"error": "REQUEST_SIGNATURE"})
            return
        try:
            value = json.loads(payload)
            request_id = value["request_id"]
            if (not isinstance(request_id, str) or len(request_id) != 64
                    or any(c not in "0123456789abcdef" for c in request_id)):
                raise ValueError
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self._send(400, {"error": "REMOTE_REQUEST_SCHEMA"})
            return
        try:
            response = self._execute(request_id, payload, float(value.get("timeout_seconds", 0)))
        except Exception:
            self._send(500, {"error": "REMOTE_TRAINING_EXECUTION_FAILED"})
            return
        if set(response) == {"error"}:
            self._send(422, response)
        else:
            self._send(200, response)

    def _execute(self, request_id: str, payload: bytes, timeout: float) -> dict:
        if not 0 < timeout <= 600:
            return {"error": "REMOTE_TIMEOUT"}
        jobs_root = (self.server.state / "jobs").resolve()
        root = (jobs_root / request_id).resolve()
        if not root.is_relative_to(jobs_root):
            return {"error": "REMOTE_REQUEST_ID"}
        request_path = root / "request.json"
        response_path = root / "response.json"
        digest_path = root / "request.sha256"
        payload_digest = hashlib.sha256(payload).hexdigest()

        def cached() -> dict | None:
            if digest_path.exists() and digest_path.read_text(encoding="ascii") != payload_digest:
                return {"error": "REMOTE_REQUEST_CONFLICT"}
            if response_path.exists():
                if response_path.stat().st_size > MAX_BODY_BYTES:
                    return {"error": "REMOTE_RESPONSE_TOO_LARGE"}
                return json.loads(response_path.read_text(encoding="utf-8"))
            return None

        existing = cached()
        if existing is not None:
            return existing
        if not self.server.execution_lock.acquire(blocking=False):
            return {"error": "REMOTE_BUSY"}
        try:
            existing = cached()
            if existing is not None:
                return existing
            self._prune_cache(jobs_root, keep=request_id)
            root.mkdir(parents=True, exist_ok=True)
            _atomic_write(request_path, payload)
            _atomic_write(digest_path, payload_digest.encode("ascii"))
            temporary_response = root / ".response.json"
            env = os.environ.copy()
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[name] = "1"
            with self.server.state_lock:
                self.server.active_jobs += 1
            try:
                completed = subprocess.run(
                    [sys.executable, "-B", "-m", "online_ts.remote_stage", str(request_path), str(temporary_response)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env=env, timeout=timeout, check=False,
                )
                if not temporary_response.exists():
                    return {"error": "REMOTE_TRAINING_INTERRUPTED" if completed.returncode else "REMOTE_RESPONSE_MISSING"}
                if temporary_response.stat().st_size > MAX_BODY_BYTES:
                    temporary_response.unlink(missing_ok=True)
                    return {"error": "REMOTE_RESPONSE_TOO_LARGE"}
                response = json.loads(temporary_response.read_text(encoding="utf-8"))
                _atomic_write(response_path, canonical(response).encode("utf-8"))
                temporary_response.unlink(missing_ok=True)
                return response
            except subprocess.TimeoutExpired:
                return {"error": "REMOTE_TRAINING_DEADLINE"}
            finally:
                with self.server.state_lock:
                    self.server.active_jobs -= 1
        finally:
            self.server.execution_lock.release()

    def _prune_cache(self, jobs_root: Path, *, keep: str) -> None:
        jobs_root.mkdir(parents=True, exist_ok=True)
        now = time.time()
        entries = []
        for path in jobs_root.iterdir():
            if (not path.is_dir() or path.name == keep or len(path.name) != 64
                    or any(c not in "0123456789abcdef" for c in path.name)):
                continue
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            entries.append((modified, path))
        expired = {path for modified, path in entries if now - modified > self.server.cache_ttl_seconds}
        retained = sorted(((modified, path) for modified, path in entries if path not in expired), reverse=True)
        excess = {path for _, path in retained[max(0, self.server.max_cached_jobs - 1):]}
        for path in expired | excess:
            resolved = path.resolve()
            if resolved.is_relative_to(jobs_root):
                shutil.rmtree(resolved, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authenticated remote training executor")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--state", type=Path, default=Path("runtime/remote-server"))
    parser.add_argument("--token-env", default="TRAINING_SERVER_API_KEY")
    parser.add_argument("--auth-token-env", default="TRAINING_SERVER_AUTH_TOKEN")
    parser.add_argument("--max-cached-jobs", type=int, default=1024)
    parser.add_argument("--cache-ttl-seconds", type=float, default=7 * 86400)
    parser.add_argument("--io-timeout-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    auth_token = os.environ.get(args.auth_token_env, "")
    if len(token) < 24:
        raise SystemExit("TRAINING_SERVER_API_KEY must contain at least 24 characters")
    if not auth_token:
        raise SystemExit("TRAINING_SERVER_AUTH_TOKEN is required")
    server = TrainingServer((args.host, args.port), token, args.state, auth_token=auth_token, max_cached_jobs=args.max_cached_jobs,
                            cache_ttl_seconds=args.cache_ttl_seconds, io_timeout_seconds=args.io_timeout_seconds)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
