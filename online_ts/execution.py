from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

from .config import GateError, TaskConfig, canonical, digest
from .data import ArtifactStore, NoRedirect
from .models import environment_fingerprint
from .resources import local_execution_status


PROTOCOL_VERSION = 1
SERVICE_NAME = "industrial-online-ts-trainer"
MAX_BODY_BYTES = 64 * 1024 * 1024
REMOTE_ERROR_CODES = {
    "REMOTE_BUSY", "REMOTE_REQUEST_CONFLICT", "REMOTE_REQUEST_ID", "REMOTE_REQUEST_SCHEMA",
    "REMOTE_RESPONSE_MISSING", "REMOTE_RESPONSE_TOO_LARGE", "REMOTE_STAGE", "REMOTE_TIMEOUT",
    "REMOTE_TRAINING_DEADLINE", "REMOTE_TRAINING_EXECUTION_FAILED", "REMOTE_TRAINING_INTERRUPTED",
    "REQUEST_LENGTH", "REQUEST_SIGNATURE", "REQUEST_TIMEOUT", "REQUEST_TOO_LARGE", "REMOTE_RESOURCE_ENVELOPE",
    "TRAINING_SERVER_CODE_MISMATCH", "TRAINING_TARGET_NOT_ALLOWED",
}


def execution_identity(status: dict, target: str | None = None) -> dict:
    gpu = status.get("gpu", {}) if isinstance(status.get("gpu"), dict) else {}
    return {"target": target or status.get("target", "local"),
            "training_backend": status.get("training_backend", "numpy-cpu"),
            "gpu_accelerated": bool(status.get("gpu_accelerated", False)),
            "gpu_name": gpu.get("name") if gpu.get("detected") else None}


def _server_credentials(config: TaskConfig) -> tuple[str, str, str]:
    url = os.environ.get(config.execution.server_url_env, "")
    auth = os.environ.get(config.execution.server_auth_env, "")
    key = os.environ.get(config.execution.server_key_env, "")
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or not auth or len(key) < 24):
        raise GateError("TRAINING_SERVER_CONFIGURATION")
    return url.rstrip("/"), auth, key


def http_transport(request: urllib.request.Request, timeout: float) -> tuple[bytes, Mapping[str, str]]:
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(MAX_BODY_BYTES + 1)
            headers = dict(response.headers.items())
    except urllib.error.HTTPError as error:
        try:
            body = error.read(MAX_BODY_BYTES + 1)
            headers = dict(error.headers.items())
        finally:
            error.close()
    if len(body) > MAX_BODY_BYTES:
        raise GateError("TRAINING_SERVER_RESPONSE_TOO_LARGE")
    return body, headers


def sign(payload: bytes, key: str) -> str:
    return hmac.new(key.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _header(headers: Mapping[str, str], name: str) -> str:
    for key, value in headers.items():
        if key.casefold() == name.casefold():
            return value
    return ""


def _verified_response(payload: bytes, headers: Mapping[str, str], key: str) -> dict:
    signature = _header(headers, "X-Training-Signature")
    if not signature or not hmac.compare_digest(signature, sign(payload, key)):
        raise GateError("TRAINING_SERVER_SIGNATURE")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GateError("TRAINING_SERVER_RESPONSE_SCHEMA") from None
    if not isinstance(value, dict):
        raise GateError("TRAINING_SERVER_RESPONSE_SCHEMA")
    return value


def _raise_remote_error(value: dict) -> None:
    if set(value) != {"error"} or not isinstance(value["error"], str):
        return
    code = value["error"]
    raise GateError(code if code in REMOTE_ERROR_CODES else "TRAINING_SERVER_REJECTED")


def _bounded_health(value: dict) -> dict:
    required = {"protocol_version", "service", "ready", "code_digest", "active_jobs", "capacity"}
    if set(value) != required or value["protocol_version"] != PROTOCOL_VERSION or value["service"] != SERVICE_NAME:
        raise GateError("TRAINING_SERVER_HEALTH_SCHEMA")
    if not isinstance(value["ready"], bool) or not isinstance(value["active_jobs"], int) or isinstance(value["active_jobs"], bool):
        raise GateError("TRAINING_SERVER_HEALTH_SCHEMA")
    if not isinstance(value["code_digest"], str) or len(value["code_digest"]) != 64 or not isinstance(value["capacity"], dict):
        raise GateError("TRAINING_SERVER_HEALTH_SCHEMA")
    capacity = value["capacity"]
    safe = {name: capacity.get(name) for name in ("cpu_count", "memory_available_mb", "training_backend", "gpu_accelerated")}
    gpu = capacity.get("gpu", {})
    safe["gpu"] = {name: gpu.get(name) for name in ("detected", "name", "memory_total_mb", "memory_free_mb", "utilization_percent") if name in gpu}
    return {"target": "server", "available": value["ready"], "active_jobs": value["active_jobs"],
            "code_match": value["code_digest"] == environment_fingerprint()["code_digest"], **safe}


def probe_training_targets(config: TaskConfig, *, transport: Callable | None = None) -> list[dict]:
    status: list[dict] = []
    if "local" in config.execution.allowed_targets:
        status.append(local_execution_status(config.execution.expected_local_gpu))
    if "server" in config.execution.allowed_targets:
        try:
            url, auth, key = _server_credentials(config)
            client_digest = environment_fingerprint()["code_digest"]
            request = urllib.request.Request(
                url + "/health?" + urllib.parse.urlencode({"client_digest": client_digest}), method="GET",
                headers={"Authorization": "Bearer " + auth,
                         "Cache-Control": "no-cache", "Pragma": "no-cache"},
            )
            payload, headers = (transport or http_transport)(request, config.execution.health_timeout_seconds)
            remote = _bounded_health(_verified_response(payload, headers, key))
            if not remote["code_match"]:
                remote["available"] = False
                remote["code"] = "TRAINING_SERVER_CODE_MISMATCH"
            status.append(remote)
        except Exception as error:
            code = str(error) if isinstance(error, GateError) else "TRAINING_SERVER_UNAVAILABLE"
            status.append({"target": "server", "available": False, "code": code})
    return status


def _artifact_record(artifacts: ArtifactStore, reference: dict) -> dict:
    return {"reference": reference, "payload": base64.b64encode(artifacts.read(reference)).decode("ascii")}


def import_artifact_records(artifacts: ArtifactStore, records: list[dict]) -> None:
    if not isinstance(records, list) or len(records) > 128:
        raise GateError("TRAINING_ARTIFACT_SCHEMA")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"reference", "payload"} or not isinstance(record["reference"], dict):
            raise GateError("TRAINING_ARTIFACT_SCHEMA")
        reference = record["reference"]
        try:
            payload = base64.b64decode(record["payload"], validate=True)
        except (ValueError, TypeError):
            raise GateError("TRAINING_ARTIFACT_SCHEMA") from None
        created = artifacts.put(payload, reference.get("media_type", "application/json")).to_dict()
        if created != reference:
            raise GateError("TRAINING_ARTIFACT_MISMATCH")


def _input_references(snapshot: dict, reports: list[dict], champion_ref: dict | None) -> list[dict]:
    values = [snapshot]
    values.extend(report["validation_artifact"] for report in reports if report.get("validation_artifact"))
    if champion_ref is not None:
        values.append(champion_ref)
    unique: dict[str, dict] = {}
    for reference in values:
        unique[reference["sha256"]] = reference
    return list(unique.values())


def remote_execute_stage(config: TaskConfig, artifacts: ArtifactStore, run_id: str, stage: str, snapshot: dict,
                         *, candidate: dict, reports: list[dict], champion_ref: dict | None,
                         transport: Callable | None = None, stage_timeout_seconds: float | None = None) -> dict:
    if stage not in {"validate", "final"}:
        raise GateError("REMOTE_STAGE")
    url, auth, key = _server_credentials(config)
    code_digest = environment_fingerprint()["code_digest"]
    request_id = digest({"run_id": run_id, "stage": stage, "candidate": candidate})
    records = [_artifact_record(artifacts, reference) for reference in _input_references(snapshot, reports, champion_ref)]
    overall_timeout = min(config.execution.request_timeout_seconds, config.budget.max_candidate_seconds,
                          stage_timeout_seconds if stage_timeout_seconds is not None else float("inf"))
    if overall_timeout <= 0.1:
        raise GateError("TRAINING_DEADLINE")
    response_grace = min(2.0, max(0.1, overall_timeout / 10))
    remote_timeout = overall_timeout - response_grace
    value = {"protocol_version": PROTOCOL_VERSION, "request_id": request_id, "code_digest": code_digest,
             "stage": stage, "config": config.to_dict(), "snapshot": snapshot, "candidate": candidate,
             "reports": reports, "champion_artifact": champion_ref, "artifacts": records,
             "max_memory_mb": config.budget.max_memory_mb,
             "timeout_seconds": remote_timeout}
    payload = canonical(value).encode("utf-8")
    if len(payload) > MAX_BODY_BYTES:
        raise GateError("TRAINING_SERVER_REQUEST_TOO_LARGE")
    request = urllib.request.Request(url + "/train-stage", data=payload, method="POST", headers={
        "Authorization": "Bearer " + auth, "Content-Type": "application/json",
        "X-Training-Signature": sign(payload, key),
    })
    try:
        response_payload, headers = (transport or http_transport)(request, overall_timeout)
        response = _verified_response(response_payload, headers, key)
        _raise_remote_error(response)
    except GateError:
        raise
    except Exception:
        raise GateError("TRAINING_SERVER_UNAVAILABLE") from None
    required = {"protocol_version", "request_id", "code_digest", "result", "artifacts", "execution"}
    if (set(response) != required or response["protocol_version"] != PROTOCOL_VERSION
            or response["request_id"] != request_id or response["code_digest"] != code_digest
            or not isinstance(response["result"], dict) or not isinstance(response["execution"], dict)
            or response["execution"].get("target") != "server"):
        raise GateError("TRAINING_SERVER_RESPONSE_SCHEMA")
    import_artifact_records(artifacts, response["artifacts"])
    expected = [response["result"].get("validation_artifact")] if stage == "validate" else [response["result"].get("artifact"), response["result"].get("reload_probe")]
    if any(not isinstance(reference, dict) for reference in expected):
        raise GateError("TRAINING_SERVER_RESPONSE_SCHEMA")
    for reference in expected:
        artifacts.read(reference)
    return response["result"]
