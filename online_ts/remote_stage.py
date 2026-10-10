from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

from .config import GateError, TaskConfig, canonical
from .data import ArtifactStore
from .execution import MAX_BODY_BYTES, PROTOCOL_VERSION, execution_identity, import_artifact_records
from .models import Candidate, environment_fingerprint
from .resources import local_execution_status, memory_limit
from .training import final_fit, validate_candidate


def execute_request(value: dict, *, apply_limits: bool = False) -> dict:
    required = {"protocol_version", "request_id", "code_digest", "stage", "config", "snapshot", "candidate",
                "reports", "champion_artifact", "artifacts", "max_memory_mb", "timeout_seconds"}
    if not isinstance(value, dict) or set(value) != required or value["protocol_version"] != PROTOCOL_VERSION:
        raise GateError("REMOTE_REQUEST_SCHEMA")
    local_digest = environment_fingerprint()["code_digest"]
    if value["code_digest"] != local_digest:
        raise GateError("TRAINING_SERVER_CODE_MISMATCH")
    if (not isinstance(value["request_id"], str) or len(value["request_id"]) != 64
            or any(c not in "0123456789abcdef" for c in value["request_id"])):
        raise GateError("REMOTE_REQUEST_ID")
    config = TaskConfig.from_dict(value["config"])
    if "server" not in config.execution.allowed_targets:
        raise GateError("TRAINING_TARGET_NOT_ALLOWED")
    if (isinstance(value["max_memory_mb"], bool) or value["max_memory_mb"] != config.budget.max_memory_mb
            or isinstance(value["timeout_seconds"], bool) or not isinstance(value["timeout_seconds"], (int, float))
            or not math.isfinite(value["timeout_seconds"]) or not 0 < value["timeout_seconds"]
            or value["timeout_seconds"] > min(config.execution.request_timeout_seconds, config.budget.max_candidate_seconds)):
        raise GateError("REMOTE_RESOURCE_ENVELOPE")
    if value["stage"] not in {"validate", "final"} or not isinstance(value["reports"], list):
        raise GateError("REMOTE_REQUEST_SCHEMA")
    candidate = Candidate.from_dict(value["candidate"])
    candidate.validate(config)
    if apply_limits:
        memory_limit(value["max_memory_mb"])
    with tempfile.TemporaryDirectory(prefix="online-ts-remote-") as name:
        artifacts = ArtifactStore(name)
        import_artifact_records(artifacts, value["artifacts"])
        capacity = local_execution_status("")
        execution = {**execution_identity(capacity, "server"), "server_code_digest": local_digest}
        if value["stage"] == "validate":
            result = validate_candidate(config, value["snapshot"], candidate, artifacts, execution,
                                        champion_artifact=value["champion_artifact"])
            output_refs = [result["validation_artifact"]]
        elif value["stage"] == "final":
            result = final_fit(config, value["snapshot"], candidate, value["reports"], artifacts,
                               value["champion_artifact"], execution)
            output_refs = [result["artifact"], result["reload_probe"]]
        import base64
        output = [{"reference": reference, "payload": base64.b64encode(artifacts.read(reference)).decode("ascii")}
                  for reference in output_refs]
    response = {"protocol_version": PROTOCOL_VERSION, "request_id": value["request_id"], "code_digest": local_digest,
                "result": result, "artifacts": output, "execution": execution}
    if len(canonical(response).encode("utf-8")) > MAX_BODY_BYTES:
        raise GateError("REMOTE_RESPONSE_TOO_LARGE")
    return response


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        return 2
    request_path, response_path = map(Path, args)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        response = execute_request(request, apply_limits=True)
    except GateError as error:
        response = {"error": str(error)}
    except Exception:
        response = {"error": "REMOTE_TRAINING_EXECUTION_FAILED"}
    response_path.write_text(json.dumps(response, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return 0 if "error" not in response else 1


if __name__ == "__main__":
    raise SystemExit(main())
