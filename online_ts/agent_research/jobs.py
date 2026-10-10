"""Stage CPU jobs for native execution on the dedicated research server.

Only trusted platform adapters supply training assets. Generated agents cannot
choose host paths or budgets. This component stages files only and does not
provide a security sandbox or execute, train, evaluate or install packages.
"""
import ast
import hashlib
from pathlib import Path
import re

from ..config import GateError, digest
from ..selection.common import file_hash, write_json


def prepare_job(folder, code, training_assets):
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > 128_000:
        raise GateError("DIAGNOSTIC_CODE_SIZE")
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise GateError("DIAGNOSTIC_CODE_SYNTAX") from error
    if not isinstance(training_assets, list) or not training_assets or len(training_assets) > 32:
        raise GateError("DIAGNOSTIC_TRAINING_ASSETS")
    # Validate all descriptors before any staging. This is not the source of
    # permission: adapters must bind descriptors to a trusted frozen T manifest.
    approved = []
    names = set()
    for asset in training_assets:
        if not isinstance(asset, dict) or set(asset) != {"path", "name", "role", "sha256"} or asset["role"] != "training":
            raise GateError("DIAGNOSTIC_TRAINING_ONLY")
        name = asset["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name) or name in names:
            raise GateError("DIAGNOSTIC_ASSET_NAME")
        names.add(name)
        source = Path(asset["path"])
        if source.is_symlink() or not source.is_file() or file_hash(source) != asset["sha256"]:
            raise GateError("DIAGNOSTIC_ASSET_IDENTITY")
        approved.append((asset, source))
    folder = Path(folder).resolve()
    if folder.exists():
        raise GateError("DIAGNOSTIC_JOB_ALREADY_EXISTS")
    folder.mkdir(parents=True)
    (folder / "code").mkdir()
    (folder / "inputs").mkdir()
    (folder / "outputs").mkdir()
    (folder / "code/agent.py").write_text(code, encoding="utf-8")
    manifests = []
    for asset, source in approved:
        destination = folder / "inputs" / asset["name"]
        # Hash bytes during copying to reject a concurrent source modification.
        hasher = hashlib.sha256()
        with source.open("rb") as reader, destination.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                hasher.update(block)
                writer.write(block)
        if hasher.hexdigest() != asset["sha256"]:
            raise GateError("DIAGNOSTIC_ASSET_CHANGED_DURING_COPY")
        manifests.append({"name": asset["name"], "role": "training", "sha256": asset["sha256"]})
    manifest = {"kind": "diagnostic_job", "code_sha256": file_hash(folder / "code/agent.py"),
                "inputs": manifests, "outputs": "untrusted", "evaluation_assets_staged": False,
                "security_sandbox": False, "execution": "prepared_only"}
    manifest["sha256"] = digest(manifest)
    write_json(folder / "job.json", manifest)
    return manifest


