from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
from dataclasses import asdict, dataclass

from ..config import GateError, canonical, digest, strict

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "agent_selection_study"


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class StudyConfig:
    model: str = "deepseek-flash"
    max_calls: int = 60
    max_candidates: int = 12
    max_llm_tokens: int = 500000
    max_call_tokens: int = 32768
    timeout_seconds: int = 120
    candidate_seconds: int = 600
    total_seconds: int = 10800
    confirmation_seconds: int = 86400
    search_seed: int = 2026
    confirmation_seeds: tuple = (2036, 2037, 2038)
    min_retention: float = 0.5
    min_group_retention: float = 0.2
    rare_group_size: int = 64
    max_rule_nodes: int = 7
    max_nmae_regression: float = 0.01
    substantive_nmse_gain: float = 0.01
    independent_runs: int = 5
    random_seed: int = 20261007

    def __post_init__(self):
        if self.model not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise GateError("DEEPSEEK_MODEL")
        for name in ("max_calls", "max_candidates", "max_llm_tokens", "max_call_tokens",
                     "timeout_seconds", "candidate_seconds", "total_seconds", "confirmation_seconds", "rare_group_size",
                     "max_rule_nodes", "independent_runs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise GateError("STUDY_BUDGET")
        if self.max_calls < 3 or self.max_candidates > 64 or self.timeout_seconds > 300 or self.max_rule_nodes > 15:
            raise GateError("STUDY_LIMIT")
        if self.candidate_seconds > 6 * self.timeout_seconds + 60:
            raise GateError("TRAINING_EXCEEDS_CONTROLLER_LEASE")
        for name in ("min_retention", "min_group_retention"):
            if type(getattr(self, name)) not in {int, float} or not 0 < getattr(self, name) <= 1:
                raise GateError("RETENTION_BOUND")
        for name in ("max_nmae_regression", "substantive_nmse_gain"):
            if type(getattr(self, name)) not in {int, float} or not 0 <= getattr(self, name) <= 1:
                raise GateError("METRIC_BOUND")
        seeds = tuple(self.confirmation_seeds)
        if (len(seeds) != 3 or len(set(seeds)) != 3 or self.search_seed in seeds
                or any(type(s) is not int or s < 0 for s in (*seeds, self.search_seed, self.random_seed))):
            raise GateError("STUDY_SEEDS")
        object.__setattr__(self, "confirmation_seeds", seeds)

    @classmethod
    def load(cls, path):
        return strict(cls, read_json(path))


def code_manifest():
    paths = list((ROOT / "online_ts/selection").glob("*.py")) + [
        ROOT / "online_ts/agent_runtime/controller.py", ROOT / "online_ts/agent_runtime/state.py",
        ROOT / "online_ts/agent_runtime/budget.py", ROOT / "online_ts/agent_runtime/llm.py",
        ROOT / "online_ts/llm_transport.py", ROOT / "online_ts/store.py",
        ROOT / "online_ts/config.py", ROOT / "online_ts/data.py", ROOT / "scripts/run_research.py",
        ROOT / "scripts/verify_dataset.py", STUDY / "prompts/selection.txt",
        STUDY / "protocol.md", STUDY / "requirements.lock"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def freeze(task_path, config_path, destination):
    import importlib.metadata
    task = read_json(task_path)
    config = StudyConfig.load(config_path)
    if (task["evaluation_scope"] != "development_only"
            or task["manifest_sha256"] != file_hash(ROOT / "data/metropt3_v1/manifest.json")
            or task["protocol_sha256"] != file_hash(ROOT / "research_runs/protocol.json")
            or task["protocol"] != read_json(ROOT / "research_runs/protocol.json")
            or task["diagnostic_code_sha256"] != file_hash(ROOT / "online_ts/selection/dataset.py")):
        raise GateError("TASK_PREPARATION_IDENTITY")
    from .dataset import TrainingData
    TrainingData(task)
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    value = {"version": "selection-harness-v1", "evaluation_scope": "development_only",
             "independent_confirmation": "requires_new_external_or_future_data",
             "task": task, "config": asdict(config), "code": code_manifest(),
             "python": platform.python_version(),
             "dependencies": {**{n: importlib.metadata.version(n) for n in ("numpy", "pandas")}, "torch": torch_version},
             "model_api": "https://api.deepseek.com/chat/completions",
             "model_version_limit": "alias; returned model and fingerprint recorded per call",
             "controller": "online_ts.agent_runtime.controller.AgentController",
             "decision_protocol": "validated JSON action, one deterministic tool per turn",
             "reasoning_effort": "low", "closing_reasoning_effort": "none",
             "random_search": "uniform single/and/or rules; frozen discrete quantile grid; see rules.py"}
    value["sha256"] = digest(value)
    destination = Path(destination)
    if destination.exists() and read_json(destination) != value:
        raise GateError("FROZEN_HARNESS_ALREADY_EXISTS")
    write_json(destination, value)
    return value


def verify_freeze(path):
    value = read_json(path)
    if digest({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]:
        raise GateError("HARNESS_MANIFEST_HASH")
    if value["code"] != code_manifest():
        raise GateError("HARNESS_CODE_CHANGED")
    import importlib.metadata
    for name in ("numpy", "pandas"):
        if importlib.metadata.version(name) != value["dependencies"][name]:
            raise GateError("HARNESS_DEPENDENCY_CHANGED")
    StudyConfig(**value["config"])
    if value["evaluation_scope"] != "development_only":
        raise GateError("INDEPENDENT_TASK_NOT_REGISTERED")
    return value
