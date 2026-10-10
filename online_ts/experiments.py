"""Compile bounded proposals. No expressions, paths, code, or policy writes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .config import GateError, TaskConfig, digest, strict


FEATURE_OPERATIONS = ("lag", "rolling_mean", "rolling_std", "difference")
PROTOCOL = "rolling-origin-v1"


@dataclass(frozen=True)
class FeatureSpec:
    operation: str
    source_field: str
    window: int
    alignment: str = "context_end"
    availability: str = "observed_before_issue"
    missing_strategy: str = "task_causal_fill_then_reject"
    entity: str = "task_entity"
    version: int = 1
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if (self.operation not in FEATURE_OPERATIONS or type(self.window) is not int
                or self.window < 1 or self.alignment != "context_end"
                or self.availability != "observed_before_issue"
                or self.missing_strategy != "task_causal_fill_then_reject"
                or self.entity != "task_entity" or self.version != 1 or self.dependencies):
            raise GateError("FEATURE_SPEC_NOT_ALLOWED")
        if not isinstance(self.source_field, str) or not self.source_field.startswith("target:"):
            raise GateError("FEATURE_SOURCE")
        suffix = self.source_field[7:]
        if not suffix.isdigit() or str(int(suffix)) != suffix:
            raise GateError("FEATURE_SOURCE")

    def validate(self, config: TaskConfig, length: int):
        if self.operation not in config.search.feature_operations:
            raise GateError("FEATURE_OPERATION_NOT_AUTHORIZED")
        if int(self.source_field[7:]) >= len(config.variables):
            raise GateError("FEATURE_SOURCE")
        needed = self.window + (1 if self.operation == "difference" else 0)
        if needed > length:
            raise GateError("FEATURE_CONTEXT_BOUNDARY")


def feature_matrix(windows: np.ndarray, specifications) -> np.ndarray:
    """The same causal transform runs in fitting, validation, reload and serving."""
    base = windows.reshape(len(windows), -1)
    columns = []
    for value in specifications:
        spec = strict(FeatureSpec, value)
        series = windows[:, :, int(spec.source_field[7:])]
        if spec.operation == "lag":
            column = series[:, -spec.window]
        elif spec.operation == "rolling_mean":
            column = series[:, -spec.window:].mean(axis=1)
        elif spec.operation == "rolling_std":
            column = series[:, -spec.window:].std(axis=1)
        else:
            column = series[:, -1] - series[:, -1 - spec.window]
        columns.append(column[:, None])
    return np.concatenate([base, *columns], axis=1) if columns else base


@dataclass(frozen=True)
class ExperimentSpec:
    objective: str
    hypothesis: str
    candidate: dict
    contract_ref: str
    evaluation_protocol: str
    resources: dict
    evidence_refs: tuple[str, ...]
    parent_candidate_id: str | None = None
    feedback_id: str | None = None
    historical_covariates: tuple[str, ...] = ()
    known_future_covariates: tuple[str, ...] = ()
    expected_direction: str = "decrease_validation_error"
    risks: tuple[str, ...] = ()
    stopping_condition: str = "workflow_budget_and_no_progress"
    schema_version: int = 1

    @classmethod
    def from_dict(cls, value):
        spec = strict(cls, value)
        for name in ("objective", "hypothesis", "expected_direction"):
            text = getattr(spec, name)
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise GateError("EXPERIMENT_DESCRIPTION")
        if (spec.schema_version != 1 or spec.evaluation_protocol != PROTOCOL
                or spec.stopping_condition != "workflow_budget_and_no_progress"):
            raise GateError("EXPERIMENT_PROTOCOL")
        from .remediation import _refs
        for name in ("evidence_refs", "risks", "historical_covariates", "known_future_covariates"):
            object.__setattr__(spec, name, _refs(getattr(spec, name), "EXPERIMENT_REFERENCES"))
        return spec


def capabilities(config: TaskConfig) -> dict:
    from .config import CHRONOS_MODEL_REVISIONS
    return {"contract_ref": config.version, "evaluation_protocol": PROTOCOL,
            "model_families": list(config.search.model_families),
            "inputs": list(config.search.inputs), "histories": list(config.search.histories),
            "ridge_alphas": list(config.search.ridge_alphas),
            "parameter_bounds": config.search.parameter_bounds,
            "feature_operations": list(config.search.feature_operations),
            "feature_sources": [f"target:{i}" for i in range(len(config.variables))],
            "feature_schema": asdict(FeatureSpec("rolling_std", "target:0", 2)),
            "feature_models": ["ridge"], "covariates": [],
            "locked_models": {m: CHRONOS_MODEL_REVISIONS[m] for m in config.foundation.model_ids},
            "adaptation_modes": list(config.foundation.adaptation_modes),
            "foundation_parameters": {n: list(getattr(config.foundation, n)) for n in
                                      ("batch_sizes", "learning_rates", "num_steps", "lora_ranks", "quantile_levels")},
            "resource_limits": {"seconds": config.budget.max_candidate_seconds,
                                "memory_mb": config.budget.max_memory_mb},
            "training_targets": list(config.execution.allowed_targets)}


def compile_experiment(value: dict, config: TaskConfig, *, evidence_refs,
                       available_targets: set[str], parent_ids=None, feedback_id=None):
    from .models import Candidate
    spec = ExperimentSpec.from_dict(value)
    if spec.contract_ref != config.version:
        raise GateError("EXPERIMENT_CONTRACT_VERSION")
    if not spec.evidence_refs or not set(spec.evidence_refs).issubset(evidence_refs):
        raise GateError("EXPERIMENT_EVIDENCE")
    if spec.feedback_id != feedback_id:
        raise GateError("EXPERIMENT_FEEDBACK_VERSION")
    if spec.parent_candidate_id is not None and spec.parent_candidate_id not in (parent_ids or ()):
        raise GateError("EXPERIMENT_PARENT")
    if spec.historical_covariates or spec.known_future_covariates:
        raise GateError("MODEL_COVARIATES_UNSUPPORTED")
    candidate = Candidate.from_dict(spec.candidate)
    candidate.validate(config)
    resource = spec.resources
    if not isinstance(resource, dict) or set(resource) != {"training_target", "seconds", "memory_mb"}:
        raise GateError("EXPERIMENT_RESOURCE_SCHEMA")
    target = resource["training_target"]
    if target not in available_targets or target not in config.execution.allowed_targets:
        raise GateError("TRAINING_TARGET_NOT_ALLOWED")
    for name, maximum in (("seconds", config.budget.max_candidate_seconds),
                          ("memory_mb", config.budget.max_memory_mb)):
        number = resource[name]
        if type(number) not in {int, float} or not math.isfinite(number) or not 0 < number <= maximum:
            raise GateError("EXPERIMENT_RESOURCE_BUDGET")
    # These are requests/estimates; the Worker reserves and enforces task limits.
    return candidate, target, {**asdict(spec), "effective_candidate": asdict(candidate),
                               "spec_id": "experiment-" + digest(asdict(spec))[:24]}
