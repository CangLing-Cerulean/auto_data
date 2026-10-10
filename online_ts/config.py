from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping


SECOND = 1_000_000_000
CHRONOS_MODEL_REVISIONS = {
    "amazon/chronos-2": "29ec3766d36d6f73f0696f85560a422f50e8498c",
    "autogluon/chronos-2-small": "ddec01313e50b6bc58ebaa92ede81bc24a3d9f9a",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class GateError(ValueError):
    """A stable, non-sensitive code suitable for events and user output."""


def strict(cls, value: Mapping[str, Any]):
    if not isinstance(value, Mapping) or set(value) - {f.name for f in fields(cls)}:
        raise GateError(f"INVALID_{cls.__name__.upper()}_FIELDS")
    try:
        return cls(**value)
    except (TypeError, OverflowError) as error:
        raise GateError(f"INVALID_{cls.__name__.upper()}") from error


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise GateError(f"INVALID_{name}")
    if value < 0 or (value == 0 and not zero):
        raise GateError(f"INVALID_{name}")


def count(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateError(f"INVALID_{name}")


@dataclass(frozen=True)
class SourceConfig:
    kind: str = "file"
    path: str = ""
    timestamp_column: str = "timestamp"
    available_at_column: str | None = None
    label_available_at_column: str | None = None
    revision_at_column: str | None = None
    regime_column: str | None = None
    label_maturity_verified: bool = False
    url_env: str = "INFLUXDB_URL"
    token_env: str = "INFLUXDB_TOKEN"
    org: str = "my-org"
    bucket: str = "sensor_data"
    measurement: str = "factory_process"
    tags: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    retries: int = 3

    def __post_init__(self):
        if self.kind not in {"file", "influxdb", "history_http"}:
            raise GateError("SOURCE_KIND")
        if self.kind == "file" and not self.path:
            raise GateError("SOURCE_PATH_REQUIRED")
        if self.regime_column is not None and (self.kind != "file" or not isinstance(self.regime_column, str)
                                               or not self.regime_column.strip()):
            raise GateError("REGIME_SOURCE_CONTRACT")
        for name in (self.url_env, self.token_env):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise GateError("CREDENTIAL_ENV_NAME")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in self.tags.items()):
            raise GateError("SOURCE_TAGS")
        if not isinstance(self.label_maturity_verified, bool):
            raise GateError("LABEL_MATURITY_VERIFICATION")
        positive(self.timeout_seconds, "SOURCE_TIMEOUT")
        count(self.retries, "SOURCE_RETRIES")
        if self.retries > 5:
            raise GateError("SOURCE_RETRIES")


@dataclass(frozen=True)
class SearchConfig:
    input_steps: int | None = None
    history_steps: int | None = None
    input_candidates: tuple[int, ...] = (8, 16, 32)
    history_candidates: tuple[int, ...] = (128, 256)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    seasonal_period: int = 8
    folds: int = 3
    validation_steps: int = 60
    test_steps: int = 120
    max_train_samples: int = 512
    model_families: tuple[str, ...] = ("last_value", "seasonal_naive", "ridge")
    parameter_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    feature_operations: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "model_families", tuple(self.model_families))
        object.__setattr__(self, "feature_operations", tuple(self.feature_operations))
        if set(self.feature_operations) - {"lag", "rolling_mean", "rolling_std", "difference"}:
            raise GateError("FEATURE_OPERATION_ALLOWLIST")
        if not isinstance(self.parameter_bounds, dict) or set(self.parameter_bounds) - {"input_steps", "history_steps", "alpha"}:
            raise GateError("PARAMETER_BOUNDS")
        for name, bounds in self.parameter_bounds.items():
            if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                    or any(type(v) not in {int, float} or not math.isfinite(v) for v in bounds)
                    or not 0 < bounds[0] <= bounds[1]):
                raise GateError("PARAMETER_BOUNDS")
            if name != "alpha" and (any(type(v) is not int for v in bounds) or bounds[0] < 2):
                raise GateError("PARAMETER_BOUNDS")
            if name == "alpha" and not 1e-6 <= bounds[0] <= bounds[1] <= 1e4:
                raise GateError("PARAMETER_BOUNDS")
        if (not self.model_families or len(set(self.model_families)) != len(self.model_families)
                or set(self.model_families) - {"last_value", "seasonal_naive", "ridge", "chronos2", "patchtst"}):
            raise GateError("MODEL_FAMILY_ALLOWLIST")
        for name in ("input_candidates", "history_candidates", "ridge_alphas"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if not getattr(self, name) or len(set(getattr(self, name))) != len(getattr(self, name)):
                raise GateError("SEARCH_GRID")
        for value in (*self.input_candidates, *self.history_candidates):
            count(value, "SEARCH_WINDOW", 2)
        for value in (self.input_steps, self.history_steps):
            if value is not None:
                count(value, "FIXED_WINDOW", 2)
        for value in self.ridge_alphas:
            positive(value, "RIDGE_ALPHA")
            if not 1e-6 <= value <= 1e4:
                raise GateError("RIDGE_ALPHA")
        for name in ("seasonal_period", "validation_steps", "test_steps", "max_train_samples"):
            count(getattr(self, name), name.upper())
        count(self.folds, "FOLDS", 3)

    @property
    def inputs(self) -> tuple[int, ...]:
        return (self.input_steps,) if self.input_steps is not None else self.input_candidates

    @property
    def histories(self) -> tuple[int, ...]:
        return (self.history_steps,) if self.history_steps is not None else self.history_candidates

    @property
    def required_history(self) -> int:
        return self.maximum_history + self.folds * self.validation_steps + self.test_steps

    @property
    def maximum_history(self) -> int:
        return max(max(self.histories), self.parameter_bounds.get("history_steps", (0, 0))[1])


@dataclass(frozen=True)
class Budget:
    max_candidates: int = 32
    max_training_seconds: float = 120.0
    max_candidate_seconds: float = 20.0
    max_attempts: int = 2
    max_repairs: int = 2
    max_memory_mb: int = 512
    max_llm_tokens: int = 8192
    max_no_improvement: int = 3
    minimum_candidate_improvement: float = 0.01

    def __post_init__(self):
        for name in ("max_candidates", "max_attempts", "max_memory_mb", "max_llm_tokens"):
            count(getattr(self, name), name.upper())
        count(self.max_repairs, "MAX_REPAIRS", 0)
        count(self.max_no_improvement, "MAX_NO_IMPROVEMENT")
        if not 0 < self.minimum_candidate_improvement < 1:
            raise GateError("CANDIDATE_PROGRESS_THRESHOLD")
        positive(self.max_training_seconds, "TRAINING_SECONDS")
        positive(self.max_candidate_seconds, "CANDIDATE_SECONDS")


@dataclass(frozen=True)
class MonitoringConfig:
    evaluation_seconds: float | None = None
    check_seconds: float = 60.0
    minimum_label_coverage: float = 0.90
    mse_ratio_trigger: float = 1.5
    mae_ratio_trigger: float = 1.25
    consecutive_checks: int = 3
    reference_windows: int = 3
    reference_max_relative_spread: float = 0.20
    max_model_age_seconds: float = 604800.0
    cooldown_seconds: float = 21600.0
    minimum_new_steps: int = 8

    def __post_init__(self):
        for name in ("check_seconds", "mse_ratio_trigger", "mae_ratio_trigger", "max_model_age_seconds"):
            positive(getattr(self, name), name.upper())
        positive(self.cooldown_seconds, "COOLDOWN", zero=True)
        if self.evaluation_seconds is not None:
            positive(self.evaluation_seconds, "EVALUATION_WINDOW")
        if not 0 < self.minimum_label_coverage <= 1 or min(self.mse_ratio_trigger, self.mae_ratio_trigger) <= 1:
            raise GateError("MONITORING_THRESHOLD")
        count(self.consecutive_checks, "CONSECUTIVE_CHECKS")
        count(self.reference_windows, "REFERENCE_WINDOWS", 3)
        positive(self.reference_max_relative_spread, "REFERENCE_SPREAD")
        count(self.minimum_new_steps, "MINIMUM_NEW_STEPS")


@dataclass(frozen=True)
class ReleaseConfig:
    shadow_seconds: float = 86400.0
    canary_seconds: float = 3600.0
    maximum_observation_seconds: float = 259200.0
    minimum_origins: int = 8
    minimum_improvement: float = 0.05
    maximum_critical_regression: float = 0.0
    maximum_inference_ms: float = 1000.0
    canary_fraction: float = 0.10
    rollback_mse_ratio: float = 2.0
    post_promotion_seconds: float = 86400.0
    critical_variables: tuple[str, ...] = ()
    business_groups: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "critical_variables", tuple(self.critical_variables))
        if (not isinstance(self.business_groups, dict) or any(not isinstance(name, str) or not name
                or not isinstance(variables, list) or not variables or len(set(variables)) != len(variables)
                for name, variables in self.business_groups.items())):
            raise GateError("BUSINESS_GROUP_DEFINITION")
        for name in ("shadow_seconds", "canary_seconds", "maximum_observation_seconds", "maximum_inference_ms", "post_promotion_seconds"):
            positive(getattr(self, name), name.upper())
        count(self.minimum_origins, "MINIMUM_ORIGINS")
        if not 0 <= self.minimum_improvement < 1 or not 0 <= self.maximum_critical_regression < 1:
            raise GateError("RELEASE_THRESHOLD")
        if not 0 < self.canary_fraction < 1 or self.rollback_mse_ratio <= 1:
            raise GateError("RELEASE_THRESHOLD")
        if self.maximum_observation_seconds < max(self.shadow_seconds, self.canary_seconds):
            raise GateError("OBSERVATION_DURATION")


@dataclass(frozen=True)
class PlannerConfig:
    kind: str = "rule"
    allow_external: bool = False
    endpoint_env: str = "DEEPSEEK_BASE_URL"
    key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-flash"
    timeout_seconds: float = 900.0
    short_output_tokens: int = 512
    spec_output_tokens: int = 1400
    feedback_output_tokens: int = 1600
    max_format_repairs: int = 1
    candidate_pool_size: int = 12
    initial_candidates: int = 2
    history_limit: int = 3
    knowledge_limit: int = 2
    example_limit: int = 1

    def __post_init__(self):
        if self.kind not in {"rule", "external"}:
            raise GateError("PLANNER_KIND")
        if self.kind == "external" and (self.allow_external is not True or not self.model):
            raise GateError("EXTERNAL_PLANNER_NOT_AUTHORIZED")
        for name in (self.endpoint_env, self.key_env):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise GateError("CREDENTIAL_ENV_NAME")
        positive(self.timeout_seconds, "PLANNER_TIMEOUT")
        for name in ("short_output_tokens", "spec_output_tokens", "feedback_output_tokens",
                     "candidate_pool_size", "initial_candidates"):
            count(getattr(self, name), name.upper())
        for name in ("max_format_repairs", "history_limit", "knowledge_limit", "example_limit"):
            count(getattr(self, name), name.upper(), 0)
        if (self.max_format_repairs > 1 or self.history_limit > 8 or self.knowledge_limit > 6
                or self.example_limit > 2 or self.candidate_pool_size > 64
                or self.initial_candidates > self.candidate_pool_size):
            raise GateError("PLANNER_CONTEXT_LIMIT")


@dataclass(frozen=True)
class ExecutionConfig:
    allowed_targets: tuple[str, ...] = ("local",)
    default_target: str = "local"
    server_url_env: str = "TRAINING_SERVER_URL"
    server_auth_env: str = "TRAINING_SERVER_AUTH_TOKEN"
    server_key_env: str = "TRAINING_SERVER_API_KEY"
    health_timeout_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    expected_local_gpu: str = "RTX 5060"

    def __post_init__(self):
        object.__setattr__(self, "allowed_targets", tuple(self.allowed_targets))
        if (not self.allowed_targets or len(set(self.allowed_targets)) != len(self.allowed_targets)
                or set(self.allowed_targets) - {"local", "server"}):
            raise GateError("TRAINING_TARGETS")
        if self.default_target not in self.allowed_targets:
            raise GateError("DEFAULT_TRAINING_TARGET")
        for name in (self.server_url_env, self.server_auth_env, self.server_key_env):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise GateError("CREDENTIAL_ENV_NAME")
        for name in ("health_timeout_seconds", "request_timeout_seconds"):
            positive(getattr(self, name), name.upper())
        if not isinstance(self.expected_local_gpu, str) or len(self.expected_local_gpu) > 120:
            raise GateError("EXPECTED_LOCAL_GPU")


@dataclass(frozen=True)
class FoundationConfig:
    """Finite, operator-approved Chronos candidate and local-weight policy."""

    enabled: bool = False
    model_roots: dict[str, str] = field(default_factory=dict)
    model_ids: tuple[str, ...] = ()
    adaptation_modes: tuple[str, ...] = ("zero_shot",)
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_sizes: tuple[int, ...] = (1,)
    learning_rates: tuple[float, ...] = (1e-5,)
    num_steps: tuple[int, ...] = (2,)
    lora_ranks: tuple[int, ...] = (4,)
    max_lora_candidates: int = 2
    device: str = "cuda"
    adapter_cache_size: int = 2

    def __post_init__(self):
        for name in ("model_ids", "adaptation_modes", "quantile_levels", "batch_sizes",
                     "learning_rates", "num_steps", "lora_ranks"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.enabled, bool):
            raise GateError("FOUNDATION_ENABLED")
        if self.device not in {"cpu", "cuda"}:
            raise GateError("CHRONOS_DEVICE")
        if not isinstance(self.model_roots, dict) or any(
                model_id not in CHRONOS_MODEL_REVISIONS or not isinstance(path, str) or not path
                for model_id, path in self.model_roots.items()):
            raise GateError("CHRONOS_MODEL_ROOTS")
        if len(set(self.model_ids)) != len(self.model_ids) or set(self.model_ids) - set(CHRONOS_MODEL_REVISIONS):
            raise GateError("CHRONOS_MODEL_ALLOWLIST")
        if self.enabled and (not self.model_ids or set(self.model_ids) - set(self.model_roots)):
            raise GateError("CHRONOS_MODEL_ROOT_REQUIRED")
        if (not self.adaptation_modes or len(set(self.adaptation_modes)) != len(self.adaptation_modes)
                or set(self.adaptation_modes) - {"zero_shot", "lora"}):
            raise GateError("CHRONOS_ADAPTATION_ALLOWLIST")
        levels = tuple(float(level) for level in self.quantile_levels)
        object.__setattr__(self, "quantile_levels", levels)
        if levels != tuple(sorted(set(levels))) or 0.5 not in levels or min(levels) <= 0 or max(levels) >= 1:
            raise GateError("CHRONOS_QUANTILES")
        if (not self.batch_sizes or set(self.batch_sizes) - {1, 2, 4, 8}
                or not self.learning_rates or set(self.learning_rates) - {1e-5, 5e-5}
                or not self.num_steps or set(self.num_steps) - {1, 2, 5, 10, 20, 50, 100}
                or not self.lora_ranks or set(self.lora_ranks) - {4, 8, 16}):
            raise GateError("CHRONOS_HYPERPARAMETER_ALLOWLIST")
        count(self.max_lora_candidates, "MAX_LORA_CANDIDATES", 0)
        if self.max_lora_candidates > 2:
            raise GateError("MAX_LORA_CANDIDATES")
        count(self.adapter_cache_size, "ADAPTER_CACHE_SIZE")
        if self.adapter_cache_size > 2:
            raise GateError("ADAPTER_CACHE_SIZE")
        if self.enabled and self.adapter_cache_size < 2:
            raise GateError("SHADOW_REQUIRES_TWO_MODEL_SLOTS")


@dataclass(frozen=True)
class PatchTSTSettings:
    """Fixed supervised PatchTST recipe; no implicit pretrained weight download."""

    patch_length: int = 8
    patch_stride: int = 4
    d_model: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    batch_size: int = 32
    num_steps: int = 50
    learning_rate: float = 0.001
    device: str = "cpu"

    def __post_init__(self):
        for name in ("patch_length", "patch_stride", "d_model", "num_hidden_layers",
                     "num_attention_heads", "batch_size", "num_steps"):
            count(getattr(self, name), "PATCHTST_" + name.upper())
        positive(self.learning_rate, "PATCHTST_LEARNING_RATE")
        if (self.device != "cpu" or self.d_model % self.num_attention_heads
                or self.d_model > 256 or self.num_hidden_layers > 6
                or self.batch_size > 256 or self.num_steps > 10000
                or self.patch_stride > self.patch_length or self.learning_rate > 0.1):
            raise GateError("PATCHTST_CONFIGURATION")


@dataclass(frozen=True)
class ForecastingContract:
    """Business-time and information-availability contract for one entity."""

    entity_id: str = "default"
    business_timezone: str = "UTC"
    target_columns: tuple[str, ...] = ()
    past_observed_covariates: tuple[str, ...] = ()
    known_future_covariates: tuple[str, ...] = ()
    cross_entity_learning: bool = False
    stale_output_policy: str = "reject"
    arrival_rule: str = "source_available_at"
    label_maturity_rule: str = "source_label_available_at"
    revision_rule: str = "append_only_as_of"

    def __post_init__(self):
        for name in ("target_columns", "past_observed_covariates", "known_future_covariates"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.entity_id or len(self.entity_id) > 160 or not self.business_timezone:
            raise GateError("FORECAST_ENTITY_OR_TIMEZONE")
        if self.stale_output_policy not in {"reject", "bridge_forecast", "delay_service"}:
            raise GateError("STALE_OUTPUT_POLICY")
        if not isinstance(self.cross_entity_learning, bool) or self.cross_entity_learning:
            raise GateError("CROSS_ENTITY_LEARNING_NOT_VALIDATED")
        if set(self.known_future_covariates) - set(self.past_observed_covariates):
            raise GateError("FUTURE_COVARIATE_WITHOUT_HISTORY")
        if set(self.known_future_covariates) & set(self.target_columns):
            raise GateError("FUTURE_TARGET_LEAKAGE")
        values = self.target_columns + self.past_observed_covariates
        if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
            raise GateError("FORECAST_CONTRACT_COLUMNS")
        if not all(isinstance(value, str) and value for value in
                   (self.arrival_rule, self.label_maturity_rule, self.revision_rule)):
            raise GateError("FORECAST_AVAILABILITY_RULE")


@dataclass(frozen=True)
class CollaborationConfig:
    permissions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    review_tokens: int = 8192
    question_tokens: int = 8192
    intent_tokens: int = 8192
    max_call_tokens: int = 4096
    max_calls_per_role: int = 8
    automatic_reviews: bool = False

    def __post_init__(self):
        if not isinstance(self.permissions, dict):
            raise GateError("HUMAN_PERMISSIONS")
        for actor, roles in self.permissions.items():
            if not isinstance(actor, str) or not actor or not isinstance(roles, (list, tuple)) or set(roles) - {"read", "experiment", "knowledge", "release"}:
                raise GateError("HUMAN_PERMISSIONS")
        for name in ("review_tokens", "question_tokens", "intent_tokens", "max_call_tokens", "max_calls_per_role"):
            count(getattr(self, name), name.upper())
        if not isinstance(self.automatic_reviews, bool):
            raise GateError("AUTOMATIC_REVIEW_FLAG")


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    variables: tuple[str, ...]
    source: SourceConfig
    frequency_seconds: int = 60
    horizon_steps: int = 30
    allowed_lateness_seconds: float = 120.0
    max_fill_steps: int = 2
    maximum_missing_fraction: float = 0.05
    seed: int = 42
    mode: str = "replay"
    search: SearchConfig = field(default_factory=SearchConfig)
    budget: Budget = field(default_factory=Budget)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    patchtst: PatchTSTSettings = field(default_factory=PatchTSTSettings)
    contract: ForecastingContract = field(default_factory=ForecastingContract)
    collaboration: CollaborationConfig = field(default_factory=CollaborationConfig)
    variable_weights: tuple[float, ...] = ()
    # Preserve the serialized layout of existing v8 tasks. This is provenance,
    # not a user-configurable hash override; active/nondefault settings always count.
    _patchtst_serialized: bool = field(default=False, repr=False, compare=False, kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", self.task_id):
            raise GateError("TASK_ID")
        if not self.variables or len(set(self.variables)) != len(self.variables) or not all(isinstance(v, str) and v for v in self.variables):
            raise GateError("VARIABLES")
        targets = self.contract.target_columns or self.variables
        if tuple(targets) != self.variables:
            raise GateError("TARGET_CONTRACT_MISMATCH")
        if self.contract.past_observed_covariates or self.contract.known_future_covariates:
            raise GateError("MODEL_COVARIATES_UNSUPPORTED")
        if not self.contract.target_columns:
            object.__setattr__(self, "contract", ForecastingContract(
                **{**asdict(self.contract), "target_columns": self.variables}))
        if self.mode not in {"replay", "live"}:
            raise GateError("TASK_MODE")
        if self.mode == "live" and self.source.kind not in {"influxdb", "history_http"}:
            raise GateError("LIVE_REQUIRES_DATA_SERVICE")
        count(self.frequency_seconds, "FREQUENCY")
        count(self.horizon_steps, "HORIZON")
        count(self.max_fill_steps, "MAX_FILL_STEPS", 0)
        count(self.seed, "SEED", 0)
        positive(self.allowed_lateness_seconds, "LATENESS", zero=True)
        if not 0 <= self.maximum_missing_fraction < 1:
            raise GateError("MISSING_THRESHOLD")
        if self.search.validation_steps < self.horizon_steps or self.search.test_steps < self.horizon_steps:
            raise GateError("EVALUATION_SHORTER_THAN_HORIZON")
        if min(self.search.histories) < min(self.search.inputs) + self.horizon_steps + 2:
            raise GateError("HISTORY_TOO_SHORT")
        if set(self.release.critical_variables) - set(self.variables):
            raise GateError("UNKNOWN_CRITICAL_VARIABLE")
        if any(set(group) - set(self.variables) for group in self.release.business_groups.values()):
            raise GateError("UNKNOWN_BUSINESS_GROUP_VARIABLE")
        if ("chronos2" in self.search.model_families) != self.foundation.enabled:
            raise GateError("FOUNDATION_SEARCH_CONFIGURATION")
        if "patchtst" in self.search.model_families and min(self.search.inputs) < self.patchtst.patch_length:
            raise GateError("PATCHTST_CONTEXT_TOO_SHORT")
        weights = tuple(self.variable_weights) or (1.0,) * len(self.variables)
        if len(weights) != len(self.variables):
            raise GateError("VARIABLE_WEIGHTS")
        for w in weights:
            positive(w, "VARIABLE_WEIGHT")
        total = sum(weights)
        object.__setattr__(self, "variable_weights", tuple(w / total for w in weights))

    @property
    def step_ns(self) -> int:
        return self.frequency_seconds * SECOND

    @property
    def evaluation_ns(self) -> int:
        seconds = self.monitoring.evaluation_seconds or self.search.test_steps * self.frequency_seconds
        return int(seconds * SECOND)

    @property
    def version(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("_patchtst_serialized")
        if (not self._patchtst_serialized and "patchtst" not in self.search.model_families
                and self.patchtst == PatchTSTSettings()):
            value.pop("patchtst")
        # New optional policy settings need not change the identity of an existing
        # explicitly configured task or its immutable model packages.
        planner_defaults = PlannerConfig()
        for name in ("short_output_tokens", "spec_output_tokens", "feedback_output_tokens",
                     "max_format_repairs", "candidate_pool_size", "initial_candidates",
                     "history_limit", "knowledge_limit", "example_limit"):
            if value["planner"][name] == getattr(planner_defaults, name):
                del value["planner"][name]
        if not value["collaboration"]["automatic_reviews"]:
            del value["collaboration"]["automatic_reviews"]
        if not value["release"]["business_groups"]:
            del value["release"]["business_groups"]
        # Preserve identities of existing approved configs and release packages.
        for name, default in (("parameter_bounds", {}), ("feature_operations", ())):
            if value["search"][name] == default:
                del value["search"][name]
        if value["budget"]["max_no_improvement"] == 3:
            del value["budget"]["max_no_improvement"]
        if self.collaboration == CollaborationConfig():
            del value["collaboration"]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskConfig":
        if not isinstance(value, Mapping) or "_patchtst_serialized" in value:
            raise GateError("INVALID_TASKCONFIG_FIELDS")
        data = dict(value)
        data["_patchtst_serialized"] = "patchtst" in value
        nested = {"source": SourceConfig, "search": SearchConfig, "budget": Budget, "monitoring": MonitoringConfig, "release": ReleaseConfig, "planner": PlannerConfig, "execution": ExecutionConfig, "foundation": FoundationConfig, "patchtst": PatchTSTSettings, "contract": ForecastingContract, "collaboration": CollaborationConfig}
        for key, typ in nested.items():
            if key in data:
                data[key] = strict(typ, data[key])
        return strict(cls, data)
