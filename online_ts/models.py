from __future__ import annotations

import platform
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import GateError, TaskConfig, digest, strict
from .data import ArtifactStore


SCALE_FLOOR = 1e-6
MODEL_FAMILIES = ("last_value", "seasonal_naive", "ridge", "chronos2", "patchtst")


@dataclass(frozen=True)
class Candidate:
    family: str
    input_steps: int
    history_steps: int
    alpha: float = 1.0
    period: int = 1
    model_id: str = ""
    revision: str = ""
    adaptation_mode: str = "classical"
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 1
    learning_rate: float = 1e-5
    num_steps: int = 2
    lora_rank: int = 4
    features: tuple[dict, ...] = ()

    def __post_init__(self):
        # Schema-level normalization makes 1 and 1.0 semantically identical.
        if isinstance(self.alpha, (int, float)) and not isinstance(self.alpha, bool):
            object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "quantile_levels", tuple(float(value) for value in self.quantile_levels))
        from .experiments import FeatureSpec
        if not isinstance(self.features, (list, tuple)) or len(self.features) > 16:
            raise GateError("FEATURE_COUNT")
        normalized = tuple(asdict(strict(FeatureSpec, item)) for item in self.features)
        if len({digest(f) for f in normalized}) != len(normalized):
            raise GateError("DUPLICATE_FEATURE")
        object.__setattr__(self, "features", tuple(sorted(normalized, key=digest)))

    @property
    def candidate_id(self) -> str:
        value = asdict(self)
        if not self.features:
            del value["features"]
        return digest(value)[:20]

    def validate(self, config: TaskConfig) -> None:
        for value in (self.input_steps, self.history_steps, self.period):
            if not isinstance(value, int) or isinstance(value, bool):
                raise GateError("CANDIDATE_INTEGER_PARAMETER")
        if not isinstance(self.alpha, (int, float)) or isinstance(self.alpha, bool) or not np.isfinite(self.alpha):
            raise GateError("CANDIDATE_ALPHA")
        if self.family not in MODEL_FAMILIES:
            raise GateError("MODEL_ALLOWLIST")
        if self.family not in config.search.model_families:
            raise GateError("MODEL_NOT_ENABLED_FOR_TASK")
        def allowed(name, value, grid):
            bounds = config.search.parameter_bounds.get(name)
            fixed = getattr(config.search, name, None)
            return value in grid or (fixed is None and bounds is not None and bounds[0] <= value <= bounds[1])
        if not allowed("input_steps", self.input_steps, config.search.inputs) or not allowed("history_steps", self.history_steps, config.search.histories):
            raise GateError("WINDOW_ALLOWLIST")
        if self.history_steps < self.input_steps + config.horizon_steps + 2:
            raise GateError("CANDIDATE_HISTORY")
        if self.family == "ridge" and not allowed("alpha", self.alpha, config.search.ridge_alphas):
            raise GateError("PARAMETER_ALLOWLIST")
        if self.family == "seasonal_naive" and (self.period != config.search.seasonal_period or self.period > self.input_steps):
            raise GateError("SEASONAL_PERIOD")
        if self.family != "ridge" and self.alpha != 1.0:
            raise GateError("UNUSED_PARAMETER")
        if self.family != "seasonal_naive" and self.period != 1:
            raise GateError("UNUSED_PARAMETER")
        if self.family == "chronos2":
            from .chronos_adapter import Chronos2Spec
            if not config.foundation.enabled or self.model_id not in config.foundation.model_ids:
                raise GateError("CHRONOS_MODEL_NOT_ENABLED")
            if self.adaptation_mode not in config.foundation.adaptation_modes:
                raise GateError("CHRONOS_ADAPTATION_NOT_ENABLED")
            if (self.batch_size not in config.foundation.batch_sizes
                    or self.quantile_levels != config.foundation.quantile_levels):
                raise GateError("CHRONOS_PARAMETER_NOT_OFFERED")
            if self.adaptation_mode == "zero_shot" and (
                    self.learning_rate != 1e-5 or self.num_steps != 2 or self.lora_rank != 4):
                raise GateError("UNUSED_LORA_PARAMETER")
            if self.adaptation_mode == "lora" and (
                    self.learning_rate not in config.foundation.learning_rates
                    or self.num_steps not in config.foundation.num_steps
                    or self.lora_rank not in config.foundation.lora_ranks):
                raise GateError("CHRONOS_PARAMETER_NOT_OFFERED")
            if self.adaptation_mode == "lora" and self.history_steps < 2 * (self.input_steps + config.horizon_steps):
                raise GateError("LORA_HISTORY_TOO_SHORT")
            Chronos2Spec(
                self.model_id, self.revision, self.adaptation_mode, self.input_steps,
                config.horizon_steps, self.quantile_levels, self.batch_size,
                self.learning_rate, self.num_steps, self.lora_rank, config.seed,
            )
        elif self.family == "patchtst":
            if (self.model_id != "patchtst" or self.revision or self.adaptation_mode != "from_scratch"
                    or self.batch_size != config.patchtst.batch_size
                    or self.learning_rate != config.patchtst.learning_rate
                    or self.num_steps != config.patchtst.num_steps
                    or self.quantile_levels != (0.1, 0.5, 0.9) or self.lora_rank != 4
                    or self.input_steps < config.patchtst.patch_length):
                raise GateError("PATCHTST_PARAMETER_NOT_OFFERED")
            if config.budget.max_memory_mb < 2048:
                raise GateError("PATCHTST_MEMORY_BUDGET")
        elif (self.model_id or self.revision or self.adaptation_mode != "classical"
              or self.quantile_levels != (0.1, 0.5, 0.9) or self.batch_size != 1
              or self.learning_rate != 1e-5 or self.num_steps != 2 or self.lora_rank != 4):
            raise GateError("UNUSED_FOUNDATION_PARAMETER")
        # Bound the actual classical allocations before constructing windows/Gram matrices.
        if self.features and self.family != "ridge":
            raise GateError("MODEL_FEATURES_UNSUPPORTED")
        from .experiments import FeatureSpec
        for feature in self.features:
            strict(FeatureSpec, feature).validate(config, self.input_steps)
        if self.family == "chronos2":
            return
        features = self.input_steps * len(config.variables) + len(self.features)
        targets = config.horizon_steps * len(config.variables)
        n = min(config.search.max_train_samples, self.history_steps)
        estimated = 8 * (4 * n * (features + targets) + 4 * min(n, features) ** 2 + 3 * features * targets)
        if estimated > config.budget.max_memory_mb * 1024 ** 2:
            raise GateError("MEMORY_BUDGET")

    @classmethod
    def from_dict(cls, value: dict) -> "Candidate":
        return strict(cls, value)


def window_arrays(values: np.ndarray, input_steps: int, horizon: int, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(origins) or np.any(origins < input_steps) or np.any(origins + horizon > len(values)):
        raise GateError("WINDOW_BOUNDARY")
    x = np.stack([values[o - input_steps:o] for o in origins])
    y = np.stack([values[o:o + horizon] for o in origins])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise GateError("WINDOW_MISSING")
    return x, y


class ForecastModel:
    """Portable direct multi-output model. No executable serialized objects."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.candidate = Candidate.from_dict(payload["candidate"])
        self.variables = tuple(payload["variables"])
        self.horizon = int(payload["horizon_steps"])
        self.mean = np.asarray(payload["scaler"]["mean"], dtype=float)
        self.scale = np.asarray(payload["scaler"]["std"], dtype=float)
        if self.mean.shape != (len(self.variables),) or self.scale.shape != self.mean.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise GateError("MODEL_SCALER")

    @classmethod
    def fit(cls, train_raw: pd.DataFrame, train_filled: pd.DataFrame, candidate: Candidate, config: TaskConfig,
            *, training_origins: np.ndarray | None = None) -> "ForecastModel":
        candidate.validate(config)
        raw = train_raw.to_numpy(dtype=float)
        values = train_filled.to_numpy(dtype=float)
        if len(raw) < candidate.input_steps + config.horizon_steps + 2 or not np.isfinite(values).all():
            raise GateError("TRAINING_DATA_INSUFFICIENT")
        mean = np.nanmean(raw, axis=0)
        raw_std = np.nanstd(raw, axis=0)
        scale = np.maximum(raw_std, SCALE_FLOOR)
        normalized = (values - mean) / scale
        origins = np.arange(candidate.input_steps, len(values) - config.horizon_steps + 1, dtype=int)
        if training_origins is not None:
            supplied = np.asarray(training_origins)
            if (supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.integer)
                    or len(np.unique(supplied)) != len(supplied)
                    or not np.isin(supplied, origins).all()):
                raise GateError("TRAINING_ORIGINS")
            origins = np.sort(supplied)
        origins = np.asarray([o for o in origins if np.isfinite(raw[o:o + config.horizon_steps]).all()], dtype=int)
        legal_training_origins = len(origins)
        if len(origins) < 2:
            raise GateError("TRAINING_LABEL_COVERAGE")
        if len(origins) > config.search.max_train_samples:
            origins = origins[np.linspace(0, len(origins) - 1, config.search.max_train_samples, dtype=int)]
        x, y = window_arrays(normalized, candidate.input_steps, config.horizon_steps, origins)
        learned: dict = {}
        if candidate.family == "ridge":
            from .experiments import feature_matrix
            xx = feature_matrix(x, candidate.features)
            yy = y.reshape(len(y), -1)
            xm, ym = xx.mean(axis=0), yy.mean(axis=0)
            centered = xx - xm
            if len(centered) <= centered.shape[1]:
                dual = np.linalg.solve(centered @ centered.T + candidate.alpha * np.eye(len(centered)), yy - ym)
                weights = centered.T @ dual
            else:
                weights = np.linalg.solve(centered.T @ centered + candidate.alpha * np.eye(centered.shape[1]), centered.T @ (yy - ym))
            learned = {"x_mean": xm.tolist(), "y_mean": ym.tolist(), "weights": weights.tolist()}
        elif candidate.family == "patchtst":
            from .patchtst_adapter import train_patchtst
            learned = train_patchtst(x, y, candidate, config)
        return cls({
            "schema": 1, "candidate": asdict(candidate), "variables": list(config.variables),
            "horizon_steps": config.horizon_steps, "frequency_seconds": config.frequency_seconds,
            "config_version": config.version, "seed": config.seed,
            "scaler": {"mean": mean.tolist(), "std": scale.tolist(), "near_constant": (raw_std < SCALE_FLOOR).tolist(),
                       "fit_start": int(train_raw.index[0]), "fit_end": int(train_raw.index[-1]), "unique_points": len(train_raw)},
            "training_target_max": int(train_raw.index[origins[-1] + config.horizon_steps - 1]),
            "training_samples": {"legal": legal_training_origins, "sampled": len(origins),
                                 "optimized": len(origins) if candidate.family in {"ridge", "patchtst"} else 0,
                                 "selection": "supplied_entity_safe_origins" if training_origins is not None else "evenly_spaced_legal_origins"},
            "learned": learned,
        })

    def predict(self, x: np.ndarray) -> np.ndarray:
        value = np.asarray(x, dtype=float)
        single = value.ndim == 2
        if single:
            value = value[None, :, :]
        if value.ndim != 3 or value.shape[1:] != (self.candidate.input_steps, len(self.variables)) or not np.isfinite(value).all():
            raise GateError("PREDICTION_INPUT_CONTRACT")
        z = (value - self.mean) / self.scale
        if self.candidate.family == "last_value":
            result = np.repeat(z[:, -1:, :], self.horizon, axis=1)
        elif self.candidate.family == "seasonal_naive":
            seed = z[:, -self.candidate.period:, :]
            result = np.tile(seed, (1, int(np.ceil(self.horizon / len(seed[0]))), 1))[:, :self.horizon]
        elif self.candidate.family == "ridge":
            learned = self.payload["learned"]
            from .experiments import feature_matrix
            result = (feature_matrix(z, self.candidate.features) - np.asarray(learned["x_mean"])) @ np.asarray(learned["weights"]) + np.asarray(learned["y_mean"])
            result = result.reshape(len(z), self.horizon, len(self.variables))
        elif self.candidate.family == "patchtst":
            from .patchtst_adapter import predict_patchtst
            result = predict_patchtst(self, z)
        else:
            raise GateError("MODEL_ALLOWLIST")
        result = result * self.scale + self.mean
        if result.shape != (len(value), self.horizon, len(self.variables)) or not np.isfinite(result).all():
            raise GateError("INVALID_PREDICTION")
        return result[0] if single else result

    def save(self, artifacts: ArtifactStore, metadata: dict) -> dict:
        return artifacts.put_json({**self.payload, "lineage": metadata}).to_dict()

    @classmethod
    def load(cls, artifacts: ArtifactStore, reference: dict, config: TaskConfig) -> "ForecastModel":
        payload = artifacts.read_json(reference)
        if payload.get("schema") != 1 or payload.get("config_version") != config.version or payload.get("variables") != list(config.variables) or payload.get("horizon_steps") != config.horizon_steps or payload.get("frequency_seconds") != config.frequency_seconds:
            raise GateError("MODEL_CONTRACT")
        model = cls(payload)
        model.candidate.validate(config)
        return model


def load_forecast_model(artifacts: ArtifactStore, reference: dict, config: TaskConfig):
    """Load a registered classical or foundation model without importing heavy deps for control commands."""
    payload = artifacts.read_json(reference)
    if payload.get("backend") == "chronos2":
        from .chronos_adapter import ChronosForecastModel
        return ChronosForecastModel.load_payload(payload, artifacts, config)
    return ForecastModel.load(artifacts, reference, config)


def metrics(actual: np.ndarray, predicted: np.ndarray, scale: np.ndarray, weights: tuple[float, ...]) -> dict:
    if actual.shape != predicted.shape or actual.ndim != 3 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise GateError("METRIC_INPUT")
    error = predicted - actual
    mae = np.abs(error).mean(axis=0)
    mse = (error ** 2).mean(axis=0)
    denominator = np.maximum(np.asarray(scale), SCALE_FLOOR)
    per_variable_mae = mae.mean(axis=0)
    per_variable_mse = mse.mean(axis=0)
    return {
        "mae_by_horizon_variable": mae.tolist(), "mse_by_horizon_variable": mse.tolist(),
        "mae_by_variable": per_variable_mae.tolist(), "mse_by_variable": per_variable_mse.tolist(),
        "bias_by_variable": error.mean(axis=(0, 1)).tolist(),
        "normalized_mae": float(np.dot(per_variable_mae / denominator, weights)),
        "normalized_mse": float(np.dot(per_variable_mse / denominator ** 2, weights)),
        "worst_variable_nmae": float((per_variable_mae / denominator).max()),
        "worst_horizon_nmae": float(((mae / denominator) @ np.asarray(weights)).max()),
        "origins": len(actual), "scale": denominator.tolist(),
    }


def environment_fingerprint() -> dict:
    from . import __version__
    from pathlib import Path
    import hashlib
    import importlib.metadata
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "pandas")}
    for name in ("pyarrow", "chronos-forecasting", "torch", "transformers", "accelerate", "peft"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    # Keep the control/classical path import-safe when optional Torch is installed.
    # Importing Torch here can consume the complete worker memory envelope and can
    # initialize CUDA even though a classical candidate never uses it.  Concrete
    # accelerator facts are recorded by the execution resource probe and by the
    # Chronos adapter; this fingerprint deliberately contains package metadata only.
    return {"package": __version__, "python": platform.python_version(), "dependencies": dependencies,
            "code_files": files, "code_digest": digest(files)}
