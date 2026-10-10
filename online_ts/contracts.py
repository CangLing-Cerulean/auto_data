from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import GateError


@dataclass(frozen=True)
class ForecastResult:
    """Backend-independent forecast with explicit business-time semantics."""

    model_id: str
    artifact_id: str
    entity_id: str
    variables: tuple[str, ...]
    feature_cutoff: int
    forecast_origin: int
    generated_at: int
    target_times: tuple[int, ...]
    frequency_ns: int
    point: np.ndarray
    quantiles: Mapping[float, np.ndarray] | None
    input_snapshot: str
    feature_version: str
    point_statistic: str = "median_q0.5"

    def __post_init__(self):
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "target_times", tuple(int(value) for value in self.target_times))
        point = np.asarray(self.point, dtype=float)
        object.__setattr__(self, "point", point)
        if not self.model_id or not self.artifact_id or not self.entity_id or not self.variables:
            raise GateError("FORECAST_IDENTITY")
        if self.feature_cutoff > self.forecast_origin or self.generated_at < self.forecast_origin:
            raise GateError("FORECAST_TIME_ORDER")
        if self.frequency_ns <= 0 or point.shape != (len(self.target_times), len(self.variables)) or not np.isfinite(point).all():
            raise GateError("FORECAST_SHAPE_OR_VALUE")
        expected = tuple(self.forecast_origin + (i + 1) * self.frequency_ns for i in range(len(self.target_times)))
        if self.target_times != expected:
            raise GateError("FORECAST_TARGET_ALIGNMENT")
        if self.point_statistic != "median_q0.5":
            raise GateError("FORECAST_POINT_STATISTIC")
        if self.quantiles is not None:
            levels = tuple(float(level) for level in self.quantiles)
            if levels != tuple(sorted(levels)) or len(levels) != len(set(levels)) or 0.5 not in levels:
                raise GateError("FORECAST_QUANTILE_ORDER")
            normalized = {float(level): np.asarray(value, dtype=float) for level, value in self.quantiles.items()}
            if any(value.shape != point.shape or not np.isfinite(value).all() for value in normalized.values()):
                raise GateError("FORECAST_QUANTILE_SHAPE_OR_VALUE")
            if not np.allclose(normalized[0.5], point, rtol=0, atol=0):
                raise GateError("FORECAST_MEDIAN_MISMATCH")
            object.__setattr__(self, "quantiles", normalized)

    @property
    def horizon(self) -> int:
        return len(self.target_times)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "artifact_id": self.artifact_id, "entity_id": self.entity_id,
            "variables": list(self.variables), "feature_cutoff": self.feature_cutoff,
            "forecast_origin": self.forecast_origin, "generated_at": self.generated_at,
            "target_times": list(self.target_times), "horizon": self.horizon,
            "frequency_ns": self.frequency_ns, "point": self.point.tolist(),
            "quantiles": None if self.quantiles is None else {str(k): v.tolist() for k, v in self.quantiles.items()},
            "input_snapshot": self.input_snapshot, "feature_version": self.feature_version,
            "point_statistic": self.point_statistic,
        }
