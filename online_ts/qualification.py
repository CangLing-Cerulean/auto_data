"""Purpose-specific data gates. Filling is for inputs; labels stay observed."""
import numpy as np

from .config import GateError
from .data import aligned, causal_fill, profile


def input_qualification(frame, config, *, now=None, freshness_seconds=None):
    issues, report = [], {}
    try:
        metadata = {"__available_at", "__label_available_at", "__revision_at", "__regime"}
        if [c for c in frame.columns if c not in metadata] != list(config.variables):
            raise GateError("SOURCE_FIELDS_MISSING")
        grid = aligned(frame, config)
        report = profile(frame, config)
        if (report["missing_fraction"] > config.maximum_missing_fraction
                or any(v > config.maximum_missing_fraction for v in report["missing_by_variable"])):
            issues.append("DATA_MISSINGNESS")
        if not np.isfinite(causal_fill(grid, config).to_numpy()).all():
            issues.append("LONG_GAP_OR_COLD_START")
        if now is not None and freshness_seconds is not None:
            if now - int(grid.index[-1]) > freshness_seconds * 1e9:
                issues.append("INPUT_STALE")
    except GateError as error:
        issues.append(str(error))
    return {"input_eligible": not issues, "issues": issues, "profile": report,
            "input_fill": {"method": "causal_forward", "maximum_steps": config.max_fill_steps,
                           "maximum_missing_fraction_per_variable": config.maximum_missing_fraction}}


def regime_context(frame, start, end, *, declared=False):
    """Explicit operational metadata is evidence; absence never proves a regime."""
    if not declared:
        return {"status": "unobserved", "regime_id": "unlabelled", "source": "not_provided"}
    mapping = frame.attrs.get("regime_by_time", {})
    times = [int(t) for t in frame.index if start <= t < end]
    values = [mapping.get(t) for t in times]
    if not values or any(v is None for v in values):
        return {"status": "unknown", "regime_id": None, "source": "explicit_metadata"}
    distinct = sorted(set(values))
    return {"status": "confirmed" if len(distinct) == 1 else "transition",
            "regime_id": distinct[0] if len(distinct) == 1 else None,
            "regimes": distinct, "source": "explicit_metadata"}


def candidate_qualification(frame, candidate, config):
    required = candidate.history_steps + config.search.folds * config.search.validation_steps + config.search.test_steps
    issues = []
    if len(frame) < required:
        issues.append("INITIAL_HISTORY_INSUFFICIENT")
    if candidate.history_steps < candidate.input_steps + config.horizon_steps + 2:
        issues.append("TRAINING_SAMPLE_INSUFFICIENT")
    samples = []
    evaluation_issues = []
    if len(frame) >= required:
        first_validation = len(frame) - config.search.test_steps - config.search.folds * config.search.validation_steps
        for fold in range(config.search.folds):
            end = first_validation + fold * config.search.validation_steps
            train = frame.iloc[end - candidate.history_steps:end]
            targets = train.to_numpy(dtype=float)
            valid = sum(np.isfinite(targets[o:o + config.horizon_steps]).all()
                        for o in range(candidate.input_steps, len(train) - config.horizon_steps + 1))
            samples.append(int(valid))
            if valid < 2:
                issues.append("TRAINING_LABEL_COVERAGE")
            if candidate.family == "chronos2" and candidate.adaptation_mode == "lora":
                internal = max(candidate.input_steps + config.horizon_steps,
                               min(config.search.validation_steps, len(train) // 3))
                if len(train) - internal < candidate.input_steps + config.horizon_steps:
                    issues.append("LORA_HISTORY_TOO_SHORT")
                if not np.isfinite(targets).all():
                    issues.append("LORA_LABEL_COVERAGE")
        if not np.isfinite(frame.iloc[first_validation:len(frame) - config.search.test_steps].to_numpy()).all():
            evaluation_issues.append("OFFLINE_LABEL_COVERAGE")
    if config.source.regime_column and len(frame):
        context = regime_context(frame.tail(required), int(frame.index[0]), int(frame.index[-1]) + 1, declared=True)
        if context["status"] != "confirmed":
            issues.append("CURRENT_REGIME_HISTORY_INSUFFICIENT")
    return {"candidate_id": candidate.candidate_id, "training_eligible": not issues,
            "required_history": required, "available_rows": len(frame), "issues": issues,
            "legal_supervised_origins_by_fold": samples,
            "evaluation_eligible": bool(samples) and not evaluation_issues, "evaluation_issues": evaluation_issues,
            "maximum_training_origins": max(0, candidate.history_steps - candidate.input_steps - config.horizon_steps + 1),
            "labels_imputed": False}


def evaluation_qualification(evidence, config):
    issues = []
    if not evidence.get("labels_complete") or not evidence.get("complete_cells"):
        issues.append("MATURE_LABEL_COVERAGE")
    if not evidence.get("window_complete", True):
        issues.append("WINDOW_INCOMPLETE")
    if evidence.get("origins", 0) < config.release.minimum_origins:
        issues.append("ORIGIN_COUNT")
    if not evidence.get("clean_inputs", False):
        issues.append("INPUT_QUALITY")
    return {"evaluation_eligible": not issues, "issues": issues, "labels_imputed": False}
