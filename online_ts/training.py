from __future__ import annotations

import time
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import GateError, TaskConfig, canonical
from .data import ArtifactStore, causal_fill, load_snapshot
from .models import (Candidate, ForecastModel, SCALE_FLOOR, environment_fingerprint,
                     load_forecast_model, metrics, window_arrays)


def final_incumbent_issues(current, incumbent, config, scale):
    """Paired sealed admission, with the same objective and protected scopes as release."""
    issues = []
    before, after = incumbent["normalized_mae"], current["normalized_mae"]
    if (before - after) / max(before, SCALE_FLOOR) < config.release.minimum_improvement:
        issues.append("OFFLINE_CHAMPION_INSUFFICIENT_GAIN")
    a, b = np.asarray(incumbent["mae_by_variable"]), np.asarray(current["mae_by_variable"])
    for variable in config.release.critical_variables:
        i = config.variables.index(variable)
        if b[i] > a[i] * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("OFFLINE_CHAMPION_CRITICAL_REGRESSION")
    scale = np.asarray(scale)
    for variables in config.release.business_groups.values():
        indices = [config.variables.index(v) for v in variables]
        if np.mean(b[indices] / scale[indices]) > np.mean(a[indices] / scale[indices]) * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("OFFLINE_CHAMPION_GROUP_REGRESSION")
    return issues


def evaluate_frozen_final(result, truth, evaluation_asof, config, artifacts):
    """Evaluate persisted predictions only; this path cannot fit or select a model."""
    from .config import digest
    from .labels import PROTOCOL_VERSION
    frozen = artifacts.read_json(result["frozen_predictions"])
    if (frozen["config_version"] != config.version
            or frozen["snapshot_sha"] != result["snapshot"]["sha256"]
            or frozen["candidate_id"] != Candidate.from_dict(result["candidate"]).candidate_id):
        raise GateError("FROZEN_EVALUATION_SCOPE")
    maturity = truth.attrs.get("label_availability_by_time", {})
    target_times = frozen["target_times"]
    flat_times = sorted({t for window in target_times for t in window})
    if any(t not in truth.index or maturity.get(t, t) > evaluation_asof for t in flat_times):
        return result
    actual = np.asarray([[truth.loc[t, list(config.variables)].to_numpy(dtype=float)
                          for t in window] for window in target_times])
    if not np.isfinite(actual).all() or len(actual) < config.release.minimum_origins:
        return result
    prediction, scale = np.asarray(frozen["prediction"]), np.asarray(frozen["scale"])
    score = metrics(actual, prediction, scale, config.variable_weights)
    baselines = {name: metrics(actual, np.asarray(frozen[name]), scale, config.variable_weights)
                 for name in ("last_value", "seasonal_naive")}
    best = min(baselines.values(), key=lambda value: value["normalized_mae"])
    issues = list(frozen["issues"])
    if score["normalized_mae"] > best["normalized_mae"] + 1e-12:
        issues.append("OFFLINE_BASELINE_REGRESSION")
    for variable in config.release.critical_variables:
        i = config.variables.index(variable)
        if score["mae_by_variable"][i] > best["mae_by_variable"][i] * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("OFFLINE_CRITICAL_REGRESSION")
    if result["inference_ms"] > min(config.release.maximum_inference_ms, config.frequency_seconds * 1000):
        issues.append("OFFLINE_INFERENCE_BUDGET")
    incumbent = None
    if frozen["incumbent"] is not None:
        incumbent = metrics(actual, np.asarray(frozen["incumbent"]), scale, config.variable_weights)
        issues.extend(final_incumbent_issues(score, incumbent, config, scale))
    version = digest({"values": actual.tolist(), "maturity": [maturity.get(t, t) for t in flat_times],
                      "revisions": [truth.attrs.get("availability_by_time", {}).get(t, t) for t in flat_times]})
    evidence = {"protocol_version": PROTOCOL_VERSION, "evaluation_asof": evaluation_asof,
                "label_version": version, "frozen_predictions": result["frozen_predictions"],
                "evaluation_id": digest({"model": result["artifact"], "window": result["frozen_predictions"],
                                         "labels": version, "protocol": PROTOCOL_VERSION}),
                "actual": actual.tolist(), "offline": score, "baselines": baselines, "issues": issues}
    return {**result, "waiting_for_labels": False, "passed": not issues, "issues": issues,
            "offline": score, "baselines": baselines, "champion_comparison": incumbent, "label_coverage": 1.,
            "probabilistic": probabilistic_metrics(actual, {float(k): np.asarray(v) for k, v in frozen["quantiles"].items()})
                if frozen["quantiles"] else {"applicable": False},
            "evaluation": artifacts.put_json(evidence).to_dict()}


def _chronos_adapter(config: TaskConfig, candidate: Candidate):
    from .chronos_adapter import Chronos2Adapter, spec_for_candidate
    try:
        root = config.foundation.model_roots[candidate.model_id]
    except KeyError:
        raise GateError("CHRONOS_MODEL_ROOT_REQUIRED") from None
    return Chronos2Adapter(
        spec_for_candidate(candidate, config), root, device=config.foundation.device,
    ).load()


def _chronos_fit_values(filled, candidate: Candidate, config: TaskConfig) -> tuple[np.ndarray, np.ndarray]:
    values = filled.to_numpy(dtype=np.float32)
    required = candidate.input_steps + config.horizon_steps
    internal_validation = max(required, min(config.search.validation_steps, len(values) // 3))
    if len(values) - internal_validation < required:
        raise GateError("LORA_HISTORY_TOO_SHORT")
    return values[:-internal_validation], values[-internal_validation:]


def _chronos_predictions(adapter, filled, origins: np.ndarray, candidate: Candidate,
                         config: TaskConfig, windows=None) -> dict[float, np.ndarray]:
    values = filled.to_numpy(dtype=float)
    rows = {level: [] for level in candidate.quantile_levels}
    for index, origin in enumerate(origins):
        context = windows[index] if windows is not None else values[origin - candidate.input_steps:origin]
        by_level = adapter.predict_array(context, variables=config.variables)
        for level in candidate.quantile_levels:
            rows[level].append(by_level[level])
    result = {level: np.stack(values) for level, values in rows.items()}
    expected = (len(origins), config.horizon_steps, len(config.variables))
    if any(values.shape != expected for values in result.values()):
        raise GateError("CHRONOS_OUTPUT_SHAPE_OR_VALUE")
    return result


def probabilistic_metrics(actual: np.ndarray, quantiles: dict[float, np.ndarray]) -> dict:
    levels = tuple(sorted(float(level) for level in quantiles))
    if not levels or 0.5 not in levels:
        raise GateError("PROBABILISTIC_EVIDENCE")
    predictions = [np.asarray(quantiles[level], dtype=float) for level in levels]
    if any(value.shape != actual.shape or not np.isfinite(value).all() for value in predictions):
        raise GateError("PROBABILISTIC_EVIDENCE")
    stacked = np.stack(predictions)
    if np.any(np.diff(stacked, axis=0) < -1e-6):
        raise GateError("CHRONOS_QUANTILE_CROSSING")
    losses = []
    for level, predicted in zip(levels, predictions):
        error = actual - predicted
        losses.append(np.maximum(level * error, (level - 1) * error))
    lower, upper = predictions[0], predictions[-1]
    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower
    return {
        "applicable": True, "calibrated": False, "quantile_levels": list(levels),
        "pinball_loss_by_quantile_variable": np.stack(losses).mean(axis=(1, 2)).tolist(),
        "pinball_loss": float(np.stack(losses).mean()),
        "central_interval": [levels[0], levels[-1]],
        "interval_coverage": float(covered.mean()),
        "interval_coverage_by_variable": covered.mean(axis=(0, 1)).tolist(),
        "interval_width": float(width.mean()),
        "interval_width_by_variable": width.mean(axis=(0, 1)).tolist(),
    }


def _persist_lora_checkpoint(checkpoint: str | Path, manifest: dict, artifacts: ArtifactStore) -> tuple[Path, str]:
    source = Path(checkpoint).resolve()
    packages = artifacts.root / "model-packages"
    packages.mkdir(parents=True, exist_ok=True)
    destination = (packages / manifest["manifest_sha256"]).resolve()
    if destination.exists():
        from .chronos_adapter import verify_weight_manifest
        existing = verify_weight_manifest(destination, expected_model_id=manifest["model_id"],
                                          expected_revision=manifest["revision"], kind="lora")
        if existing != manifest:
            raise GateError("MODEL_PACKAGE_CONFLICT")
    else:
        shutil.move(str(source), str(destination))
    relative = destination.relative_to(artifacts.root.resolve()).as_posix()
    return destination, relative


def split_plan(length: int, config: TaskConfig, *, history_steps: int | None = None) -> dict:
    search = config.search
    test_start = length - search.test_steps
    first_validation = test_start - search.folds * search.validation_steps
    if first_validation < (history_steps if history_steps is not None else min(search.histories)):
        raise GateError("BACKTEST_HISTORY_INSUFFICIENT")
    return {"folds": [{"train_end": first_validation + k * search.validation_steps,
                       "validation_end": first_validation + (k + 1) * search.validation_steps} for k in range(search.folds)],
            "test_start": test_start, "test_end": length}


def evaluation_arrays(raw, filled, start: int, end: int, candidate: Candidate, config: TaskConfig, *, require_labels=True):
    origins = np.arange(start, end - config.horizon_steps + 1, dtype=int)
    if not len(origins) or start < candidate.input_steps:
        raise GateError("EVALUATION_ORIGINS")
    values = filled.to_numpy(dtype=float)
    x, _ = window_arrays(values, candidate.input_steps, config.horizon_steps, origins)
    availability = raw.attrs.get("availability_by_time", {})
    if availability:
        windows = []
        for origin in origins:
            issue = int(raw.index[origin - 1])
            begin = max(0, origin - candidate.input_steps - config.max_fill_steps)
            history = raw.iloc[begin:origin].copy()
            late = [availability.get(int(t), int(t)) > issue for t in history.index]
            history.loc[late, :] = np.nan
            visible = causal_fill(history, config).tail(candidate.input_steps).to_numpy(dtype=float)
            if not np.isfinite(visible).all():
                raise GateError("HISTORICAL_INPUT_UNAVAILABLE")
            windows.append(visible)
        x = np.stack(windows)
    actual = np.stack([raw.iloc[o:o + config.horizon_steps].to_numpy(dtype=float) for o in origins])
    evaluation_asof = raw.attrs.get("evaluation_asof", int(raw.index[-1]))
    maturity = raw.attrs.get("label_availability_by_time", {})
    if require_labels and any(maturity.get(int(t), int(t)) > evaluation_asof for t in raw.index[start:end]):
        raise GateError("WAITING_FOR_MATURE_EVALUATION_LABELS")
    valid = np.isfinite(actual).all(axis=(1, 2))
    coverage = float(valid.mean())
    if not require_labels:
        return x, actual, origins, coverage
    if coverage != 1.0 or int(valid.sum()) < config.release.minimum_origins:
        raise GateError("OFFLINE_LABEL_COVERAGE")
    return x[valid], actual[valid], origins[valid], coverage


def training_as_of(raw, start, end, config):
    """A fold cannot fit labels or revisions that arrived after its training cutoff."""
    issue = raw.attrs.get("training_asof", int(raw.index[end - 1]))
    train = raw.iloc[start:end].copy()
    available = raw.attrs.get("label_availability_by_time", {})
    if any(available.get(int(t), int(t)) > issue for t in train.index):
        raise GateError("HISTORICAL_TRAINING_UNAVAILABLE")
    filled = causal_fill(train, config)
    if not np.isfinite(filled.to_numpy()).all():
        raise GateError("HISTORICAL_TRAINING_UNAVAILABLE")
    return train, filled


def baseline_predictions(filled, origins, config: TaskConfig, raw=None, *, require_labels=True) -> tuple[np.ndarray, np.ndarray]:
    values = filled.to_numpy(dtype=float)
    last = np.stack([np.repeat(values[o - 1:o], config.horizon_steps, axis=0) for o in origins])
    period = config.search.seasonal_period
    if np.any(origins < period):
        raise GateError("BASELINE_HISTORY")
    seasonal = np.stack([np.tile(values[o - period:o], (int(np.ceil(config.horizon_steps / period)), 1))[:config.horizon_steps] for o in origins])
    if raw is not None and raw.attrs.get("availability_by_time"):
        probe = Candidate("last_value", period, max(config.search.histories))
        x, _, baseline_origins, _ = evaluation_arrays(raw, filled, int(origins[0]),
            int(origins[-1]) + config.horizon_steps, probe, config, require_labels=require_labels)
        if not np.array_equal(origins, baseline_origins):
            raise GateError("BASELINE_PAIR_MISMATCH")
        last = np.repeat(x[:, -1:, :], config.horizon_steps, axis=1)
        seasonal = np.tile(x, (1, int(np.ceil(config.horizon_steps / period)), 1))[:, :config.horizon_steps]
    return last, seasonal


def validate_candidate(config: TaskConfig, snapshot_ref: dict, candidate: Candidate, artifacts: ArtifactStore,
                       execution: dict | None = None, champion_artifact: dict | None = None) -> dict:
    candidate.validate(config)
    raw, meta = load_snapshot(snapshot_ref, config, artifacts)
    filled = causal_fill(raw, config)
    from .qualification import candidate_qualification
    qualification = candidate_qualification(raw, candidate, config)
    if not qualification["training_eligible"]:
        raise GateError(qualification["issues"][0])
    split = split_plan(len(raw), config, history_steps=candidate.history_steps)
    scale_raw, _ = training_as_of(raw, 0, split["folds"][0]["train_end"], config)
    common_scale = np.maximum(np.nanstd(scale_raw.to_numpy(), axis=0), SCALE_FLOOR)
    fold_reports = []
    incumbent = load_forecast_model(artifacts, champion_artifact, config) if champion_artifact else None
    incumbent_cutoff = incumbent.payload["scaler"]["fit_end"] if incumbent else None
    comparable = incumbent is not None and incumbent_cutoff < int(raw.index[split["folds"][0]["train_end"] - 1])
    for fold in split["folds"]:
        end, val_end = fold["train_end"], fold["validation_end"]
        start = end - candidate.history_steps
        train_raw, train_filled = training_as_of(raw, start, end, config)
        x, actual, origins, coverage = evaluation_arrays(raw, filled, end, val_end, candidate, config)
        adaptation_evidence = None
        if candidate.family == "chronos2":
            adapter = _chronos_adapter(config, candidate)
            if candidate.adaptation_mode == "lora":
                if not np.isfinite(train_raw.to_numpy()).all():
                    raise GateError("LORA_LABEL_COVERAGE")
                fit_values, internal_values = _chronos_fit_values(
                    train_filled, candidate, config)
                with tempfile.TemporaryDirectory(prefix="chronos-fold-", dir=artifacts.root) as folder:
                    adaptation_evidence = adapter.adapt_lora(
                        fit_values, folder, validation_values=internal_values)["evidence"]
            quantile_predictions = _chronos_predictions(adapter, filled, origins, candidate, config, x)
            prediction = quantile_predictions[0.5]
            scaler_fit_end = int(raw.index[end - 1])
            weight_updates = 0 if candidate.adaptation_mode == "zero_shot" else adaptation_evidence["optimizer_steps"]
        else:
            model = ForecastModel.fit(train_raw, train_filled, candidate, config)
            if model.payload["training_target_max"] >= int(raw.index[end]):
                raise GateError("TRAINING_TARGET_LEAKAGE")
            prediction = model.predict(x)
            scaler_fit_end = model.payload["scaler"]["fit_end"]
            weight_updates = 1
            if candidate.family == "patchtst":
                adaptation_evidence = model.payload["learned"]["training_evidence"]
                weight_updates = adaptation_evidence["optimizer_steps"]
        report = metrics(actual, prediction, common_scale, config.variable_weights)
        last, seasonal = baseline_predictions(filled, origins, config, raw=raw)
        report["baselines"] = {"last_value": metrics(actual, last, common_scale, config.variable_weights),
                               "seasonal_naive": metrics(actual, seasonal, common_scale, config.variable_weights)}
        if comparable:
            incumbent_x, incumbent_actual, incumbent_origins, _ = evaluation_arrays(raw, filled, end, val_end, incumbent.candidate, config)
            if not np.array_equal(origins, incumbent_origins) or not np.array_equal(actual, incumbent_actual):
                raise GateError("DEVELOPMENT_CHAMPION_PAIR_MISMATCH")
            report["deployed_champion"] = metrics(actual, incumbent.predict(incumbent_x), common_scale, config.variable_weights)
        report["probabilistic"] = (probabilistic_metrics(actual, quantile_predictions)
                                   if candidate.family == "chronos2" else {"applicable": False})
        report.update(train_start=int(raw.index[start]), train_end=int(raw.index[end - 1]),
                      validation_start=int(raw.index[end]), validation_end=int(raw.index[val_end - 1]),
                      scaler_fit_end=scaler_fit_end, label_coverage=coverage,
                      issue_times=[int(raw.index[o - 1]) for o in origins],
                      adaptation_mode=candidate.adaptation_mode, weight_updates=weight_updates,
                      training_samples=model.payload["training_samples"] if candidate.family != "chronos2" else None,
                      adaptation_evidence=adaptation_evidence)
        report.update(backtest_kind="retrospective_mature_history", training_asof=meta["as_of"],
                      evaluation_asof=meta["as_of"], input_asof=report["issue_times"])
        fold_reports.append(report)
    baseline_name = min(("last_value", "seasonal_naive"), key=lambda name: np.mean(
        [f["baselines"][name]["normalized_mae"] for f in fold_reports]))
    issues = []
    for variable in config.release.critical_variables:
        index = config.variables.index(variable)
        measured = np.mean([f["mae_by_variable"][index] for f in fold_reports])
        reference = np.mean([f["baselines"][baseline_name]["mae_by_variable"][index] for f in fold_reports])
        if measured > reference * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("VALIDATION_CRITICAL_REGRESSION")
    report = {"candidate": asdict(candidate), "candidate_id": candidate.candidate_id, "eligible": not issues,
            "data_qualification": qualification,
            "deployed_champion_comparison": {"comparable": comparable, "artifact": champion_artifact,
                "training_cutoff": incumbent_cutoff,
                "reason": "paired_frozen_artifact" if comparable else "training_overlaps_development" if incumbent else "no_champion"},
            "issues": issues, "config_version": config.version,
            "score": float(np.mean([f["normalized_mae"] for f in fold_reports])),
            "score_std": float(np.std([f["normalized_mae"] for f in fold_reports])),
            "folds": fold_reports, "snapshot_sha": snapshot_ref["sha256"], "split": split,
            "normalization": {"version": "first-fold-training-std-v1", "scale": common_scale.tolist(),
                              "source_end": int(raw.index[split["folds"][0]["train_end"] - 1])},
            "execution": {**(execution or {"target": "local", "training_backend": "numpy-cpu"}),
                          **({"training_backend": "torch-cpu"} if candidate.family == "patchtst" else {})}}
    report["development_admission"] = development_admission(report, config)
    return {**report, "validation_artifact": artifacts.put_json(report).to_dict()}


def development_admission(report, config):
    comparison = report["deployed_champion_comparison"]
    if not comparison["artifact"]:
        return {"worthy_of_final": report["eligible"], "reason": "initial_model", "comparison": comparison}
    if not comparison["comparable"]:
        return {"worthy_of_final": False, "reason": "DEVELOPMENT_CHAMPION_COMPARISON_REQUIRED", "comparison": comparison}
    folds = report["folds"]
    before = float(np.mean([f["deployed_champion"]["normalized_mae"] for f in folds]))
    gain = (before - report["score"]) / max(before, SCALE_FLOOR)
    issues = []
    if gain < config.release.minimum_improvement:
        issues.append("DEVELOPMENT_INSUFFICIENT_GAIN")
    if any(f["normalized_mae"] > f["deployed_champion"]["normalized_mae"] + 1e-12 for f in folds):
        issues.append("DEVELOPMENT_TEMPORAL_REGRESSION")
    current = np.mean([f["mae_by_variable"] for f in folds], axis=0)
    reference = np.mean([f["deployed_champion"]["mae_by_variable"] for f in folds], axis=0)
    for variable in config.release.critical_variables:
        i = config.variables.index(variable)
        if current[i] > reference[i] * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("DEVELOPMENT_CRITICAL_REGRESSION")
    scale = np.asarray(report["normalization"]["scale"])
    for variables in config.release.business_groups.values():
        indices = [config.variables.index(v) for v in variables]
        if np.mean(current[indices] / scale[indices]) > np.mean(reference[indices] / scale[indices]) * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("DEVELOPMENT_GROUP_REGRESSION")
    return {"worthy_of_final": report["eligible"] and not issues, "reason": issues[0] if issues else "paired_gain",
            "issues": issues, "relative_improvement": gain, "comparison": comparison,
            "minimum_improvement": config.release.minimum_improvement}


def choose(reports: list[dict], config: TaskConfig, *, artifacts: ArtifactStore | None = None, snapshot_sha: str | None = None) -> dict:
    qualified = []
    for report in reports:
        if not report.get("eligible"):
            continue
        if report.get("config_version") != config.version or len(report.get("folds", [])) != config.search.folds or not np.isfinite(report.get("score", float("nan"))):
            raise GateError("VALIDATION_EVIDENCE")
        candidate = Candidate.from_dict(report["candidate"])
        candidate.validate(config)
        if report.get("candidate_id") != candidate.candidate_id or (snapshot_sha and report.get("snapshot_sha") != snapshot_sha):
            raise GateError("VALIDATION_EVIDENCE_SCOPE")
        if any(f["train_end"] >= f["validation_start"] or f["scaler_fit_end"] > f["train_end"] for f in report["folds"]):
            raise GateError("VALIDATION_TIME_BOUNDARY")
        if artifacts is not None:
            reference = report.get("validation_artifact")
            if reference is None or canonical(artifacts.read_json(reference)) != canonical({k: v for k, v in report.items() if k != "validation_artifact"}):
                raise GateError("VALIDATION_ARTIFACT_MISMATCH")
        qualified.append(report)
    if not qualified:
        raise GateError("NO_QUALIFIED_CANDIDATE")
    complexity = {"last_value": 0, "seasonal_naive": 1, "ridge": 2, "patchtst": 3, "chronos2": 4}
    best = min(qualified, key=lambda r: (r["score"], r["score_std"], complexity[r["candidate"]["family"]], r["candidate_id"]))
    return best["candidate"]


def final_fit(config: TaskConfig, snapshot_ref: dict, selected: Candidate, reports: list[dict], artifacts: ArtifactStore,
              champion_ref: dict | None = None, execution: dict | None = None) -> dict:
    """A sealed configuration gets one logical holdout operation; never search on this result."""
    verified = Candidate.from_dict(
        choose(reports, config, artifacts=artifacts, snapshot_sha=snapshot_ref["sha256"])
    )
    if selected != verified:
        raise GateError("UNSEALED_SELECTION")
    raw, meta = load_snapshot(snapshot_ref, config, artifacts)
    filled = causal_fill(raw, config)
    split = split_plan(len(raw), config, history_steps=selected.history_steps)
    end = split["test_start"]
    start = end - selected.history_steps
    train_raw, train_filled = training_as_of(raw, start, end, config)
    from .qualification import regime_context
    training_regime = regime_context(train_raw, int(train_raw.index[0]), int(train_raw.index[-1]) + 1,
                                     declared=bool(config.source.regime_column))
    cold_started = time.perf_counter()
    adaptation_evidence = None
    if selected.family == "chronos2":
        from .chronos_adapter import ChronosForecastModel
        adapter = _chronos_adapter(config, selected)
        cold_start_seconds = time.perf_counter() - cold_started
        if selected.adaptation_mode == "lora":
            if not np.isfinite(train_raw.to_numpy()).all():
                raise GateError("LORA_LABEL_COVERAGE")
            fit_values, internal_values = _chronos_fit_values(
                train_filled, selected, config)
            with tempfile.TemporaryDirectory(prefix="chronos-final-", dir=artifacts.root) as folder:
                adapted = adapter.adapt_lora(fit_values, folder, validation_values=internal_values)
                _, relative = _persist_lora_checkpoint(
                    adapted["checkpoint"], adapted["manifest"], artifacts)
                adaptation_evidence = adapted["evidence"]
            package = {"kind": "lora", "manifest_sha256": adapted["manifest"]["manifest_sha256"],
                       "artifact_relative_path": relative,
                       "base_manifest_sha256": adapted["base_manifest"]["manifest_sha256"]}
        else:
            package = {"kind": "base", "manifest_sha256": adapter.manifest["manifest_sha256"]}
        model = ChronosForecastModel.build(
            selected, config, train_raw, adapter, package=package, lineage={})
    else:
        model = ForecastModel.fit(train_raw, train_filled, selected, config)
        cold_start_seconds = time.perf_counter() - cold_started
        if selected.family == "patchtst":
            adaptation_evidence = model.payload["learned"]["training_evidence"]
            execution = {**(execution or {"target": "local"}), "training_backend": "torch-cpu"}
    x, actual, origins, coverage = evaluation_arrays(raw, filled, end, len(raw), selected, config, require_labels=False)
    latency_ms = []
    for _ in range(5):
        before = time.perf_counter()
        model.predict(x[0])
        latency_ms.append((time.perf_counter() - before) * 1000)
    inference_ms = float(np.percentile(latency_ms, 95))
    if selected.family == "chronos2":
        quantile_predictions = model.predict_quantiles(x)
        prediction = quantile_predictions[0.5]
        probability = {"applicable": True, "pending": True}
    else:
        prediction = model.predict(x)
        probability = {"applicable": False}
    maturity = raw.attrs.get("label_availability_by_time", {})
    if coverage != 1.0 or any(maturity.get(int(t), int(t)) > meta["as_of"] for t in raw.index[end:]):
        last, seasonal = baseline_predictions(filled, origins, config, raw=raw, require_labels=False)
        incumbent_prediction = None
        pending_issues = []
        if champion_ref is not None:
            champion = load_forecast_model(artifacts, champion_ref, config)
            if champion.payload["scaler"]["fit_end"] >= int(raw.index[end]):
                pending_issues.append("CHAMPION_TEST_OVERLAP")
            else:
                champion_x, _, _, _ = evaluation_arrays(raw, filled, end, len(raw), champion.candidate, config, require_labels=False)
                incumbent_prediction = champion.predict(champion_x).tolist()
        lineage = {"snapshot": snapshot_ref, "snapshot_cutoff": meta["cutoff"], "split": split,
                   "environment": environment_fingerprint(), "evaluation_status": "waiting_for_labels"}
        reference = model.save(artifacts, lineage)
        loaded = load_forecast_model(artifacts, reference, config)
        if not np.allclose(prediction, loaded.predict(x), rtol=1e-5 if selected.family == "chronos2" else 1e-10, atol=1e-10):
            raise GateError("MODEL_RELOAD_MISMATCH")
        probe = artifacts.put_json({"input": x[0].tolist(), "prediction": prediction[0].tolist()}).to_dict()
        frozen = artifacts.put_json({"prediction": prediction.tolist(), "scale": model.scale.tolist(),
            "last_value": last.tolist(), "seasonal_naive": seasonal.tolist(),
            "incumbent": incumbent_prediction, "issues": pending_issues,
            "quantiles": {str(k): v.tolist() for k, v in quantile_predictions.items()} if selected.family == "chronos2" else {},
            "target_times": [[int(t) for t in raw.index[o:o + config.horizon_steps]] for o in origins],
            "issue_times": [int(raw.index[o - 1]) for o in origins],
            "snapshot_sha": snapshot_ref["sha256"], "candidate_id": selected.candidate_id,
            "config_version": config.version}).to_dict()
        return {"artifact": reference, "reload_probe": probe, "candidate": asdict(selected), "snapshot": snapshot_ref,
                "training_regime": training_regime,
                "offline": None, "passed": False, "waiting_for_labels": True,
                "issues": ["WAITING_FOR_MATURE_EVALUATION_LABELS"], "frozen_predictions": frozen,
                "test_start": int(raw.index[end]), "test_end": int(raw.index[-1]), "inference_ms": inference_ms,
                "validation_reports": reports, "label_coverage": coverage,
                "holdout_operation": {"snapshot_sha": snapshot_ref["sha256"], "candidate_id": selected.candidate_id}}
    if selected.family == "chronos2":
        probability = probabilistic_metrics(actual, quantile_predictions)
    # All baselines and incumbent use this final Train scale and identical test origins.
    score = metrics(actual, prediction, model.scale, config.variable_weights)
    last, seasonal = baseline_predictions(filled, origins, config, raw=raw)
    baselines = {"last_value": metrics(actual, last, model.scale, config.variable_weights), "seasonal_naive": metrics(actual, seasonal, model.scale, config.variable_weights)}
    best_baseline = min(baselines.values(), key=lambda r: r["normalized_mae"])
    issues = []
    if score["normalized_mae"] > best_baseline["normalized_mae"] + 1e-12:
        issues.append("OFFLINE_BASELINE_REGRESSION")
    for variable in config.release.critical_variables:
        i = config.variables.index(variable)
        if score["mae_by_variable"][i] > best_baseline["mae_by_variable"][i] * (1 + config.release.maximum_critical_regression) + 1e-12:
            issues.append("OFFLINE_CRITICAL_REGRESSION")
    if inference_ms > min(config.release.maximum_inference_ms, config.frequency_seconds * 1000):
        issues.append("OFFLINE_INFERENCE_BUDGET")
    incumbent = None
    if champion_ref is not None:
        champion = load_forecast_model(artifacts, champion_ref, config)
        # A current model trained after this test starts cannot be fairly evaluated here.
        if champion.payload["scaler"]["fit_end"] >= int(raw.index[end]):
            issues.append("CHAMPION_TEST_OVERLAP")
        else:
            champion_x, incumbent_actual, incumbent_origins, _ = evaluation_arrays(raw, filled, end, len(raw), champion.candidate, config)
            if not np.array_equal(origins, incumbent_origins) or not np.array_equal(actual, incumbent_actual):
                raise GateError("OFFLINE_PAIR_MISMATCH")
            incumbent = metrics(actual, champion.predict(champion_x), model.scale, config.variable_weights)
            issues.extend(final_incumbent_issues(score, incumbent, config, model.scale))
    lineage = {"snapshot": snapshot_ref, "snapshot_cutoff": meta["cutoff"], "split": split,
               "train_start": int(raw.index[start]), "train_end": int(raw.index[end - 1]),
               "test_start": int(raw.index[end]), "test_end": int(raw.index[-1]),
               "validation_reports": reports, "offline_metrics": score,
               "probabilistic_metrics": probability, "baselines": baselines,
               "baseline_probability_applicability": {"last_value": False, "seasonal_naive": False},
               "environment": environment_fingerprint(),
               "execution": execution or {"target": "local", "training_backend": "numpy-cpu"},
               "adaptation": adaptation_evidence,
               "resource_observation": {
                   "kind": "measured", "cold_start_seconds": cold_start_seconds,
                   "inference_samples": len(latency_ms),
                   "inference_p50_ms": float(np.percentile(latency_ms, 50)),
                   "inference_p95_ms": inference_ms,
                   "inference_p99_ms": float(np.percentile(latency_ms, 99)),
               }}
    reference = model.save(artifacts, lineage)
    loaded = load_forecast_model(artifacts, reference, config)
    tolerance = 1e-5 if selected.family == "chronos2" else 1e-10
    if not np.allclose(prediction, loaded.predict(x), rtol=tolerance, atol=tolerance):
        raise GateError("MODEL_RELOAD_MISMATCH")
    if selected.family == "chronos2":
        reloaded_quantiles = loaded.predict_quantiles(x)
        if any(not np.allclose(quantile_predictions[level], reloaded_quantiles[level],
                               rtol=tolerance, atol=tolerance)
               for level in selected.quantile_levels):
            raise GateError("MODEL_RELOAD_MISMATCH")
    probe = artifacts.put_json({"input": x[0].tolist(), "prediction": prediction[0].tolist()}).to_dict()
    from .labels import PROTOCOL_VERSION
    from .config import digest
    evaluation = artifacts.put_json({"protocol_version": PROTOCOL_VERSION, "evaluation_asof": meta["as_of"],
        "issue_times": [int(raw.index[o - 1]) for o in origins],
        "target_times": [[int(t) for t in raw.index[o:o + config.horizon_steps]] for o in origins],
        "prediction": prediction.tolist(), "actual": actual.tolist(),
        "label_version": digest({"values": actual.tolist(), "maturity": meta["label_availability_by_time"],
                                  "availability": meta["availability_by_time"]}),
        "model": reference, "snapshot": snapshot_ref}).to_dict()
    return {"artifact": reference, "reload_probe": probe, "candidate": asdict(selected), "snapshot": snapshot_ref,
            "evaluation": evaluation, "training_regime": training_regime,
            "offline": score, "baselines": baselines, "champion_comparison": incumbent,
            "test_start": int(raw.index[end]), "test_end": int(raw.index[-1]), "inference_ms": inference_ms,
            "label_coverage": coverage, "probabilistic": probability,
            "baseline_probability_applicability": {"last_value": False, "seasonal_naive": False},
            "passed": not issues, "issues": issues, "validation_reports": reports,
            "holdout_operation": {"snapshot_sha": snapshot_ref["sha256"], "candidate_id": selected.candidate_id},
            "execution": execution or {"target": "local", "training_backend": "numpy-cpu"},
            "adaptation": adaptation_evidence, "resource_observation": lineage["resource_observation"]}
