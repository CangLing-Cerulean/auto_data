from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .config import CHRONOS_MODEL_REVISIONS, GateError, canonical, digest
from .contracts import ForecastResult


CHRONOS_VERSION = "2.3.1"
TESTED_RUNTIME = {
    "chronos-forecasting": CHRONOS_VERSION,
    "transformers": "5.16.1",
    "accelerate": "1.14.0",
    "peft": "0.18.1",
}
MODEL_REVISIONS = CHRONOS_MODEL_REVISIONS
MODEL_WEIGHT_SHA256 = {
    "amazon/chronos-2": "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42",
    "autogluon/chronos-2-small": "492290ae82bb89f9769e3479ce90b3179de1f33e600c34daa0352531538b23cd",
}
LORA_TARGET_MODULES = (
    "self_attention.q", "self_attention.v", "self_attention.k", "self_attention.o",
    "output_patch_embedding.output_layer",
)


@dataclass(frozen=True)
class AdapterCapabilities:
    multivariate: bool = True
    past_observed_covariates: bool = True
    known_future_covariates: bool = True
    quantiles: bool = True
    zero_shot: bool = True
    lora: bool = True
    full_finetune: bool = False
    exact_step_resume: bool = False
    safe_stage_retry: bool = True


@dataclass(frozen=True)
class Chronos2Spec:
    model_id: str
    revision: str
    adaptation_mode: str
    context_length: int
    prediction_length: int
    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 1
    learning_rate: float = 1e-5
    num_steps: int = 2
    lora_rank: int = 4
    seed: int = 42
    cross_learning: bool = False

    def __post_init__(self):
        object.__setattr__(self, "quantile_levels", tuple(float(v) for v in self.quantile_levels))
        if self.model_id not in MODEL_REVISIONS or self.revision != MODEL_REVISIONS[self.model_id]:
            raise GateError("CHRONOS_MODEL_OR_REVISION_NOT_ALLOWED")
        if self.adaptation_mode not in {"zero_shot", "lora"}:
            raise GateError("CHRONOS_ADAPTATION_MODE")
        if self.cross_learning is not False:
            raise GateError("CROSS_ENTITY_LEARNING_NOT_VALIDATED")
        if self.context_length < 8 or self.prediction_length < 1:
            raise GateError("CHRONOS_LENGTH")
        if self.quantile_levels != tuple(sorted(set(self.quantile_levels))) or 0.5 not in self.quantile_levels:
            raise GateError("CHRONOS_QUANTILES")
        if min(self.quantile_levels) <= 0 or max(self.quantile_levels) >= 1:
            raise GateError("CHRONOS_QUANTILES")
        if self.batch_size not in {1, 2, 4, 8}:
            raise GateError("CHRONOS_BATCH_ALLOWLIST")
        if self.learning_rate not in {1e-5, 5e-5} or self.num_steps not in {1, 2, 5, 10, 20, 50, 100}:
            raise GateError("CHRONOS_LORA_BUDGET")
        if self.lora_rank not in {4, 8, 16} or self.seed < 0:
            raise GateError("CHRONOS_LORA_ALLOWLIST")

    @property
    def candidate_id(self) -> str:
        return "chronos2-" + digest(asdict(self))[:20]


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def directory_files(root: str | Path) -> list[dict]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise GateError("MODEL_WEIGHTS_MISSING")
    rows = []
    # Path ordering differs between Windows and Linux. Keep the manifest order
    # stable so moving verified weights does not change their manifest hash.
    for path in sorted(base.rglob("*"), key=lambda item: (
            tuple(part.casefold() for part in item.relative_to(base).parts),
            item.relative_to(base).as_posix())):
        relative = path.relative_to(base).as_posix()
        if path.is_file() and path.name != "model-manifest.json" and not relative.startswith(".cache/"):
            rows.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def build_weight_manifest(root: str | Path, model_id: str, revision: str, *, kind: str = "base") -> dict:
    if model_id not in MODEL_REVISIONS or MODEL_REVISIONS[model_id] != revision:
        raise GateError("CHRONOS_MODEL_OR_REVISION_NOT_ALLOWED")
    files = directory_files(root)
    names = {item["path"] for item in files}
    if kind == "base" and not {"config.json", "model.safetensors"}.issubset(names):
        raise GateError("MODEL_WEIGHTS_MISSING")
    if kind == "base":
        weight = next(item for item in files if item["path"] == "model.safetensors")
        if weight["sha256"] != MODEL_WEIGHT_SHA256[model_id]:
            raise GateError("MODEL_WEIGHT_HASH_MISMATCH")
    body = {"schema": 1, "kind": kind, "model_id": model_id, "revision": revision,
            "chronos_forecasting": CHRONOS_VERSION, "files": files}
    return {**body, "manifest_sha256": digest(body)}


def write_weight_manifest(root: str | Path, model_id: str, revision: str, *, kind: str = "base") -> dict:
    base = Path(root).resolve()
    manifest = build_weight_manifest(base, model_id, revision, kind=kind)
    pending = base / ".model-manifest.pending"
    pending.write_text(canonical(manifest), encoding="utf-8")
    os.replace(pending, base / "model-manifest.json")
    return manifest


def verify_weight_manifest(root: str | Path, *, expected_model_id: str | None = None,
                           expected_revision: str | None = None, kind: str = "base") -> dict:
    base = Path(root).resolve()
    path = base / "model-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise GateError("MODEL_MANIFEST_MISSING_OR_INVALID") from None
    required = {"schema", "kind", "model_id", "revision", "chronos_forecasting", "files", "manifest_sha256"}
    if set(manifest) != required or manifest["schema"] != 1 or manifest["kind"] != kind:
        raise GateError("MODEL_MANIFEST_SCHEMA")
    body = {key: manifest[key] for key in required - {"manifest_sha256"}}
    if digest(body) != manifest["manifest_sha256"]:
        raise GateError("MODEL_MANIFEST_HASH_MISMATCH")
    if expected_model_id and manifest["model_id"] != expected_model_id:
        raise GateError("MODEL_ID_MISMATCH")
    if expected_revision and manifest["revision"] != expected_revision:
        raise GateError("MODEL_REVISION_MISMATCH")
    rebuilt = build_weight_manifest(base, manifest["model_id"], manifest["revision"], kind=kind)
    if rebuilt != manifest:
        raise GateError("MODEL_FILE_HASH_MISMATCH")
    return manifest


def dependency_status(require_lora: bool = False) -> dict:
    required = ["chronos-forecasting", "transformers", "accelerate"] + (["peft"] if require_lora else [])
    result = {}
    for name in required:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


class Chronos2Adapter:
    capabilities = AdapterCapabilities()

    def __init__(self, spec: Chronos2Spec, model_root: str | Path, *, device: str = "cpu",
                 pipeline_factory: Callable | None = None, manifest_kind: str = "base",
                 base_model_root: str | Path | None = None):
        self.spec = spec
        self.model_root = Path(model_root).resolve()
        self.device = device
        self.pipeline_factory = pipeline_factory
        if manifest_kind not in {"base", "lora"}:
            raise GateError("MODEL_MANIFEST_SCHEMA")
        self.manifest_kind = manifest_kind
        self.base_model_root = None if base_model_root is None else Path(base_model_root).resolve()
        self.pipeline = None
        self.manifest: dict | None = None
        self._cancel = threading.Event()

    def validate_config(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise GateError("CHRONOS_DEVICE")
        if self.spec.adaptation_mode == "lora" and not self.capabilities.lora:
            raise GateError("LORA_NOT_SUPPORTED")
        if self.capabilities.full_finetune:
            raise GateError("FULL_FINETUNE_POLICY")

    def resource_estimate(self) -> dict:
        manifest = verify_weight_manifest(self.model_root, expected_model_id=self.spec.model_id,
                                          expected_revision=self.spec.revision, kind=self.manifest_kind)
        package_mb = sum(item["size"] for item in manifest["files"]) / 1024 ** 2
        # This is a conservative admission estimate, never reported as a measurement.
        return {"kind": "estimate", "package_mb": package_mb,
                "minimum_ram_mb": math.ceil(package_mb * (6 if self.spec.adaptation_mode == "lora" else 3)),
                "minimum_vram_mb": math.ceil(package_mb * (5 if self.spec.adaptation_mode == "lora" else 2))}

    def _check_dependencies(self, *, require_lora: bool = False) -> dict:
        status = dependency_status(require_lora)
        missing = [name for name, version in status.items() if version is None]
        if missing:
            raise GateError("LORA_DEPENDENCY_MISSING" if require_lora and "peft" in missing else "CHRONOS_DEPENDENCY_MISSING")
        if status["chronos-forecasting"] != CHRONOS_VERSION:
            raise GateError("CHRONOS_VERSION_INCOMPATIBLE")
        return status

    def load(self) -> "Chronos2Adapter":
        self.validate_config()
        self.manifest = verify_weight_manifest(self.model_root, expected_model_id=self.spec.model_id,
                                               expected_revision=self.spec.revision, kind=self.manifest_kind)
        if self.pipeline_factory is not None:
            self.pipeline = self.pipeline_factory(self.model_root, self.device)
            return self
        self._check_dependencies(require_lora=self.spec.adaptation_mode == "lora")
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError:
            raise GateError("CHRONOS_DEPENDENCY_MISSING") from None
        if self.device == "cuda" and not torch.cuda.is_available():
            raise GateError("GPU_UNAVAILABLE")
        try:
            if self.manifest_kind == "lora":
                if self.base_model_root is None:
                    raise GateError("LORA_BASE_MODEL_ROOT_REQUIRED")
                verify_weight_manifest(self.base_model_root, expected_model_id=self.spec.model_id,
                                       expected_revision=self.spec.revision, kind="base")
                from peft import PeftModel
                base = Chronos2Pipeline.from_pretrained(
                    str(self.base_model_root), device_map=self.device, local_files_only=True)
                model = PeftModel.from_pretrained(
                    base.model, str(self.model_root), is_trainable=False, local_files_only=True)
                self.pipeline = Chronos2Pipeline(model=model)
                if not hasattr(self.pipeline.model, "peft_config"):
                    raise GateError("LORA_ADAPTER_NOT_ACTIVE")
            else:
                self.pipeline = Chronos2Pipeline.from_pretrained(
                    str(self.model_root), device_map=self.device, local_files_only=True)
        except GateError:
            raise
        except OSError:
            raise GateError("MODEL_OFFLINE_LOAD_FAILED") from None
        except Exception:
            raise GateError("CHRONOS_VERSION_OR_MODEL_INCOMPATIBLE") from None
        return self

    def _payload(self, context: np.ndarray, past_covariates: Mapping[str, np.ndarray] | None,
                 future_covariates: Mapping[str, np.ndarray] | None) -> dict:
        value = np.asarray(context, dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != self.spec.context_length or not np.isfinite(value).all():
            raise GateError("CHRONOS_INPUT_SHAPE_OR_VALUE")
        past = {} if past_covariates is None else {str(k): np.asarray(v) for k, v in past_covariates.items()}
        future = {} if future_covariates is None else {str(k): np.asarray(v) for k, v in future_covariates.items()}
        if set(future) - set(past):
            raise GateError("FUTURE_COVARIATE_WITHOUT_HISTORY")
        if any(v.ndim != 1 or len(v) != self.spec.context_length or not np.isfinite(v.astype(float)).all() for v in past.values()):
            raise GateError("PAST_COVARIATE_CONTRACT")
        if any(v.ndim != 1 or len(v) != self.spec.prediction_length or not np.isfinite(v.astype(float)).all() for v in future.values()):
            raise GateError("FUTURE_COVARIATE_CONTRACT")
        payload = {"target": value.T}
        if past:
            payload["past_covariates"] = past
        if future:
            payload["future_covariates"] = future
        return payload

    def predict_array(self, context: np.ndarray, *, variables: tuple[str, ...],
                      past_covariates: Mapping[str, np.ndarray] | None = None,
                      future_covariates: Mapping[str, np.ndarray] | None = None) -> dict[float, np.ndarray]:
        if self.pipeline is None or self.manifest is None:
            raise GateError("MODEL_NOT_LOADED")
        if np.asarray(context).ndim != 2 or np.asarray(context).shape[1] != len(variables):
            raise GateError("VARIABLE_ORDER_OR_SHAPE")
        payload = self._payload(context, past_covariates, future_covariates)
        try:
            quantiles, _ = self.pipeline.predict_quantiles(
                inputs=[payload], prediction_length=self.spec.prediction_length,
                quantile_levels=list(self.spec.quantile_levels), batch_size=self.spec.batch_size,
                context_length=self.spec.context_length, cross_learning=False,
                limit_prediction_length=True,
            )
            raw = quantiles[0]
            raw = raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)
        except GateError:
            raise
        except Exception:
            raise GateError("CHRONOS_PREDICTION_FAILED") from None
        expected = (len(variables), self.spec.prediction_length, len(self.spec.quantile_levels))
        if raw.shape != expected or not np.isfinite(raw).all():
            raise GateError("CHRONOS_OUTPUT_SHAPE_OR_VALUE")
        by_level = {level: raw[:, :, i].T.copy() for i, level in enumerate(self.spec.quantile_levels)}
        ordered = np.stack([by_level[level] for level in self.spec.quantile_levels])
        if np.any(np.diff(ordered, axis=0) < -1e-6):
            raise GateError("CHRONOS_QUANTILE_CROSSING")
        return by_level

    def predict(self, context: np.ndarray, *, variables: tuple[str, ...], entity_id: str,
                feature_cutoff: int, forecast_origin: int, generated_at: int, frequency_ns: int,
                input_snapshot: str, feature_version: str,
                past_covariates: Mapping[str, np.ndarray] | None = None,
                future_covariates: Mapping[str, np.ndarray] | None = None) -> ForecastResult:
        by_level = self.predict_array(context, variables=variables, past_covariates=past_covariates,
                                      future_covariates=future_covariates)
        artifact_id = self.manifest["manifest_sha256"]
        return ForecastResult(
            model_id=self.spec.candidate_id, artifact_id=artifact_id, entity_id=entity_id,
            variables=variables, feature_cutoff=feature_cutoff, forecast_origin=forecast_origin,
            generated_at=generated_at,
            target_times=tuple(forecast_origin + (i + 1) * frequency_ns for i in range(self.spec.prediction_length)),
            frequency_ns=frequency_ns, point=by_level[0.5], quantiles=by_level,
            input_snapshot=input_snapshot, feature_version=feature_version,
        )

    def cancel(self) -> None:
        self._cancel.set()

    def adapt_lora(self, train_values: np.ndarray, output_dir: str | Path,
                   *, validation_values: np.ndarray | None = None) -> dict:
        if self.spec.adaptation_mode != "lora":
            raise GateError("LORA_MODE_REQUIRED")
        self._check_dependencies(require_lora=True)  # blocks the upstream silent full-finetune fallback
        if self.pipeline is None or self.manifest is None:
            self.load()
        if self._cancel.is_set():
            raise GateError("TRAINING_CANCELLED")
        train = np.asarray(train_values, dtype=np.float32).copy()
        if train.ndim != 2 or train.shape[0] < self.spec.context_length + self.spec.prediction_length or not np.isfinite(train).all():
            raise GateError("LORA_TRAINING_DATA")
        validation = None if validation_values is None else np.asarray(validation_values, dtype=np.float32).copy()
        if validation is not None and (validation.ndim != 2 or validation.shape[1] != train.shape[1] or not np.isfinite(validation).all()):
            raise GateError("LORA_VALIDATION_DATA")
        destination = Path(output_dir).resolve()
        if destination.exists() and any(destination.iterdir()):
            raise GateError("LORA_OUTPUT_NOT_EMPTY")
        destination.mkdir(parents=True, exist_ok=True)
        base_before = verify_weight_manifest(self.model_root, expected_model_id=self.spec.model_id,
                                             expected_revision=self.spec.revision)
        probe = train[-self.spec.context_length:]
        base_result = self.predict(
            probe, variables=tuple(f"v{i}" for i in range(train.shape[1])), entity_id="training-probe",
            feature_cutoff=0, forecast_origin=0, generated_at=0, frequency_ns=1,
            input_snapshot="training-probe", feature_version="training-probe")
        try:
            import torch
            from transformers import TrainerCallback, EarlyStoppingCallback
        except ImportError:
            raise GateError("LORA_DEPENDENCY_MISSING") from None

        logs: list[dict] = []
        cancel_flag = self._cancel
        progress = {"optimizer_steps": 0, "best_step": None, "best_internal_validation_loss": None}
        early_stop_patience, early_stop_delta = 3, 1e-4

        class EvidenceCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    safe = {key: float(value) for key, value in logs.items()
                            if isinstance(value, (int, float)) and math.isfinite(value)}
                    safe["step"] = int(state.global_step)
                    locals_logs.append(safe)

            def on_step_end(self, args, state, control, **kwargs):
                progress["optimizer_steps"] = int(state.global_step)
                if cancel_flag.is_set():
                    control.should_training_stop = True
                return control

            def on_train_end(self, args, state, control, **kwargs):
                progress["optimizer_steps"] = int(state.global_step)
                progress["best_internal_validation_loss"] = state.best_metric
                if state.best_model_checkpoint:
                    progress["best_step"] = int(Path(state.best_model_checkpoint).name.rsplit("-", 1)[-1])
                return control

        locals_logs = logs
        torch.manual_seed(self.spec.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.spec.seed)
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            adapted = self.pipeline.fit(
                inputs=[train.T], prediction_length=self.spec.prediction_length,
                validation_inputs=None if validation is None else [validation.T],
                finetune_mode="lora",
                lora_config={"r": self.spec.lora_rank, "lora_alpha": 2 * self.spec.lora_rank,
                             "target_modules": list(LORA_TARGET_MODULES)},
                context_length=self.spec.context_length, learning_rate=self.spec.learning_rate,
                num_steps=self.spec.num_steps, batch_size=self.spec.batch_size,
                min_past=self.spec.prediction_length, output_dir=destination,
                callbacks=[EvidenceCallback()] + ([EarlyStoppingCallback(
                    early_stopping_patience=early_stop_patience, early_stopping_threshold=early_stop_delta)]
                    if validation is not None else []), remove_printer_callback=True,
                logging_steps=1, eval_steps=1, save_steps=1, disable_tqdm=True,
            )
        except RuntimeError as error:
            if "out of memory" in str(error).casefold():
                raise GateError("GPU_OOM") from None
            raise GateError("LORA_TRAINING_FAILED") from None
        except Exception:
            raise GateError("LORA_TRAINING_FAILED") from None
        duration = time.perf_counter() - started
        if self._cancel.is_set():
            raise GateError("TRAINING_CANCELLED")
        if not hasattr(adapted.model, "peft_config") or not hasattr(adapted.model, "get_nb_trainable_parameters"):
            raise GateError("LORA_SILENT_FULL_FINETUNE_BLOCKED")
        trainable, total = adapted.model.get_nb_trainable_parameters()
        if not 0 < int(trainable) < int(total):
            raise GateError("LORA_TRAINABLE_PARAMETER_SCOPE")
        checkpoint = destination / "finetuned-ckpt"
        peak_vram = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        if verify_weight_manifest(self.model_root, expected_model_id=self.spec.model_id,
                                  expected_revision=self.spec.revision) != base_before:
            raise GateError("BASE_MODEL_MUTATED")
        # Trainer/Accelerate may leave inference hooks around the in-memory training copy.
        # Never evaluate or serve that mutable instance: reload the just-saved adapter onto
        # a freshly verified base through an explicit local PEFT chain.
        # PEFT writes the local absolute base path into adapter_config.json.  The
        # loader supplies an already verified base model explicitly, so retain only
        # immutable identity here to keep the adapter package portable.
        adapter_config_path = checkpoint / "adapter_config.json"
        try:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            adapter_config["base_model_name_or_path"] = f"{self.spec.model_id}@{self.spec.revision}"
            pending_config = checkpoint / ".adapter_config.pending"
            pending_config.write_text(canonical(adapter_config), encoding="utf-8")
            os.replace(pending_config, adapter_config_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise GateError("LORA_ADAPTER_CONFIG_INVALID") from None
        write_weight_manifest(checkpoint, self.spec.model_id, self.spec.revision, kind="lora")
        self.pipeline = None
        clean = Chronos2Adapter(
            self.spec, checkpoint, device=self.device, manifest_kind="lora",
            base_model_root=self.model_root,
        ).load()
        self.pipeline = clean.pipeline
        self.manifest = clean.manifest
        adapted_result = self.predict(
            probe, variables=tuple(f"v{i}" for i in range(train.shape[1])), entity_id="training-probe",
            feature_cutoff=0, forecast_origin=0, generated_at=0, frequency_ns=1,
            input_snapshot="training-probe", feature_version="training-probe")
        changed = not np.allclose(base_result.point, adapted_result.point, rtol=1e-7, atol=1e-7)
        if not changed:
            raise GateError("LORA_NO_WEIGHT_EFFECT")
        # The official high-level API saves model-only checkpoints, so exact optimizer-step
        # resume is intentionally not claimed; a retry restarts the charged stage from the base.
        evidence = {
            "schema": 1, "adaptation_mode": "lora", "base_manifest_sha256": base_before["manifest_sha256"],
            "trainable_parameters": int(trainable), "total_parameters": int(total),
            "trainable_fraction": float(trainable / total), **progress,
            "requested_steps": self.spec.num_steps,
            "termination_reason": "internal_validation_stagnation" if progress["optimizer_steps"] < self.spec.num_steps else "step_budget_reached",
            "checkpoint_selection": "best_internal_validation" if validation is not None else "last_step",
            "early_stopping": {"patience": early_stop_patience, "minimum_delta": early_stop_delta,
                               "validation_scope": "inside_training_segment"},
            "convergence_claimed": False,
            "loss_log": logs, "seed": self.spec.seed, "learning_rate": self.spec.learning_rate,
            "lora_rank": self.spec.lora_rank, "target_modules": list(LORA_TARGET_MODULES),
            "context_length": self.spec.context_length, "prediction_length": self.spec.prediction_length,
            "train_rows": len(train), "validation_rows": 0 if validation is None else len(validation),
            "duration_seconds": duration, "peak_vram_bytes": peak_vram,
            "base_prediction_sha256": digest(base_result.point.tolist()),
            "adapted_prediction_sha256": digest(adapted_result.point.tolist()),
            "weight_effect_verified": changed, "base_unchanged_verified": True,
            "clean_reload_used_for_evaluation": True,
            "resume_mode": "charged_stage_restart_from_locked_base",
        }
        (checkpoint / "training-evidence.json").write_text(canonical(evidence), encoding="utf-8")
        # Evidence is immutable package content; it deliberately does not contain the
        # package hash because that would create an impossible self-referential digest.
        package_manifest = write_weight_manifest(checkpoint, self.spec.model_id, self.spec.revision, kind="lora")
        self.manifest = package_manifest
        return {"checkpoint": str(checkpoint), "manifest": package_manifest,
                "base_manifest": base_before, "evidence": evidence}


def spec_for_candidate(candidate, config) -> Chronos2Spec:
    return Chronos2Spec(
        model_id=candidate.model_id,
        revision=candidate.revision,
        adaptation_mode=candidate.adaptation_mode,
        context_length=candidate.input_steps,
        prediction_length=config.horizon_steps,
        quantile_levels=candidate.quantile_levels,
        batch_size=candidate.batch_size,
        learning_rate=candidate.learning_rate,
        num_steps=candidate.num_steps,
        lora_rank=candidate.lora_rank,
        seed=config.seed,
        cross_learning=False,
    )


class ChronosForecastModel:
    """Engine-compatible immutable serving wrapper around one verified Chronos package."""

    def __init__(self, payload: dict, adapter: Chronos2Adapter):
        from .models import Candidate
        self.payload = payload
        self.candidate = Candidate.from_dict(payload["candidate"])
        self.variables = tuple(payload["variables"])
        self.horizon = int(payload["horizon_steps"])
        self.mean = np.asarray(payload["scaler"]["mean"], dtype=float)
        self.scale = np.asarray(payload["scaler"]["std"], dtype=float)
        self.adapter = adapter
        if (self.mean.shape != (len(self.variables),) or self.scale.shape != self.mean.shape
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)):
            raise GateError("MODEL_SCALER")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_quantiles(x)[0.5]

    def predict_quantiles(self, x: np.ndarray) -> dict[float, np.ndarray]:
        value = np.asarray(x, dtype=float)
        single = value.ndim == 2
        if single:
            value = value[None, :, :]
        expected = (self.candidate.input_steps, len(self.variables))
        if value.ndim != 3 or value.shape[1:] != expected or not np.isfinite(value).all():
            raise GateError("PREDICTION_INPUT_CONTRACT")
        rows = [self.adapter.predict_array(item, variables=self.variables) for item in value]
        result = {level: np.stack([row[level] for row in rows])
                  for level in self.candidate.quantile_levels}
        expected_shape = (len(value), self.horizon, len(self.variables))
        if any(values.shape != expected_shape or not np.isfinite(values).all()
               for values in result.values()):
            raise GateError("INVALID_PREDICTION")
        return {level: values[0] if single else values for level, values in result.items()}

    def save(self, artifacts, metadata: dict) -> dict:
        return artifacts.put_json({**self.payload, "lineage": metadata}).to_dict()

    @staticmethod
    def _package_root(payload: dict, artifacts, config) -> tuple[Path, str]:
        package = payload["model_package"]
        kind = package.get("kind")
        if kind == "base":
            try:
                root = Path(config.foundation.model_roots[payload["candidate"]["model_id"]]).resolve()
            except (KeyError, TypeError):
                raise GateError("MODEL_PACKAGE_INVALID") from None
        elif kind == "lora":
            relative = package.get("artifact_relative_path")
            if not isinstance(relative, str) or not relative:
                raise GateError("MODEL_PACKAGE_INVALID")
            root = (artifacts.root / relative).resolve()
            try:
                root.relative_to(artifacts.root.resolve())
            except ValueError:
                raise GateError("MODEL_PACKAGE_PATH_ESCAPE") from None
        else:
            raise GateError("MODEL_PACKAGE_INVALID")
        return root, kind

    @classmethod
    def load_payload(cls, payload: dict, artifacts, config) -> "ChronosForecastModel":
        from .models import Candidate
        required = {"schema", "backend", "candidate", "variables", "horizon_steps",
                    "frequency_seconds", "config_version", "scaler", "model_package", "lineage"}
        if (set(payload) != required or payload.get("schema") != 2 or payload.get("backend") != "chronos2"
                or payload.get("config_version") != config.version
                or payload.get("variables") != list(config.variables)
                or payload.get("horizon_steps") != config.horizon_steps
                or payload.get("frequency_seconds") != config.frequency_seconds):
            raise GateError("MODEL_CONTRACT")
        candidate = Candidate.from_dict(payload["candidate"])
        candidate.validate(config)
        root, kind = cls._package_root(payload, artifacts, config)
        adapter = Chronos2Adapter(
            spec_for_candidate(candidate, config), root, device=config.foundation.device,
            manifest_kind=kind,
            base_model_root=config.foundation.model_roots[candidate.model_id] if kind == "lora" else None,
        ).load()
        package = payload["model_package"]
        if adapter.manifest["manifest_sha256"] != package.get("manifest_sha256"):
            raise GateError("MODEL_PACKAGE_MANIFEST_MISMATCH")
        return cls(payload, adapter)

    @classmethod
    def build(cls, candidate, config, train_raw, adapter: Chronos2Adapter,
              *, package: dict, lineage: dict) -> "ChronosForecastModel":
        from .models import SCALE_FLOOR
        candidate.validate(config)
        raw = train_raw.to_numpy(dtype=float)
        mean = np.nanmean(raw, axis=0)
        scale = np.maximum(np.nanstd(raw, axis=0), SCALE_FLOOR)
        payload = {
            "schema": 2, "backend": "chronos2", "candidate": asdict(candidate),
            "variables": list(config.variables), "horizon_steps": config.horizon_steps,
            "frequency_seconds": config.frequency_seconds, "config_version": config.version,
            "scaler": {"mean": mean.tolist(), "std": scale.tolist(),
                       "fit_start": int(train_raw.index[0]), "fit_end": int(train_raw.index[-1]),
                       "unique_points": len(train_raw)},
            "model_package": package, "lineage": lineage,
        }
        return cls(payload, adapter)


class AdapterCache:
    """Small version-keyed cache; mutable training adapters are never inserted."""

    def __init__(self, maximum: int = 1):
        if maximum < 1:
            raise GateError("ADAPTER_CACHE_SIZE")
        self.maximum = maximum
        self._values: OrderedDict[str, Chronos2Adapter] = OrderedDict()

    def get_or_load(self, key: str, factory: Callable[[], Chronos2Adapter]) -> Chronos2Adapter:
        if key in self._values:
            self._values.move_to_end(key)
            return self._values[key]
        value = factory().load()
        self._values[key] = value
        while len(self._values) > self.maximum:
            self._values.popitem(last=False)
        return value
