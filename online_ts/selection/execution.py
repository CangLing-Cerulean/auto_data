"""Deterministic local worker boundary. No SSH, automatic installation or model-written commands."""
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from ..config import GateError, digest
from .common import ROOT, STUDY, StudyConfig, file_hash, read_json, write_json
from .rules import mask_hash


class CandidateEvaluator:
    def __init__(self, repo, data, freeze_path):
        self.repo, self.data, self.freeze_path = repo, data, str(Path(freeze_path).resolve())

    def __call__(self, mask, seed):
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            raise GateError("TORCH_REQUIRED_ON_TRAINING_HOST")
        import torch
        if not torch.cuda.is_available():
            raise GateError("CUDA_REQUIRED_FOR_FROZEN_TRAINER")
        if self.repo.frozen["dependencies"]["torch"] != torch.__version__:
            raise GateError("FREEZE_ON_THE_REGISTERED_TRAINING_RUNTIME")
        config = self.repo.config
        scientific = {"task": self.data.task["features_sha256"], "protocol": self.data.task["protocol_sha256"],
                      "manifest": self.data.task["manifest_sha256"], "mask": mask_hash(mask), "seed": seed,
                      "worker": file_hash(ROOT / "online_ts/selection/trainer.py"),
                      "core": file_hash(ROOT / "scripts/run_research.py"),
                      "dependencies": self.repo.frozen["dependencies"], "python": self.repo.frozen["python"],
                      "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
                      "sampling": "uniform-retained-common-inverse-cdf-v1"}
        job_id = digest(scientific)
        folder = STUDY / "jobs" / job_id
        result_path = folder / "result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result["scientific_identity"] != scientific:
                raise GateError("CACHED_RESULT_IDENTITY")
            if any(file_hash(folder / name) != expected for name, expected in result["assets"].items()):
                raise GateError("CACHED_RESULT_ASSET")
            self.record_cost(job_id, 0., True)
            return {**result["metrics"], "job_id": job_id, "cache_hit": True,
                    "original_elapsed_seconds": result["elapsed_seconds"]}
        if (folder / "failure.json").exists():
            raise GateError("FAILED_JOB_REQUIRES_EXPLICIT_PLATFORM_REVIEW")
        owner = uuid.uuid4().hex
        if not self.repo.store.acquire_resource("selection-gpu", owner, self.repo.key, job_id, "training", config.candidate_seconds + 30):
            raise GateError("TRAINING_RESOURCE_BUSY")
        started = time.monotonic()
        try:
            elapsed = sum(r["seconds"] for r in self.repo.iter_records("training_cost"))
            limit = config.confirmation_seconds if self.repo.run_id == "development_confirmation" else config.total_seconds
            if elapsed + config.candidate_seconds > limit:
                raise GateError("TRAINING_TIME_RESERVATION")
            folder.mkdir(parents=True, exist_ok=True)
            mask_path = folder / "mask.npz"
            if not mask_path.exists():
                np.savez_compressed(mask_path, retained=mask, start_row=self.data.starts)
            job = {"job_id": job_id, "scientific_identity": scientific,
                   "freeze_path": self.freeze_path, "mask_sha256": file_hash(mask_path),
                   "seed": seed, "time_limit_seconds": config.candidate_seconds}
            if (folder / "job.json").exists() and read_json(folder / "job.json") != job:
                raise GateError("JOB_RESUME_IDENTITY")
            write_json(folder / "job.json", job)
            with (folder / "worker.log").open("a", encoding="utf-8") as log:
                process = subprocess.run([sys.executable, "-B", "-m", "online_ts.selection.trainer", "--job", str(folder / "job.json")],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=config.candidate_seconds)
            if process.returncode or not result_path.exists():
                raise GateError("TRAINING_WORKER_FAILED")
            result = read_json(result_path)
            if result["scientific_identity"] != scientific:
                raise GateError("WORKER_RESULT_IDENTITY")
            return {**result["metrics"], "job_id": job_id, "cache_hit": False,
                    "elapsed_seconds": time.monotonic() - started}
        except Exception as error:
            if folder.exists():
                write_json(folder / "failure.json", {"error": str(error) if isinstance(error, GateError) else type(error).__name__,
                    "scientific_identity": scientific, "elapsed_seconds": time.monotonic() - started})
            raise
        finally:
            self.record_cost(job_id, time.monotonic() - started, False)
            self.repo.store.release_resource("selection-gpu", owner)

    def record_cost(self, job_id, seconds, cached):
        record_id = "cost-" + uuid.uuid4().hex
        self.repo.put(record_id, "training_cost", {"record_id": record_id, "job_id": job_id,
                      "seconds": seconds, "cache_hit": cached}, time.time_ns())


def planned_runs(config):
    return [{"run_id": f"{arm}_{i + 1:02d}", "arm": arm} for i in range(config.independent_runs)
            for arm in ("random", "agent")]


def preflight(frozen):
    import os
    import importlib.util
    available = importlib.util.find_spec("torch") is not None
    cuda, version = False, None
    if available:
        import torch
        cuda, version = torch.cuda.is_available(), torch.__version__
    return {"api_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY")), "torch_installed": available,
            "cuda_available": cuda, "torch_version": version,
            "training_ready": cuda and version == frozen["dependencies"]["torch"],
            "independent_evaluation_ready": False, "scope": "development_only",
            "network_calls": 0, "training_started": False}


def freeze_batch(frozen, run_ids, destination):
    """Evaluator boundary: require EVERY preregistered submission before confirmation."""
    expected = [r["run_id"] for r in planned_runs(StudyConfig(**frozen["config"]))]
    if sorted(run_ids) != sorted(expected) or len(set(run_ids)) != len(run_ids):
        raise GateError("ALL_PREREGISTERED_RUNS_REQUIRED")
    submissions = []
    for run_id in run_ids:
        folder = STUDY / "runs" / run_id
        policy_path = folder / "final_policy.json"
        if not policy_path.exists():
            raise GateError("BATCH_SUBMISSION_MISSING:" + run_id)
        policy = read_json(policy_path)
        if policy["harness_sha256"] != frozen["sha256"] or file_hash(folder / "selected_windows.npz") != policy["selected_windows_sha256"]:
            raise GateError("BATCH_SUBMISSION_IDENTITY")
        submissions.append({"run_id": run_id, "policy_sha256": file_hash(policy_path),
                            "windows_sha256": policy["selected_windows_sha256"]})
    batch = {"harness_sha256": frozen["sha256"], "submissions": submissions,
             "confirmation_seeds": frozen["config"]["confirmation_seeds"], "membership_seeds": [20261021, 20261022, 20261023],
             "scope": "development_confirmation_only", "E": "absent"}
    batch["sha256"] = digest(batch)
    destination = Path(destination)
    if destination.exists() and read_json(destination) != batch:
        raise GateError("BATCH_ALREADY_FROZEN")
    write_json(destination, batch)
    return batch


def confirm(freeze_path, batch_path):
    """Explicit evaluator command; confirmation reports never enter agent context."""
    from .common import verify_freeze
    from .dataset import TrainingData
    from .rules import matched_controls
    from .runtime import StudyRepository
    frozen = verify_freeze(freeze_path)
    batch = read_json(batch_path)
    if digest({k: v for k, v in batch.items() if k != "sha256"}) != batch["sha256"]:
        raise GateError("BATCH_HASH")
    # Rechecking the full batch prevents confirm-after-only-one-final leakage.
    actual = freeze_batch(frozen, [s["run_id"] for s in batch["submissions"]], batch_path)
    if actual != batch:
        raise GateError("BATCH_CHANGED")
    data = TrainingData(frozen["task"])
    repo = StudyRepository(frozen, "development_confirmation", "uniform")
    evaluator = CandidateEvaluator(repo, data, freeze_path)
    output = STUDY / "confirmation" / batch["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in batch["confirmation_seeds"]:
        baseline = evaluator(np.ones(len(data.starts), bool), seed)
        rows.append({"run_id": "B0", "kind": "uniform", "seed": seed, "metrics": baseline})
        write_json(output / "results.json", {"scope": batch["scope"], "complete": False, "rows": rows})
        for submission in batch["submissions"]:
            with np.load(STUDY / "runs" / submission["run_id"] / "selected_windows.npz", allow_pickle=False) as archive:
                mask = archive["retained"]
                if not np.array_equal(archive["start_row"], data.starts):
                    raise GateError("CONFIRMATION_WINDOW_MAP")
            metrics = evaluator(mask, seed)
            rows.append({"run_id": submission["run_id"], "kind": "selected", "seed": seed, "metrics": metrics})
            for member_seed in batch["membership_seeds"]:
                for kind, control in matched_controls(mask, data, member_seed).items():
                    rows.append({"run_id": submission["run_id"], "kind": kind, "member_seed": member_seed,
                                 "seed": seed, "metrics": evaluator(control, seed)})
                    write_json(output / "results.json", {"scope": batch["scope"], "complete": False, "rows": rows})
    write_json(output / "results.json", {"scope": batch["scope"], "complete": True, "rows": rows})
    return {"path": str(output / "results.json"), "rows": len(rows), "scope": batch["scope"]}
