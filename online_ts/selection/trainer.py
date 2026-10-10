"""Fixed original DLinear trainer. Uses only train/validation; no final-test command."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..config import GateError
from .common import ROOT, STUDY, file_hash, read_json, write_json, verify_freeze
from .rules import mask_hash


def train(job_path):
    folder = Path(job_path).resolve().parent
    if not folder.is_relative_to((STUDY / "jobs").resolve()):
        raise GateError("TRAINING_JOB_SCOPE")
    job = read_json(job_path)
    frozen = verify_freeze(job["freeze_path"])
    task, p = frozen["task"], frozen["task"]["protocol"]
    if p != read_json(ROOT / "research_runs/protocol.json"):
        raise GateError("FIXED_PROTOCOL_CHANGED")
    if file_hash(ROOT / "data/metropt3_v1/manifest.json") != task["manifest_sha256"]:
        raise GateError("DATASET_MANIFEST_CHANGED")
    if file_hash(folder / "mask.npz") != job["mask_sha256"]:
        raise GateError("TRAINING_MASK_HASH")
    # Integrity verification may hash test bytes; it produces no model feedback.
    import subprocess
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify_dataset.py")], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    import torch
    if frozen["dependencies"]["torch"] != torch.__version__:
        raise GateError("FREEZE_ON_THE_REGISTERED_TRAINING_RUNTIME")
    from scripts.run_research import DLinear, prepare, evaluate
    if not torch.cuda.is_available():
        raise GateError("CUDA_REQUIRED_FOR_FROZEN_TRAINER")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(job["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arrays, starts, groups, audit = prepare(p, read_json(ROOT / "data/metropt3_v1/manifest.json"))
    with np.load(folder / "mask.npz", allow_pickle=False) as saved:
        mask = saved["retained"].copy()
        if (mask.dtype != np.bool_ or not np.array_equal(saved["start_row"], starts["train"])
                or mask.shape != starts["train"].shape or not mask.any()):
            raise GateError("TRAINING_MASK_MAPPING")
    if mask_hash(mask) != job["scientific_identity"]["mask"]:
        raise GateError("TRAINING_MASK_IDENTITY")
    probability = mask.astype(np.float64) / mask.sum()
    data = {s: torch.from_numpy(v).cuda() for s, v in arrays.items()}
    indices = {s: torch.from_numpy(v).cuda() for s, v in starts.items()}
    cdf = torch.from_numpy(np.cumsum(probability)).cuda()
    cdf[-1] = 1.
    model = DLinear(p).cuda()
    if sum(v.numel() for v in model.parameters()) != 7920:
        raise GateError("FIXED_MODEL_PARAMETER_COUNT")
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    rng = torch.Generator(device="cuda").manual_seed(job["seed"] + 1000)
    counts = torch.zeros(len(mask), dtype=torch.int64, device="cuda")
    step0, best, best_step, history, previous_seconds = 0, float("inf"), 0, [], 0.
    if (folder / "latest.pt").exists():
        saved = torch.load(folder / "latest.pt", map_location="cuda", weights_only=False)
        if saved["scientific_identity"] != job["scientific_identity"]:
            raise GateError("TRAINING_CHECKPOINT_IDENTITY")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["sampling_rng"].cpu())
        counts, step0, best, best_step, history, previous_seconds = (
            saved[k] for k in ("draw_counts", "step", "best_score", "best_step", "history", "elapsed_seconds"))
    offsets = torch.arange(p["input_length"] + p["prediction_length"], device="cuda")
    for step in range(step0 + 1, p["steps"] + 1):
        if previous_seconds + time.monotonic() - started >= job["time_limit_seconds"]:
            raise GateError("CANDIDATE_WALL_TIME_LIMIT")
        model.train()
        uniform = torch.rand(p["batch_size"], dtype=torch.float64, device="cuda", generator=rng)
        sampled = torch.searchsorted(cdf, uniform, right=True)
        counts.scatter_add_(0, sampled, torch.ones_like(sampled))
        batch = data["train"][indices["train"][sampled, None] + offsets]
        x, y = batch[:, :p["input_length"]], batch[:, p["input_length"]:]
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean((1, 2)).mean()
        if not torch.isfinite(loss):
            raise GateError("NONFINITE_LOSS")
        loss.backward()
        optimizer.step()
        if step % p["validation_every"] == 0:
            metrics = evaluate(model, data["validation"], indices["validation"], groups, p, "baseline")
            score = metrics["groups"]["all"]["NMSE"]
            history.append({"step": step, "metrics": metrics["groups"]["all"]})
            if score < best:
                best, best_step = score, step
                torch.save(model.state_dict(), folder / "best.tmp")
                (folder / "best.tmp").replace(folder / "best.pt")
            torch.save({"scientific_identity": job["scientific_identity"], "step": step,
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "sampling_rng": rng.get_state(),
                        "draw_counts": counts, "best_score": best, "best_step": best_step, "history": history,
                        "elapsed_seconds": previous_seconds + time.monotonic() - started}, folder / "latest.tmp")
            (folder / "latest.tmp").replace(folder / "latest.pt")
            write_json(folder / "history.json", history)
    model.load_state_dict(torch.load(folder / "best.pt", weights_only=True))
    metrics = evaluate(model, data["validation"], indices["validation"], groups, p, "baseline")
    draws = counts.cpu().numpy()
    if draws.sum() != p["steps"] * p["batch_size"] or np.any(draws[~mask]):
        raise GateError("SAMPLING_EXPOSURE")
    np.savez_compressed(folder / "sampling.npz", probability=probability, draw_count=draws, start_row=starts["train"])
    write_json(folder / "data_audit.json", audit)
    result = {"scientific_identity": job["scientific_identity"], "metrics": metrics["groups"]["all"],
              "groups": metrics["groups"], "per_channel_NMSE": metrics["per_channel_NMSE"],
              "per_channel_NMAE": metrics["per_channel_NMAE"], "best_step": best_step,
              "steps": p["steps"], "elapsed_seconds": previous_seconds + time.monotonic() - started,
              "scope": "development_only", "parameters": 7920,
              "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
                              "device": torch.cuda.get_device_name()},
              "assets": {n: file_hash(folder / n) for n in ("mask.npz", "best.pt", "latest.pt", "sampling.npz", "history.json", "data_audit.json")}}
    write_json(folder / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    train(args.job)
