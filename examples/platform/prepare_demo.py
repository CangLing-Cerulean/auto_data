"""Create fresh synthetic live/replay configurations under outputs/."""
import argparse
import csv
import datetime
import getpass
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/platform_demo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    folder = Path(args.output).resolve()
    if not folder.is_relative_to(root / "outputs") or folder == root / "outputs":
        raise ValueError("Choose a new directory within this project's outputs/")
    folder.mkdir(parents=True, exist_ok=False)
    config = {
        "task_id": "demo_stream", "variables": ["signal"], "mode": "live",
        "source": {"kind": "history_http", "url_env": "PLATFORM_STREAM_URL",
                   "timeout_seconds": 5, "retries": 1},
        "frequency_seconds": 1, "horizon_steps": 2, "allowed_lateness_seconds": 0,
        "search": {"input_steps": 8, "history_steps": 64, "validation_steps": 12,
                   "test_steps": 12, "seasonal_period": 1, "ridge_alphas": [1.],
                   "model_families": ["last_value", "ridge"], "max_train_samples": 64},
        "budget": {"max_candidates": 2, "max_candidate_seconds": 20,
                   "max_training_seconds": 80, "max_memory_mb": 1024, "max_attempts": 1},
        "monitoring": {"evaluation_seconds": 8, "check_seconds": 2,
                       "cooldown_seconds": 60, "minimum_new_steps": 16,
                       "max_model_age_seconds": 300},
        "release": {"minimum_origins": 4, "shadow_seconds": 12, "canary_seconds": 12,
                    "post_promotion_seconds": 12, "maximum_observation_seconds": 120},
        "collaboration": {"permissions": {getpass.getuser(): ["read", "experiment", "release"]}},
    }
    (folder / "live.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    config.update(task_id="demo_replay", mode="replay",
                  source={"kind": "file", "path": "synthetic.csv", "timestamp_column": "timestamp"})
    (folder / "replay.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    origin = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    with (folder / "synthetic.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "signal"])
        for i in range(480):
            writer.writerow([(origin + datetime.timedelta(seconds=i)).isoformat(),
                             10 + .03 * i + .1 * math.sin(i / 10)])
    print(json.dumps({"directory": str(folder), "live_warmup_seconds": 112,
                      "data": "synthetic", "external_model_calls": 0}))


if __name__ == "__main__":
    main()
