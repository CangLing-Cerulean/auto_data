"""Report every preregistered run, including absent/failed runs. No confirmation feedback to agents."""
import numpy as np

from .common import STUDY, StudyConfig, read_json, write_json
from .execution import planned_runs


def report(frozen, destination):
    rows = []
    for expected in planned_runs(StudyConfig(**frozen["config"])):
        folder = STUDY / "runs" / expected["run_id"]
        policy = read_json(folder / "final_policy.json") if (folder / "final_policy.json").exists() else None
        status = read_json(folder / "status.json") if (folder / "status.json").exists() else None
        failed = policy is not None and policy["termination"] == "agent_execution_failed"
        row = {**expected, "state": "missing_submission" if policy is None else "failed_with_submission" if failed else "submitted",
               "autonomous_completed": policy is not None and not failed,
               "interventions": 0, "policy": policy, "budget": status["budget"] if status else None}
        rows.append(row)
    summaries = {}
    config = StudyConfig(**frozen["config"])
    for arm in ("agent", "random"):
        members = [r for r in rows if r["arm"] == arm]
        gains = [r["policy"]["nmse_gain"] for r in members if r["policy"]]
        summaries[arm] = {"preregistered": len(members), "submitted": sum(r["policy"] is not None for r in members),
            "autonomous_completed": sum(r["autonomous_completed"] for r in members),
            "missing_or_failed": sum(not r["autonomous_completed"] for r in members),
            "full_retention": sum(r["policy"] is not None and r["policy"]["rule"] == {"kind": "all"} for r in members),
            "substantive_success_over_all_runs": sum(r["autonomous_completed"] and r["policy"]["nmse_gain"] >= config.substantive_nmse_gain for r in members) / len(members),
            "median_gain_submissions": float(np.median(gains)) if gains else None,
            "worst_gain_submissions": float(min(gains)) if gains else None}
    result = {"harness_sha256": frozen["sha256"], "scope": "development_only", "runs": rows, "summary": summaries,
              "independent_generalization_established": False,
              "training_seed_repeats_are_not_independent_agent_runs": True,
              "intervention_policy": "no strategy edits during frozen runs; code changes require a new manifest",
              "statistics": "descriptive only; no window-level independence or significance claim"}
    write_json(destination, result)
    return result
