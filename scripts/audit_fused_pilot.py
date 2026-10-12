"""Adjudicate completed pilot evidence without changing frozen run receipts."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(root):
    begun = time.monotonic()
    protocol, summary = load(root / "protocol.json"), load(root / "summary.json")
    if not summary["confirmation_gate"]["exploration_matrix_complete"]:
        raise ValueError("Exploration matrix incomplete")
    locks = {phase: load(root / f"decision_{phase}_lock.json") for phase in ("initial", "after_up", "final")}
    for lock in locks.values():
        if sha(root / lock["decision_file"]) != lock["decision_sha256"]:
            raise ValueError("Decision identity changed")
        if "evidence_file" in lock and sha(root / lock["evidence_file"]) != lock["evidence_sha256"]:
            raise ValueError("Decision evidence changed")
    if sha(root / "agent_state_before.json") != locks["initial"]["state_sha256"]:
        raise ValueError("Initial evidence changed")
    if sha(root / "PROTOCOL.md") != locks["initial"]["protocol_sha256"]:
        raise ValueError("Readable protocol changed")
    visibility = {a: load(root / f"visibility_{a.lower()}.json") for a in ("UP", "DOWN")}
    for event in visibility.values():
        if sha(root / event["evidence_file"]) != event["sha256"]:
            raise ValueError("Feedback identity changed")
    rows = [json.loads(line) for line in (root / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        action = row["action"]
        start = stamp(row["computed_at"]) - timedelta(seconds=row["elapsed_seconds"])
        if row["training_repeat"] == protocol["exploration_seed"]:
            phase = "initial" if action == "UP" else "after_up" if action == "DOWN" else "final" if action.startswith("CTRL_") else None
            if phase and start <= stamp(locks[phase]["locked_at_utc"]):
                raise ValueError("Execution predates decision lock")
            if action in visibility:
                if stamp(row["computed_at"]) >= stamp(visibility[action]["visible_to_policy_at"]):
                    raise ValueError("Feedback exposure predates result")
                row["visible_to_policy_at"] = {"sol_high_r1": visibility[action]["visible_to_policy_at"]}
            elif action == "B0":
                row["visible_to_policy_at"] = {"sol_high_r1": "initial_context_before_initial_lock; exact_read_time_not_recorded"}
            else:
                row["visible_to_policy_at"] = {"sol_high_r1": None}
            diff = summary["exploration"][row["task_id"]][action]
            row["guardrail_status"] = "pass_observed" if diff["NMAE_deterioration_pct"] <= 1 else "fail_observed"
        else:
            row["visible_to_policy_at"] = {"sol_high_r1": None}
            row["guardrail_status"] = "see_confirmation_aggregate"
        row["scientific_status"] = "reference" if action == "B0" else "insufficient_evidence_at_single_run_level"
        row["action_rule_id"] = "recent_switch_weight_v1:" + action
        row["action_instance_id"] = row["task_id"] + ":" + action + ":" + row["data_audit_sha256"][:12]
        row["dataset_id"] = protocol["dataset_id"]
        row["manifest_sha256"] = protocol["manifest_sha256"]
    (root / "adjudicated_responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows), encoding="utf-8")
    controls = {}
    for task in protocol["tasks"]:
        with np.load(root / "tasks" / task / "prepared.npz", allow_pickle=False) as z:
            values, starts, target, control = z["train"], z["train_starts"], z["target"], z["control"]
            cumulative = np.r_[0., np.cumsum(np.abs(np.diff(values[:, :7].astype(np.float64), axis=0)).mean(axis=1))]
            variation = (cumulative[starts + 31] - cumulative[starts]) / 31
            last = values[starts + 31, 7:]
            controls[task] = {
                "target_count": int(target.sum()), "control_count": int(control.sum()),
                "matched_by_design": ["target_start_month", "last_COMP", "treated_count"],
                "target_equals_control": bool(np.array_equal(target, control)),
                "mean_standardized_analog_input_variation": {"target": float(variation[target].mean()), "control": float(variation[control].mean())},
                "mean_standardized_last_switch_channels": {"channels": protocol["selector_channels"], "target": last[target].mean(0, dtype=np.float64).tolist(), "control": last[control].mean(0, dtype=np.float64).tolist()},
                "limitations": ["other_switch_channels_unmatched", "input_variation_unmatched", "temporal_dependence_unmatched", "same_source_development_only"]}
        e = summary["exploration"][task]
        controls[task]["observed_target_vs_control"] = {a: {
            "overall_NMSE_gain_pct": 100 * (1 - e[a]["raw_overall"]["NMSE"] / e["CTRL_" + a]["raw_overall"]["NMSE"]),
            "local_NMSE_gain_pct": 100 * (1 - e[a]["raw_local"]["NMSE"] / e["CTRL_" + a]["raw_local"]["NMSE"]),
            "single_seed_not_confirmation": True} for a in ("UP", "DOWN")}
    initial, final = load(root / "decision_initial.json"), load(root / "decision_final.json")
    initial_actions = {r["task_id"]: r["initial_recommendation"] for r in initial["decisions"]}
    final_actions = {r["task_id"]: r["final_recommendation"] for r in final["decisions"]}
    choices = {}
    for task in protocol["tasks"]:
        first, last = initial_actions[task], final_actions[task]
        if last not in protocol["selectable_actions"]:
            raise ValueError("Invalid final recommendation")
        e = summary["exploration"][task]
        choices[task] = {"initial": first, "final": last,
            "exploration_NMSE_gain_vs_initial_pct": 100 * (1 - e[last]["raw_overall"]["NMSE"] / e[first]["raw_overall"]["NMSE"]),
            "causal_feedback_value_identified": False,
            "qualified_exploration_actions": [a for a in protocol["selectable_actions"] if a != "B0" and e[a]["NMSE_gain_pct"] >= 1 and e[a]["NMAE_deterioration_pct"] <= 1]}
    write(root / "audit.json", {"status": "passed", "computed_at": datetime.now(timezone.utc).isoformat(),
        "decision_before_execution": True, "feedback_after_result": True, "controls_hidden_until_final": True,
        "paired_assets": summary["paired_initialization_and_exposure"], "controls": controls, "choices": choices,
        "initial_semantic_correction": "O80 initial response calls 26.35% a target loss share; it is normalized weight mass, not measured loss contribution. Original decision is preserved unchanged.",
        "scientific_scope": "development pilot; no independent source or time audit; no masked-feedback branch", "new_training": 0, "DeepSeek_API_calls": 0,
        "audit_process_seconds": time.monotonic() - begun})
    print(json.dumps({"audit": "passed", "choices": choices}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
