import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from ...config import digest


SCENARIOS = {
    "1_normal_hold": "test_normal_operation_keeps_model_without_training",
    "2_immature_labels_wait_wake": "test_wait_persists_and_wakes_only_matching_event",
    "3_input_delay": "test_data_delay_is_measured_without_training",
    "4_real_experiment": "test_real_worker_development_then_agent_can_stop",
    "5_human_knowledge_influence": "test_confirmed_human_knowledge_changes_investigation_with_evidence",
    "6_experiment_memory_reuse": "test_real_worker_development_then_agent_can_stop",
    "7_diagnostic_contrasts": "test_diagnostic_factorial_produces_only_scoped_verified_contrasts",
    "8_worse_candidate_hold": "test_worse_real_experiment_keeps_good_champion",
    "9_controlled_release": "test_controlled_release_uses_real_shadow_canary_and_promotion_gates",
    "10_llm_failure_recovery": "test_llm_failure_keeps_real_champion_serving_and_honors_failure_policy",
}


def source_manifest(root):
    root = Path(root)
    paths = {root / "pyproject.toml"}
    for folder in ("online_ts", "tests", "scripts", "configs"):
        paths.update(p for p in (root / folder).rglob("*") if p.is_file()
                     and p.suffix in {".py", ".json", ".txt", ".ps1"} and "__pycache__" not in p.parts)
    files = [{"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]
    return {"scope": "online_ts + tests + scripts + configs + pyproject.toml; runtime/artifacts/docs excluded",
            "files": files, "source_sha256": digest(files)}


def scenario_report(junit_path):
    cases = {c.attrib["name"]: c for c in ET.parse(junit_path).iter("testcase")}
    return {"real_llm_evaluated": False, "real_llm_status": "deferred_by_user_while_Qwen_is_being_optimized",
            "evaluation_kind": "scripted_controller_choices_with_real_local_tools_and_safety_invariants",
            "scenarios": [{"scenario": name, "test": test, "local_invariants_passed": test in cases and not any(
                child.tag in {"failure", "error", "skipped"} for child in cases[test]),
                "real_llm_action_quality": "not_evaluated"} for name, test in SCENARIOS.items()]}


def execution_metrics(repo, elapsed_seconds):
    records = list(reversed(list(repo.iter_records("decision"))))
    wait_seconds = 0
    for left, right in zip(records, records[1:]):
        if (left.get("actual_result") or {}).get("wait_condition"):
            wait_seconds += max(0, right["created_at"] - left["created_at"]) / 1e9
    return {"tool_calls": len(records), "experiments": len(list(repo.iter_records("experiment"))),
            "failed_decisions": sum(r["decision_outcome"] == "failed" for r in records),
            "recovered_effects": sum(bool((r.get("actual_result") or {}).get("recovered")) for r in records),
            "wait_event_time_seconds": wait_seconds, "wall_seconds": elapsed_seconds,
            "human_release_grants": sum(e["kind"] == "AGENT_RELEASE_HUMAN_GRANTED" for e in repo.store.events(repo.key)),
            "memory_refs_used": sorted({ref for r in records for ref in r["cited_memory_refs"]}),
            "knowledge_refs_used": sorted({ref for r in records for ref in r["cited_knowledge_refs"]})}
