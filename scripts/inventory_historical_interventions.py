"""Read-only inventory of an explicitly scoped historical fixed-policy cohort.

Uses disclosed result metadata and hashes saved assets; never loads sensor arrays
or model tensors. No historical scientific result or archive file is modified.
"""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("C:/Users/31227/Desktop/auto_data_history/20261011_204809")
OUT = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_intervention_coverage_r1"
ARCHIVE_SHA = "ea9271e648676b2de94d69b8daf42214cb3dcc21b8252cca05dc2a3f52c0c884"
CORE_SHA = "4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052"
FAMILIES = {
    "cluster": [f"T{i:03d}" for i in (17, 18, 19)],
    "coverage": [f"T{i:03d}" for i in (23, 24, 25, 26, 27, 28, 29, 50)],
    "covariate": [f"T{i:03d}" for i in (45, 46, 47, 48, 49)],
    "cadence": [f"T{i:03d}" for i in (52, 53, 54)],
    "leverage": [f"T{i:03d}" for i in (57, 58, 59, 60, 61)],
    "completion_scalar_or_subset": [f"C{i:03d}" for i in (*range(100, 115), *range(120, 135)) if i not in (102, 103, 122, 123)],
}
PARENTS = {0.6: "T000_prefix60_full", 0.8: "T005_prefix80_full"}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    begun = time.perf_counter()
    manifest_path = ARCHIVE / "archive_manifest.json"
    assert sha(manifest_path) == ARCHIVE_SHA
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    registered = {r["path"]: r for r in manifest["files"]}
    result_root = ARCHIVE / "files/research_runs/results"
    checked = {}

    def verify(path):
        rel = path.relative_to(ARCHIVE / "files").as_posix()
        entry = registered[rel]
        assert not entry["keep"], rel
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"], rel
        checked[rel] = {"bytes": entry["bytes"], "sha256": entry["sha256"]}
        return entry["sha256"]

    def read(path):
        verify(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    parents = {}
    parent_records = {}
    for origin, run_id in PARENTS.items():
        run = result_root / run_id
        result = read(run / "result.json")
        identity = read(run / "identity.json")
        assert result["identity"] == identity
        assert result["run_id"] == run_id
        assert identity["origin"] == origin and identity["seed"] == 2026
        assert identity["core_sha256"] == CORE_SHA and result["steps_completed"] == 3000
        assert identity["drop_block"] == -1
        assert result["samples_presented"] == 768000
        data_audit = read(run / "data_audit.json")
        parent_records[origin] = (result, data_audit)
        for name in ("final.pt", "latest.pt", "windows.npz", "code_snapshot.py", "core_snapshot.py", "protocol.json", "environment.json"):
            verify(run / name)
        parents[origin] = {"id": run_id, "nmse": result["future"]["groups"]["all"]["NMSE"],
                           "windows": result["future"]["groups"]["all"]["windows"],
                           "initial_file_present": (run / "initial.pt").exists()}

    rows = []
    for family, prefixes in FAMILIES.items():
        for prefix in prefixes:
            matches = list(result_root.glob(prefix + "_*/result.json"))
            assert len(matches) == 1, (prefix, matches)
            path = matches[0]
            result = read(path)
            identity = read(path.parent / "identity.json")
            assert result["run_id"] == path.parent.name
            assert result["identity"] == identity and identity["seed"] == 2026
            origin = identity["origin"]
            assert origin in parents and result["steps_completed"] == 3000
            parent_result, parent_audit = parent_records[origin]
            assert identity["manifest_sha256"] == parent_result["identity"]["manifest_sha256"], prefix
            protocol_sha = identity["sources_sha256"]["protocol.json"] if family in (
                "cadence", "leverage", "completion_scalar_or_subset"
            ) else identity["protocol_sha256"]
            assert protocol_sha == parent_result["identity"]["protocol_sha256"], prefix
            assert result["sensor_columns"] == parent_result["sensor_columns"]
            data_audit = read(path.parent / "data_audit.json")
            for key in ("dataset_id", "source_train_sha256", "train_raw_rows", "probe_raw_rows",
                        "train_windows", "mean", "scale"):
                assert data_audit[key] == parent_audit[key], (prefix, key)
            # The original ten T017..T029 records and later policy runners
            # deliberately used different names for this same recorded count.
            count_key = "probe_windows" if prefix in {f"T{i:03d}" for i in (17, 18, 19, 23, 24, 25, 26, 27, 28, 29)} else "evaluation_windows"
            assert data_audit[count_key] == parent_audit["probe_windows"], prefix
            assert result["samples_presented"] == 768000
            core = identity.get("core_sha256", identity.get("sources_sha256", {}).get("core_snapshot.py"))
            assert core == CORE_SHA
            nmse = result["future"]["groups"]["all"]["NMSE"]
            assert math.isfinite(nmse) and nmse >= 0
            assert result["future"]["groups"]["all"]["windows"] == parents[origin]["windows"]
            for name in ("final.pt", "latest.pt", "windows.npz", "code_snapshot.py", "core_snapshot.py", "protocol.json"):
                verify(path.parent / name)
            initial = path.parent / "initial.pt"
            initial_sha = verify(initial) if initial.exists() else None
            rows.append({"id": result["run_id"], "family": family, "origin": origin,
                         "parent": parents[origin]["id"], "nmse": nmse,
                         "parent_nmse": parents[origin]["nmse"],
                         "gain_percent": 100 * (parents[origin]["nmse"] - nmse) / parents[origin]["nmse"],
                         "initial_sha256": initial_sha, "evaluation": "disclosed_train_internal_fixed_step_3000"})
    assert len(rows) == 50 and len({r["id"] for r in rows}) == 50
    summaries = []
    for origin, parent in parents.items():
        group = [r for r in rows if r["origin"] == origin]
        best = max(group, key=lambda r: r["gain_percent"])
        summaries.append({"origin": origin, "candidates": len(group),
                          "best_disclosed_candidate": best["id"], "best_gain_percent": best["gain_percent"],
                          "oracle_including_parent_gain_percent": max(0, best["gain_percent"]),
                          "candidates_above_1_percent": sum(r["gain_percent"] >= 1 for r in group),
                          "parent": parent})
    assert sha(manifest_path) == ARCHIVE_SHA
    output = {"status": "metadata_and_selected_asset_hashes_verified", "retrospective": True,
              "archive_manifest_sha256": ARCHIVE_SHA, "cohort": rows, "summaries": summaries,
              "assets": checked, "asset_count": len(checked),
              "scope": "Six explicitly enumerated static-policy families; not all historical interventions",
              "excluded": {"selection_S_runs": "base validation scoring outside this train-internal review",
                           "learning": "dynamic model-dependent probabilities",
                           "completion_target": "per-output weights outside scalar-window M1 interface"},
              "binary_tensor_semantics_verified": False, "proxy_ready": False,
              "new_full_trainings": 0, "cumulative_full_trainings": 409,
              "seconds": time.perf_counter() - begun}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "historical_inventory.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (OUT / "historical_cohort.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"asset_count": len(checked), "summaries": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
