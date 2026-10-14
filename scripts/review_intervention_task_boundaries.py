"""Summarize admissible source-file metadata without opening sensor tables."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_runs/agent_research/intervention_diagnosis/20261014_intervention_coverage_r1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = ROOT / "data/skab_v1/manifest.json"
    expected = "372d99fd9188151d3429171c1978c3ea13a34b735bf3ec7d031d8b6845c5ecc2"
    assert sha(manifest_path) == expected
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    train = next(s for s in manifest["splits"] if s["name"] == "train")
    series = train["series_coverage"]
    assert sum(s["rows"] for s in series) == train["rows"] == 32802
    assert len({s["series_id"] for s in series}) == len(series) == 23
    by_directory = {}
    for s in series:
        name = s["series_id"].split("/")[0]
        group = by_directory.setdefault(name, {"files": 0, "rows": 0})
        group["files"] += 1
        group["rows"] += s["rows"]
    sources = ["scripts/prepare_intervention_stage1.py",
               "research_runs/agent_research/intervention_diagnosis/20261014_existing_sources_r1/PROTOCOL.md"]
    result = {
        "status": "metadata_review_only",
        "dataset_id": manifest["dataset_id"], "manifest_sha256": expected,
        "train_file_sha256_from_manifest": train["sha256"],
        "sensor_tables_opened": [], "model_tensors_loaded": False,
        "source_documents": {p: sha(ROOT / p) for p in sources},
        "base_train_rows": train["rows"], "base_train_source_files": len(series),
        "by_directory_provenance_only": by_directory,
        "source_file_train_segments": series,
        "interpretation": [
            "Stage1a partitions source files between S1 and S2, then uses base train and validation within each task; it does not guarantee file-disjoint fit and score sets within a task.",
            "The closed existing-sources M1 task splits each source file chronologically into A/B/C; file identities recur across roles.",
            "A future source-file-disjoint A/B/C task using only base-training segments would test transfer between experimental recordings, not an unseen dataset or independent apparatus.",
            "The 23 files include base-training segments; do not call all of them complete original experiments.",
            "Directory names are provenance, not model features or anomaly labels. No valve2 source appears in base train.",
            "No fold assignment, candidate, prediction, training launch, or scientific acceptance decision is frozen by this metadata review."
        ],
        "new_full_trainings": 0, "cumulative_full_trainings": 409,
    }
    with (OUT / "task_boundary_review.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"files": len(series), "rows": train["rows"], "provenance": by_directory}))


if __name__ == "__main__":
    main()
