"""Compile bounded, attributable historical development excerpts for retrieval."""
from pathlib import Path

from ..config import GateError, digest
from ..selection.common import file_hash, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "research_runs/agent_research"
SOURCE_FILES = frozenset({
    "research_runs/decisions.md", "research_runs/selection/decisions.md",
    "research_runs/cluster_selection/REPORT.md", "research_runs/learning_sampling/REPORT.md",
    "research_runs/covariate_sampling/REPORT.md", "research_runs/cadence_selection/decisions.md",
    "research_runs/leverage_sampling/REPORT.md", "research_runs/completion_study/REPORT.md",
})


def build_library(spec_path=STUDY / "experience_specs.json", destination=STUDY / "experience_library.json", *, root=ROOT):
    specs = read_json(spec_path)
    records = []
    for case in specs["cases"]:
        if case["source"] not in SOURCE_FILES:
            raise GateError("EXPERIENCE_SOURCE_NOT_APPROVED")
        source = Path(root) / case["source"]
        lines = source.read_text(encoding="utf-8").splitlines()
        if type(case["start"]) is not int or type(case["end"]) is not int or not 1 <= case["start"] <= case["end"] <= len(lines):
            raise GateError("EXPERIENCE_RANGE")
        excerpt = "\n".join(lines[case["start"] - 1:case["end"]])
        if not case["anchors"] or any(a not in excerpt for a in case["anchors"]):
            raise GateError("EXPERIENCE_CLAIM_ANCHOR")
        records.append({"record_id": "experience:" + case["id"], "dataset_id": specs["dataset_id"],
                        "evaluation_scope": "historical_development", "historical_intervention": "unknown",
                        "provenance": "curated_from_existing_records_not_new_agent_run",
                        **{k: case[k] for k in ("tags", "lesson", "status", "limits")},
                        "source": {"path": case["source"], "sha256": file_hash(source),
                                   "start": case["start"], "end": case["end"],
                                   "anchors": case["anchors"], "excerpt": excerpt, "excerpt_sha256": digest(excerpt)}})
    library = {"version": "development-experience-library-v1", "spec_sha256": file_hash(spec_path),
               "scope": "development_only", "records": records}
    library["sha256"] = digest(library)
    verify_library(library, root=root)
    destination = Path(destination)
    if destination.exists() and read_json(destination) != library:
        raise GateError("EXPERIENCE_LIBRARY_CHANGED_USE_NEW_DESTINATION")
    write_json(destination, library)
    return library


def verify_library(library, *, root=ROOT):
    if library.get("scope") != "development_only" or digest({k: v for k, v in library.items() if k != "sha256"}) != library.get("sha256"):
        raise GateError("EXPERIENCE_LIBRARY_IDENTITY")
    ids = set()
    for record in library["records"]:
        if record["record_id"] in ids or record["evaluation_scope"] != "historical_development":
            raise GateError("EXPERIENCE_SCOPE_OR_DUPLICATE")
        ids.add(record["record_id"])
        ref = record["source"]
        if ref["path"] not in SOURCE_FILES:
            raise GateError("EXPERIENCE_SOURCE_NOT_APPROVED")
        path = Path(root) / ref["path"]
        if file_hash(path) != ref["sha256"]:
            raise GateError("EXPERIENCE_SOURCE_CHANGED")
        lines = path.read_text(encoding="utf-8").splitlines()
        if type(ref["start"]) is not int or type(ref["end"]) is not int or not 1 <= ref["start"] <= ref["end"] <= len(lines):
            raise GateError("EXPERIENCE_RANGE")
        excerpt = "\n".join(lines[ref["start"] - 1:ref["end"]])
        if excerpt != ref["excerpt"] or digest(excerpt) != ref["excerpt_sha256"] or any(a not in excerpt for a in ref["anchors"]):
            raise GateError("EXPERIENCE_EXCERPT_CHANGED")
    return len(ids)


def retrieve(library, *, target_dataset, track, query="", limit=5, root=ROOT):
    verify_library(library, root=root)
    if track not in {"experience_combination", "new_task_discovery"} or not isinstance(target_dataset, str) or not target_dataset:
        raise GateError("EXPERIENCE_TRACK")
    if type(limit) is not int or not 1 <= limit <= 10 or not isinstance(query, str) or len(query) > 1000:
        raise GateError("EXPERIENCE_QUERY")
    records = [r for r in library["records"] if track != "new_task_discovery" or r["dataset_id"] != target_dataset]
    terms = query.lower().split()
    def score(record):
        text = " ".join(record["tags"] + [record["lesson"], record["limits"]]).lower()
        return sum(term in text for term in terms)
    if terms:
        records = [r for r in records if score(r)]
    return {"library_sha256": library["sha256"], "track": track, "target_dataset": target_dataset,
            "records": sorted(records, key=lambda r: (-score(r), r["record_id"]))[:limit]}
