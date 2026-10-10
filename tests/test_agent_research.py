"""Engineering metadata and mock metrics only; no synthetic sensor records or research runs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from online_ts.config import GateError, digest
from online_ts.agent_research.evidence import EvidenceJournal
from online_ts.agent_research.experience import build_library, retrieve, verify_library
from online_ts.agent_research.jobs import prepare_job
from online_ts.selection.common import file_hash
from online_ts.selection.common import write_json


def engineering_temporary():
    root = Path(__file__).resolve().parents[1] / "outputs/agent-research-tests"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = engineering_temporary()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        path = self.root / "research_runs/decisions.md"
        path.parent.mkdir()
        path.write_text("approved development excerpt\n", encoding="utf-8")
        spec = {"dataset_id": "metadata_fixture", "cases": [{"id": "fixture", "tags": ["coverage"],
            "source": "research_runs/decisions.md", "start": 1, "end": 1, "anchors": ["development"],
            "lesson": "Engineering metadata", "status": "fixture", "limits": "Not experimental evidence"}]}
        self.spec = self.root / "spec.json"
        write_json(self.spec, spec)
        self.library = build_library(self.spec, self.root / "library.json", root=self.root)

    def test_source_and_excerpt_are_verified(self):
        self.assertEqual(verify_library(self.library, root=self.root), 1)
        (self.root / "research_runs/decisions.md").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(GateError):
            verify_library(self.library, root=self.root)

    def test_discovery_excludes_target_history(self):
        args = {"target_dataset": "metadata_fixture", "root": self.root}
        self.assertEqual(len(retrieve(self.library, track="experience_combination", **args)["records"]), 1)
        self.assertEqual(retrieve(self.library, track="new_task_discovery", **args)["records"], [])
        self.assertEqual(len(retrieve(self.library, target_dataset="new_fixture", track="new_task_discovery", root=self.root)["records"]), 1)

    def test_query_has_no_implicit_irrelevant_fallback(self):
        result = retrieve(self.library, target_dataset="metadata_fixture", track="experience_combination", query="unmatched", root=self.root)
        self.assertEqual(result["records"], [])

    def test_test_source_is_not_approved(self):
        spec = {"dataset_id": "fixture", "cases": [{"source": "research_runs/model_exploration/test_result.json"}]}
        write_json(self.spec, spec)
        with self.assertRaises(GateError):
            build_library(self.spec, self.root / "unapproved.json", root=self.root)

    def test_tampered_excerpt_even_with_rehashed_library_is_rejected(self):
        library = deepcopy(self.library)
        library["records"][0]["source"]["excerpt"] = "fabricated"
        library["sha256"] = digest({k: v for k, v in library.items() if k != "sha256"})
        with self.assertRaises(GateError):
            verify_library(library, root=self.root)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = engineering_temporary()
        self.addCleanup(self.temp.cleanup)
        self.args = {"run_id": "engineering_fixture", "target_dataset": "metadata_fixture",
            "track": "experience_combination", "scope": "engineering_only",
            "harness_sha256": digest("engineering harness"), "library_sha256": digest("engineering library")}
        self.path = Path(self.temp.name) / "journal.sqlite"
        self.journal = EvidenceJournal(self.path, **self.args)
        self.journal.evidence("diagnostic", dataset_id="metadata_fixture", role="training_diagnostics",
                              artifact_sha256=digest("metadata fixture"), summary="Mock engineering evidence")
        self.proposal = {"observation": "Mock engineering observation", "hypothesis": "Mock hypothesis",
            "expected_effect": {"metric": "fixture_metric", "target_scope": "engineering", "direction": "decrease", "criterion": "Mock only"},
            "falsification": "Mock falsification", "control": "Mock control", "evidence_refs": ["diagnostic"],
            "code_sha256": digest("fixture code")}
        self.update = {"assessment": "inconclusive", "reason": "Mock result insufficient", "revised_hypothesis": "Unverified",
            "next_action": "Stop fixture", "receipt_sha256": digest("fixture receipt")}

    def completed_receipt(self):
        self.journal.receipt("c1", status="completed", metrics={"fixture_metric": 1.0}, artifact_sha256=digest("fixture receipt"), reason="Mock receipt")

    def test_three_separate_records_and_resumption(self):
        self.journal.proposal("c1", self.proposal)
        self.completed_receipt()
        self.assertEqual(self.journal.summary()["updates"], 0)
        self.journal.update("c1", self.update)
        reopened = EvidenceJournal(self.path, **self.args)
        self.assertEqual(reopened.summary()["updates"], 1)
        self.assertEqual(reopened.summary()["diagnosis_correctness"], "unassessed")
        self.assertFalse(reopened.summary()["autonomy_eligible"])

    def test_before_and_after_cannot_be_rewritten(self):
        self.journal.proposal("c1", self.proposal)
        self.completed_receipt()
        self.journal.proposal("c1", self.proposal)
        changed = deepcopy(self.proposal) | {"hypothesis": "Post-hoc rewrite"}
        with self.assertRaises(GateError):
            self.journal.proposal("c1", changed)
        self.journal.update("c1", self.update)
        with self.assertRaises(GateError):
            self.journal.update("c1", self.update | {"assessment": "supported"})

    def test_no_update_before_result_or_wrong_receipt(self):
        self.journal.proposal("c1", self.proposal)
        with self.assertRaises(GateError):
            self.journal.update("c1", self.update)
        self.completed_receipt()
        with self.assertRaises(GateError):
            self.journal.update("c1", self.update | {"receipt_sha256": digest("different")})

    def test_execution_failure_does_not_refute_hypothesis(self):
        self.journal.proposal("c1", self.proposal)
        self.journal.receipt("c1", status="failed", metrics={}, artifact_sha256=digest("fixture receipt"), reason="Mock platform failure")
        with self.assertRaises(GateError):
            self.journal.update("c1", self.update | {"assessment": "contradicted"})
        self.journal.update("c1", self.update)

    def test_task_scope_citations_and_nonfinite_metrics(self):
        with self.assertRaises(GateError):
            self.journal.evidence("bad", dataset_id="metadata_fixture", role="test", artifact_sha256=digest("fixture"), summary="Forbidden")
        with self.assertRaises(GateError):
            self.journal.proposal("c1", self.proposal | {"evidence_refs": ["unknown"]})
        self.journal.proposal("c1", self.proposal)
        with self.assertRaises(GateError):
            self.journal.receipt("c1", status="completed", metrics={"mock": float("nan")}, artifact_sha256=digest("fixture"), reason="Mock")

    def test_discovery_history_and_changed_identity_rejected(self):
        args = self.args | {"run_id": "discovery_fixture", "track": "new_task_discovery"}
        journal = EvidenceJournal(self.path, **args)
        with self.assertRaises(GateError):
            journal.evidence("target_history", dataset_id="metadata_fixture", role="historical_development", artifact_sha256=digest("fixture"), summary="Excluded")
        with self.assertRaises(GateError):
            EvidenceJournal(self.path, **(self.args | {"harness_sha256": digest("changed")}))

    def test_human_intervention_is_retained(self):
        self.journal.intervention("i1", reason="Mock human strategy edit", changed_assets=["fixture_code.py"])
        self.assertEqual(len(self.journal.summary()["interventions"]), 1)
        self.assertFalse(self.journal.summary()["autonomy_eligible"])


class JobPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = engineering_temporary()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.asset = self.root / "metadata.json"
        self.asset.write_text('{"kind":"engineering metadata"}', encoding="utf-8")
        self.descriptor = {"path": str(self.asset), "name": "metadata.json", "role": "training", "sha256": file_hash(self.asset)}
        self.job = self.root / "job"

    def test_training_staging_preserves_asset_identity_without_claiming_sandbox(self):
        manifest = prepare_job(self.job, "print('engineering fixture')", [self.descriptor])
        self.assertFalse(manifest["evaluation_assets_staged"])
        self.assertFalse(manifest["security_sandbox"])
        self.assertEqual((self.job / "inputs/metadata.json").read_bytes(), self.asset.read_bytes())
        self.assertEqual(file_hash(self.job / "inputs/metadata.json"), self.descriptor["sha256"])
        self.assertEqual(list((self.job / "outputs").iterdir()), [])

    def test_nontraining_asset_and_changed_identity_rejected_before_staging(self):
        for changed in (self.descriptor | {"role": "test"}, self.descriptor | {"name": "../test.csv"}, self.descriptor | {"sha256": digest("wrong")}):
            with self.assertRaises(GateError):
                prepare_job(self.job, "pass", [changed])
            self.assertFalse(self.job.exists())

    def test_invalid_code_and_repeated_job_rejected(self):
        with self.assertRaises(GateError):
            prepare_job(self.job, "this is invalid python!", [self.descriptor])
        prepare_job(self.job, "pass", [self.descriptor])
        with self.assertRaises(GateError):
            prepare_job(self.job, "pass", [self.descriptor])


if __name__ == "__main__":
    unittest.main()
