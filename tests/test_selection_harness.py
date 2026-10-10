"""Engineering tests with feature-only fixtures and explicitly mocked model feedback.

These fixtures are not sensor records, experimental model results, or independent
agent ability evidence. They never read the dataset or update research ledgers.
"""
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from online_ts.config import GateError, canonical
from online_ts.agent_runtime.controller import AgentController
from online_ts.agent_runtime.state import Event
from online_ts.agent_runtime.llm import DeepSeekModel
from online_ts.agent_runtime.budget import snapshot, AgentBudgetError
from online_ts.llm_transport import read_chat_stream
from online_ts.selection.common import StudyConfig
from online_ts.selection.dataset import valid_starts, FEATURES
from online_ts.selection.rules import compile_rule, check_mask, mask_hash, matched_controls, random_rule
from online_ts.selection.runtime import StudyRepository, SelectionGateway, context_builder, run
from online_ts.selection.execution import freeze_batch, planned_runs


def feature_fixture():
    values = np.arange(20, dtype=float)
    return SimpleNamespace(starts=np.arange(20), features={k: values.copy() for k in FEATURES},
        groups=np.zeros(20, dtype=int), arrays={"group_names": np.array(["fixture"]),
            "source_index_start": np.arange(20), "source_index_end_inclusive": np.arange(20) + 39},
        task={"historical_masks": [], "dataset_id": "feature_fixture", "protocol": {"model": "fixed-fixture"}})


def leaf(q=.25):
    return {"kind": "compare", "feature": "input_variation", "op": "ge", "value": q, "scale": "quantile"}


def frozen(config=None):
    return {"sha256": "engineering-fixture", "config": asdict(config or StudyConfig(rare_group_size=1)), "task": {}}


def decision(tool, args=None):
    return {"action": {"tool": tool, "arguments": args or {}}, "reason": "工程测试模拟决策", "hypotheses": [], "evidence_refs": []}


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.data = feature_fixture()
        self.config = StudyConfig(rare_group_size=1)

    def test_gaps_and_role_boundaries(self):
        times = np.datetime64("2020-01-01") + np.array([0, 10, 20, 100, 110, 120]).astype("timedelta64[s]")
        self.assertEqual(valid_starts(times, 3, 30).tolist(), [0, 3])
        self.assertEqual(valid_starts(times[:2], 3, 30).tolist(), [])

    def test_rule_uses_training_quantile_and_combination(self):
        mask = compile_rule({"kind": "and", "children": [leaf(.25), {"kind": "not", "child": leaf(.9)}]}, self.data, self.config)
        self.assertEqual(np.flatnonzero(mask).tolist(), list(range(5, 18)))

    def test_no_code_paths_unknown_features_or_nonfinite_thresholds(self):
        for rule in ({"kind": "python", "code": "print(1)"}, {**leaf(), "feature": "test.csv"}, {**leaf(), "value": float("nan")}, {**leaf(), "path": "../test.csv"}):
            with self.assertRaises(GateError):
                compile_rule(rule, self.data, self.config)

    def test_node_bound(self):
        rule = {"kind": "all"}
        for _ in range(8):
            rule = {"kind": "not", "child": rule}
        with self.assertRaises(GateError):
            compile_rule(rule, self.data, self.config)

    def test_local_coverage_and_rare_support(self):
        self.data.groups = np.r_[np.zeros(18, int), np.ones(2, int)]
        self.data.arrays["group_names"] = np.array(["common", "rare"])
        config = StudyConfig(rare_group_size=2)
        mask = np.ones(20, bool)
        mask[-1] = False
        audit = check_mask(mask, self.data, config)
        self.assertFalse(audit["valid"])
        self.assertIn("RARE_GROUP:rare", audit["issues"])

    def test_historical_repeat_detection(self):
        mask = compile_rule(leaf(), self.data, self.config)
        self.data.task["historical_masks"] = [{"path": "historical-feature-mask", "sha256": "fixture"}]
        self.data.arrays.update(historical_domain_0=np.ones(20, bool), historical_removed_0=~mask)
        self.assertFalse(check_mask(mask, self.data, self.config)["valid"])

    def test_matching_controls_preserve_count_and_quota(self):
        self.data.groups = np.arange(20) % 2
        mask = np.arange(20) % 3 != 0
        controls = matched_controls(mask, self.data, 71)
        self.assertEqual(controls["same_count"].sum(), mask.sum())
        for g in (0, 1):
            self.assertEqual(controls["same_group_quota"][self.data.groups == g].sum(), mask[self.data.groups == g].sum())

    def test_random_distribution_reproducible_and_in_public_grammar(self):
        a, b = np.random.default_rng(44), np.random.default_rng(44)
        for _ in range(20):
            rule = random_rule(a)
            self.assertEqual(rule, random_rule(b))
            self.assertEqual(compile_rule(rule, self.data, self.config).dtype, np.bool_)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs/selection-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.root = Path(self.temp.name)
        self.scope = patch("online_ts.selection.runtime.STUDY", self.root)
        self.scope.start()
        self.repo = StudyRepository(frozen(), "agent_01", "agent")
        self.data = feature_fixture()
        self.evaluations = []

        def evaluator(mask, seed):
            self.evaluations.append((mask_hash(mask), seed))
            return {"NMSE": .9, "NMAE": 1.0}

        self.gateway = SelectionGateway(self.repo, self.data, evaluator)
        self.repo.put("baseline", "baseline", {"metrics": {"NMSE": 1.0, "NMAE": 1.0}}, 1)

    def tearDown(self):
        self.scope.stop()
        self.temp.cleanup()

    def step(self, value):
        controller = AgentController(self.repo, self.gateway, lambda c: value, context_builder=context_builder)
        return controller.step(Event("fixture", "selection", 100 + self.repo.state().decision_version))

    def test_duplicate_mask_does_not_train_twice(self):
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        self.assertEqual(len(self.evaluations), 1)
        self.assertTrue(self.repo.state().last_result["result"]["recovered"])

    def test_submit_selects_measured_best_and_stops(self):
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        self.step(decision("submit_policy"))
        final = json.loads((self.repo.folder / "final_policy.json").read_text(encoding="utf-8"))
        self.assertEqual(final["rule"], leaf())
        self.assertEqual(final["metrics"]["NMSE"], .9)
        self.assertTrue(self.repo.state().incident_finished)
        with np.load(self.repo.folder / "selected_windows.npz") as selected:
            self.assertEqual(len(selected["selected_start_row"]), 15)

    def test_nmae_constraint_prevents_nmse_only_selection(self):
        self.gateway.evaluator = lambda m, s: {"NMSE": .1, "NMAE": 1.5}
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        final = self.gateway.submit("fixture")["final_policy"]
        self.assertEqual(final["rule"], {"kind": "all"})

    def test_final_replay_same_assets_no_training(self):
        final = self.gateway.submit("fixture")
        again = self.gateway.submit("fixture")
        self.assertEqual(final["final_policy"], again["final_policy"])
        self.assertEqual(len(self.evaluations), 0)

    def test_final_asset_tampering_is_rejected(self):
        self.gateway.submit("fixture")
        (self.repo.folder / "selected_windows.npz").write_bytes(b"corrupted engineering artifact")
        with self.assertRaises(GateError):
            self.gateway.submit("fixture")

    def test_prepared_attempt_recovery_does_not_duplicate_completed_training(self):
        mask = compile_rule(leaf(), self.data, self.repo.config)
        audit = check_mask(mask, self.data, self.repo.config)
        row = {"record_id": "candidate-" + mask_hash(mask), "index": 0, "rule": leaf(),
               "mask_sha256": mask_hash(mask), "audit": audit, "status": "prepared", "seed": 2026}
        self.repo.put(row["record_id"], "candidate_attempt", row, 1)
        self.gateway.recover_attempts()
        self.gateway.recover_attempts()
        self.assertEqual(len(self.evaluations), 1)
        self.assertEqual(self.gateway.results()[0]["status"], "completed")

    def test_nonfinite_feedback_cannot_be_selected(self):
        self.gateway.evaluator = lambda m, s: {"NMSE": float("nan"), "NMAE": .1}
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        self.assertEqual(self.gateway.results()[0]["status"], "failed")
        self.assertEqual(self.gateway.submit("fixture")["final_policy"]["rule"], {"kind": "all"})

    def test_run_identity_and_memory_isolation(self):
        self.step(decision("inspect_results"))
        other = StudyRepository(frozen(), "agent_02", "agent")
        self.assertEqual(list(other.iter_records("decision")), [])
        with self.assertRaises(GateError):
            StudyRepository({**frozen(), "sha256": "changed"}, "agent_01", "agent")

    def test_unknown_tool_and_untrained_submission_blocked(self):
        row = self.step(decision("read_file", {"path": "test.csv"}))
        self.assertEqual(row["decision_outcome"], "failed")
        row = self.step(decision("submit_policy", {"rule": leaf()}))
        self.assertEqual(row["decision_outcome"], "failed")
        self.assertFalse((self.repo.folder / "final_policy.json").exists())

    def test_candidate_budget_failed_training_counted(self):
        self.gateway.config = StudyConfig(max_candidates=1, rare_group_size=1)
        self.gateway.evaluator = lambda m, s: (_ for _ in ()).throw(RuntimeError("mocked worker failure"))
        self.step(decision("evaluate_candidate", {"rule": leaf()}))
        self.assertEqual(self.gateway.results()[0]["status"], "failed")
        with self.assertRaises(GateError):
            self.gateway.evaluate(leaf(.5), compile_rule(leaf(.5), self.data, self.gateway.config), {})

    def test_batch_requires_all_preregistered_submissions(self):
        with patch("online_ts.selection.execution.STUDY", self.root):
            with self.assertRaises(GateError):
                freeze_batch(frozen(), ["agent_01"], self.root / "batch.json")

    def test_interleaved_batch(self):
        self.assertEqual([p["arm"] for p in planned_runs(StudyConfig())][:4], ["random", "agent", "random", "agent"])


class ApiTests(unittest.TestCase):
    setUp = RuntimeTests.setUp
    tearDown = RuntimeTests.tearDown
    def test_deepseek_json_request_and_usage_accounting(self):
        captured = []
        answer = decision("submit_policy")

        def transport(request, timeout):
            captured.append((request.full_url, json.loads(request.data)))
            return canonical({"model": "deepseek-flash", "usage": {"total_tokens": 20},
                "choices": [{"finish_reason": "stop", "message": {"content": canonical(answer), "reasoning_content": "do not persist"}}]}).encode()

        config = SimpleNamespace(planner=SimpleNamespace(kind="external", allow_external=True,
            endpoint_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY", model="deepseek-flash", timeout_seconds=120))
        client = DeepSeekModel(self.repo, config, system_prompt="Return JSON only.", transport=transport)
        context = context_builder(self.repo, self.gateway, Event("api", "selection", 100))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"}):
            self.assertEqual(client(context), answer)
            self.assertEqual(client(context), answer)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(captured[0][1]["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", captured[0][1])
        self.assertEqual(snapshot(self.repo)["tokens_charged_or_reserved"], 20)
        self.repo.export()
        audit = (self.repo.folder / "model_calls.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("fixture-key", audit)
        self.assertNotIn("do not persist", audit)

    def test_untrusted_endpoint_rejected_before_transport(self):
        config = SimpleNamespace(planner=SimpleNamespace(kind="external", allow_external=True,
            endpoint_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY", model="deepseek-flash", timeout_seconds=120))
        client = DeepSeekModel(self.repo, config, transport=lambda r, t: self.fail("network should not be called"))
        context = context_builder(self.repo, self.gateway, Event("api", "selection", 100))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key", "DEEPSEEK_BASE_URL": "https://untrusted.example"}):
            with self.assertRaises(GateError):
                client(context)

    def test_budget_charges_incomplete_response(self):
        config = SimpleNamespace(planner=SimpleNamespace(kind="external", allow_external=True,
            endpoint_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY", model="deepseek-flash", timeout_seconds=120))
        raw = canonical({"model": "deepseek-flash", "usage": {"total_tokens": 15},
                         "choices": [{"finish_reason": "length", "message": {"content": "{"}}]}).encode()
        client = DeepSeekModel(self.repo, config, transport=lambda r, t: raw)
        context = context_builder(self.repo, self.gateway, Event("api", "selection", 100))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key"}):
            with self.assertRaises(GateError):
                client(context)
        self.assertEqual(snapshot(self.repo)["tokens_charged_or_reserved"], 15)

    def test_sse_reasoning_discarded_usage_preserved(self):
        chunks = [{"model": "deepseek-flash", "choices": [{"index": 0, "delta": {"reasoning_content": "hidden"}}]},
                  {"choices": [{"index": 0, "delta": {"content": "{}"}, "finish_reason": "stop"}]},
                  {"choices": [], "usage": {"total_tokens": 12}}]
        stream = io.BytesIO(("".join("data: " + canonical(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n").encode())
        result = read_chat_stream(stream, __import__("time").monotonic() + 2)
        self.assertNotIn(b"hidden", result)
        self.assertEqual(json.loads(result)["usage"]["total_tokens"], 12)

    def test_call_budget_counts_failed_requests_and_protects_closing(self):
        self.repo.config = StudyConfig(max_calls=3, rare_group_size=1)
        config = SimpleNamespace(planner=SimpleNamespace(kind="external", allow_external=True,
            endpoint_env="DEEPSEEK_BASE_URL", key_env="DEEPSEEK_API_KEY", model="deepseek-flash", timeout_seconds=120))
        raw = canonical({"model": "deepseek-flash", "usage": {"total_tokens": 15},
                         "choices": [{"finish_reason": "length", "message": {"content": "{"}}]}).encode()
        client = DeepSeekModel(self.repo, config, transport=lambda r, t: raw)
        context = context_builder(self.repo, self.gateway, Event("api", "selection", 100))
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key", "DEEPSEEK_BASE_URL": "https://api.deepseek.com"}):
            with self.assertRaises(GateError):
                client(context)
            context["context_version"] = "new-call"
            with self.assertRaises(AgentBudgetError):
                client(context)
        self.assertEqual(snapshot(self.repo)["calls_used"], 1)


class RunnerTests(unittest.TestCase):
    def test_complete_agent_loop_then_resume_without_new_calls(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs/selection-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            calls, evaluations = [], []
            answers = [decision("inspect_results"), decision("evaluate_candidate", {"rule": leaf()}), decision("submit_policy")]

            def model(context):
                calls.append(context)
                return answers.pop(0)

            def evaluator(mask, seed):
                evaluations.append((mask_hash(mask), seed))
                return {"NMSE": 1.0 if mask.all() else .9, "NMAE": 1.0}

            with patch("online_ts.selection.runtime.STUDY", Path(temporary)), \
                    patch("online_ts.selection.runtime.verify_freeze", return_value=frozen()), \
                    patch("online_ts.selection.runtime.TrainingData", return_value=feature_fixture()):
                result = run("fixture-manifest", "agent_01", "agent", evaluator=evaluator, model=model)
                resumed = run("fixture-manifest", "agent_01", "agent", evaluator=evaluator, model=model)
                self.assertEqual(result, resumed)
                self.assertEqual(result["rule"], leaf())
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(evaluations), 2)

    def test_random_loop_obeys_same_candidate_cap(self):
        temporary_root = Path(__file__).resolve().parents[1] / "outputs/selection-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            config = StudyConfig(max_candidates=2, rare_group_size=1)
            evaluations = []

            def evaluator(mask, seed):
                evaluations.append(mask_hash(mask))
                return {"NMSE": 1.0 if mask.all() else .95, "NMAE": 1.0}

            with patch("online_ts.selection.runtime.STUDY", Path(temporary)), \
                    patch("online_ts.selection.runtime.verify_freeze", return_value=frozen(config)), \
                    patch("online_ts.selection.runtime.TrainingData", return_value=feature_fixture()):
                result = run("fixture-manifest", "random_01", "random", evaluator=evaluator)
                repo = StudyRepository(frozen(config), "random_01", "random")
                self.assertLessEqual(len(list(repo.iter_records("candidate_attempt"))), 2)
                self.assertLessEqual(len(evaluations), 3)  # baseline plus at most two different candidates
                self.assertEqual(result["evaluation_scope"], "development_only")


if __name__ == "__main__":
    unittest.main()
