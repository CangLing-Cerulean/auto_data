"""Workflow integration checks using metadata-only commands in fresh workspaces."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from online_ts.config import GateError, digest
from online_ts.platform import studies, study_worker
from online_ts.platform.common import file_hash, file_lock, read_json, write_json


REPOSITORY = Path(__file__).resolve().parents[1]


class StudyWorkflowTests(unittest.TestCase):
    def setUp(self):
        base = REPOSITORY / "outputs" / "platform-tests"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.config = self.workspace / "config.json"
        self.root = self.workspace / "outputs" / "child"

    def script(self, name, text):
        path = self.workspace / name
        path.write_text(text, encoding="utf-8")
        return name

    def step(self, key, script, **changes):
        result = {"id": key, "depends_on": [],
                  "command": ["{python}", "{workspace}/" + script, "{context}"],
                  "timeout_seconds": 10, "outputs": ["result.txt"]}
        result.update(changes)
        return result

    def freeze(self, steps, scripts):
        self.spec = {"version": 1, "name": "MetadataCheck", "workspace": ".",
                     "inputs": {"code" + str(i): {"path": name, "role": "code"}
                                for i, name in enumerate(scripts)}, "steps": steps}
        write_json(self.config, self.spec)
        return studies.create(self.config, self.root)

    def simple(self, **step_changes):
        script = self.script("success.py", "from pathlib import Path\nPath('result.txt').write_text('ok', encoding='utf-8')\nprint('executed once')\n")
        return self.freeze([self.step("first", script, **step_changes)], [script])

    def test_dependency_context_and_completed_resume_preserve_receipts(self):
        first = self.script("first.py", "from pathlib import Path\nPath('result.txt').write_text('metadata', encoding='utf-8')\n")
        second = self.script("second.py", "import json, sys\nfrom pathlib import Path\nc = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\np = Path(c['dependencies']['first']) / 'result.txt'\nassert Path(c['inputs']['code0']).is_file()\nPath('result.txt').write_text(p.read_text(encoding='utf-8') + ' chained', encoding='utf-8')\n")
        self.freeze([self.step("first", first), self.step("second", second, depends_on=["first"])], [first, second])
        result = studies.run(self.root)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["attempts"], 2)
        output = self.root / "steps/second/001/result.txt"
        self.assertEqual(output.read_text(encoding="utf-8"), "metadata chained")
        receipts = {p: p.read_bytes() for p in self.root.glob("steps/*/*/receipt.json")}
        self.assertEqual(studies.run(self.root)["attempts"], 2)
        self.assertEqual({p: p.read_bytes() for p in receipts}, receipts)
        self.assertTrue(studies.report(self.root)["verified"])
        with self.assertRaisesRegex(GateError, "ONLY_UNCERTAIN"):
            studies.reconcile(self.root, "second", 1, outcome="failed", returncode=75,
                              evidence=self.config, actor="tester", reason="success must remain durable")

    def test_explicit_retry_keeps_failed_cost_and_logs(self):
        script = self.script("retry.py", "import sys\nfrom pathlib import Path\nprint('attempt-' + Path.cwd().name, flush=True)\nif Path.cwd().name == '001':\n    print('transient metadata failure', file=sys.stderr, flush=True)\n    sys.exit(75)\nPath('result.txt').write_text('recovered', encoding='utf-8')\n")
        self.freeze([self.step("first", script, max_attempts=2, retry_exit_codes=[75])], [script])
        result = studies.run(self.root)
        attempts = result["steps"][0]["attempts"]
        self.assertEqual([a["state"] for a in attempts], ["failed", "completed"])
        self.assertEqual(attempts[0]["receipt"]["returncode"], 75)
        self.assertTrue(all(a["receipt"]["seconds"] > 0 for a in attempts))
        self.assertAlmostEqual(result["known_command_seconds"], sum(a["receipt"]["seconds"] for a in attempts))
        failed = Path(attempts[0]["directory"])
        self.assertIn("attempt-001", (failed / "stdout.log").read_text(encoding="utf-8"))
        self.assertIn("transient metadata failure", (failed / "stderr.log").read_text(encoding="utf-8"))
        original = (failed / "receipt.json").read_bytes()
        self.assertEqual(studies.run(self.root)["attempts"], 2)
        self.assertEqual((failed / "receipt.json").read_bytes(), original)

    def test_unlisted_failure_is_not_retried(self):
        script = self.script("fail.py", "import sys\nprint('permanent metadata failure')\nsys.exit(2)\n")
        self.freeze([self.step("first", script, max_attempts=2, retry_exit_codes=[75])], [script])
        result = studies.run(self.root)
        self.assertEqual(result["steps"][0]["state"], "failed")
        self.assertEqual(result["steps"][0]["attempts"][0]["receipt"]["returncode"], 2)
        self.assertEqual(studies.run(self.root)["attempts"], 1)
        self.assertGreater(result["known_command_seconds"], 0)

    def test_timeout_is_not_retried(self):
        script = self.script("sleep.py", "import time\nprint('waiting', flush=True)\ntime.sleep(30)\n")
        self.freeze([self.step("first", script, timeout_seconds=0.2, max_attempts=2, retry_exit_codes=[75])], [script])
        result = studies.run(self.root)
        self.assertEqual(result["steps"][0]["state"], "timed_out")
        self.assertGreater(result["known_command_seconds"], 0)
        self.assertEqual(studies.run(self.root)["attempts"], 1)

    def test_unknown_and_running_attempts_are_not_restarted(self):
        self.simple(max_attempts=2, retry_exit_codes=[75])
        attempt = self.root / "steps/first/001"
        attempt.mkdir(parents=True)
        self.assertEqual(studies.run(self.root)["steps"][0]["state"], "uncertain")
        with file_lock(attempt / "worker.lock"):
            result = studies.run(self.root)
            self.assertEqual(result["steps"][0]["state"], "running")
            with self.assertRaisesRegex(GateError, "ONLY_UNCERTAIN"):
                studies.reconcile(self.root, "first", 1, outcome="completed", returncode=0,
                                  evidence=self.config, actor="tester", reason="metadata evidence")
        self.assertEqual(result["attempts"], 1)
        self.assertFalse((attempt / "started.json").exists())

    def test_reconciliation_retains_evidence_and_unknown_seconds(self):
        self.simple()
        attempt = self.root / "steps/first/001"
        attempt.mkdir(parents=True)
        (attempt / "result.txt").write_text("externally observed", encoding="utf-8")
        evidence = self.workspace / "evidence.txt"
        evidence.write_text("external metadata execution receipt", encoding="utf-8")
        result = studies.reconcile(self.root, "first", 1, outcome="completed", returncode=0,
                                   evidence=evidence, actor="tester", reason="verified metadata output")
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["unknown_command_seconds"], 1)
        self.assertEqual(result["known_command_seconds"], 0)
        self.assertIsNone(result["steps"][0]["attempts"][0]["receipt"]["seconds"])
        self.assertEqual((attempt / "reconciliation-evidence.bin").read_bytes(), evidence.read_bytes())
        self.assertEqual(studies.run(self.root)["attempts"], 1)
        original = (attempt / "reconciliation.json").read_bytes()
        with self.assertRaisesRegex(GateError, "ONLY_UNCERTAIN"):
            studies.reconcile(self.root, "first", 1, outcome="failed", returncode=75,
                              evidence=evidence, actor="tester", reason="cannot overwrite success")
        self.assertEqual((attempt / "reconciliation.json").read_bytes(), original)

    def test_input_tampering_is_rejected(self):
        self.simple()
        (self.workspace / "success.py").write_text("raise RuntimeError('changed')", encoding="utf-8")
        with self.assertRaisesRegex(GateError, "STUDY_INPUT_CHANGED"):
            studies.run(self.root)
        self.assertEqual(studies.status(self.root)["attempts"], 0)

    def test_worker_cannot_restart_an_externally_reconciled_attempt(self):
        self.simple(max_attempts=2, retry_exit_codes=[75])
        attempt = self.root / "steps/first/001"
        attempt.mkdir(parents=True)
        studies.reconcile(self.root, "first", 1, outcome="failed", returncode=75,
                          evidence=self.config, actor="tester", reason="verified failed metadata command")
        original = (attempt / "reconciliation.json").read_bytes()
        with self.assertRaises((ValueError, GateError)):
            study_worker.work(self.root, "first", attempt)
        self.assertFalse((attempt / "started.json").exists())
        self.assertFalse((attempt / "result.txt").exists())
        self.assertFalse((attempt / "receipt.json").exists())
        self.assertEqual((attempt / "reconciliation.json").read_bytes(), original)

    def test_completed_output_tampering_is_rejected(self):
        self.simple()
        self.assertEqual(studies.run(self.root)["state"], "completed")
        (self.root / "steps/first/001/result.txt").write_text("altered", encoding="utf-8")
        for operation in (studies.run, studies.report):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(GateError, "STUDY_OUTPUT_CHANGED"):
                    operation(self.root)

    def test_frozen_code_and_lock_tampering_are_rejected(self):
        self.simple()
        lock_path = self.root / "study.json"
        lock = read_json(lock_path)
        lock["spec"]["name"] = "Changed"
        write_json(lock_path, lock)
        with self.assertRaisesRegex(GateError, "STUDY_LOCK_CHANGED"):
            studies.run(self.root)
        # Freeze a temporary source as a hash sentinel without touching shared code.
        sentinel = self.workspace / "frozen-source.py"
        sentinel.write_text("VALUE = 1\n", encoding="utf-8")
        lock["platform_code"][str(sentinel)] = file_hash(sentinel)
        lock["sha256"] = digest({k: v for k, v in lock.items() if k != "sha256"})
        write_json(lock_path, lock)
        sentinel.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(GateError, "STUDY_PLATFORM_CODE_CHANGED"):
            studies.run(self.root)

    def test_concurrent_runner_kernel_lock_prevents_duplicate_execution(self):
        script = self.script("slow.py", "import time\nfrom pathlib import Path\nPath('entered.txt').write_text('entered', encoding='utf-8')\ntime.sleep(1.5)\nPath('result.txt').write_text('once', encoding='utf-8')\n")
        self.freeze([self.step("first", script)], [script])
        process = subprocess.Popen([sys.executable, "-B", "-c",
            "from online_ts.platform.studies import run; import sys; run(sys.argv[1])", str(self.root)],
            cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 15
            marker = self.root / "steps/first/001/entered.txt"
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(marker.exists(), "First runner did not reach metadata command")
            with self.assertRaisesRegex(GateError, "PLATFORM_BUSY"):
                studies.run(self.root)
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 0, (stdout + stderr).decode(errors="replace"))
            self.assertEqual(studies.status(self.root)["attempts"], 1)
            self.assertEqual(studies.report(self.root)["state"], "completed")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_invalid_config_roles_paths_and_run_location_are_refused(self):
        self.simple()
        cases = []
        for role in ("D", "test"):
            value = copy.deepcopy(self.spec)
            value["inputs"]["code0"]["role"] = role
            cases.append((role, value))
        for path in ("../outside.py", str(self.workspace / "success.py")):
            value = copy.deepcopy(self.spec)
            value["inputs"]["code0"]["path"] = path
            cases.append((path, value))
        for field, value in (("depends_on", ["later"]), ("timeout_seconds", 0),
                             ("max_attempts", True), ("retry_exit_codes", [0]),
                             ("outputs", ["../escape.txt"]),
                             ("command", ["{unknown}"])):
            spec = copy.deepcopy(self.spec)
            spec["steps"][0][field] = value
            cases.append((field, spec))
        for name, spec in cases:
            with self.subTest(case=name):
                write_json(self.config, spec)
                with self.assertRaises((GateError, ValueError)):
                    studies.validate_config(self.config)
        write_json(self.config, self.spec)
        for location in (self.workspace / "outside", self.workspace / "outputs", self.root):
            with self.subTest(run=location):
                with self.assertRaisesRegex(GateError, "NEW_OUTPUTS_DIRECTORY"):
                    studies.create(self.config, location)


if __name__ == "__main__":
    unittest.main()
