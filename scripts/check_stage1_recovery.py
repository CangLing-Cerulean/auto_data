"""No-training regression checks for detached regular-file output and ID guards."""
import importlib.util
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).with_name('supervise_stage1_recovery.py'))
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


class RecoveryTests(unittest.TestCase):
    def test_stdout_and_stderr_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'child.log'
            command = [sys.executable, '-c', 'import os,stat,sys; assert stat.S_ISREG(os.fstat(1).st_mode); assert stat.S_ISREG(os.fstat(2).st_mode); print("x"*200000, flush=True); print("saved",file=sys.stderr,flush=True)']
            child = supervisor.spawn_logged(command, path)
            self.assertEqual(child.wait(timeout=10), 0)
            self.assertTrue(path.read_text().endswith('saved\n'))
            self.assertGreater(path.stat().st_size, 200000)

    def test_error_exit_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            child = supervisor.spawn_logged([sys.executable, '-c', 'raise RuntimeError("deliberate")'], Path(directory)/'error.log')
            self.assertNotEqual(child.wait(timeout=10), 0)

    def test_log_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.log'
            path.write_text('original')
            with self.assertRaises(FileExistsError):
                supervisor.spawn_logged([sys.executable, '-c', 'pass'], path)
            self.assertEqual(path.read_text(), 'original')

    def test_exact_missing_ids(self):
        jobs = [dict(task='C2', action=action, seed=seed, id=f'C2_{action}_s{seed}')
                for seed in (2026101402,2026101403,2026101404) for action in ('PARENT','P1','CP1')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supervisor.validate_missing_jobs(root, jobs)
            with self.assertRaises(ValueError):
                supervisor.validate_missing_jobs(root, jobs[:-1])
            (root/'candidate_runs'/jobs[0]['id']).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                supervisor.validate_missing_jobs(root, jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    root = Path(args.root)
    plan = supervisor.read(root/'confirmation_recovery_protocol.json')
    assert supervisor.sha(__file__) == plan['checker_sha256']
    assert supervisor.sha(root/'supervise_stage1_recovery.py') == plan['supervisor_sha256']
    assert supervisor.sha(root/'audit_stage1_confirmation_recovered.py') == plan['auditor_sha256']
    assert supervisor.sha(root/'run_stage1_candidate.py') == plan['runner_sha256']
    assert supervisor.sha(root/'candidate_protocol.json') == plan['candidate_protocol_sha256']
    for relative,digest in plan['preserved_files_sha256'].items():
        assert supervisor.sha(root/relative) == digest, relative
    supervisor.validate_missing_jobs(root, plan['new_jobs'])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    receipt = dict(status='PASS', tests_passed=result.testsRun, new_trainings=0,
                   protocol_sha256=supervisor.sha(root/'confirmation_recovery_protocol.json'),
                   preserved_files_checked=len(plan['preserved_files_sha256']),
                   missing_runs_verified=9, test_model_access=False,
                   process_seconds=time.monotonic()-started)
    supervisor.write(root/'confirmation_recovery_preflight.json', receipt)
    print(json.dumps(receipt))
