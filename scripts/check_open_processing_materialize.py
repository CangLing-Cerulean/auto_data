"""Synthetic parent-adapter checks. Subprocess mocks do not verify Linux isolation."""

import hashlib
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import open_processing_materialize as adapter
import open_processing_sandbox as sandbox


class MaterializeChecks(unittest.TestCase):
    def setUp(self):
        self.rows = np.zeros((41, 370), dtype=np.float64)
        self.source = 'synthetic source only'

    def reply(self, arrays, **metadata):
        return subprocess.CompletedProcess([], 0, sandbox.pack(dict(status='PASS', **metadata), arrays), b'diagnostic')

    def pipeline(self, n=2, target=False):
        arrays = dict(model_input=np.zeros((n, 32, 370), dtype=np.float32),
                      offset=np.zeros((n, 370), dtype=np.float32),
                      scale=np.ones((n, 370), dtype=np.float32),
                      sampling=np.ones(n, dtype=np.float64), loss=np.ones(n, dtype=np.float64))
        if target:
            arrays['target'] = np.ones((n, 8, 370), dtype=bool)
        return arrays

    def test_request_and_success_receipt(self):
        state = dict(center=np.arange(3.))
        with patch.object(adapter.subprocess, 'run', return_value=self.reply({'measurement': np.ones(2)}, windows=2)) as run:
            arrays, receipt = adapter.apply_program(self.source, 'measurement', self.rows, state)
        args, kwargs = run.call_args
        self.assertEqual(args[0], [sys.executable, '-B', str(adapter.Path(adapter.__file__).with_name('open_processing_sandbox.py')), 'worker'])
        self.assertIs(kwargs['close_fds'], True)
        self.assertEqual(kwargs['timeout'], 150)
        self.assertIsInstance(kwargs['input'], bytes)
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(kwargs['env'][name], '1')
        metadata, supplied = sandbox.unpack(kwargs['input'])
        self.assertEqual(metadata['source'], self.source)
        self.assertEqual(metadata['action'], 'apply')
        self.assertEqual(metadata['kind'], 'measurement')
        np.testing.assert_array_equal(supplied['rows'], self.rows)
        np.testing.assert_array_equal(sandbox.decode_state(metadata['state'], supplied)['center'], state['center'])
        self.assertEqual(receipt['status'], 'PASS')
        self.assertEqual(receipt['stderr'], 'diagnostic')
        self.assertEqual(receipt['returncode'], 0)
        self.assertFalse(receipt['timed_out'])
        self.assertGreaterEqual(receipt['seconds'], 0)
        for key in ('source_sha256', 'rows_sha256', 'state_sha256', 'request_sha256', 'stdout_sha256'):
            self.assertEqual(len(receipt[key]), 64)
        self.assertEqual(set(arrays), {'measurement'})

    def test_fit_state_and_budget(self):
        state = dict(center=np.ones(370), nested=(1, 2.))
        tree, arrays = sandbox.encode_state(state)
        with patch.object(adapter.subprocess, 'run', return_value=self.reply(arrays, state=tree)):
            fitted, receipt = adapter.fit_program(self.source, 'pipeline', self.rows)
        np.testing.assert_array_equal(fitted['center'], state['center'])
        self.assertEqual(len(receipt['fitted_state_sha256']), 64)
        for invalid in (np.zeros(sandbox.MAX_STATE_BYTES // 8), np.array([np.nan])):
            tree, arrays = sandbox.encode_state(invalid)
            with patch.object(adapter.subprocess, 'run', return_value=self.reply(arrays, state=tree)):
                with self.assertRaises(adapter.ProgramExecutionError) as caught:
                    adapter.fit_program(self.source, 'pipeline', self.rows)
            self.assertEqual(caught.exception.receipt['status'], 'FAIL')
        tree, arrays = sandbox.encode_state(1)
        arrays['extra'] = np.ones(1)
        with patch.object(adapter.subprocess, 'run', return_value=self.reply(arrays, state=tree)):
            with self.assertRaises(adapter.ProgramExecutionError):
                adapter.fit_program(self.source, 'pipeline', self.rows)

    def test_pipeline_valid_variants(self):
        for target in (False, True):
            with self.subTest(target=target), patch.object(adapter.subprocess, 'run', return_value=self.reply(self.pipeline(target=target), windows=2)):
                arrays, receipt = adapter.apply_program(self.source, 'pipeline', self.rows, {}, True, target)
                self.assertEqual('target' in arrays, target)
                self.assertEqual(receipt['status'], 'PASS')

    def test_invalid_materialized_outputs(self):
        cases = []
        for name in self.pipeline():
            missing = self.pipeline(); del missing[name]; cases.append(missing)
        extra = self.pipeline(); extra['extra'] = np.ones(2); cases.append(extra)
        for name in self.pipeline():
            wrong_shape = self.pipeline(); wrong_shape[name] = wrong_shape[name][1:]; cases.append(wrong_shape)
            wrong_dtype = self.pipeline(); wrong_dtype[name] = wrong_dtype[name].astype(np.int64); cases.append(wrong_dtype)
            nonfinite = self.pipeline(); nonfinite[name].flat[0] = np.nan; cases.append(nonfinite)
        for name in ('scale', 'sampling', 'loss'):
            negative = self.pipeline(); negative[name].flat[0] = -1; cases.append(negative)
        zero_scale = self.pipeline(); zero_scale['scale'].flat[0] = 0; cases.append(zero_scale)
        unexpected_target = self.pipeline(target=True); cases.append(unexpected_target)
        for index, arrays in enumerate(cases):
            with self.subTest(index=index), patch.object(adapter.subprocess, 'run', return_value=self.reply(arrays, windows=2)):
                with self.assertRaises(adapter.ProgramExecutionError) as caught:
                    adapter.apply_program(self.source, 'pipeline', self.rows, {})
                self.assertEqual(caught.exception.receipt['status'], 'FAIL')
                self.assertEqual(caught.exception.receipt['stderr'], 'diagnostic')
        for arrays in (self.pipeline(), dict(self.pipeline(), target=np.ones((2, 8, 370), dtype=np.float32))):
            with patch.object(adapter.subprocess, 'run', return_value=self.reply(arrays, windows=2)):
                with self.assertRaises(adapter.ProgramExecutionError):
                    adapter.apply_program(self.source, 'pipeline', self.rows, {}, include_target=True)
        for measurement in (np.ones(2, dtype=np.float32), np.ones(3), np.array([1., np.inf])):
            with patch.object(adapter.subprocess, 'run', return_value=self.reply(dict(measurement=measurement), windows=2)):
                with self.assertRaises(adapter.ProgramExecutionError):
                    adapter.apply_program(self.source, 'measurement', self.rows, {})
        with patch.object(adapter.subprocess, 'run', return_value=self.reply(self.pipeline(), windows=3)):
            with self.assertRaises(adapter.ProgramExecutionError):
                adapter.apply_program(self.source, 'pipeline', self.rows, {})

    def test_process_failure_evidence(self):
        cases = [subprocess.CompletedProcess([], 7, b'partial', b'x' * 9000),
                 subprocess.CompletedProcess([], 0, b'not an archive', b'parse failure'),
                 None]
        # A valid archive with an explicitly unsuccessful worker status.
        cases[-1] = subprocess.CompletedProcess([], 0, sandbox.pack(dict(status='INVALID_PROGRAM', error='bad program'), {}), b'worker error')
        for result in cases:
            with self.subTest(code=result.returncode), patch.object(adapter.subprocess, 'run', return_value=result) as run:
                with self.assertRaises(adapter.ProgramExecutionError) as caught:
                    adapter.run_worker(self.source, 'pipeline', 'fit', self.rows)
                receipt = caught.exception.receipt
                self.assertEqual(receipt['returncode'], result.returncode)
                self.assertEqual(receipt['stdout_bytes'], len(result.stdout))
                self.assertEqual(receipt['stdout_sha256'], hashlib.sha256(result.stdout).hexdigest())
                self.assertLessEqual(len(receipt['stderr']), 8000)
                self.assertEqual(receipt['stderr_truncated'], len(result.stderr) > 8000)
                self.assertEqual(run.call_count, 1)
        timeout = subprocess.TimeoutExpired('worker', 150, output=b'partial', stderr=b'timeout detail')
        for error in (timeout, OSError('launch failure')):
            with patch.object(adapter.subprocess, 'run', side_effect=error) as run:
                with self.assertRaises(adapter.ProgramExecutionError) as caught:
                    adapter.run_worker(self.source, 'pipeline', 'fit', self.rows)
                receipt = caught.exception.receipt
                self.assertEqual(receipt['status'], 'FAIL')
                self.assertEqual(receipt['timed_out'], error is timeout)
                self.assertEqual(run.call_count, 1)
                if error is timeout:
                    self.assertEqual(receipt['stdout_bytes'], 7)
                    self.assertEqual(receipt['stderr'], 'timeout detail')

    def test_weights_expected_values(self):
        sampling = np.tile([1., 3.], 256)
        loss = np.tile([2., 4.], 256)
        p, normalized, stats = adapter.normalize_weights(sampling, loss)
        np.testing.assert_allclose(p, sampling / 1024)
        np.testing.assert_allclose(normalized, loss / 3.5)
        self.assertAlmostEqual(np.sum(p * normalized), 1)
        self.assertEqual(stats['n'], 512)
        self.assertEqual(stats['positive_effective'], 512)
        self.assertEqual(stats['positive_fraction'], 1)
        self.assertAlmostEqual(stats['ess_sampling'], 409.6)
        self.assertAlmostEqual(stats['max_sampling_multiplier'], 1.5)
        self.assertAlmostEqual(stats['max_loss'], 4 / 3.5)
        self.assertAlmostEqual(stats['loss_std_under_p'], np.sqrt(3) / 7)
        p, normalized, stats = adapter.normalize_weights(np.ones(256), np.ones(256))
        self.assertEqual(stats['positive_effective'], 256)
        self.assertEqual(stats['loss_std_under_p'], 0)
        # Exact upper bound 20 is accepted, including zero sampling or loss.
        sampling = np.r_[np.ones(256), np.zeros(4864)]
        loss = np.ones(5120)
        _, _, stats = adapter.normalize_weights(sampling, loss)
        self.assertEqual(stats['max_sampling_multiplier'], 20)
        loss = np.r_[np.ones(256), 20., np.zeros(255)]
        sampling = np.r_[np.ones(256), np.zeros(256)]
        _, _, stats = adapter.normalize_weights(sampling, loss)
        self.assertEqual(stats['max_loss'], 20)

    def test_invalid_request_does_not_launch(self):
        cases = [(self.source, 'unknown', 'fit', self.rows, None),
                 (self.source, 'pipeline', 'unknown', self.rows, None),
                 (self.source, 'pipeline', 'fit', np.zeros((40, 369)), None),
                 (self.source, 'pipeline', 'apply', self.rows, {'bad': np.array([np.inf])}),
                 (self.source, 'pipeline', 'apply', self.rows, np.zeros(sandbox.MAX_STATE_BYTES // 8))]
        for arguments in cases:
            with self.subTest(arguments=arguments[1:3]), patch.object(adapter.subprocess, 'run') as run:
                with self.assertRaises(adapter.ProgramExecutionError) as caught:
                    adapter.run_worker(*arguments)
                self.assertEqual(caught.exception.receipt['status'], 'FAIL')
                run.assert_not_called()

    def test_weights_reject_boundaries(self):
        cases = [(np.ones(255), np.ones(255)),
                 (np.ones(256), np.ones(257)),
                 (np.ones((256, 1)), np.ones((256, 1))),
                 (np.ones(256, dtype=np.complex128), np.ones(256)),
                 (np.ones(256), np.full(256, '1')),
                 (np.zeros(256), np.ones(256)),
                 (np.ones(256), np.zeros(256)),
                 (np.r_[np.ones(255), 0], np.ones(256)),
                 (np.ones(256), np.r_[np.ones(255), 0]),
                 (np.r_[np.ones(256), np.zeros(4865)], np.ones(5121)),
                 (np.r_[np.ones(256), np.zeros(256)], np.r_[np.ones(256), 20.01, np.zeros(255)]),
                 (np.array([]), np.array([])),
                 (np.full(256, np.finfo(np.float64).max), np.ones(256)),
                 (np.ones(256), np.full(256, np.finfo(np.float64).tiny) * np.finfo(np.float64).tiny)]
        for bad in (-1., np.inf, np.nan):
            cases.append((np.r_[bad, np.ones(256)], np.ones(257)))
            cases.append((np.ones(257), np.r_[bad, np.ones(256)]))
        for index, (sampling, loss) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ValueError):
                adapter.normalize_weights(sampling, loss)


if __name__ == '__main__':
    print('Synthetic IPC mocks only; real Linux sandbox and real pipeline execution are NOT verified.', flush=True)
    unittest.main(verbosity=2)
