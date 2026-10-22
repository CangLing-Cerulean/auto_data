import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import transfer_diagnosis_core as core
from run_transfer_diagnosis import validate_fold


def row(i):
    polynomial = np.zeros((6, 2, 8, 3))
    polynomial[..., 0] = .5
    polynomial[..., 1] = .2
    polynomial[..., 2] = .1
    polynomial[0, 0, :, 0] = 1.
    return dict(id=str(i), x=(np.ones((8, 3)) * i).tolist(), polynomial=polynomial.tolist(),
        prefix_p=[1., .5], parent_prefix_p=[1., 1.], parent_c=1.,
        z=[.5] * 8, z_energy=[1.] * 8, y=10. + i, m1_gain=20.)


class TransferDiagnosisTests(unittest.TestCase):
    def test_response_endpoints_tie_and_known_prefix(self):
        r = row(0)
        for a, expected in ((0., 50.), (1., 20.)):
            gain, steps, _ = core.response(core.tensor([[a] * 8]), core.tensor([r['polynomial']]), core.tensor([1.]))
            self.assertAlmostEqual(float(gain[0]), expected)
            self.assertEqual(int(steps[0]), 1)
        gain, steps, _ = core.response(core.tensor([[0.] * 8]), core.tensor([r['polynomial']]), core.tensor([1.]), core.tensor([[.1, .7]]), torch.tensor([True]))
        self.assertEqual(int(steps[0]), 0)
        self.assertAlmostEqual(float(gain[0]), 30.)

    def test_actual_separate_fit_blocks_value_gradient(self):
        original_updates = core.UPDATES
        core.UPDATES = 2
        try:
            fold = dict(history=[row(0), row(1)], probes=[row(2)])
            changed = copy.deepcopy(fold)
            for r in changed['history']:
                r['y'] += 80.
            separate_a, separate_b = [core.fit(f, 'SEPARATE') for f in (fold, changed)]
            np.testing.assert_array_equal(separate_a['weights'], separate_b['weights'])
            self.assertFalse(np.array_equal(separate_a['beta'], separate_b['beta']))
            joint_a, joint_b = [core.fit(f, 'JOINT') for f in (fold, changed)]
            self.assertFalse(np.array_equal(joint_a['weights'], joint_b['weights']))
        finally:
            core.UPDATES = original_updates

    def test_fold_rejects_target_truth_and_unopened_prefix(self):
        history = [dict(id=f'h{i}', task='H', y=i) for i in range(24)]
        targets = [dict(id=f't{i}', task='T', action='A' if i < 6 else 'B', probed=i < 6,
                        **({'prefix_p': [1., 1.]} if i < 6 else {})) for i in range(12)]
        probes = [dict(id=f't{i}', task='T', action='A') for i in range(6)]
        fold = dict(task='T', probe_actions=['A'], history=history, probes=probes, targets=targets)
        validate_fold(fold)
        wrong = copy.deepcopy(fold)
        wrong['targets'][6]['y'] = 99.
        with self.assertRaises(AssertionError):
            validate_fold(wrong)
        wrong = copy.deepcopy(fold)
        wrong['targets'][6]['prefix_p'] = [1., 1.]
        with self.assertRaises(AssertionError):
            validate_fold(wrong)


if __name__ == '__main__':
    unittest.main()
