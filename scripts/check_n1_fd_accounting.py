"""Asset-free checks of the registered FD directions and low-signal arithmetic."""
import unittest
import numpy as np
import n1_finite_difference as fd


class Checks(unittest.TestCase):
    def test_registered_matrix_is_exact(self):
        rows=fd.specifications()
        self.assertEqual(len(rows),16)
        self.assertEqual(len({r['id'] for r in rows}),16)
        self.assertEqual({(r['direction'],r['denominator'],r['sign']) for r in rows},
            {(j,h,s) for j in range(4) for h in (64,128) for s in (-1,1)})
        self.assertEqual({(r['task'],r['seed']) for r in rows},{('G1',2026101701)})

    def test_realized_float32_direction_recovers_linear_model(self):
        W=np.column_stack((np.array([1.,-1.,0.,0.])/np.sqrt(2.),
            np.array([1.,1.,-2.,0.])/np.sqrt(6.),np.array([1.,1.,1.,-3.])/np.sqrt(12.),np.ones(4)/2))
        matrix=np.arange(60,dtype=np.float64).reshape(15,4)/17
        parent=np.linspace(-.3,.7,15)
        for j in range(4):
            for h in (1/64,1/128):
                plus=(1+h*W[:,j]).astype(np.float32).astype(np.float64)-1
                minus=(1-h*W[:,j]).astype(np.float32).astype(np.float64)-1
                observed=((parent+matrix@plus)-(parent+matrix@minus))/(2*h)
                effective=W.T@(plus-minus)/(2*h)
                predicted=(matrix@W)@effective
                result=fd.error_comparison(observed,predicted,parent,1e-6)
                self.assertLess(result['normalized_error'],1e-12)

    def test_low_signal_parent_floor_keeps_absolute_error(self):
        result=fd.error_comparison(np.array([.01]),np.array([0.]),np.array([1e6]),1e-6)
        self.assertTrue(result['low_signal'])
        self.assertEqual(result['denominator'],1.)
        self.assertEqual(result['absolute_error'],.01)
        self.assertEqual(result['normalized_error'],.01)

    def test_zero_and_nonzero_discrepancy(self):
        zero=np.zeros(3)
        self.assertEqual(fd.error_comparison(zero,zero,zero,1e-6)['normalized_error'],0.)
        one=np.array([1.,0.,0.])
        self.assertEqual(fd.error_comparison(one,zero,zero,1e-6)['normalized_error'],1.)
        self.assertGreater(fd.error_comparison(one,one*1.03,one,1e-6)['normalized_error'],.02)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    raise SystemExit(not result.wasSuccessful())
