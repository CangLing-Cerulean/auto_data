"""Small algebraic fixtures test contracts; never inserted into sensor datasets."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from online_ts.agent_research.intervention import (
    paired_response, equivalent_processing, context, register_prediction,
    dual_update, counterexamples, measure_signal, signal_value)


class InterventionContracts(unittest.TestCase):
    def test_elemental_complement(self):
        p=np.array([[[2.,1.],[2.,1.]],[[1.,1.],[1.,1.]]])
        c=p.copy();c[0,:,0]=1.;c[0,:,1]=2.
        target=np.zeros_like(p,dtype=bool);target[0,:,0]=True
        r=paired_response(p,c,target)
        self.assertEqual(r['target']['elements'],2)
        self.assertEqual(r['complement']['elements'],6)
        self.assertGreater(r['target']['delta']['NMSE'],0)
        self.assertLess(r['complement']['delta']['NMSE'],0)
        self.assertAlmostEqual(r['global']['delta']['NMSE'],0)
        self.assertAlmostEqual(sum(r[k]['elements']*r[k]['delta']['NMSE'] for k in ('target','complement')),
                               r['global']['elements']*r['global']['delta']['NMSE'])

    def test_bad_alignment_and_nonfinite(self):
        p=np.ones((2,2,2));mask=np.zeros_like(p,dtype=bool);mask[0]=True
        with self.assertRaises(ValueError):paired_response(p,p[:1],mask)
        with self.assertRaises(ValueError):paired_response(p,p,np.ones_like(mask))
        p[0,0,0]=np.nan
        with self.assertRaises(ValueError):paired_response(p,p,mask)

    def test_parent_and_hypothesis_separated(self):
        response={k:dict(gain_pct=dict(NMSE=v,NMAE=.2)) for k,v in [('global',2.),('target',-2.),('complement',3.)]}
        out=dual_update(response,preregistered=True,matched_control_passed=True,independent_confirmation=True,
                        directions=dict(global_='improve'))
        self.assertEqual(out['recipe'],'confirmed');self.assertEqual(out['hypothesis'],'insufficient_prediction')
        out=dual_update(response,preregistered=True,matched_control_passed=True,independent_confirmation=True,
                        directions={'global':'improve','target':'improve','complement':'unchanged'})
        self.assertEqual(out['hypothesis'],'not_supported')

    def test_prediction_failure_blocks_support(self):
        response={k:dict(gain_pct=dict(NMSE=v,NMAE=.2)) for k,v in [('global',2.),('target',3.),('complement',-2.)]}
        out=dual_update(response,preregistered=True,matched_control_passed=True,independent_confirmation=True,
                        directions={'global':'improve','target':'improve','complement':'unchanged'})
        self.assertEqual(out['hypothesis'],'response_prediction_contradicted')

    def test_equal_shared_facts(self):
        facts={'parent_id':'p','failures':['x']}
        a,b=context(facts,'strong_prompt'),context(facts,'full')
        self.assertEqual(a['facts_sha256'],b['facts_sha256']);self.assertEqual(a['tools'],b['tools'])

    def test_lock_is_create_only_and_not_retrospective(self):
        decision=dict(parent_id='p',action='a',slice={'group':'g'},evidence_ids=['x'],falsifiers=['target harm'],
                      directions={'global':'improve','target':'improve','complement':'unchanged'})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'lock.json';register_prediction(path,{},decision)
            with self.assertRaises(FileExistsError):register_prediction(path,{},decision)
            with self.assertRaises(ValueError):register_prediction(Path(folder)/'late.json',{'response_visible':True},decision)

    def test_equivalence_conservative(self):
        a=dict(source_identity='s',input_program_sha256='x',start_row=[1,2],probability=[.5,.5],loss_weight=[1,1])
        self.assertEqual(equivalent_processing(a,a),'exact')
        self.assertEqual(equivalent_processing(a,dict(a,input_program_sha256='y')),'unknown_program_equivalence')

    def test_counterexamples_exclude_missing_prestate(self):
        row=dict(record_id='r',parent_id='p',action='a',prestate={},scope='development',
                 prestate_locked=True,predicted_direction=1,observed_direction=-1)
        self.assertEqual(len(counterexamples([row,dict(row,prestate_locked=False)])),1)

    def test_signal_missing_and_identity_guard(self):
        rule=dict(operator='all',conditions=[dict(feature='coverage',comparison='le',threshold=.2)])
        self.assertIsNone(measure_signal({},rule,['coverage']))
        self.assertTrue(measure_signal({'coverage':.1},rule,['coverage']))
        with self.assertRaises(ValueError):measure_signal({'coverage':.1},rule,[])
        with self.assertRaises(ValueError):signal_value([1],[1],[1],construction_sources=['s'],selection_sources=['s'])


if __name__=='__main__':unittest.main()
