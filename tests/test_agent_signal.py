"""Meaningful regressions: hidden outcomes, empty exits, fact and gate separation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from online_ts.config import GateError
from online_ts.agent_research.code_boundary import check_builder
from online_ts.agent_research.signal_gateway import SignalGateway,scope_summary,verify_training_claim,NEW_TOOLS
from online_ts.agent_research.signal_scoring import outcome_gates

class SignalChecks(unittest.TestCase):
    def test_processing_scope_does_not_admit_forecast_training_or_dynamic_escape(self):
        for code in ('import torch.nn as nn','import torch\nx=torch.nn.Linear(2,1)',
                     'import subprocess','import ctypes','x=__import__("os")','import torch\nx.backward()'):
            with self.assertRaises(GateError):check_builder(code)
        check_builder('import numpy as np,json,sys\nfrom pathlib import Path\np=Path(sys.argv[2])\n')

    def test_no_known_case_tool_or_preset_processing_menu_is_added(self):
        self.assertNotIn('read_registered_case',NEW_TOOLS)
        self.assertIn('register_signal',NEW_TOOLS)
        self.assertIn('control_code',NEW_TOOLS['propose_processing'])

    def test_structure_example_contains_all_required_proposal_fields(self):
        from online_ts.agent_research.signal_study import FORMAL_PROMPT
        start=FORMAL_PROMPT.index('{\n  "action": {')
        example,_=json.JSONDecoder().raw_decode(FORMAL_PROMPT[start:])
        self.assertEqual(set(example['action']['arguments']),set(NEW_TOOLS['propose_processing']))

    def test_scope_uses_exact_original_channel_names_and_reference(self):
        row={'windows':10,'per_channel_NMSE':[1.,3.],'per_channel_NMAE':[.5,1.],
             'persistence_per_channel_NMSE':[2.,2.],'persistence_per_channel_NMAE':[.7,.7]}
        out=scope_summary({'groups':{'best.pt':{'all':row}}},'all',['sensor_b'],['sensor_a','sensor_b'])
        self.assertEqual(out['NMSE'],3.);self.assertEqual(out['persistence_NMSE'],2.)
        with self.assertRaises(GateError):scope_summary({'groups':{'best.pt':{'all':row}}},'all',['renamed'],['sensor_a','sensor_b'])

    def test_factual_reference_rejects_wrong_report_even_if_diagnostic_is_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'inputs.npz'
            np.savez(path,values=np.array([[1.,2.],[3.,4.]]),start_row=np.array([0,1]),feature=np.array([2.,8.]))
            claim={'array':'feature','channel':'','statistic':'mean','quantile':.5,'value':5.}
            self.assertTrue(verify_training_claim(claim,path,['a','b'])['matches'])
            claim['value']=6.;self.assertFalse(verify_training_claim(claim,path,['a','b'])['matches'])

    def test_local_success_and_global_success_remain_separate(self):
        rows=[{'baseline':{'NMSE':1.,'NMAE':1.},'candidate':{'NMSE':1.02,'NMAE':1.},
               'control':{'NMSE':1.04,'NMAE':1.},'local_baseline':{'NMSE':2.},
               'local_candidate':{'NMSE':1.8},'local_control':{'NMSE':2.1}} for _ in range(3)]
        gate=outcome_gates(rows,3)
        self.assertFalse(gate['E_numeric']);self.assertTrue(gate['L_numeric']);self.assertTrue(gate['C_numeric'])
        self.assertEqual(gate['C_semantics'],'unassessed_pending_primary_review')

    def test_global_gain_does_not_supply_a_missing_control_or_local_claim(self):
        rows=[{'baseline':{'NMSE':1.,'NMAE':1.},'candidate':{'NMSE':.95,'NMAE':1.},
               'control':None,'local_baseline':None,'local_candidate':None,'local_control':None} for _ in range(3)]
        gate=outcome_gates(rows,3);self.assertTrue(gate['E_numeric'])
        self.assertIsNone(gate['L_numeric']);self.assertIsNone(gate['C_numeric'])

    def test_empty_search_can_honestly_finalize_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            g=object.__new__(SignalGateway);g.candidates=lambda:[];g.baseline=lambda:{'metrics':{'NMSE':1.,'NMAE':1.}}
            g.signals=lambda:[];g.journal=SimpleNamespace(summary=lambda:{'autonomy_eligible':True})
            recorded={};g.repo=SimpleNamespace(run_id='test',config={'group':'full','feedback_mode':'visible'},
                folder=Path(folder),count_records=lambda k:0,put=lambda key,kind,value,ns:recorded.update({key:value}))
            result=g.submit({'candidate_id':'B0','reason':'证据不足，保留基线'})
            self.assertTrue(result['agent_finalized']);self.assertTrue(result['final_policy']['empty_search'])

    def test_failed_candidate_updates_must_be_completed_before_exit(self):
        g=object.__new__(SignalGateway);g.candidates=lambda:[{'candidate_id':'C1','status':'failed','hypothesis_updated':False}]
        with self.assertRaisesRegex(GateError,'POST_RESULT_UPDATE_REQUIRED'):g.submit({'candidate_id':'B0','reason':'结束'})

    def test_hidden_group_cannot_claim_support_from_unknown_candidate_outcome(self):
        g=object.__new__(SignalGateway);g.repo=SimpleNamespace(config={'feedback_mode':'withheld'},check_binding=lambda _:None)
        g.candidate=lambda _: {'candidate_id':'C1','status':'trained'}
        with self.assertRaisesRegex(GateError,'SIGNAL_OUTCOME_WITHHELD'):
            g.execute({'selected_action':'update_hypothesis','arguments':{'candidate_id':'C1','update':{'assessment':'supported'}},'binding':{}})

if __name__=='__main__':unittest.main()
