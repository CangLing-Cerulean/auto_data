"""Interface and real-T-only checks; no training or fabricated sensor records."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from online_ts.config import GateError
from online_ts.agent_research.phase4 import PilotGateway, PilotModel, FORMAL_PROMPT, conceal
from online_ts.agent_research.runtime import ResearchGateway, CONFIG
from online_ts.agent_runtime.budget import snapshot,check_request,AgentBudgetError
from online_ts.agent_research.task import BASE_ROOT
from online_ts.agent_research.transforms import processor_metadata,load_processor,apply_processor


class ProgramMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.code=Path(self.temp.name)/'processor.py';self.state=Path(self.temp.name)/'processor_state.json'
        self.code.write_text('import torch\ndef transform(x,state):\n    return torch.clamp(x,min=-1,max=1)\n')
        self.state.write_text('{}')

    def test_agent_code_not_a_fixed_mapping_menu(self):
        self.assertEqual(processor_metadata(self.code,self.state),{})

    def test_files_and_network_imports_rejected(self):
        for text in ('import os\ndef transform(x,state):\n    return x',
                     'import torch\ndef transform(x,state):\n    return torch.load("host.pt")',
                     'def transform(x,state):\n    open("host.csv")\n    return x'):
            self.code.write_text(text)
            with self.assertRaises(GateError):processor_metadata(self.code,self.state)

    def test_new_trainable_model_rejected(self):
        self.code.write_text('import torch\ndef transform(x,state):\n    return torch.nn.Linear(15,15)(x)')
        with self.assertRaisesRegex(GateError,'EXTERNAL_EFFECT'):processor_metadata(self.code,self.state)

    def test_state_is_bounded_finite_json(self):
        self.state.write_text('{"value":1e999}')
        with self.assertRaisesRegex(GateError,'STATE'):processor_metadata(self.code,self.state)
        self.state.write_text('[1,2]')
        with self.assertRaisesRegex(GateError,'STATE'):processor_metadata(self.code,self.state)


class FeedbackTests(unittest.TestCase):
    def test_open_search_retains_tools_after_old_slot_limits(self):
        gateway=object.__new__(PilotGateway)
        gateway.repo=SimpleNamespace(config={'max_diagnostics':None,'max_candidates':None},count_records=lambda kind:50)
        gateway.candidates=lambda:[{}]*30
        self.assertIn('run_diagnostic',gateway.catalog())
        self.assertIn('propose_processing',gateway.catalog())

    def test_open_search_passes_old_call_and_token_totals_without_forced_closing(self):
        repo=SimpleNamespace(binding=lambda:{'policy':{'execution_profile':'research_open','max_call_tokens':32768}})
        with patch('online_ts.agent_runtime.budget.usage',return_value={
            'calls_used':500,'tokens_charged_or_reserved':5_000_000,'closing_calls_used':0}):
            check_request(repo,8000,'exploration')
            self.assertIsNone(snapshot(repo)['remaining_calls'])
            self.assertIsNone(snapshot(repo)['remaining_tokens'])
            with self.assertRaises(AgentBudgetError):check_request(repo,40000,'exploration')

    def test_open_durable_state_reports_no_finite_remaining_slots(self):
        gateway=object.__new__(ResearchGateway)
        gateway.repo=SimpleNamespace(config={'max_diagnostics':None,'max_candidates':None},iter_records=lambda kind:[])
        gateway.candidates=lambda:[];gateway.baseline=lambda:{'metrics':{}}
        gateway.journal=SimpleNamespace(summary=lambda:{})
        value=gateway.results()
        self.assertIsNone(value['remaining_diagnostic_slots'])
        self.assertIsNone(value['remaining_candidate_slots'])

    def test_exhausted_slots_are_not_advertised_as_available_tools(self):
        gateway=object.__new__(PilotGateway)
        gateway.repo=SimpleNamespace(config={'max_diagnostics':4,'max_candidates':3},count_records=lambda kind:4)
        gateway.candidates=lambda:[{}, {}, {}]
        tools=gateway.catalog()
        self.assertNotIn('run_diagnostic',tools);self.assertNotIn('propose_processing',tools)
        self.assertIn('submit_policy',tools)

    def test_completed_diagnostics_remain_in_durable_result_state(self):
        gateway=object.__new__(ResearchGateway)
        diagnostic={'record_id':'d1','purpose':'actual scope','status':'completed','facts':{'measured_value':1}}
        gateway.repo=SimpleNamespace(config={'max_diagnostics':4,'max_candidates':3},iter_records=lambda kind:[diagnostic])
        gateway.candidates=lambda:[];gateway.baseline=lambda:{'metrics':{}}
        gateway.journal=SimpleNamespace(summary=lambda:{})
        value=gateway.results()
        self.assertEqual(value['remaining_diagnostic_slots'],3)
        self.assertEqual(value['remaining_candidate_slots'],3)
        self.assertEqual(value['diagnostics'][0]['record_id'],'d1')
        self.assertIn('measured_value',value['diagnostics'][0]['facts_preview'])

    def test_nested_candidate_outcomes_are_removed(self):
        raw={'candidate_id':'C1','status':'trained','result':{'metrics':{'NMSE':.2}},
             'metrics':{'NMSE':.2},'best_step':2500,'receipt_sha256':'a'*64,
             'proposal':{'observation':'Visible pre-result observation'}}
        hidden=conceal(raw)
        self.assertNotIn('result',hidden);self.assertNotIn('metrics',hidden);self.assertNotIn('best_step',hidden)
        self.assertEqual(hidden['receipt_sha256'],'a'*64)
        self.assertIn('proposal',hidden)

    def test_repeated_processing_and_proposal_cannot_reveal_candidate_feedback(self):
        gateway=object.__new__(PilotGateway)
        gateway.repo=SimpleNamespace(config={'feedback_mode':'withheld'},put=lambda *args,**kwargs:None)
        outcome={'status':'trained','candidate_id':'C1','result':{'metrics':{'NMSE':.2}},'receipt_sha256':'b'*64}
        with patch.object(ResearchGateway,'execute',return_value=outcome):
            for tool in ('run_processing','evaluate_candidate','propose_processing'):
                args={'parameters':{},'proposal':{'expected_effect':{'metric':'NMSE','target_scope':'all','direction':'decrease'}}}
                value=gateway.execute({'selected_action':tool,'arguments':args,'decision_id':'d1'})
                self.assertNotIn('result',value)
                self.assertEqual(value['development_outcome'],'withheld_until_frozen_submission')

    def test_withheld_updates_cannot_claim_supported(self):
        gateway=object.__new__(PilotGateway);gateway.repo=SimpleNamespace(config={'feedback_mode':'withheld'})
        with self.assertRaisesRegex(GateError,'OUTCOME_WITHHELD'):
            gateway.execute({'selected_action':'update_hypothesis','arguments':{'update':{'assessment':'supported'}}})

    def test_parameter_adaptation_is_disabled_equally(self):
        gateway=object.__new__(PilotGateway);gateway.repo=SimpleNamespace(config={'feedback_mode':'visible'})
        with self.assertRaisesRegex(GateError,'FIXED_PARAMETERS'):
            gateway.execute({'selected_action':'propose_processing','arguments':{'parameters':{'learning_rate':.003}}})

    def test_model_generation_and_operation_bind_temperature(self):
        repo=SimpleNamespace(config=CONFIG);config=SimpleNamespace(planner=SimpleNamespace(model=CONFIG['model']))
        model=PilotModel(repo,config,system_prompt=FORMAL_PROMPT)
        context={'phase':'exploration','context_version':'phase4-interface-test'}
        body,operation=model.request_spec(context,0)
        self.assertEqual(body['temperature'],.8);self.assertEqual(body['reasoning_effort'],'none')
        self.assertNotEqual(operation,model.request_spec(context,1)[1])


@unittest.skipUnless((BASE_ROOT/'data/metropt3_v1/train.csv').is_file(),'Registered server T required')
class ActualTrainingHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        raw=np.loadtxt(BASE_ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17),max_rows=41)
        cls.torch=torch;cls.original=torch.tensor(raw,dtype=torch.float32)
        cls.x=cls.original[:32][None].clone()

    def test_freely_written_clipping_program_and_output_contract(self):
        program=(lambda x,state:self.torch.clamp(x,min=state['min'],max=state['max']),{'min':-1,'max':1})
        value=apply_processor(self.x,program)
        self.assertEqual(value.shape,self.x.shape)
        self.assertTrue(self.torch.equal(value,self.torch.clamp(self.x,min=-1,max=1)))

    def test_known_half_input_can_be_written_as_a_program(self):
        def processing(x,state):
            source=self.torch.tensor(state['source'],device=x.device)
            mixed=x[:,-1:]+.5*(x-x[:,-1:])+.5*(x[:,:,source]-x[:,-1:,source])
            changed=source!=self.torch.arange(15,device=x.device)
            return self.torch.where(changed[None,None,:],mixed,x)
        value=apply_processor(self.x,(processing,{'source':[1]+list(range(1,15))}))
        expected=self.x[:,-1:,0]+.5*(self.x[:,:,0]-self.x[:,-1:,0])+.5*(self.x[:,:,1]-self.x[:,-1:,1])
        self.assertTrue(self.torch.allclose(value[:,:,0],expected,atol=1e-6,rtol=0))
        self.assertTrue(self.torch.equal(value[:,:,1:],self.x[:,:,1:]))

    def test_identity_control_bypasses_processor_without_mutating_input(self):
        value=apply_processor(self.x,(lambda x,state:x*2,{}),identity_control=True)
        self.assertTrue(self.torch.equal(value,self.x))

    def test_shape_change_and_nonfinite_output_rejected(self):
        for function in (lambda x,state:x[:,:,:1],lambda x,state:x*self.torch.tensor(float('nan'))):
            with self.assertRaises(GateError):apply_processor(self.x,(function,{}))


if __name__=='__main__':unittest.main()
