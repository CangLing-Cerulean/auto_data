"""Regression checks for real failure classes: paging, request fitting and feedback grading."""
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_ts.agent_research.capability import grade_assessment,LOOP_PROMPT,BANK_PROMPT,check_known_code
from online_ts.config import GateError
from online_ts.agent_runtime.controller import AgentController,DecisionSchemaError
from online_ts.agent_research.runtime import EngineeringModel,ResearchGateway,CONFIG,TOOLS
from online_ts.agent_runtime.llm import reservation_for
from online_ts.agent_research.experience import STUDY
from online_ts.agent_research.task import TaskAssets


class CapabilityChecks(unittest.TestCase):
    def test_observed_processing_model_training_is_rejected_before_execution(self):
        with self.assertRaisesRegex(GateError,'KNOWN_NO_MODEL_TRAINING_IN_CODE'):
            check_known_code('import torch.nn as nn\nmodel=nn.Linear(24,8)\n')
        with self.assertRaisesRegex(GateError,'KNOWN_NO_MODEL_TRAINING_IN_CODE'):
            check_known_code('import torch\nmodel=torch.nn.Linear(24,8)\n')
        with self.assertRaisesRegex(GateError,'KNOWN_NO_MODEL_TRAINING_IN_CODE'):
            check_known_code('from torch import optim\n')
        check_known_code('import sys,json,numpy as np\nfrom pathlib import Path\noutput=Path(sys.argv[2])\n')

    def test_case_example_has_the_executable_controller_shape(self):
        example=BANK_PROMPT.split('STRUCTURE-ONLY EXAMPLE, replace the case ID with the assigned ID:\n',1)[1].split('\n',1)[0]
        value=json.loads(example)
        AgentController._validate(value)
        self.assertEqual(value['action']['tool'],'read_case')
        with self.assertRaises(DecisionSchemaError):
            AgentController._validate(value|{'action':'read_case','arguments':value['action']['arguments']})

    def test_multi_seed_scope_and_one_sided_metric_conflict_are_scored_separately(self):
        case={'sources':[{'source_id':'result'}]}
        expected={'claim_scope':'confirmed_development','meets_processing_criterion':True}
        value=expected|{'source_refs':['result'],'summary':'多种子确认仍限于本数据开发证据'}
        self.assertTrue(grade_assessment(case,expected,value,1)['passed'])
        self.assertFalse(grade_assessment(case,expected,value|{'claim_scope':'single_seed_development'},1)['passed'])

    def test_visible_case_reference_is_not_an_original_source_reference(self):
        case={'sources':[{'source_id':'baseline'},{'source_id':'candidate'}]}
        value={'source_refs':['baseline','candidate'],'summary':'有来源的判断'}
        self.assertTrue(grade_assessment(case,{},value,1)['passed'])
        self.assertFalse(grade_assessment(case,{},value|{'source_refs':['case:C02','baseline','candidate']},1)['passed'])

    def test_cli_spawn_import_does_not_start_another_bank(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_capability.py'
        with patch('online_ts.agent_research.capability.run_bank',side_effect=AssertionError('Spawn reentered bank')):
            namespace=runpy.run_path(str(script),run_name='__mp_main__')
        self.assertTrue(callable(namespace['main']))
    @unittest.skipUnless((STUDY/'tasks/metropt3_dev/task.json').is_file(),'Registered server T required')
    def test_profile_reports_the_actual_row_matrix_and_legal_window_contract(self):
        task=TaskAssets(STUDY/'tasks/metropt3_dev')
        gateway=object.__new__(ResearchGateway);gateway.task=task;gateway.inputs=task.export_inputs()
        gateway.repo=SimpleNamespace(config=CONFIG,check_binding=lambda binding:None)
        gateway.journal=SimpleNamespace(evidence=lambda *args,**kwargs:None)
        value=gateway.execute({'selected_action':'profile_training_data','arguments':{},'binding':{}})
        contract=value['input_contract']
        self.assertEqual(contract['values_shape'],[1061863,15])
        self.assertEqual(contract['start_row_shape'],[task.task['legal_windows']])
        self.assertTrue(contract['values_are_rows_not_windows'])
        self.assertEqual(contract['input_length'],32)

    def test_actual_request_with_recovery_fits_and_original_code_is_preserved(self):
        repo=SimpleNamespace(config=CONFIG);planner=SimpleNamespace(model=CONFIG['model'])
        model=EngineeringModel(repo,SimpleNamespace(planner=planner),system_prompt=LOOP_PROMPT)
        code='print(1)\n'*850
        context={'phase':'exploration','context_version':'request-regression',
            'binding':{'policy':{'max_call_tokens':32768}},'history':[{'full_error':'x'*16000} for _ in range(8)],
            'memory':[],'results':{},'format_recovery':{'rejected_action':{'arguments':{'code':code}}}}
        body,operation=model.request_spec(context,0)
        self.assertLessEqual(reservation_for(body['messages'],'exploration'),32768)
        self.assertEqual(context['format_recovery']['rejected_action']['arguments']['code'],code)
        self.assertEqual(len(context['history']),8)
        self.assertNotEqual(operation,model.request_spec(context,1)[1])

    def test_many_candidates_and_diagnostics_do_not_expand_active_state_without_bound(self):
        gateway=object.__new__(ResearchGateway)
        rows=[{'candidate_id':'C'+str(i),'index':i,'status':'failed','error':'short error','proposal':{'text':'x'*4000}} for i in range(100)]
        diagnostics=[{'record_id':'d'+str(i),'purpose':'actual record','status':'completed','facts':{'count':i}} for i in range(100)]
        gateway.repo=SimpleNamespace(config={'max_candidates':None,'max_diagnostics':None},iter_records=lambda kind:diagnostics)
        gateway.candidates=lambda:rows;gateway.baseline=lambda:{'metrics':{}}
        gateway.journal=SimpleNamespace(summary=lambda:{})
        value=gateway.results()
        self.assertEqual(value['candidate_count'],100);self.assertEqual(value['diagnostic_count'],100)
        self.assertEqual(len(value['candidates']),4);self.assertEqual(len(value['diagnostics']),4)
        self.assertLess(len(json.dumps(value)),6000)

    def test_candidate_detail_pages_preserve_feedback_withholding(self):
        gateway=object.__new__(ResearchGateway)
        gateway.repo=SimpleNamespace(config={'feedback_mode':'withheld'},check_binding=lambda binding:None)
        gateway.candidate=lambda name:{'candidate_id':name,'status':'trained','proposal':{'observation':'T fact'},
            'receipt_sha256':'a'*64,'result':{'metrics':{'NMSE':.2,'NMAE':.1}}}
        value=gateway.execute({'selected_action':'read_candidate','arguments':{'candidate_id':'C1','offset':0,'limit':1000},'binding':{}})
        record=json.loads(value['text'])
        self.assertNotIn('metrics',record);self.assertNotIn('result',record)
        self.assertIn('receipt_sha256',record)

    def test_direct_evidence_pages_preserve_feedback_withholding(self):
        gateway=object.__new__(ResearchGateway)
        gateway.repo=SimpleNamespace(config={'feedback_mode':'withheld'},check_binding=lambda binding:None)
        evidence={'evidence_id':'result:C1','role':'development_metrics','summary':'NMSE=0.2', 'artifact_sha256':'a'*64}
        gateway.journal=SimpleNamespace(repo=SimpleNamespace(get=lambda key:evidence))
        value=gateway.execute({'selected_action':'read_evidence','arguments':{'evidence_id':'result:C1','offset':0,'limit':1000},'binding':{}})
        self.assertNotIn('NMSE=0.2',value['text']);self.assertIn('withheld',value['text'])

    def test_nested_update_contract_names_all_required_fields(self):
        self.assertEqual(set(TOOLS['update_hypothesis']['update']),
            {'assessment','reason','revised_hypothesis','next_action','receipt_sha256'})

    def test_failed_execution_cannot_pass_as_scientific_refutation(self):
        case={'sources':[{'source_id':'execution'}]}
        expected={'assessment':'inconclusive','nmse_change_pct':None,'nmae_change_pct':None,
            'meets_processing_criterion':None,'physical_causality_proven':False}
        wrong=expected|{'assessment':'contradicted','source_refs':['execution'],'summary':'未完成执行'}
        score=grade_assessment(case,expected,wrong,1)
        self.assertFalse(score['passed']);self.assertFalse(score['checks']['assessment'])

    def test_unsupported_causality_claim_is_not_hidden_by_correct_metric_fields(self):
        case={'sources':[{'source_id':'result'}]}
        expected={'physical_causality_proven':False,'nmse_change_pct':-4.2385}
        value=expected|{'physical_causality_proven':True,'source_refs':['result'],'summary':'数据处理改善'}
        self.assertFalse(grade_assessment(case,expected,value,1)['passed'])


if __name__=='__main__':unittest.main()
