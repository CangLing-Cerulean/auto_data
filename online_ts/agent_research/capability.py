"""Small auditable historical-evidence acceptance, separate from discovery experiments."""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

from ..agent_runtime.controller import AgentController,DecisionSchemaError
from ..agent_runtime.state import Event
from ..agent_runtime.budget import snapshot
from ..config import GateError,canonical,digest
from ..selection.common import read_json,write_json,file_hash
from .experience import ROOT,STUDY
from .memory import MEMORY,verify_memory
from .processing import DEFAULTS
from .runtime import CONFIG,ResearchRepository,ResearchGateway,EngineeringModel,context_builder,code_identity,PROMPT
from .task import TaskAssets

PILOT=STUDY/'capability_acceptance/20261010_v4'
OUTPUT=Path('/tmp/auto_data_capability_acceptance_20261010_v4')
FIELDS={'execution_status','nmse_change_pct','nmae_change_pct','meets_processing_criterion',
    'assessment','claim_scope','physical_causality_proven','independent_task_proven','action','source_refs','summary'}
BANK_PROMPT='''Assess historical experiment evidence using tools, not guesses. Read the assigned case before submitting.
The criterion is NMSE decrease >=1%, NMAE deterioration <=1%; positive change_pct means deterioration.
A completed candidate can contradict this criterion even if its NMAE improves. Failed execution means
the scientific hypothesis is untested: metrics and criterion should be null, assessment inconclusive.
Multiple confirmation seeds and matched controls can support scoped development evidence, but do not
establish independent-dataset generalization or sensor physical causality. A single training seed is
single_seed_development. You can calculate exact metric changes through calculate_case_metrics.
Use only the case and its source IDs; retain limitations in a concise Chinese summary. No new training.
confirmed_development means multiple confirmation TRAINING SEEDS with matched controls. A single
dataset or model configuration does not make a multi-seed result single_seed_development. Independent
task evaluation is a separate dimension. Distinguish aggregate changes from individual seed changes;
report disagreement instead of describing every seed as equally improved or negligible.
NMAE deterioration <=1% is ONE-SIDED: larger improvements are allowed, not a symmetric +/-1% interval.
source_refs inside the assessment must contain exactly the source_id values returned by read_case.
The visible case:ID may be cited in top-level evidence_refs, but is not an original source_id.
An execution error does not establish an installation defect; keep repair suggestions grounded in
the recorded failure and distinguish a proposed investigation from a proven cause.
Every assessment must contain exactly the fields in the advertised tool schema. Grading answers are not accessible.
Return JSON ONLY. The outer action is an OBJECT containing tool and arguments, not a tool-name string.
reason, hypotheses and evidence_refs belong at the TOP LEVEL, outside action and its arguments.
The assessment's action is a separate scientific next-action label, nested inside arguments.assessment;
never confuse it with the outer tool-call action. Top-level hypotheses is an array of strings or [].
STRUCTURE-ONLY EXAMPLE, replace the case ID with the assigned ID:
{"action":{"tool":"read_case","arguments":{"case_id":"ASSIGNED_CASE_ID"}},"reason":"读取原始案例证据","hypotheses":[],"evidence_refs":[]}
For submit_assessment, place the complete assessment object at action.arguments.assessment, alongside
action.arguments.case_id. Do not place assessment fields at the top level. Choose content from evidence.
'''
LOOP_PROMPT='''This is an ENGINEERING REPLICATION acceptance on an already explored dataset, not new discovery.
Read the supplied registered historical processing case, diagnose real T, and independently write a
processing program to reproduce it with the fixed trusted trainer. You may reject it after measured feedback.
read_registered_case returns the exact original processing_spec, not just a prose lesson. Use its
equation and mapping; a centered history means history minus its own last value, not a window mean.
The code argument to propose_processing is an EXECUTABLE ASSET BUILDER, not a standalone transform.
When run as a script it reads sys.argv[1], writes to sys.argv[2], and exits. It creates three files:
processing.npz (start_row, weight, sampling_multiplier), processor.py, processor_state.json.
processor.py is a SEPARATE output file, defining transform(x,state); do not submit only that function
as the builder. Write the builder's entry point explicitly so it actually runs. Never use EXP_INPUT_DIR.
Do not build or save a full (all_windows,32,15) tensor or processed window dataset. The trusted trainer
applies processor.py on small batches. The builder writes only window arrays and fitted state/code.
Do not create any forecasting model, optimizer, training loop, prediction metrics or result.json in
diagnostic/processing programs. Training, targets and D metrics belong exclusively to the trusted trainer.
The only processing input file is sys.argv[1]/train_inputs.npz; both sys.argv[1] and sys.argv[2] are DIRECTORIES.
values has shape (training_rows,15), not (windows,32,15). Legal past windows are
values[start_row[:,None]+numpy.arange(32)[None,:]]. The data are already T-normalized.
Write diagnostic.json to the output directory. Processing always writes processing.npz with unchanged int64
start_row and finite floating weight/sampling_multiplier in [0,4]. For input processing also write
processor.py defining transform(x,state), x=(batch,32,15), same finite shape/dtype/device, and processor_state.json.
The function uses torch/math, no files/network/random/global state, no mutation or cross-window statistics.
Fit all parameters on T. Do not use future targets as inputs, read other host paths or install dependencies.
No new trainable model. All proposal parameters={}, DLinear and its 3000-step dose stay fixed.
History and candidate details are durable; read older evidence through page tools when absent from context.
Use exact nested proposal and update schemas. Copy the actual receipt_sha256. Execution failures are inconclusive.
Every attempted candidate must be executed, evaluated or recorded failed, and updated before final submission.
At least one successfully trained candidate, completed new diagnostic, and registered-case read are required.
Choose the lowest measured eligible NMSE including B0 (candidate NMAE<=1.01*B0). A single seed is not final efficacy.
No cumulative API/token/diagnostic/candidate/training-job/time cap. Decide when this acceptance is complete.
Scientific conclusions must distinguish measured outcome from mechanism and generalization.
''' + PROMPT[PROMPT.index('Keep Chinese decision summaries'):]


def open_config():
    return CONFIG|{'version':'capability-acceptance-20261010-v4','scope':'engineering_only',
        'execution_profile':'research_open','max_calls':None,'max_llm_tokens':None,
        'max_candidates':None,'max_diagnostics':None,'max_training_jobs':None,
        'seed':20261010,'input_transforms_allowed':True,'formal_research_protocol':False,
        'parameter_ranges':{k:[v] for k,v in DEFAULTS.items()}}


def model_config(config):
    key=Path('/root/.config/auto_data_research/deepseek_api_key')
    if key.stat().st_mode&0o077:raise GateError('SERVER_KEY_PERMISSIONS')
    os.environ['DEEPSEEK_API_KEY']=key.read_text().strip()
    return SimpleNamespace(planner=SimpleNamespace(model=config['model'],kind='external',allow_external=True,
        timeout_seconds=config['timeout_seconds'],endpoint_env='DEEPSEEK_BASE_URL',key_env='DEEPSEEK_API_KEY'))


def recover_format(repo,error):
    number=repo.count_records('format_failure')+1
    repo.put('format:'+str(number),'format_failure',{'number':number,'error':str(error),
        'validation_errors':error.validation_errors,'action_executed':False},time.time_ns())


def require_preflight():
    ready=read_json(PILOT/'preflight.json');protocol=read_json(PILOT/'protocol.json')
    if ready['status']!='passed' or ready['code']!=code_identity() or ready['protocol_sha256']!=file_hash(PILOT/'protocol.json'):
        raise GateError('CAPABILITY_PREFLIGHT_BINDING')
    if protocol['expected_sha256']!=file_hash(PILOT/'private_expected.json') or protocol['cases_sha256']!=file_hash(PILOT/'cases.json'):
        raise GateError('CAPABILITY_CASE_BINDING')
    if protocol['bank_summary_sha256']!=file_hash(PILOT/'bank_summary.json'):
        raise GateError('CAPABILITY_ACCEPTED_BANK_CHANGED')


def grade_assessment(case,expected,value,read_count):
    checks={'case_read':read_count>0}
    for key,target in expected.items():
        answer=value.get(key)
        if key in {'nmse_change_pct','nmae_change_pct'} and target is not None:
            checks[key]=type(answer) in {int,float} and abs(answer-target)<=.05
        else:checks[key]=type(answer)is type(target) and answer==target
    checks['source_refs']=set(value.get('source_refs',[]))=={s['source_id'] for s in case['sources']}
    checks['summary_present']=isinstance(value.get('summary'),str) and 1<=len(value['summary'])<=1200
    return {'passed':all(checks.values()),'checks':checks,
        'free_text_truth':'unassessed_requires_independent_review','structured_score_only':True}


class CaseGateway:
    environment_kind='historical_evidence_capability_acceptance'
    def __init__(self,repo,case):self.repo,self.case=repo,case
    def catalog(self):
        return {'read_case':{'case_id':'assigned case ID'},'calculate_case_metrics':{'case_id':'already read case ID'},
            'submit_assessment':{'case_id':'assigned case ID','assessment':{
                'execution_status':'completed | failed','nmse_change_pct':'number or null; positive is deterioration',
                'nmae_change_pct':'number or null; positive is deterioration','meets_processing_criterion':'boolean or null',
                'assessment':'supported | contradicted | inconclusive',
                'claim_scope':'confirmed_development | single_seed_development | execution_only',
                'physical_causality_proven':'boolean','independent_task_proven':'boolean',
                'action':'reuse_as_known_example | retain_baseline | repair_execution_and_retest',
                'source_refs':'array of case source_id values','summary':'Chinese evidence-based explanation <=1200 characters'}}}
    def execute(self,record):
        tool,args=record['selected_action'],record['arguments']
        if tool not in self.catalog() or set(args)!=set(self.catalog()[tool]) or args['case_id']!=self.case['case_id']:
            raise GateError('CASE_TOOL_ARGUMENTS')
        if tool=='read_case':
            for source in self.case['sources']:
                if file_hash(PILOT/source['artifact'])!=source['sha256']:raise GateError('CASE_SOURCE_CHANGED')
            self.repo.put('read:'+record['decision_id'],'case_read',{'case_id':self.case['case_id']},time.time_ns())
            return deepcopy(self.case)
        if not self.repo.count_records('case_read'):raise GateError('CASE_READ_REQUIRED')
        if tool=='calculate_case_metrics':
            data=self.case['measurement']
            if data['execution_status']!='completed':return {'NMSE_change_pct':None,'NMAE_change_pct':None,'no_model_result':True}
            return {key+'_change_pct':(data['candidate'][key]/data['baseline'][key]-1)*100 for key in ('NMSE','NMAE')}
        value=args['assessment']
        if not isinstance(value,dict) or set(value)!=FIELDS:raise GateError('CASE_ASSESSMENT_SCHEMA')
        if not isinstance(value['summary'],str) or not 1<=len(value['summary'])<=1200:raise GateError('CASE_SUMMARY_SIZE')
        self.repo.put('assessment','case_assessment',value,time.time_ns())
        return {'agent_finalized':True,'case_id':self.case['case_id'],'assessment_recorded':True}


def case_context(repo,gateway,event):
    reads=repo.count_records('case_read');failures=list(repo.iter_records('format_failure'))
    decisions=list(repo.iter_records('decision'))
    value={'phase':'exploration','task':{'goal':'assess_historical_evidence','case_id':gateway.case['case_id']},
        'tools':gateway.catalog(),'visible_evidence_refs':(['case:'+gateway.case['case_id']]+
            [row['source_id'] for row in gateway.case['sources']]) if reads else [],
        'history':[{'tool':row['selected_action'],'result':row.get('actual_result')} for row in decisions[-3:]],
        'memory':[],'knowledge':[],'budget':snapshot(repo)}
    if failures:value['format_failure']=failures[-1]
    value['context_version']=digest(value);value['binding']=repo.binding();return value


def run_bank():
    raise GateError('V4_INHERITS_ACCEPTED_BANK_NO_REPEAT')


def check_known_code(code):
    """Reject observed torch model/training APIs before executing a known-case builder."""
    ResearchGateway.check_code(code)
    try:tree=ast.parse(code)
    except SyntaxError as error:raise GateError('KNOWN_CODE_SYNTAX') from error
    forbidden={'nn','optim','backward','parameters','requires_grad_','__import__','eval','exec','compile','getattr'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import) and any(alias.name.startswith(('torch.nn','torch.optim')) for alias in node.names):
            raise GateError('KNOWN_NO_MODEL_TRAINING_IN_CODE')
        if isinstance(node,ast.ImportFrom) and ((node.module or '').startswith(('torch.nn','torch.optim'))
                or node.module=='torch' and any(alias.name in {'nn','optim'} for alias in node.names)):
            raise GateError('KNOWN_NO_MODEL_TRAINING_IN_CODE')
        if isinstance(node,ast.Attribute) and node.attr in forbidden:
            raise GateError('KNOWN_NO_MODEL_TRAINING_IN_CODE')
        if isinstance(node,ast.Name) and node.id in forbidden:
            raise GateError('KNOWN_NO_MODEL_TRAINING_IN_CODE')


class KnownGateway(ResearchGateway):
    def catalog(self):return super().catalog()|{'read_registered_case':{}}
    def execute(self,record):
        if record['selected_action']=='read_registered_case':
            if record['arguments']:raise GateError('KNOWN_CASE_ARGUMENTS')
            value=read_json(PILOT/'registered_case.json')
            for name,key in (('path','sha256'),('recipe_path','recipe_sha256')):
                if file_hash(ROOT/value['source'][name])!=value['source'][key]:raise GateError('KNOWN_CASE_SOURCE_CHANGED')
            self.journal.evidence(value['record_id'],dataset_id=self.task.task['dataset_id'],role='historical_development',
                artifact_sha256=value['source']['excerpt_sha256'],summary=value['lesson']+'; '+value['limits'])
            self.repo.put('known:'+record['decision_id'],'known_case_read',{'record_id':value['record_id']},time.time_ns())
            recipe=read_json(ROOT/value['source']['recipe_path'])
            spec={key:recipe[key] for key in ('transformation','mapping','changed_channels','alpha','data_membership','physical_record_changes')}
            self.journal.evidence(value['record_id']+':recipe',dataset_id=self.task.task['dataset_id'],
                role='historical_development',artifact_sha256=value['source']['recipe_sha256'],summary=canonical(spec))
            return value|{'processing_spec':spec,'spec_inputs':'already_T_normalized_history_only',
                'builder_contract':{'input':'directory sys.argv[1]/train_inputs.npz','output':'directory sys.argv[2]',
                    'required_files':['processing.npz','processor.py','processor_state.json'],
                    'processing_arrays':['start_row','weight','sampling_multiplier'],
                    'processor_entry':'transform(x,state), x=(batch,32,15)',
                    'forecast_models_and_training_in_builder':False}}
        if record['selected_action'] in {'propose_processing','run_diagnostic'}:
            check_known_code(record['arguments'].get('code'))
        if record['selected_action']=='propose_processing' and record['arguments'].get('parameters')!={}:raise GateError('KNOWN_FIXED_PARAMETERS')
        return super().execute(record)
    def submit(self,args):
        if not self.repo.count_records('known_case_read'):raise GateError('KNOWN_CASE_READ_REQUIRED')
        return super().submit(args)


def known_context(repo,gateway,event):
    value=context_builder(repo,gateway,event)
    value['task']['goal']='known_case_replication_and_feedback_engineering_acceptance'
    failures=list(repo.iter_records('format_failure'))
    if failures:value['format_failure']=failures[-1]
    value['context_version']=digest({k:v for k,v in value.items() if k not in ('context_version','binding')})
    return value


def run_known_loop():
    require_preflight()
    protocol=read_json(PILOT/'protocol.json')
    if not read_json(PILOT/'bank_summary.json')['all_structured_checks_passed']:raise GateError('CASE_BANK_NOT_PASSED')
    review=read_json(PILOT/'bank_review.json')
    if review.get('status')!='passed' or review.get('bank_summary_sha256')!=file_hash(PILOT/'bank_summary.json'):
        raise GateError('CASE_BANK_TEXT_REVIEW_NOT_PASSED')
    folder=OUTPUT/'known_loop'
    if folder.exists():raise GateError('KNOWN_LOOP_ALREADY_STARTED_NO_REPLAY')
    task=TaskAssets(STUDY/'tasks/metropt3_dev');library=read_json(MEMORY/'library.json');verify_memory(library)
    config=open_config();frozen=code_identity()
    identity={'config':config,'code':frozen,'protocol_sha256':file_hash(PILOT/'protocol.json'),
        'task_sha256':file_hash(task.folder/'task.json'),'library_sha256':library['sha256'],
        'case_sha256':file_hash(PILOT/'registered_case.json'),'prompt_sha256':digest(LOOP_PROMPT),
        'bank_review_sha256':file_hash(PILOT/'bank_review.json'),
        'accepted_bank_summary_sha256':file_hash(PILOT/'bank_summary.json')}
    identity['sha256']=digest(identity);repo=ResearchRepository(protocol['known_loop_run_id'],identity,folder=folder)
    write_json(folder/'preregistration.json',identity);gateway=KnownGateway(repo,task,library)
    controller=AgentController(repo,gateway,EngineeringModel(repo,model_config(config),system_prompt=LOOP_PROMPT),context_builder=known_context)
    started=time.monotonic()
    try:
        gateway.baseline()
        while not repo.state().incident_finished:
            if code_identity()!=frozen:raise GateError('KNOWN_LOOP_CODE_CHANGED')
            try:controller.step(Event('known-'+str(repo.state().decision_version),'known_case_acceptance',time.time_ns()))
            except DecisionSchemaError as error:recover_format(repo,error)
            repo.export()
        baseline=gateway.baseline()['metrics'];rows=gateway.candidates();feedback=[]
        for row in rows:
            if row['status']!='trained':continue
            metric=row['result']['metrics'];gain=(1-metric['NMSE']/baseline['NMSE'])*100
            mae=(metric['NMAE']/baseline['NMAE']-1)*100
            feedback.append({'candidate_id':row['candidate_id'],'NMSE_gain_pct':gain,'NMAE_deterioration_pct':mae,
                'recorded_assessment':row['post_result_update']['assessment'],'criterion_supported':gain>=1 and mae<=1,
                'receipt_traceable':row['post_result_update']['receipt_sha256']==row['receipt_sha256']})
        result={'status':'completed_engineering','final_policy':repo.get('final_policy'),'baseline':baseline,
            'known_case_reads':repo.count_records('known_case_read'),'feedback_checks':feedback,
            'API':snapshot(repo),'elapsed_seconds':time.monotonic()-started,'independent_task':False,
            'autonomous_discovery':False,'formal_processing_effectiveness':'not_established_by_single_seed',
            'diagnosis_facts':'requires_recomputation_review'}
    except Exception as error:
        result={'status':'failed_preserved','error':str(error) if isinstance(error,GateError) else type(error).__name__,
            'API':snapshot(repo),'elapsed_seconds':time.monotonic()-started,'same_ID_replay':False}
    finally:repo.export()
    write_json(folder/'result_summary.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('bank','loop'),required=True);args=parser.parse_args()
    result=run_bank() if args.stage=='bank' else run_known_loop()
    print(canonical({k:result[k] for k in ('status','passed','registered','error') if k in result}),flush=True)
