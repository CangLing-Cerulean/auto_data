"""Engineering autonomous data-processing loop on the dedicated Linux server."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

DEPENDENCY_ROOT=Path(__file__).resolve().parents[2]/'runtime_deps'
if DEPENDENCY_ROOT.is_dir():
    sys.path.insert(0,str(DEPENDENCY_ROOT))

import numpy as np

from ..agent_runtime.budget import snapshot, AgentBudgetError
from ..agent_runtime.controller import AgentController
from ..agent_runtime.llm import DeepSeekModel
from ..agent_runtime.state import Event, Repository
from ..config import GateError, canonical, digest
from ..store import Store
from ..selection.common import file_hash, read_json, write_json
from .evidence import EvidenceJournal
from .experience import ROOT, STUDY
from .memory import MEMORY, EvidenceMemory, verify_memory
from .jobs import prepare_job
from .native import execute_job,run_process
from .processing import DEFAULTS, PARAMETERS, diagnostic_json, parameters, validate_processing
from .task import TaskAssets
from .training import TrainingAdapter

CONFIG={'version':'native-agent-engineering-memory-v2','scope':'engineering_only','seed':20261009,
        'model':'deepseek-flash','max_candidates':2,'max_diagnostics':2,'max_calls':36,
        'reasoning_effort':'none',
        'experience_mode':'full','dataset_family':'metropt3',
        'max_llm_tokens':500000,'max_call_tokens':32768,'timeout_seconds':90,
        'track':'experience_combination','architecture':'original_individual_DLinear',
        'input_length':32,'prediction_length':8,'steps':3000,
        'parameter_ranges':{k:list(v) for k,v in PARAMETERS.items()},
        'human_strategy_intervention':False,'formal_research_protocol':False}

PROMPT="""You are the agent in an ENGINEERING acceptance run of autonomous time-series data processing.
Choose one tool per turn. Your objective is to diagnose training-data issues, propose a testable
processing hypothesis, write actual new NumPy processing code, test it with the trusted fixed
DLinear trainer, and update your hypothesis AFTER observing results. Read historical experience
but treat all lessons as conditional, revisable interpretations of scoped observations.
Your primary research goal is to locate measured prediction deficits, connect diagnostic evidence
to a testable processing hypothesis, and evaluate whether the predicted change actually occurs.
A skewed feature distribution alone does not establish a performance cause or a correct signal.
Memory is optional evidence, never a method-family ban or a list of approved directions.
You may challenge, extend, or replicate past recipes. Historical model changes are context only;
the current fixed architecture still applies. In mode=none, memory tools expose no records and
experience retrieval is not required. In new-task discovery or other_tasks mode, target-family
history is excluded. Use list_experience for coverage and read_experience pages for original details.
No strategy suggestions or candidate code are supplied by humans during this run.
Experience access is optional, including in full mode. This ENGINEERING run exercises
at least one coded diagnostic, one processing candidate,
its execution, training feedback (or an honestly reported failed attempt), an explicit post-result
update, and final submission. This workflow requirement is engineering coverage, not proof of research ability.
You have two diagnostic slots and two processing candidate slots; use remaining slots carefully.
Do not assume you need to exhaust them. Model architecture, forecast target/horizon and data partition
are fixed. Input history is fixed at 32 for this engineering run; its future adaptation is not tested here.
Diagnose T, fit processing on T only, and get D metrics only through the trainer. Never read other
host paths, credentials, other experiment folders, test, E, or ground_truth; do not install dependencies
or use network/subprocesses in generated code. Native execution is not an OS security sandbox.
All generated code uses sys.argv[1] as the input folder and sys.argv[2] as the output folder.
Load inputs/train_inputs.npz with numpy.load(..., allow_pickle=False). It contains values (T-normalized
row matrix), start_row (legal T window starts), groups/group_names, source-index mapping, sensor_columns,
and input_variation, analog_mean, analog_std, digital_transitions, elapsed_days, month, state, group_size.
Diagnostics must write outputs/diagnostic.json: one bounded finite JSON object with measured facts.
Processing must write outputs/processing.npz with exactly start_row (unchanged int64), weight and
sampling_multiplier (floating arrays of the same length, finite within [0,4]). Zeros delete windows.
Weight affects loss; sampling_multiplier affects draw probability. The platform normalizes the expected
loss scale, requires 50% active windows, per-group coverage and protected rare groups, and rejects
historically repeated deletion. A scalar weight alone is normalized away and cannot improve the recipe.
The proposal object has observation, hypothesis, expected_effect {metric,target_scope,direction,criterion},
falsification, control, evidence_refs. Cite ONLY visible_evidence_refs. Code hash is filled by the platform.
Post-result update has assessment (supported/contradicted/inconclusive), reason, revised_hypothesis,
next_action, receipt_sha256 (copy the actual receipt's artifact_sha256). Execution failure is inconclusive.
Agent self-assessment is not a scientific correctness label. Report conflicting metrics and limitations.
Select the lowest measured comparable development NMSE with NMAE no worse than B0 by 1%; baseline
is eligible if no candidate qualifies. Hyperparameter changes confound data attribution unless matched
controls are run; this acceptance does not establish data gains or independent generalization.
Keep Chinese decision summaries brief and evidence-based. Return JSON ONLY with exactly:
{"action":{"tool":"name","arguments":{}},"reason":"summary","hypotheses":[],"evidence_refs":[]}.
The top-level hypotheses is ALWAYS an array of short strings (or []), never proposal objects.
The proposal belongs ONLY in action.arguments.proposal. reason, hypotheses, evidence_refs
are siblings of action, not fields inside action or arguments. Escape Python newlines as JSON
string escapes. This STRUCTURE-ONLY example contains placeholders, not a proposed experiment:
{
  "action": {
    "tool": "propose_processing",
    "arguments": {
      "candidate_id": "YOUR_UNIQUE_ID",
      "code": "YOUR_PYTHON_SOURCE",
      "parameters": {},
      "proposal": {
        "observation": "YOUR_MEASURED_OBSERVATION",
        "hypothesis": "YOUR_TESTABLE_HYPOTHESIS",
        "expected_effect": {
          "metric": "YOUR_METRIC", "target_scope": "YOUR_SCOPE",
          "direction": "decrease", "criterion": "YOUR_CRITERION"
        },
        "falsification": "YOUR_FALSIFICATION",
        "control": "YOUR_CONTROL",
        "evidence_refs": ["COPY_A_VISIBLE_EVIDENCE_ID"]
      }
    }
  },
  "reason": "YOUR_BRIEF_DECISION_SUMMARY",
  "hypotheses": [],
  "evidence_refs": ["COPY_A_VISIBLE_EVIDENCE_ID"]
}
Replace all placeholders with your own evidence-based content. The final response must parse
as one JSON object; do not close the outer object before writing its reason field.
"""

TOOLS={
 'describe_task':{},'profile_training_data':{},
 'retrieve_experience':{'query':'string, may be empty','limit':'integer 1..10'},
 'list_experience':{'offset':'integer >=0','limit':'integer 1..20'},
 'read_experience':{'record_id':'visible memory record id','offset':'character offset >=0','limit':'integer 100..1000'},
 'run_diagnostic':{'purpose':'short string','code':'Python source, <=8000 UTF-8 bytes'},
 'propose_processing':{'candidate_id':'short unique id','code':'Python source, <=8000 bytes',
    'proposal':{'observation':'measured scoped observation','hypothesis':'testable prediction',
        'expected_effect':{'metric':'NMSE','target_scope':'all','direction':'decrease','criterion':'numeric criterion'},
        'falsification':'what would refute the prediction','control':'proposed matched control',
        'evidence_refs':'array of visible evidence IDs'},'parameters':'allowed overrides; {} means original settings'},
 'run_processing':{'candidate_id':'registered id'},
 'evaluate_candidate':{'candidate_id':'processing-validated id'},
 'update_hypothesis':{'candidate_id':'evaluated/failed id','update':{
    'assessment':'supported | contradicted | inconclusive','reason':'scoped result-based explanation',
    'revised_hypothesis':'updated statement; do not invent a result','next_action':'next step',
    'receipt_sha256':'exact candidate receipt_sha256; failures require inconclusive'}},
 'list_candidates':{'offset':'integer >=0','limit':'integer 1..10'},
 'read_candidate':{'candidate_id':'existing candidate ID','offset':'integer >=0','limit':'integer 100..1000'},
 'read_evidence':{'evidence_id':'existing evidence ID from this run','offset':'integer >=0','limit':'integer 100..1000'},
 'inspect_results':{},'submit_policy':{'candidate_id':'B0 or eligible trained id','reason':'short explanation'}}


class EngineeringModel(DeepSeekModel):
    """Keep the shared transport/accounting; use non-thinking tool generation.

    The first acceptance call exhausted all 4096 completion tokens in provider
    reasoning and returned no JSON. This setting belongs to this engineering
    protocol, not the frozen selection harness or its default model settings.
    """
    def request_spec(self,context,attempt):
        body,_=super().request_spec(context,attempt)
        from copy import deepcopy
        from ..agent_runtime.llm import messages_for,reservation_for
        public=deepcopy(context)
        limit=(context.get('binding',{}).get('policy') or {}).get('max_call_tokens',self.repo.config['max_call_tokens'])
        # Fit the actual serialized request, including format recovery and system prompt.
        # Complete records remain in the ledger and can be retrieved through page tools.
        while reservation_for(body['messages'],context.get('phase','exploration'))>limit:
            if public.get('history'):public['history'].pop(0)
            elif public.get('memory'):public['memory'].pop(0)
            elif public.get('format_recovery',{}).get('rejected_action',{}).get('arguments',{}).get('code'):
                arguments=public['format_recovery']['rejected_action']['arguments']
                public['format_recovery']['omitted_code_sha256']=digest(arguments.pop('code'))
                public['format_recovery']['code_omitted_from_request']=True
            elif public.get('results',{}).get('diagnostics'):public['results']['diagnostics'].pop(0)
            elif public.get('results',{}).get('candidates'):public['results']['candidates'].pop(0)
            else:raise GateError('RESEARCH_REQUEST_VIEW_TOO_LARGE')
            body['messages']=messages_for(public,self.system_prompt)
        body['reasoning_effort']=self.repo.config['reasoning_effort']
        generation={k:v for k,v in body.items() if k!='messages'}
        operation='controller-call:'+digest({'context':context['context_version'],
            'messages':digest(body['messages']),'generation':generation,'transport_attempt':attempt})
        return body,operation


def code_identity():
    paths=list((ROOT/'online_ts/agent_research').glob('*.py'))
    paths += [ROOT/'online_ts/agent_runtime'/n for n in ('controller.py','state.py','budget.py','llm.py')]
    paths += [ROOT/'online_ts'/n for n in ('llm_transport.py','store.py','config.py')]
    paths += [ROOT/'scripts/run_capability.py']
    return {str(p):file_hash(p) for p in sorted(paths)}


class ResearchRepository(Repository):
    def __init__(self,run_id,identity,*,folder=None):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',run_id):raise GateError('RESEARCH_RUN_ID')
        self.run_id,self.identity,self.config=run_id,identity,identity['config']
        self.folder=Path(folder) if folder is not None else STUDY/'runs'/run_id
        self.folder.mkdir(parents=True,exist_ok=True)
        super().__init__(Store(self.folder/'runtime.sqlite'),'native-research:'+run_id)
        self.initialize('Autonomous data-processing engineering acceptance with auditable evidence')
        self.put('identity','run_identity',identity,time.time_ns())
        state=self.state()
        if state.incident_id is None:
            state.incident_id=run_id;state.status='investigating';self.save_state(state,0)

    def binding(self):
        return {'task_key':self.key,'owner':'agent','epoch':1,'stop_requested':False,
          'harness_sha256':self.identity['sha256'],'policy':{'execution_profile':self.config.get('execution_profile','bounded'),
            **{k:self.config[k] for k in ('max_calls','max_llm_tokens','max_call_tokens','timeout_seconds')}}}

    def finish(self,record,result):
        with self.store.transaction() as db:
            old=self.get(record['decision_id'])
            if old['decision_outcome']!='prepared':return old
            state=self.state();ref='tool:'+record['decision_id']
            self.put(ref,'tool_evidence',{'record_id':ref,'tool':record['selected_action'],'result':result},
                     record['created_at'],db=db)
            # A proposal before execution is never copied into a post-result update.
            record.update(actual_result=result,decision_outcome='failed' if result.get('error') else 'completed',
                          hypothesis_update=[],result_refs=[ref],actual_tool_effect={'evidence_id':ref})
            state.decision_version+=1
            state.last_result={'decision_id':record['decision_id'],'tool':record['selected_action'],'result':result}
            state.current_hypotheses=record['hypotheses']
            if result.get('agent_finalized'):state.incident_finished=True;state.status='completed'
            self.save_state(state,state.decision_version-1,db=db)
            self.put(record['decision_id'],'decision',record,record['created_at'],replace=True,db=db)
        self.export();return record

    def export(self):
        rows=sorted(self.iter_records('decision'),key=lambda r:(r['created_at'],r['decision_id']))
        (self.folder/'trace.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
        with self.store.connection() as db:
            calls=[dict(r) for r in db.execute('SELECT * FROM agent_calls WHERE task_key=? ORDER BY created_ns',(self.key,))]
        for call in calls:call['payload']=json.loads(call['payload'])
        (self.folder/'model_calls.jsonl').write_text(''.join(canonical(r)+'\n' for r in calls),encoding='utf-8')
        write_json(self.folder/'status.json',{'identity':self.identity,'state':asdict(self.state()),
                    'budget':snapshot(self),'trainings':list(self.iter_records('research_training')),
                    'formal_research_run':self.config['formal_research_protocol']})


class ResearchGateway:
    environment_kind='native_research_engineering'

    def __init__(self,repo,task,library):
        self.repo,self.task,self.library=repo,task,library
        self.memory=EvidenceMemory(library,target_dataset=task.task['dataset_id'],
            target_family=repo.config['dataset_family'],track=repo.config['track'],mode=repo.config['experience_mode'])
        self.guards={**task.guards,**code_identity(),str(MEMORY/'library.json'):file_hash(MEMORY/'library.json'),
            str(MEMORY/'input_catalog.json'):file_hash(MEMORY/'input_catalog.json'),
            str(STUDY/'experience_library.json'):file_hash(STUDY/'experience_library.json')}
        self.inputs=task.export_inputs()
        self.guards[str(self.inputs)]=file_hash(self.inputs)
        self.guards[str(task.folder/'task.json')]=file_hash(task.folder/'task.json')
        for record in library['records']:
            source=record['source']
            path=MEMORY/source['storage_path'] if record['record_type']=='measured_run' else ROOT/source['path']
            self.guards[str(path)]=source['sha256']
        self.journal=EvidenceJournal(repo.folder/'evidence.sqlite',run_id=repo.run_id,
            target_dataset=task.task['dataset_id'],track=repo.config['track'],scope=repo.config['scope'],
            harness_sha256=repo.identity['sha256'],library_sha256=library['sha256'])
        self.adapter=TrainingAdapter(repo,task,self.guards)

    def catalog(self):return TOOLS
    def candidates(self):return sorted(self.repo.iter_records('processing_candidate'),key=lambda r:r['index'])
    def visible_refs(self):
        return [r['evidence_id'] for r in self.journal.repo.iter_records('research_evidence')]
    def assets(self):return [{'path':str(self.inputs),'name':'train_inputs.npz','role':'training','sha256':file_hash(self.inputs)}]
    def candidate(self,name):return self.repo.get('candidate:'+name)
    def save(self,value):self.repo.put('candidate:'+value['candidate_id'],'processing_candidate',value,time.time_ns(),replace=True)

    def baseline(self):
        try:result=self.repo.get('baseline')['result']
        except GateError:
            result=self.adapter.evaluate('B0',None,DEFAULTS)
            self.repo.put('baseline','baseline',{'result':result},time.time_ns())
        self.journal.evidence('baseline',dataset_id=self.task.task['dataset_id'],role='development_metrics',
            artifact_sha256=digest(result),summary=canonical({'metrics':result['metrics'],'per_channel_NMSE':result['per_channel_NMSE'],
              'config':result['config'],'scope':self.repo.config['scope'],'independent_E':False}))
        job_folder=self.repo.folder/'training'/result['job_id']
        if read_json(job_folder/'result.json')!=result:raise GateError('BASELINE_RECEIPT_CHANGED')
        for name,expected in result['assets'].items():
            if file_hash(job_folder/name)!=expected:raise GateError('BASELINE_ARTIFACT_CHANGED')
            self.guards[str(job_folder/name)]=expected
        self.guards[str(job_folder/'result.json')]=file_hash(job_folder/'result.json')
        return result

    def failed(self,value,error):
        if value.get('status') in {'trained','failed'}:return value
        receipt={'candidate_id':value['candidate_id'],'status':'failed','error':str(error),
                 'stage':value['status'],'scope':self.repo.config['scope']}
        path=self.repo.folder/'failures'/(value['candidate_id']+'.json');write_json(path,receipt)
        sha=file_hash(path)
        self.journal.receipt(value['candidate_id'],status='failed',metrics={},artifact_sha256=sha,
                             reason='Execution/validation failure; scientific hypothesis remains untested')
        value.update(status='failed',error=str(error),receipt_sha256=sha);self.save(value)
        return value

    def execute(self,record):
        tool,args=record['selected_action'],record['arguments']
        if tool not in TOOLS or set(args)!=set(TOOLS[tool]):raise GateError('RESEARCH_TOOL_ARGUMENTS')
        self.repo.check_binding(record['binding'])
        if tool=='describe_task':return {'config':self.repo.config,'parameter_ranges':PARAMETERS,'tools':TOOLS}
        if tool=='profile_training_data':
            value=self.task.profile()
            with np.load(self.inputs,allow_pickle=False) as inputs:
                value['input_contract']={'values_shape':list(inputs['values'].shape),
                    'start_row_shape':list(inputs['start_row'].shape),'input_length':self.repo.config['input_length'],
                    'prediction_length':self.repo.config['prediction_length'],
                    'window_expression':'values[start_row[:,None]+numpy.arange(input_length)[None,:]]',
                    'values_are_rows_not_windows':True,'input_argument':'directory sys.argv[1]',
                    'output_argument':'directory sys.argv[2]','normalization':'already T-normalized'}
            self.journal.evidence('T_profile',dataset_id=self.task.task['dataset_id'],
                role='training_diagnostics',artifact_sha256=digest(value),summary=canonical(value))
            return value
        if tool in {'list_candidates','read_candidate','read_evidence'}:
            if type(args['offset'])is not int or args['offset']<0 or type(args['limit'])is not int:
                raise GateError('RESEARCH_RECORD_PAGE')
            if tool=='list_candidates':
                if not 1<=args['limit']<=10:raise GateError('RESEARCH_RECORD_PAGE')
                rows=self.candidates();start,end=args['offset'],args['offset']+args['limit']
                return {'records':[self.candidate_brief(row) for row in rows[start:end]],'total':len(rows),
                    'next_offset':end if end<len(rows) else None}
            if not 100<=args['limit']<=1000:raise GateError('RESEARCH_RECORD_PAGE')
            if tool=='read_candidate':
                row=self.candidate(args['candidate_id'])
                value={key:row[key] for key in ('candidate_id','status','parameters','proposal','post_result_update',
                    'receipt_sha256','error') if key in row}
                if row.get('result') and self.repo.config.get('feedback_mode')!='withheld':value['metrics']=row['result']['metrics']
            else:
                value=dict(self.journal.repo.get('evidence:'+args['evidence_id']))
                if self.repo.config.get('feedback_mode')=='withheld' and value['role']=='development_metrics' and value['evidence_id']!='baseline':
                    value['summary']='Candidate development outcome withheld until frozen submission'
            text=canonical(value);start,end=args['offset'],args['offset']+args['limit']
            return {'record_id':args.get('candidate_id',args.get('evidence_id')),'text':text[start:end],
                'total_chars':len(text),'next_offset':end if end<len(text) else None}
        if tool in {'retrieve_experience','list_experience','read_experience'}:
            if tool=='retrieve_experience':value=self.memory.retrieve(args['query'],args['limit'])
            elif tool=='list_experience':value=self.memory.list(args['offset'],args['limit'])
            else:
                if type(args['offset'])is not int or args['offset']<0 or type(args['limit'])is not int or not 100<=args['limit']<=1000:
                    raise GateError('MEMORY_PAGE')
                item=self.memory.read(args['record_id']);text=canonical(item)
                start,end=args['offset'],args['offset']+args['limit']
                value={'library_sha256':self.library['sha256'],'records':[self.memory.brief(item)],
                    'record_id':item['record_id'],'text':text[start:end],'total_chars':len(text),
                    'next_offset':end if end<len(text) else None,'policy':self.library['policy']}
            for item in value['records']:
                self.journal.evidence(item['record_id'],dataset_id=item['dataset_id'],role='historical_development',
                    artifact_sha256=item['source']['excerpt_sha256'],summary=item['lesson']+'; '+item['limits'])
            kind='experience_retrieval' if tool=='retrieve_experience' else 'experience_read'
            value['access_created_at']=record['created_at']
            self.repo.put('memory:'+record['decision_id'],kind,value,time.time_ns())
            return value
        if tool=='run_diagnostic':
            diagnostic_id='diagnostic:'+record['decision_id']
            previous=next((v for v in self.repo.iter_records('diagnostic_job') if v['record_id']==diagnostic_id),None)
            if previous and previous['status']!='started':return previous
            limit=self.repo.config['max_diagnostics']
            if not previous and limit is not None and self.repo.count_records('diagnostic_job')>=limit:raise GateError('DIAGNOSTIC_BUDGET')
            self.check_code(args['code'])
            if not isinstance(args['purpose'],str) or not 1<=len(args['purpose'])<=1000:raise GateError('DIAGNOSTIC_PURPOSE')
            folder=self.repo.folder/'diagnostics'/record['decision_id']
            if not folder.exists():prepare_job(folder,args['code'],self.assets())
            value={'record_id':diagnostic_id,'purpose':args['purpose'],'status':'started'}
            self.repo.put(value['record_id'],'diagnostic_job',value,time.time_ns())
            execution=execute_job(folder,self.guards)
            if execution['status']=='completed':
                try:
                    facts=diagnostic_json(folder/'outputs/diagnostic.json')
                    value.update(status='completed',facts=facts,execution_sha256=file_hash(folder/'execution.json'))
                    self.journal.evidence(value['record_id'],dataset_id=self.task.task['dataset_id'],role='training_diagnostics',
                        artifact_sha256=file_hash(folder/'outputs/diagnostic.json'),summary=canonical(facts)[:3900])
                except (GateError,ValueError,KeyError) as error:value.update(status='failed',error=str(error))
            else:value.update(status='failed',error=(execution['error'] or 'DIAGNOSTIC_WORKER_FAILED')+': '+execution['stdout'][-1000:])
            self.repo.put(value['record_id'],'diagnostic_job',value,time.time_ns(),replace=True)
            return value
        if tool=='propose_processing':
            name=args['candidate_id'];self.check_code(args['code'])
            if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',name) or name=='B0':raise GateError('PROCESSING_ID')
            proposal=args['proposal']
            expected={'observation','hypothesis','expected_effect','falsification','control','evidence_refs'}
            if not isinstance(proposal,dict) or set(proposal)!=expected:raise GateError('PROCESSING_PROPOSAL_SCHEMA')
            chosen_parameters=parameters(args['parameters'])
            existing=next((v for v in self.candidates() if v['candidate_id']==name),None)
            if existing:
                if existing['code_sha256']!=digest(args['code']) or existing['parameters']!=chosen_parameters or existing['proposal']!=proposal:raise GateError('PROCESSING_IMMUTABLE')
                return existing
            limit=self.repo.config['max_candidates']
            if limit is not None and len(self.candidates())>=limit:raise GateError('PROCESSING_CANDIDATE_BUDGET')
            folder=self.repo.folder/'processing'/name
            manifest=prepare_job(folder,args['code'],self.assets())
            self.journal.proposal(name,proposal|{'code_sha256':manifest['code_sha256']})
            value={'candidate_id':name,'index':len(self.candidates()),'status':'proposed',
                'code_sha256':digest(args['code']),'source_sha256':manifest['code_sha256'],
                'parameters':chosen_parameters,'proposal':proposal,'folder':str(folder)}
            self.save(value);return value
        if tool=='run_processing':
            value=self.candidate(args['candidate_id'])
            if value['status']!='proposed':return value
            try:
                folder=Path(value['folder']);execution=execute_job(folder,self.guards,seconds=45)
                if execution['status']!='completed':raise GateError((execution['error'] or 'PROCESSING_WORKER_FAILED')+': '+execution['stdout'][-1000:])
                asset=folder/'outputs/processing.npz'
                probability,weight,audit=validate_processing(asset,self.task.data)
                transform_path=folder/'outputs/processor.py'
                state_path=folder/'outputs/processor_state.json'
                transform=None
                if transform_path.exists() or state_path.exists():
                    if not self.repo.config.get('input_transforms_allowed',False):raise GateError('INPUT_TRANSFORM_NOT_REGISTERED')
                    from .transforms import processor_metadata
                    processor_metadata(transform_path,state_path)
                    check=run_process([sys.executable,'-B','-m','online_ts.agent_research.processor_check','--folder',str(folder)],
                        cwd=ROOT,outputs=folder/'outputs',seconds=45,gpu=True)
                    write_json(folder/'processor_check_execution.json',check)
                    if check['status']!='completed' or not (folder/'outputs/processor_validation.json').exists():
                        raise GateError('INPUT_PROCESSOR_VALIDATION_FAILED: '+check['stdout'][-1000:])
                    transform={'code_sha256':file_hash(transform_path),'state_sha256':file_hash(state_path)}
                if np.allclose(probability,1/len(probability),rtol=1e-12,atol=0) and np.allclose(weight,1,rtol=1e-7) and value['parameters']==DEFAULTS and transform is None:
                    raise GateError('PROCESSING_NO_CHANGE_FROM_BASELINE')
                scientific=digest({'probability':digest(probability.tobytes().hex()),
                    'loss_weight':digest(weight.astype(np.float32).tobytes().hex()),'parameters':value['parameters'],'input_transform':transform})
                if any(v.get('scientific_policy')==scientific and v['candidate_id']!=value['candidate_id'] for v in self.candidates()):
                    raise GateError('PROCESSING_DUPLICATE_POLICY')
                value.update(status='validated',asset=str(asset),asset_sha256=file_hash(asset),audit=audit,scientific_policy=scientific,
                    input_transform_path=str(transform_path) if transform is not None else None,
                    input_transform=transform)
                self.save(value);return value
            except (GateError,ValueError,KeyError,OSError) as error:
                if str(error)=='NATIVE_PROTECTED_ASSET_CHANGED':raise
                return self.failed(value,error)
        if tool=='evaluate_candidate':
            value=self.candidate(args['candidate_id'])
            if value['status'] in {'trained','failed'}:return value
            if value['status']!='validated':raise GateError('PROCESSING_VALIDATION_REQUIRED')
            try:
                result=self.adapter.evaluate(value['candidate_id'],Path(value['asset']),value['parameters'],transform=value.get('input_transform_path'))
                sha=digest(result)
                self.journal.receipt(value['candidate_id'],status='completed',metrics={k:result['metrics'][k] for k in ('NMSE','NMAE')},
                    artifact_sha256=sha,reason='Full fixed D feedback; single training seed, not independent E')
                self.journal.evidence('result:'+value['candidate_id'],dataset_id=self.task.task['dataset_id'],role='development_metrics',
                    artifact_sha256=sha,summary=canonical(result['metrics']))
                value.update(status='trained',result=result,receipt_sha256=sha);self.save(value);return value
            except GateError as error:
                if 'SOURCE_CHANGED' in str(error) or 'ASSET_CHANGED' in str(error):raise
                return self.failed(value,error)
        if tool=='update_hypothesis':
            value=self.candidate(args['candidate_id'])
            self.journal.update(value['candidate_id'],args['update'])
            value['hypothesis_updated']=True;value['post_result_update']=args['update'];self.save(value)
            return {'candidate_id':value['candidate_id'],'recorded_update':args['update'],'diagnosis_correctness':'unassessed'}
        if tool=='inspect_results':return self.results()
        if tool=='submit_policy':return self.submit(args)
        raise GateError('RESEARCH_TOOL')

    @staticmethod
    def check_code(code):
        if not isinstance(code,str) or len(code.encode())>8000:raise GateError('RESEARCH_CODE_SIZE')

    def candidate_brief(self,row):
        value={k:row[k] for k in ('candidate_id','index','status','receipt_sha256','hypothesis_updated','error') if k in row}
        if row.get('result') and self.repo.config.get('feedback_mode')!='withheld':value['metrics']=row['result']['metrics']
        if row.get('post_result_update'):value['assessment']=row['post_result_update']['assessment']
        return value

    def results(self):
        candidates=self.candidates()
        diagnostics=list(self.repo.iter_records('diagnostic_job'))
        diagnostic_rows=[{k:r[k] for k in ('record_id','purpose','status','error') if k in r} |
            ({'facts_preview':canonical(r['facts'])[:1000]} if 'facts' in r else {}) for r in diagnostics]
        return {'baseline':self.baseline()['metrics'],'candidates':[self.candidate_brief(row) for row in candidates[-4:]],
            'candidate_count':len(candidates),'journal':self.journal.summary(),
            'diagnostics':diagnostic_rows[-4:],'diagnostic_count':len(diagnostics),
            'older_records':'Use list_candidates/read_candidate/read_evidence to recover complete older details',
            'remaining_diagnostic_slots':None if self.repo.config['max_diagnostics'] is None else max(0,self.repo.config['max_diagnostics']-len(diagnostics)),
            'remaining_candidate_slots':None if self.repo.config['max_candidates'] is None else max(0,self.repo.config['max_candidates']-len(candidates))}

    def submit(self,args):
        rows=self.candidates();completed=[v for v in rows if v['status']=='trained']
        diagnostics=list(self.repo.iter_records('diagnostic_job'))
        if not completed or not any(v['status']=='completed' for v in diagnostics):
            raise GateError('ENGINEERING_CLOSED_LOOP_NOT_COMPLETE')
        if any(v['status'] not in {'trained','failed'} or not v.get('hypothesis_updated') for v in rows):
            raise GateError('POST_RESULT_UPDATE_REQUIRED')
        baseline=self.baseline();eligible=[('B0',baseline['metrics'],DEFAULTS)]
        eligible += [(v['candidate_id'],v['result']['metrics'],v['parameters']) for v in completed
                     if v['result']['metrics']['NMAE']<=baseline['metrics']['NMAE']*1.01]
        best=min(eligible,key=lambda v:v[1]['NMSE'])
        if args['candidate_id']!=best[0]:raise GateError('SELECT_MEASURED_ELIGIBLE_BEST')
        if not isinstance(args['reason'],str) or not 1<=len(args['reason'])<=2000:raise GateError('SUBMISSION_REASON')
        final={'run_id':self.repo.run_id,'selected_candidate':best[0],'metrics':best[1],'parameters':best[2],
            'baseline':baseline['metrics'],'reason':args['reason'],'scope':'engineering_development_only',
            'full_chain_passed':True,'formal_agent_run':False,'independent_evaluation':False,
            'correct_signal':'unassessed','data_gain_attribution':'unassessed_without_matched_parameter_controls',
            'journal':self.journal.summary(),'candidate_statuses':[{k:v[k] for k in ('candidate_id','status')} for v in rows]}
        write_json(self.repo.folder/'final_policy.json',final)
        self.repo.put('final_policy','final_policy',final,time.time_ns())
        return {'agent_finalized':True,'final_policy':final}


def context_builder(repo,gateway,event):
    # Durable complete decision history stays on disk; bounded context summaries
    # avoid blindly re-sending submitted Python programs on every call.
    history=sorted(repo.iter_records('decision'),key=lambda r:r['created_at'])
    summaries=[]
    for row in history:
        result=row.get('actual_result') or {}
        text=canonical(result)
        summaries.append({'tool':row['selected_action'],'reason':row['reason_summary'],
            'result':result if len(text)<3500 else {'summary':text[:3500],'truncated':True}})
    accesses=sorted(list(repo.iter_records('experience_retrieval'))+list(repo.iter_records('experience_read')),
        key=lambda r:r['access_created_at'])
    memory={}
    for access in accesses:
        for item in access['records']:
            memory[item['record_id']]={k:item[k] for k in ('record_id','lesson','limits','status')}
    known_refs=gateway.visible_refs()
    recent_reads=[row['arguments']['evidence_id'] for row in history[-4:]
        if row['selected_action']=='read_evidence' and 'evidence_id' in row['arguments']]
    visible_refs=list(dict.fromkeys([ref for ref in ('baseline','T_profile') if ref in known_refs]+known_refs[-28:]+recent_reads))
    context={'phase':'exploration','task':{'dataset_id':gateway.task.task['dataset_id'],
        'goal':'autonomous data-processing engineering closed-loop acceptance','profile':gateway.task.profile(),
        'baseline':{k:gateway.baseline()[k] for k in ('metrics','per_channel_NMSE','per_channel_NMAE')},
        'input_asset':'train_inputs.npz','roles':{'T':'code input','D':'trusted metrics only','E':'absent'},
        'constraints':repo.config},'tools':gateway.catalog(),'visible_evidence_refs':visible_refs,
        'history':summaries[-4:],'results':gateway.results(),'budget':snapshot(repo),
        'memory':list(memory.values())[-4:], 'knowledge':[],
        'memory_policy':gateway.library['policy'],'memory_mode':repo.config['experience_mode']}
    while len(canonical(context).encode())>19000:
        if context['history']:context['history'].pop(0)
        elif context['memory']:context['memory'].pop(0)
        elif context['results'].get('diagnostics'):context['results']['diagnostics'].pop(0)
        elif context['results'].get('candidates'):context['results']['candidates'].pop(0)
        else:break
    if len(canonical(context).encode())>19000:raise GateError('ENGINEERING_CONTEXT_TOO_LARGE')
    context['context_version']=digest(context);context['binding']=repo.binding()
    # Do not truncate proposals or post-result updates; candidate state supplies
    # them separately even when old decision summaries fall outside this window.
    return context


def run_engineering(run_id,*,experience_mode='full',track='experience_combination'):
    if os.name!='posix':raise GateError('RESEARCH_SERVER_REQUIRED')
    key=Path('/root/.config/auto_data_research/deepseek_api_key')
    if key.stat().st_mode&0o077:raise GateError('SERVER_KEY_PERMISSIONS')
    os.environ['DEEPSEEK_API_KEY']=key.read_text().strip()
    if DEPENDENCY_ROOT.is_dir():
        os.environ['PYTHONPATH']=str(DEPENDENCY_ROOT)
    task=TaskAssets(STUDY/'tasks/metropt3_dev')
    library=read_json(MEMORY/'library.json');verify_memory(library)
    run_config=CONFIG|{'experience_mode':experience_mode,'track':track}
    import pandas
    identity={'config':run_config,'dependencies':{'numpy':np.__version__,'pandas':pandas.__version__},
              'task_sha256':file_hash(task.folder/'task.json'),
              'features_sha256':file_hash(task.feature_path),'code':code_identity(),
              'library_sha256':library['sha256'],'prompt_sha256':digest(PROMPT)}
    identity['sha256']=digest(identity)
    repo=ResearchRepository(run_id,identity)
    gateway=ResearchGateway(repo,task,library)
    config=SimpleNamespace(planner=SimpleNamespace(model=CONFIG['model'],kind='external',allow_external=True,
        timeout_seconds=CONFIG['timeout_seconds'],endpoint_env='DEEPSEEK_BASE_URL',key_env='DEEPSEEK_API_KEY'))
    controller=AgentController(repo,gateway,EngineeringModel(repo,config,system_prompt=PROMPT),context_builder=context_builder)
    write_json(repo.folder/'preregistration.json',identity)
    try:
        gateway.baseline()
        while not repo.state().incident_finished:
            controller.step(Event('engineering-'+str(repo.state().decision_version),'engineering',time.time_ns()))
            repo.export()
        return repo.get('final_policy')
    except Exception as error:
        write_json(repo.folder/'failure.json',{'status':'failed','error':str(error) if isinstance(error,GateError) else type(error).__name__,
                   'scope':'engineering_only','candidate_count':len(gateway.candidates()),'budget':snapshot(repo)})
        raise
    finally:
        repo.export()
