"""Preregistered development pilot: independent searches and matched ablations."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np

from ..agent_runtime.budget import snapshot
from ..agent_runtime.controller import AgentController, DecisionSchemaError
from ..agent_runtime.state import Event
from ..config import GateError, canonical, digest
from ..selection.common import file_hash, read_json, write_json
from .experience import ROOT, STUDY
from .memory import MEMORY, verify_memory
from .processing import DEFAULTS
from .runtime import (CONFIG, PROMPT, TOOLS, EngineeringModel, ResearchGateway,
                      ResearchRepository, code_identity, context_builder)
from .task import TaskAssets

PILOT = STUDY / 'phase4/20261009_open'
OUTPUT = Path('/tmp/auto_data_phase4_20261009_open')
RUN_PREFIX = 'p4open'
GROUPS = ('full', 'no_memory', 'no_feedback', 'random_reference')
SEEDS = (20261101, 20261102, 20261103)
CONFIRMATION_SEEDS = (20261201, 20261202, 20261203)

FORMAL_PROMPT = """You are an autonomous researcher in a preregistered DEVELOPMENT pilot on a previously explored dataset.
Choose one available tool per turn. Diagnose measurable training-data properties and prediction deficits,
propose falsifiable processing hypotheses, write NumPy code, test them with a fixed trusted trainer,
and record result-based updates. Distribution skew alone does not establish a predictive cause.
All targets, partitions, original DLinear structure and training parameters are fixed. Return parameters={}
for every proposal. The user has removed cumulative exploration limits: no maximum number of coded
diagnostics, processing candidates, API calls, aggregate tokens, GPU jobs or total search wall time.
Actual costs and all failed attempts are still recorded. Choose when to submit based on the evidence;
you do not need to exhaust a quota. Individual model context/output and executable-job resource bounds remain.
At least one completed coded diagnostic and one attempted processing candidate are required before final submission.
Memory tools are optional. Historical evidence is conditional; replicating a supplied recipe is experience
combination, not new discovery. Do not read other host paths, experiment folders, credentials, D, test,
E or ground_truth, install dependencies, use network or subprocesses. No human strategy updates occur during search.
Generated code reads sys.argv[1]/train_inputs.npz with allow_pickle=False, and writes only sys.argv[2].
sys.argv[1] IS A DIRECTORY, never the npz file. Use pathlib.Path(sys.argv[1])/'train_inputs.npz'.
sys.argv[2] IS THE OUTPUT DIRECTORY. Write pathlib.Path(sys.argv[2])/'diagnostic.json', not stdout.
CPU fitting/diagnostics can import NumPy or CPU PyTorch; CUDA is unavailable in that fitting process.
The separate input function runs on CPU/CUDA as supplied by the trusted trainer.
Input values are T-normalized sensor rows; start_row are legal T window starts, alongside training feature
arrays, groups, source-index mappings and sensor_columns. Fit everything on T, never use future targets as inputs.
Diagnostics write diagnostic.json containing bounded finite measured facts, with named channels and scopes.
Processing writes processing.npz with exactly unchanged int64 start_row, floating weight and sampling_multiplier,
finite in [0,4], all arrays of the same length. Window coverage, rare-group protection and loss scale normalization apply.
For an input transformation, write your OWN processor.py and processor_state.json in the output folder.
processor.py defines transform(x,state), using torch/math, taking a T-normalized tensor of shape
(batch,32,15) and returning a finite tensor with identical shape, dtype and device. You choose the actual
processing algorithm freely within the fixed task: no new trainable model, file/network access or random/global
state. The function sees ONLY past x and frozen JSON state fitted on T by your processing code. It cannot see
future targets or D. No predetermined transformation menu is required. Make operations device-independent,
do not mutate inputs/state or mix different windows via batch statistics. Input code is checked on real T
windows on CPU/CUDA before admission, then executed during fixed training and evaluation in clean subprocesses.
The state is one bounded finite JSON object; processing source <=16000 bytes, state <=24000 bytes.
Always write processing.npz too, using uniform weights/sampling for a pure input transform. Targets are unchanged.
Proposal fields are observation, hypothesis, expected_effect, falsification, control, evidence_refs.
For this pilot expected_effect must use metric=NMSE, target_scope=all, direction=decrease;
criterion should state the >=1% overall improvement and <=1% NMAE deterioration criterion.
Cite only visible_evidence_refs. A prose control is a prediction, not evidence that a control has been run.
Post-result updates record supported, contradicted or inconclusive, reason, revised_hypothesis, next_action,
receipt_sha256. Scientific claims remain unassessed until external confirmation and control checks.
For feedback-visible runs, select the lowest measured candidate NMSE satisfying NMAE<=1.01*B0;
B0 is eligible and must be retained if no candidate qualifies. Negative evidence is a valid result.
For feedback-withheld runs, candidate execution status and receipt hashes are available, but candidate D
metrics and result-based ranking are hidden. Select your policy using T evidence and prior knowledge,
write inconclusive updates for withheld outcomes, and do not claim you saw performance. After final submission
the external evaluator reveals and confirms the frozen choice without allowing revisions.
Independent confirmation seeds and automated controls are external to the search budget; no confirmation
feedback is returned to this or peer runs. Search repetitions have fresh contexts and cannot read peer discoveries.
""" + PROMPT[PROMPT.index('Keep Chinese decision summaries'):]


def conceal(value):
    """Remove candidate outcomes, including nested trainer receipts, before model exposure."""
    hidden = {'metrics', 'groups', 'per_channel_NMSE', 'per_channel_NMAE', 'best_step',
              'result_sha256', 'data_audit', 'history', 'result'}
    if isinstance(value, dict):
        return {k: conceal(v) for k, v in value.items() if k not in hidden}
    if isinstance(value, list):
        return [conceal(v) for v in value]
    return value


class PilotGateway(ResearchGateway):
    environment_kind = 'native_agent_development_pilot'

    def __init__(self, repo, task, library):
        super().__init__(repo, task, library)
        self.guards.update({str(PILOT/name):file_hash(PILOT/name)
                            for name in ('protocol.json','PROTOCOL.md','registered_case.json')})
        source=read_json(PILOT/'registered_case.json')['source']
        for path_key,hash_key in (('path','sha256'),('recipe_path','recipe_sha256')):
            path=ROOT/source[path_key]
            if file_hash(path)!=source[hash_key]:raise GateError('PILOT_CASE_SOURCE_CHANGED')
            self.guards[str(path)]=source[hash_key]

    def catalog(self):
        tools=TOOLS | {'read_registered_case': {}}
        limit=self.repo.config['max_diagnostics']
        if limit is not None and self.repo.count_records('diagnostic_job')>=limit:tools.pop('run_diagnostic')
        limit=self.repo.config['max_candidates']
        if limit is not None and len(self.candidates())>=limit:tools.pop('propose_processing')
        return tools

    def results(self):
        value = super().results()
        if self.repo.config['feedback_mode'] == 'withheld':
            value['candidates'] = conceal(value['candidates'])
            value['candidate_metrics'] = 'withheld_until_frozen_submission'
        return value

    def execute(self, record):
        tool, args = record['selected_action'], record['arguments']
        if tool == 'describe_task':
            if args: raise GateError('RESEARCH_TOOL_ARGUMENTS')
            return {'config':self.repo.config, 'tools':self.catalog(), 'parameter_ranges':{k:[v] for k,v in DEFAULTS.items()}}
        if tool == 'read_registered_case':
            if args: raise GateError('RESEARCH_TOOL_ARGUMENTS')
            records = [read_json(PILOT/'registered_case.json')] if self.repo.config['experience_mode']=='full' else []
            for row in records:
                self.journal.evidence(row['record_id'],dataset_id=self.task.task['dataset_id'],role='historical_development',
                    artifact_sha256=row['source']['excerpt_sha256'],summary=row['lesson']+'; '+row['limits'])
            value={'records':records,'access_created_at':record['created_at'],'known_recipe_not_new_discovery':True}
            self.repo.put('case:'+record['decision_id'],'experience_read',value,time.time_ns())
            return value
        if tool == 'propose_processing':
            if args.get('parameters') != {}: raise GateError('PILOT_FIXED_PARAMETERS')
            effect = args.get('proposal',{}).get('expected_effect',{})
            if (effect.get('metric'),effect.get('target_scope'),effect.get('direction')) != ('NMSE','all','decrease'):
                raise GateError('PILOT_EXPECTATION_SCOPE')
        if tool == 'update_hypothesis' and self.repo.config['feedback_mode']=='withheld':
            if args.get('update',{}).get('assessment') != 'inconclusive': raise GateError('PILOT_OUTCOME_WITHHELD')
        value = super().execute(record)
        if tool in ('evaluate_candidate','run_processing','propose_processing') and self.repo.config['feedback_mode']=='withheld':
            self.repo.put('private:'+record['decision_id'],'private_trainer_result',{'value':value},time.time_ns())
            return conceal(value) | {'development_outcome':'withheld_until_frozen_submission'}
        return value

    def submit(self, args):
        rows=self.candidates()
        if not rows or not any(v['status']=='completed' for v in self.repo.iter_records('diagnostic_job')):
            raise GateError('PILOT_DIAGNOSTIC_AND_ATTEMPT_REQUIRED')
        if any(v['status'] not in ('trained','failed') or not v.get('hypothesis_updated') for v in rows):
            raise GateError('POST_RESULT_UPDATE_REQUIRED')
        baseline=self.baseline()
        completed=[v for v in rows if v['status']=='trained']
        if self.repo.config['feedback_mode']=='visible':
            choices=[('B0',baseline['metrics'])]+[(v['candidate_id'],v['result']['metrics']) for v in completed
                if v['result']['metrics']['NMAE'] <= baseline['metrics']['NMAE']*1.01]
            selected=min(choices,key=lambda x:x[1]['NMSE'])[0]
            if args['candidate_id']!=selected: raise GateError('SELECT_MEASURED_ELIGIBLE_BEST')
        elif args['candidate_id']!='B0' and args['candidate_id'] not in {v['candidate_id'] for v in completed}:
            raise GateError('PILOT_SELECT_TRAINED_POLICY')
        if not isinstance(args['reason'],str) or not 1<=len(args['reason'])<=2000:raise GateError('SUBMISSION_REASON')
        final={'run_id':self.repo.run_id,'selected_candidate':args['candidate_id'],'reason':args['reason'],
            'group':self.repo.config['group'],'scope':'preregistered_development_pilot',
            'formal_agent_run':self.repo.config['group']!='random_reference','independent_evaluation':False,'human_strategy_intervention':False,
            'correct_signal':'unassessed_pending_external_scoring','journal':self.journal.summary()}
        write_json(self.repo.folder/'final_policy.json',final)
        self.repo.put('final_policy','final_policy',final,time.time_ns())
        return {'agent_finalized':True,'final_policy':final}


def pilot_context(repo, gateway, event):
    value=context_builder(repo,gateway,event)
    value['task']['goal']='diagnose_and_test_processing_with_fixed_model_in_controlled_development_pilot'
    value['task']['budget_note']='No cumulative exploration limits; actual calls, candidates, training jobs and elapsed cost are reported'
    failures=list(repo.iter_records('format_failure'))
    if failures:
        value['task']['last_format_failure']=failures[-1]
        value['task']['format_failure_count']=len(failures)
    value['task']['code_interface']={'input_argument':'directory: sys.argv[1]',
        'read_example':"numpy.load(pathlib.Path(sys.argv[1]) / 'train_inputs.npz', allow_pickle=False)",
        'output_argument':'directory: sys.argv[2]',
        'diagnostic_write_example':"(pathlib.Path(sys.argv[2]) / 'diagnostic.json').write_text(json.dumps(facts))",
        'cpu_fit':'numpy_or_CPU_torch_no_CUDA_8GiB_virtual_address_limit',
        'processor_function':'transform(x,state), x tensor (batch,32,15), identical shape/dtype/device'}
    if repo.config['feedback_mode']=='withheld':
        for row in value['history']:
            if row['tool']=='evaluate_candidate':row['result']=conceal(row['result'])
    value['context_version']=digest({k:v for k,v in value.items() if k not in ('context_version','binding')})
    return value


class PilotModel(EngineeringModel):
    def request_spec(self, context, attempt):
        body, _ = super().request_spec(context, attempt)
        body['temperature'] = 0.8
        operation = 'controller-call:' + digest({'context':context['context_version'],
            'messages':digest(body['messages']), 'generation':{k:v for k,v in body.items() if k!='messages'},
            'transport_attempt':attempt})
        return body, operation


def random_code(index, random_seed):
    # Fixed secondary reference; its narrower recipe space is disclosed.
    return f'''import json,sys,pathlib,numpy as np
inp=np.load(pathlib.Path(sys.argv[1])/"train_inputs.npz",allow_pickle=False)
out=pathlib.Path(sys.argv[2]); n=len(inp["start_row"]); rng=np.random.default_rng({random_seed})
w=np.ones(n);s=np.ones(n)
family=rng.integers(0,4)
if family<2:
 feature=inp[rng.choice(["input_variation","digital_transitions","elapsed_days"])]
 order=np.argsort(feature,kind="stable"); rank=np.empty(n);rank[order]=np.arange(n)/(n-1)
 score=0.5+rank if rng.integers(0,2) else 1.5-rank
 if family==0:w=score
 else:s=score
else:
 values=inp["values"]; correlation=np.corrcoef(values[::17],rowvar=False);np.fill_diagonal(correlation,-np.inf)
 mapping=np.argmax(np.nan_to_num(correlation,nan=-1),axis=1) if family==2 else rng.integers(0,15,15)
 fraction=np.full(15,rng.choice([0.25,0.5,0.75]))
 processor='import torch\\ndef transform(x,state):\\n    source=torch.tensor(state["mapping"],device=x.device)\\n    fraction=state["fraction"]\\n    mixed=x[:,-1:]+fraction*(x-x[:,-1:])+(1-fraction)*(x[:,:,source]-x[:,-1:,source])\\n    unchanged=(source==torch.arange(15,device=x.device))[None,None,:]\\n    return torch.where(unchanged,x,mixed)\\n'
 (out/"processor.py").write_text(processor)
 (out/"processor_state.json").write_text(json.dumps({{"mapping":mapping.tolist(),"fraction":float(fraction[0])}}))
np.savez_compressed(out/"processing.npz",start_row=inp["start_row"],weight=w,sampling_multiplier=s)
'''


def run_random(repo,gateway):
    def action(tool,args):
        record={'selected_action':tool,'arguments':args,'decision_id':'reference-'+str(repo.state().decision_version),
                'created_at':time.time_ns(),'binding':repo.binding()}
        value=gateway.execute(record)
        state=repo.state();state.decision_version+=1;repo.save_state(state,state.decision_version-1)
        repo.put(record['decision_id'],'reference_action',record|{'result':value},record['created_at'])
        return value
    action('run_diagnostic',{'purpose':'Fixed reference training variation summary',
        'code':'import numpy as np,json,sys,pathlib; x=np.load(pathlib.Path(sys.argv[1])/"train_inputs.npz",allow_pickle=False); pathlib.Path(sys.argv[2],"diagnostic.json").write_text(json.dumps({"windows":len(x["start_row"]),"mean_input_variation":float(x["input_variation"].mean())}))'})
    for index in range(repo.config['random_reference_candidates']):
        name='R'+str(index+1)
        action('propose_processing',{'candidate_id':name,'code':random_code(index,repo.config['seed']+701+index),'parameters':{},
            'proposal':{'observation':'Fixed random-reference plan; no adaptive diagnosis claim',
                'hypothesis':'Random permitted processing may improve overall forecast error',
                'expected_effect':{'metric':'NMSE','target_scope':'all','direction':'decrease','criterion':'>=1% NMSE gain with <=1% NMAE deterioration'},
                'falsification':'Fails registered confirmation gates','control':'External automated control',
                'evidence_refs':['baseline']}})
        value=action('run_processing',{'candidate_id':name})
        if value['status']!='failed':value=action('evaluate_candidate',{'candidate_id':name})
        action('update_hypothesis',{'candidate_id':name,'update':{'assessment':'inconclusive','reason':'Reference search does not claim diagnosis correctness',
            'revised_hypothesis':'Await external confirmation','next_action':'Continue fixed registered reference plan','receipt_sha256':value['receipt_sha256']}})
    baseline=gateway.baseline()['metrics']
    choices=[('B0',baseline)]+[(v['candidate_id'],v['result']['metrics']) for v in gateway.candidates()
                            if v['status']=='trained' and v['result']['metrics']['NMAE']<=baseline['NMAE']*1.01]
    action('submit_policy',{'candidate_id':min(choices,key=lambda x:x[1]['NMSE'])[0],'reason':'Fixed reference selects the measured eligible best'})


def run_search(group,repeat):
    if group not in GROUPS or repeat not in (1,2,3):raise GateError('PILOT_RUN_PLAN')
    protocol=read_json(PILOT/'protocol.json')
    task=TaskAssets(STUDY/'tasks/metropt3_dev')
    library=read_json(MEMORY/'library.json');verify_memory(library)
    run_id=f'{RUN_PREFIX}_{group}_r{repeat}'
    folder=OUTPUT/'runs'/run_id
    if folder.exists():raise GateError('PILOT_RUN_ALREADY_EXISTS_NO_REPLAY')
    if shutil.disk_usage(OUTPUT.parent).free<6*1024**3:raise GateError('PILOT_DISK_RESERVE')
    config=CONFIG|{'version':'phase4-open-development-v3','scope':'development_only',
        'group':group,'seed':SEEDS[repeat-1], 'max_candidates':None,'max_diagnostics':None,'max_calls':None,
        'max_llm_tokens':None,'max_training_jobs':None,'execution_profile':'research_open',
        'random_reference_candidates':3,
        'experience_mode':'none' if group in ('no_memory','random_reference') else 'full',
        'feedback_mode':'withheld' if group=='no_feedback' else 'visible',
        'input_transforms_allowed':True,'formal_research_protocol':True,
        'parameter_ranges':{k:[v] for k,v in DEFAULTS.items()}}
    identity={'config':config,'task_sha256':file_hash(task.folder/'task.json'),'features_sha256':file_hash(task.feature_path),
        'library_sha256':library['sha256'],'code':code_identity(),'protocol_sha256':file_hash(PILOT/'protocol.json'),
        'protocol_document_sha256':file_hash(PILOT/'PROTOCOL.md'),
        'registered_case_sha256':file_hash(PILOT/'registered_case.json'),'prompt_sha256':digest(FORMAL_PROMPT),
        'plan':protocol['scope']}
    identity['sha256']=digest(identity)
    repo=ResearchRepository(run_id,identity,folder=folder);gateway=PilotGateway(repo,task,library)
    write_json(folder/'preregistration.json',identity)
    source=folder/'source_snapshot';source.mkdir()
    for path in identity['code']:
        rel=Path(path).relative_to(ROOT);destination=source/rel;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    started=time.monotonic()
    try:
        gateway.baseline()
        if group=='random_reference':run_random(repo,gateway)
        else:
            key=Path('/root/.config/auto_data_research/deepseek_api_key')
            if key.stat().st_mode&0o077:raise GateError('SERVER_KEY_PERMISSIONS')
            os.environ['DEEPSEEK_API_KEY']=key.read_text().strip()
            planner=SimpleNamespace(model=config['model'],kind='external',allow_external=True,
                timeout_seconds=config['timeout_seconds'],endpoint_env='DEEPSEEK_BASE_URL',key_env='DEEPSEEK_API_KEY')
            model=PilotModel(repo,SimpleNamespace(planner=planner),system_prompt=FORMAL_PROMPT)
            controller=AgentController(repo,gateway,model,context_builder=pilot_context)
            while not repo.state().incident_finished:
                try:
                    controller.step(Event('pilot-'+str(repo.state().decision_version),'development_pilot',time.time_ns()))
                except DecisionSchemaError as error:
                    # Format recovery only; no scientific proposal is supplied or altered by the platform.
                    number=repo.count_records('format_failure')+1
                    repo.put('format-failure:'+str(number),'format_failure',
                        {'record_id':'format-failure:'+str(number),'error':str(error),
                         'validation_errors':error.validation_errors,'action_executed':False,
                         'instruction':'Return a legal response using the existing tool schema; no scientific strategy was supplied.'},time.time_ns())
                repo.export()
        final=repo.get('final_policy')
        write_json(folder/'search_result.json',{'status':'completed','final_policy':final,'elapsed_seconds':time.monotonic()-started,
            'API':snapshot(repo),'candidate_count':len(gateway.candidates())})
    except Exception as error:
        write_json(folder/'failure.json',{'status':'failed','error':str(error) if isinstance(error,GateError) else type(error).__name__,
            'group':group,'repeat':repeat,'elapsed_seconds':time.monotonic()-started,'API':snapshot(repo)})
    finally:repo.export()
    return folder


def confirm_search(folder):
    folder=Path(folder)
    if (folder/'external_confirmation.json').exists():return read_json(folder/'external_confirmation.json')
    if not (folder/'search_result.json').exists():
        value={'status':'search_failed','effective_processing':False,'denominator_retained':True}
        write_json(folder/'external_confirmation.json',value);return value
    identity=read_json(folder/'preregistration.json');repo=ResearchRepository(folder.name,identity,folder=folder)
    task=TaskAssets(STUDY/'tasks/metropt3_dev');gateway=PilotGateway(repo,task,read_json(MEMORY/'library.json'))
    selected=read_json(folder/'final_policy.json')['selected_candidate']
    if selected=='B0':
        value={'status':'retained_baseline','effective_processing':False,'denominator_retained':True}
        write_json(folder/'external_confirmation.json',value);return value
    candidate=gateway.candidate(selected);asset=Path(candidate['asset']);transform=candidate['input_transform_path']
    # The final submission predates all external calls; never return these results to an agent.
    frozen={'selected_candidate':selected,'policy_sha256':file_hash(folder/'final_policy.json'),
        'processing_sha256':file_hash(asset),'input_transform_sha256':file_hash(transform) if transform else None,
        'confirmation_seeds':list(CONFIRMATION_SEEDS),'scope':'same_development_dataset_not_independent_task'}
    if not (folder/'confirmation_freeze.json').exists():write_json(folder/'confirmation_freeze.json',frozen)
    elif read_json(folder/'confirmation_freeze.json')!=frozen:raise GateError('PILOT_CONFIRMATION_FREEZE_CHANGED')
    outcomes=[]
    for seed in CONFIRMATION_SEEDS:
        baseline=gateway.adapter.evaluate('external_B0',None,DEFAULTS,seed=seed)
        treated=gateway.adapter.evaluate('external_selected',asset,DEFAULTS,transform=transform,seed=seed)
        control='input_identity' if transform else 'group_shuffle'
        matched=gateway.adapter.evaluate('external_control',asset,DEFAULTS,transform=transform,seed=seed,control=control)
        outcomes.append({'seed':seed,'baseline':baseline['metrics'],'candidate':treated['metrics'],'control':matched['metrics'],
            'control_kind':control,'candidate_result_sha256':digest(treated)})
        write_json(folder/'confirmation_progress.json',{'completed_seeds':len(outcomes),'results':outcomes})
    avg=lambda kind,metric:float(np.mean([v[kind][metric] for v in outcomes]))
    gain=(1-avg('candidate','NMSE')/avg('baseline','NMSE'))*100
    mae=(avg('candidate','NMAE')/avg('baseline','NMAE')-1)*100
    advantage=(1-avg('candidate','NMSE')/avg('control','NMSE'))*100
    gates={'each_seed_NMSE_improves':all(v['candidate']['NMSE']<v['baseline']['NMSE'] for v in outcomes),
        'mean_NMSE_gain_at_least_1pct':gain>=1,'mean_NMAE_deterioration_at_most_1pct':mae<=1,
        'matched_control_advantage_at_least_0_25pct':advantage>=.25,
        'no_human_strategy_intervention':gateway.journal.summary()['autonomy_eligible']}
    updates=[]
    for row in gateway.candidates():
        if not row.get('post_result_update'):continue
        update=row['post_result_update'];supported=False
        if row['status']=='trained':
            base=gateway.baseline()['metrics'];metric=row['result']['metrics']
            supported=metric['NMSE']<=base['NMSE']*.99 and metric['NMAE']<=base['NMAE']*1.01
        updates.append({'candidate_id':row['candidate_id'],'assessment':update['assessment'],
            'outcome_criterion_supported':supported,'overclaimed_global_gain':update['assessment']=='supported' and not supported,
            'receipt_traceable':update['receipt_sha256']==row['receipt_sha256']})
    value={'status':'confirmed','effective_processing':all(gates.values()),'gates':gates,'mean_NMSE_gain_pct':gain,
        'mean_NMAE_deterioration_pct':mae,'matched_control_advantage_pct':advantage,'results':outcomes,
        'feedback_checks':updates,'fact_accuracy':'unassessed_requires_independent_code_and_evidence_review',
        'mechanism_scope':'input_program_increment_over_same_weights_or_membership_control_not_physical_causality',
        'independent_task':False,'search_freeze':frozen}
    write_json(folder/'external_confirmation.json',value);repo.export();return value


def run_batch():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'batch_started.json').exists():raise GateError('PILOT_BATCH_ALREADY_STARTED_NO_REPLAY')
    frozen={'protocol_sha256':file_hash(PILOT/'protocol.json'),'document_sha256':file_hash(PILOT/'PROTOCOL.md'),
            'case_sha256':file_hash(PILOT/'registered_case.json'),'code':code_identity(),
            'library_sha256':file_hash(MEMORY/'library.json'),'started_ns':time.time_ns()}
    write_json(OUTPUT/'batch_started.json',frozen)
    protocol=read_json(PILOT/'protocol.json')
    searches=[]
    for entry in protocol['search_order']:
        if (code_identity()!=frozen['code'] or file_hash(PILOT/'protocol.json')!=frozen['protocol_sha256']
            or file_hash(PILOT/'PROTOCOL.md')!=frozen['document_sha256'] or file_hash(PILOT/'registered_case.json')!=frozen['case_sha256']
            or file_hash(MEMORY/'library.json')!=frozen['library_sha256']):raise GateError('PILOT_BATCH_BINDING_CHANGED')
        write_json(OUTPUT/'progress.json',{'phase':'search','active':entry,'completed_searches':len(searches),'queue_active':True})
        folder=run_search(entry['group'],entry['repeat']);searches.append(str(folder))
        print(canonical({'event':'search_finished','run_id':folder.name,'completed_searches':len(searches)}),flush=True)
    confirmations=[]
    for folder in searches:
        write_json(OUTPUT/'progress.json',{'phase':'external_confirmation','active_run':Path(folder).name,
            'completed_searches':len(searches),'completed_confirmations':len(confirmations),'queue_active':True})
        result=confirm_search(folder);confirmations.append({'run_id':Path(folder).name,**result})
        print(canonical({'event':'confirmation_finished','run_id':Path(folder).name,'effective':result['effective_processing']}),flush=True)
    training_files=sorted((OUTPUT/'runs').glob('*/training/*/result.json'))
    costs=[]
    for folder in searches:
        status=read_json(Path(folder)/'status.json')
        costs.append({'run_id':Path(folder).name,'API_budget':status['budget'],
            'training_attempts':len(status['trainings']),
            'completed_trainings':sum(r['status']=='completed' for r in status['trainings']),
            'training_process_seconds':sum(r.get('elapsed_seconds',0) for r in status['trainings'])})
    summary={'scope':protocol['scope'],'status':'completed','runs':confirmations,'costs':costs,
        'completed_new_training_count':len(training_files),'previous_total_training_count':protocol['previous_total_trainings'],
        'groups':{group:{'registered':3,'successful':sum(x['effective_processing'] for x in confirmations if x['run_id'].startswith(RUN_PREFIX+'_'+group+'_')),
                        'denominator_includes_failed_and_baseline_runs':True} for group in GROUPS},
        'independent_task':False,'random_reference_space_narrower_than_coded_agents':True,'statistical_agent_superiority':'unassessed'}
    write_json(OUTPUT/'summary.json',summary)
    write_json(OUTPUT/'progress.json',{'phase':'completed','queue_active':False,'completed_searches':12,'completed_confirmations':12})
    print(canonical({'event':'batch_completed','groups':summary['groups']}),flush=True)


if __name__=='__main__':
    try:run_batch()
    except Exception as error:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        write_json(OUTPUT/'batch_failure.json',{'status':'failed','error':str(error) if isinstance(error,GateError) else type(error).__name__})
        write_json(OUTPUT/'progress.json',{'phase':'failed','queue_active':False,'preserve_all_run_assets':True})
        raise
