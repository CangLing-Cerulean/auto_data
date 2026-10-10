"""New phase-4 study; all historical attempts and frozen projects remain separate."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from ..agent_runtime.budget import snapshot
from ..agent_runtime.controller import AgentController,DecisionSchemaError
from ..agent_runtime.state import Event
from ..config import GateError,canonical,digest
from ..selection.common import file_hash,read_json,write_json
from .experience import ROOT,STUDY
from .memory import MEMORY,verify_memory
from .runtime import CONFIG,PROMPT,EngineeringModel,ResearchRepository,code_identity,context_builder
from .signal_gateway import SignalGateway
from .task import TaskAssets

PILOT=STUDY/'phase4/20261010_signal_v2'
OUTPUT=Path('/tmp/auto_data_signal_study_20261010_v2')
RUN_PREFIX='p4signal2'
GROUPS=('full','no_memory','no_feedback')

FORMAL_PROMPT='''You are the autonomous data researcher in a registered DEVELOPMENT comparison on MetroPT3.
Use evidence to locate prediction weaknesses, form falsifiable processing hypotheses, write and execute
your own diagnostics and processing code, and update beliefs from the evidence available to YOUR group.
No cumulative calls, tokens, diagnostics, candidates, GPU jobs or search-time quotas apply. Decide when
the evidence warrants final submission. Fixed model/parameters and per-job resource rules still apply.
There is no requirement to discover an improvement. B0 is a legitimate final choice, including no coded
candidate or all execution failures. Every existing candidate needs execution/failure and one immutable
post-result update before submission. An engineering error leaves the scientific hypothesis inconclusive.

describe_task exposes tool schemas, exact sensor names and baseline_error_groups. read_baseline_error
provides measured B0 local errors, last-value persistence and overall comparisons. These are baseline
observations available to ALL groups; they do not reveal a hidden candidate outcome. run_diagnostic lets
you write any supported new T analysis. Distribution skew, shapes and memory alone are not evidence of
prediction deficits. register_signal is OPTIONAL for program execution but REQUIRED to claim a scored
diagnostic signal: choose an existing input-defined group and exact sensor subset, state a testable
hypothesis/alternative/falsification and a predicted local improvement >=3%, and cite a completed coded
diagnostic plus the relevant baseline evidence. observed_gap_pct must be the relative local NMSE excess
over persistence or overall B0. The platform scores it later, without correcting your scientific claim.
training_claim refers to one independently checkable measured T statistic: values for a named sensor,
or a one-dimensional named T feature. Values already have T normalization; mean/std use all T rows,
window features use all legal T windows. This numeric crosscheck does NOT prove the proposed mechanism.
Other original diagnoses are allowed but remain unassessed until separate independent verification.

Historical memory is optional and conditional. In no_memory no records are provided. Do not read peer
results, host experiment files, credentials, D/test/E/ground_truth, use network/subprocesses, install
dependencies, or train a replacement forecast model. Only supported NumPy, CPU torch, math, json,
pathlib, sys, collections and statistics imports are available to generated builders. Host file reads
outside the assigned inputs/outputs are denied and logged by a Python audit hook; this is not an OS sandbox.

Executable code gets positional DIRECTORY arguments: sys.argv[1] for inputs, sys.argv[2] for outputs.
np.load(pathlib.Path(sys.argv[1])/'train_inputs.npz',allow_pickle=False) returns values (1061863,15),
start_row (1053433,) of legal windows and named T features. Do not reconstruct consecutive starts,
fit on D, build the full huge window tensor, or treat rows as windows. Stream or sample T if needed.
Diagnostics write one small finite JSON object to outputs/diagnostic.json (<=8000 bytes).
A PROCESSING builder must write outputs/processing.npz with UNCHANGED int64 start_row, floating
weight and sampling_multiplier, all length 1053433, finite [0,4]. Pure transformations still write
uniform weights/sampling. Deletions use zeros, preserving the original start_row array; coverage and
expected loss normalization apply. Returning a transform function alone is not a runnable asset builder.
For a transform, the builder ALSO writes processor.py AND processor_state.json. processor.py imports
only torch/math and defines transform(x,state), x=(batch,32,15); output has identical shape, device,
dtype and finite values, sees only past x and frozen T-fitted JSON state, no trainable module, file/network
access, batch mixing, random/global state or input/state mutation. Original targets stay unchanged.
Processing builder source <=8000 bytes; processor.py <=16000, state <=24000. No model training in builders.

propose_processing requires parameters={}, the ordinary proposal object, signal_id (or empty string if
not scored), and control_code (or empty string if no executable matched control). control_code is YOUR
independent asset builder with the SAME I/O contract, registered BEFORE candidate execution. State in
proposal.control which registered competing explanation it tests and which differences remain.
For weighting/sampling only, an empty control_code permits external within-T-group shuffle; a pure
transform without supplied matched code has no mechanism control. Identity is only B0, not mechanism
proof. Confirmation and controls run after all searches finish, never returned as search feedback.
proposal.expected_effect uses metric=NMSE,target_scope=all,direction=decrease,criterion describing
>=1% overall gain and <=1% NMAE deterioration; your separate registered local prediction is also scored.

In visible-feedback groups choose the lowest observed NMSE among B0 and candidates whose NMAE is
<=1.01*B0. Mark supported only for the stated measured criterion, not for a mechanism or independent
generalization. If a candidate improves less than 1%, say that explicitly. When feedback_mode=withheld,
execution success/failure and receipt hashes are visible, but candidate metrics/ranking are not: select
using T and prior knowledge, keep updates inconclusive, never claim to know hidden outcomes.
Use list_candidates/read_candidate/read_evidence pages for complete older records. pending_update
shows candidates still needing updates. An already frozen update is returned unchanged, not rewritten.
Register new hypotheses/IDs for revised experiments. Cite only currently visible_evidence_refs.
Keep diagnostic.json small and avoid printing arrays. Stop honestly with B0 when further experiments
are not warranted. Do not assume a quota remains to be spent.

''' + PROMPT[PROMPT.index('Keep Chinese decision summaries'):].replace(
    '"parameters": {},', '"parameters": {},\n      "signal_id": "",\n      "control_code": "",')
FORMAL_PROMPT+='''
The above proposal structure-only example must also include action.arguments.signal_id and
action.arguments.control_code, each a string. register_signal's training_claim has exactly
array, channel, statistic, quantile, value; sensor channel is empty only for one-dimensional features.
Final submission uses submit_policy with candidate_id and reason. Never alter the fixed model.
'''

def study_code():
    code=code_identity()
    for name in ('run_signal_study.py','check_signal_readiness.py','run_signal_checks.py'):
        path=ROOT/'scripts'/name;code[str(path)]=file_hash(path)
    return code

class StudyModel(EngineeringModel):
    def request_spec(self,context,attempt):
        body,_=super().request_spec(context,attempt);body['temperature']=0.8
        return body,'controller-call:'+digest({'context':context['context_version'],'messages':digest(body['messages']),
            'generation':{k:v for k,v in body.items() if k!='messages'},'transport_attempt':attempt})

def study_context(repo,gateway,event):
    value=context_builder(repo,gateway,event)
    value['task']['goal']='measured_prediction_deficit_and_targeted_processing_in_registered_development_comparison'
    value['task']['budget_note']='No cumulative search limits; per-job bounds and all actual costs remain'
    value['task']['input_contract']={'input_directory':'sys.argv[1]','output_directory':'sys.argv[2]',
        'values_shape':[1061863,15],'legal_start_row_shape':[1053433],
        'builder_outputs':['processing.npz','processor.py and processor_state.json when transforming'],
        'diagnostic_output':'diagnostic.json; small finite JSON <=8000 bytes'}
    failures=list(repo.iter_records('format_failure'))
    if failures:value['task']['last_format_failure']=failures[-1]
    return value

def require_ready():
    protocol=read_json(PILOT/'protocol.json');preflight=read_json(PILOT/'preflight.json')
    if (protocol['protocol_document_sha256']!=file_hash(PILOT/'PROTOCOL.md') or preflight['status']!='passed'
        or preflight['protocol_sha256']!=file_hash(PILOT/'protocol.json') or preflight['code']!=study_code()):
        raise GateError('SIGNAL_PREFLIGHT_BINDING')
    return protocol

def run_search(entry,protocol):
    group,repeat=entry['group'],entry['repeat'];run_id=f'{RUN_PREFIX}_{group}_r{repeat}'
    folder=OUTPUT/'runs'/run_id
    if folder.exists():raise GateError('SIGNAL_RUN_NOT_REPLAYED')
    if shutil.disk_usage(OUTPUT.parent).free<6*1024**3:raise GateError('SIGNAL_DISK_RESERVE')
    task=TaskAssets(STUDY/'tasks/metropt3_dev');library=read_json(MEMORY/'library.json');verify_memory(library)
    config=CONFIG|{'version':'signal-study-development-v2','scope':'development_only','seed':entry['training_seed'],
        'group':group,'max_candidates':None,'max_diagnostics':None,'max_calls':None,'max_llm_tokens':None,
        'max_training_jobs':None,'execution_profile':'research_open','experience_mode':'none' if group=='no_memory' else 'full',
        'feedback_mode':'withheld' if group=='no_feedback' else 'visible','input_transforms_allowed':True,
        'formal_research_protocol':True,'parameter_ranges':protocol['fixed_parameters']}
    identity={'config':config,'code':study_code(),'prompt_sha256':digest(FORMAL_PROMPT),
        'protocol_sha256':file_hash(PILOT/'protocol.json'),'protocol_document_sha256':file_hash(PILOT/'PROTOCOL.md'),
        'task_sha256':file_hash(task.folder/'task.json'),'library_sha256':library['sha256']}
    identity['sha256']=digest(identity);repo=ResearchRepository(run_id,identity,folder=folder)
    gateway=SignalGateway(repo,task,library,PILOT)
    gateway.guards.update(identity['code'])
    write_json(folder/'preregistration.json',identity)
    for p in identity['code']:
        target=folder/'source_snapshot'/Path(p).relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    started=time.monotonic()
    try:
        gateway.baseline()
        key=Path('/root/.config/auto_data_research/deepseek_api_key')
        if key.stat().st_mode&0o077:raise GateError('SERVER_KEY_PERMISSIONS')
        os.environ['DEEPSEEK_API_KEY']=key.read_text().strip()
        planner=SimpleNamespace(model=config['model'],kind='external',allow_external=True,
            timeout_seconds=config['timeout_seconds'],endpoint_env='DEEPSEEK_BASE_URL',key_env='DEEPSEEK_API_KEY')
        controller=AgentController(repo,gateway,StudyModel(repo,SimpleNamespace(planner=planner),system_prompt=FORMAL_PROMPT),context_builder=study_context)
        while not repo.state().incident_finished:
            try:controller.step(Event('signal-'+str(repo.state().decision_version),'signal_study',time.time_ns()))
            except DecisionSchemaError as error:
                number=repo.count_records('format_failure')+1
                repo.put('format-failure:'+str(number),'format_failure',{'record_id':'format-failure:'+str(number),
                    'error':str(error),'validation_errors':error.validation_errors,'action_executed':False,
                    'instruction':'Repair only JSON/schema format using the existing tool contracts.'},time.time_ns())
            repo.export()
        write_json(folder/'search_result.json',{'status':'completed','final_policy':repo.get('final_policy'),
            'elapsed_seconds':time.monotonic()-started,'API':snapshot(repo)})
    except Exception as error:
        write_json(folder/'failure.json',{'status':'failed','error':str(error),'error_type':type(error).__name__,
            'group':group,'repeat':repeat,'elapsed_seconds':time.monotonic()-started,'API':snapshot(repo)})
    finally:
        repo.export();os.environ.pop('DEEPSEEK_API_KEY',None)
    return folder

def run_batch():
    protocol=require_ready();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'batch_started.json').exists():raise GateError('SIGNAL_BATCH_NOT_REPLAYED')
    frozen={'code':study_code(),'protocol_sha256':file_hash(PILOT/'protocol.json'),
        'document_sha256':file_hash(PILOT/'PROTOCOL.md'),'library_sha256':file_hash(MEMORY/'library.json'),
        'started_ns':time.time_ns(),'previous_total_trainings':protocol['previous_total_trainings'],
        'previous_API_attempts':protocol['previous_API_attempts']}
    write_json(OUTPUT/'batch_started.json',frozen);searches=[]
    for entry in protocol['search_order']:
        if study_code()!=frozen['code'] or file_hash(PILOT/'protocol.json')!=frozen['protocol_sha256'] or file_hash(MEMORY/'library.json')!=frozen['library_sha256']:
            raise GateError('SIGNAL_FROZEN_BINDING_CHANGED')
        write_json(OUTPUT/'progress.json',{'phase':'search','active':entry,'completed_searches':len(searches),'queue_active':True})
        folder=run_search(entry,protocol);searches.append(folder)
        print(canonical({'event':'search_finished','run_id':folder.name,'completed_searches':len(searches)}),flush=True)
    from .signal_scoring import confirm_and_score
    scores=[]
    for folder in searches:
        if study_code()!=frozen['code'] or file_hash(PILOT/'protocol.json')!=frozen['protocol_sha256'] or file_hash(MEMORY/'library.json')!=frozen['library_sha256']:
            raise GateError('SIGNAL_FROZEN_BINDING_CHANGED')
        write_json(OUTPUT/'progress.json',{'phase':'confirmation','active_run':folder.name,'completed_searches':9,
            'completed_confirmations':len(scores),'queue_active':True})
        scores.append(confirm_and_score(folder,protocol))
    groups={g:{'registered':3,'completed':sum(s['search_completed'] for s in scores if s['group']==g),
        'effective_processing':sum(s['E'] is True for s in scores if s['group']==g),
        'operational_signal_verified':sum(s['operational_signal'] is True for s in scores if s['group']==g),
        'denominator_includes_failure_B0_and_information_violations':True} for g in GROUPS}
    write_json(OUTPUT/'summary.json',{'status':'completed_pending_primary_semantic_review','groups':groups,'runs':scores,
        'independent_task':False,'equal_search_budget':False,'matched_control_semantics':'primary_review_required_before_C_pass'})
    write_json(OUTPUT/'progress.json',{'phase':'completed_pending_review','completed_searches':9,'completed_confirmations':9,'queue_active':False})

def main():
    try:run_batch()
    except BaseException as error:
        OUTPUT.mkdir(parents=True,exist_ok=True)
        write_json(OUTPUT/'batch_failure.json',{'status':'failed','error':str(error),'error_type':type(error).__name__})
        write_json(OUTPUT/'progress.json',{'phase':'failed','queue_active':False,'preserve_all_attempts':True})
        raise
