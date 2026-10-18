"""Freeze and run four budget-matched, source-only executable-measurement calls."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from diagnosis_pool_features import read,sha,write
from window_measurement_dsl import evaluate

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261018_measurement_generation_r4'
OUT=STAGE/'generation'
SANDBOX=Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/window_signal_calls')
SCRIPTS=('window_measurement_dsl.py','prepare_window_signal_inputs.py','replay_window_signals.py','generate_window_signals.py')

def prepare():
    assert read(STAGE/'interface_audit.json')['status']=='PASS'
    assert read(BASE/'20261018_parent_measurements_r3/correction_audit.json')['status']=='PASS'
    OUT.mkdir();SANDBOX.mkdir()
    item=dict(type='object',additionalProperties=False,properties={k:{'type':'string'} for k in ('name','expression','measurement_meaning','limitation','falsifying_response')},required=['name','expression','measurement_meaning','limitation','falsifying_response'])
    write(OUT/'schema.json',dict(type='object',additionalProperties=False,properties={'signals':dict(type='array',items=item,minItems=0,maxItems=2),'reasoning':{'type':'string'}},required=['signals','reasoning']))
    calls=[]
    for family in ('ETT','Hydraulic'):
        packet=STAGE/'corrected_inputs'/f'{family}_facts.json'
        for mode in ('ordinary','mismatch'):
            folder=OUT/f'{family}_{mode}';folder.mkdir()
            focus=('Use ordinary supervised error-driven feature engineering to improve future Y response prediction. Complete conditional Z error information is available, but no special diagnosis reliability objective is required.' if mode=='ordinary' else 'Use qualified conditional Z mismatches as counterexamples to explanations, distinguish recipe Y from diagnostic reliability, and propose reusable parent-only measurements with specific future falsifying responses. Missing information is only one possible source of mismatch.')
            prompt='''You are an experimental measurement generator. Return only the required JSON. Do not use tools, files, web, prior conversations, task identity, or hidden target data. All permitted evidence is in FACTS. At most two expressions; zero is allowed. This is retrospective source development; few tasks and residual mismatch do not establish causal mechanisms.
Y=(parent global MSE-child global MSE)/parent global MSE. Z=target improvement minus complement improvement, both using that SAME parent global denominator. The target is the latter half of the horizon, p=0.5. Y and Z and the observed-seed-Y conditional residuals may guide your reasoning, but are NEVER runtime measurement inputs. Three seeds are not independent tasks. S uses nonuniform sampling; W uses corresponding weights under uniform sampling. Source-task-out errors have not used the held source task for fitting.
The executable interface accepts vectors A_<input> and B_<input> for each of the 8 input_window_columns, and E_<error> for each of the 6 error_window_columns. A/B inputs are standardized by the original A parameters. E vectors are aligned B parent errors, averaged over three seeds. E_loss, E_target_loss, E_complement_loss are normalized by each seed's global parent B MSE; E_bias/E_absolute_error by its global RMSE; E_contrast=E_target_loss-E_complement_loss. Scalar action variables: exposure_shift_first, removed_fraction, weighted_implementation. No other identifiers are allowed. first_mean/second_mean use the frozen early/late A groups. Positive exposure_shift_first favors the first A group.
Expressions allow +,-,*, unary -, constants -1,0,0.5,1,2; abs(x), tanh(x), square(x), log1p_abs(x)=log(1+abs(x)); safe_ratio(a,b)=a/(1+abs(b)); mean(vector), std(vector), center(vector)=vector-mean(vector), unit(vector)=(vector-mean(vector))/(1+std(vector)); first_mean(A_vector), second_mean(A_vector). No division, powers, indexing, comparisons, strings, arbitrary functions, or Python code. A vectors may combine only with A vectors or scalars; B and E vectors only with B/E vectors or scalars. Reductions return scalars which may combine. Final result must be a finite scalar. At most 95 Python AST nodes, including operator/load nodes.
You may define a new nonlinear window measurement rather than only combining existing task scalars. Summary evidence contains correlations and B-error quartile profiles; it does not reveal every nonlinear statistic. Do not invent measured values for an uncomputed expression. Strong controls include all original 26 features, generic parent-window correlations, action-aligned coverage/correlation statistics, and their full quadratic expansion. Downstream ridge is task-balanced, lambda=1 with feature variance floor 1; predictive scale is leave-task-out residual error floored at 0.01. Ordinary and conditional generators have exactly the same factual input and two-expression budget. An expression does not prove information gain, generalization or causal validity. State uncertainty and allow abstention.
'''+focus+'\nFACTS\n'+packet.read_text(encoding='utf-8')
            (folder/'prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
            calls.append(dict(name=folder.name,family=family,mode=mode,input_sha256=sha(packet),prompt_sha256=sha(folder/'prompt.txt')))
    scripts={name:sha(ROOT/'scripts'/name) for name in SCRIPTS}
    table_root=BASE/'20261018_parent_measurements_r3/frequency_corrected'
    tables={p.name:sha(p) for p in table_root.glob('*.npz')};assert len(tables)==5
    write(STAGE/'evaluation_lock.json',dict(script_sha256=scripts,table_sha256=tables,protocol_sha256=sha(STAGE/'PROTOCOL.md'),pool_sha256=sha(BASE/'20261018_conditional_diagnosis_r1/development_pool.json'),methods=__import__('replay_window_signals').METHODS,source_splits=['ETT_to_Hydraulic','Hydraulic_to_ETT'],nodes=[2,4,8],new_full_trainings=0))
    write(OUT/'generation_lock.json',dict(calls=calls,model='gpt-6.1-sol',reasoning='high',max_calls=4,no_regeneration=True,script_sha256=scripts,protocol_sha256=sha(STAGE/'PROTOCOL.md'),evaluation_lock_sha256=sha(STAGE/'evaluation_lock.json'),schema_sha256=sha(OUT/'schema.json')))
    print(dict(status='FROZEN',calls=4,target_data_in_prompts=False))

def run(name):
    lock=read(OUT/'generation_lock.json');entry=next(c for c in lock['calls'] if c['name']==name);folder=OUT/name
    for script,h in lock['script_sha256'].items():assert sha(ROOT/'scripts'/script)==h
    assert sha(folder/'prompt.txt')==entry['prompt_sha256'] and sha(OUT/'schema.json')==lock['schema_sha256']
    assert sha(STAGE/'evaluation_lock.json')==lock['evaluation_lock_sha256'] and sha(STAGE/'PROTOCOL.md')==lock['protocol_sha256']
    write(folder/'started.json',dict(started=time.time(),model=lock['model'],reasoning=lock['reasoning']))
    command=[shutil.which('codex'),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-s','read-only','-C',str(SANDBOX),'-m',lock['model'],'-c','model_reasoning_effort="high"','--json','--output-schema',str(OUT/'schema.json'),'-o',str(folder/'response.json'),'-']
    start=time.perf_counter()
    with (folder/'events.jsonl').open('xb') as output,(folder/'stderr.log').open('xb') as error:
        execution=subprocess.run(command,input=(folder/'prompt.txt').read_bytes(),stdout=output,stderr=error)
    receipt=dict(exit_code=execution.returncode,seconds=time.perf_counter()-start,model_requested=lock['model'],reasoning_requested='high',usage=None,tool_items=[],transport_errors=[]);completed=False
    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line);item=event.get('item',{})
        if event.get('type')=='turn.completed':receipt['usage']=event.get('usage');completed=True
        if item.get('type')=='error':receipt['transport_errors'].append(item)
        elif item.get('type') not in (None,'agent_message','reasoning'):receipt['tool_items'].append(item.get('type'))
    try:
        assert execution.returncode==0 and completed and not receipt['tool_items']
        signals=read(folder/'response.json')['signals'];assert len(signals)<=2 and len({s['name'] for s in signals})==len(signals)
        # Validate only on the source. Target execution waits until all four outputs freeze.
        records=read(BASE/'20261018_conditional_diagnosis_r1/development_pool.json')['records']
        for r in records:
            if r['family']!=entry['family']:continue
            with np.load(BASE/'20261018_parent_measurements_r3/frequency_corrected'/f"{r['family']}_{r['task']}.npz",allow_pickle=False) as z:table=dict(z)
            for signal in signals:evaluate(signal['expression'],table,r['features'])
        receipt.update(status='PASS',signals=len(signals),response_sha256=sha(folder/'response.json'))
    except Exception as exc:receipt.update(status='FAILED',error=repr(exc))
    write(folder/'receipt.json',receipt);print(json.dumps(dict(name=name,**receipt)))

def seal():
    lock=read(OUT/'generation_lock.json');calls={}
    assert sha(STAGE/'evaluation_lock.json')==lock['evaluation_lock_sha256']
    for entry in lock['calls']:
        folder=OUT/entry['name'];receipt=read(folder/'receipt.json')
        assert receipt['status']=='PASS' and sha(folder/'response.json')==receipt['response_sha256']
        assert sha(folder/'prompt.txt')==entry['prompt_sha256']
        calls[entry['name']]={name:sha(folder/name) for name in ('response.json','receipt.json','prompt.txt','events.jsonl')}
    write(STAGE/'generated_outputs_lock.json',dict(calls=calls,evaluation_lock_sha256=sha(STAGE/'evaluation_lock.json'),generation_lock_sha256=sha(OUT/'generation_lock.json'),target_measurements_not_yet_evaluated=True))
    print(dict(status='SEALED',calls=len(calls)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','seal'));parser.add_argument('--name');args=parser.parse_args()
    if args.action=='run':run(args.name)
    else:globals()[args.action]()
