"""Bounded two-arm cold-start orchestration; no imported research outcomes."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from coldstart_reference import predict, evidence

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10'
REMOTE='/root/auto_data_experiments/20261018_cold_start_r10'
SANDBOX=Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/cold_start_calls')
ARMS=('STRONG','SEPARATE')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def transport():
    c=read(ROOT/'research_runs/ett_confirmation/private/connection.json')
    options=['-i',c['identity_file'],'-o','BatchMode=yes','-o','ConnectTimeout=15']
    return ['ssh',*options,'-p',str(c['port']),c['host']]


def remote(script):
    p=subprocess.run(transport()+['/opt/miniconda3/bin/python','-B','-'],input=script.encode(),capture_output=True,check=True)
    return json.loads(p.stdout.decode())


def rpc(action,identifier,request=None):
    """Server process writes normal files; an SSH stdout interruption cannot kill training."""
    assert identifier.replace('_','').isalnum()
    payload=base64.b64encode(json.dumps(request).encode()).decode()
    setup=f'''import pathlib,json,subprocess,sys,base64
r=pathlib.Path({REMOTE!r});p=r/'control'/{identifier!r}
assert r.resolve()==r and r.is_dir()
if not p.exists():
 p.mkdir(parents=True)
 (p/'request.json').write_bytes(base64.b64decode({payload!r}))
 command=[sys.executable,'-B',str(r/'coldstart_runtime.py'),{action!r},'--root',str(r),'--request',str(p/'request.json'),'--output',str(p/'output.json')]
 wrapper="import subprocess,json,pathlib; p=pathlib.Path("+repr(str(p))+"); result=subprocess.run("+repr(command)+"); (p/'exit.json').write_text(json.dumps(dict(returncode=result.returncode)))"
 with (p/'run.log').open('xb') as log:
  child=subprocess.Popen([sys.executable,'-B','-c',wrapper],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (p/'started.json').write_text(json.dumps(dict(pid=child.pid,action={action!r})))
else:
 assert json.loads((p/'request.json').read_text())==json.loads(base64.b64decode({payload!r}))
print(json.dumps(dict(started=True)))
'''
    remote(setup)
    while True:
        value=remote(f'''import pathlib,json,os
p=pathlib.Path({REMOTE!r})/'control'/{identifier!r}
if (p/'exit.json').exists():
 e=json.loads((p/'exit.json').read_text());print(json.dumps(dict(done=True,exit=e,output=json.loads((p/'output.json').read_text()) if (p/'output.json').exists() else None,log=(p/'run.log').read_text()[-4000:])))
else:
 pid=json.loads((p/'started.json').read_text())['pid']
 try:os.kill(pid,0);alive=True
 except ProcessLookupError:alive=False
 print(json.dumps(dict(done=False,alive=alive)))
''')
        if value['done']:
            assert value['exit']['returncode']==0,value
            return value['output']
        assert value['alive'],'Worker stopped without completion receipt; retain files and investigate'
        time.sleep(10)


def obj(properties):
    return dict(type='object',properties=properties,required=list(properties),additionalProperties=False)


def schemas():
    string=dict(type='string');number=dict(type='number')
    measurement=obj(dict(expression=string,meaning=string,expected_information=string))
    first=obj(dict(measurement=dict(anyOf=[measurement,dict(type='null')]),reasoning=string))
    target=obj(dict(window_expression=string,channels=dict(type='array',items=dict(type='integer')),horizons=dict(type='array',items=dict(type='integer'))))
    hypothesis=obj({k:number for k in ('mean_y','mean_z','std_y','std_z','rho')})
    second=obj(dict(mode=dict(type='string',enum=['S','W']),weight_expression=string,target=target,
                    diagnostic_claim=string,falsifying_response=string,forecast=hypothesis,reasoning=string))
    return first,second


COMMON='''You are an autonomous experimental agent. Use only FACTS and your own recorded fresh observations. No tools, files, shell, web, external papers, other conversations, prior research records, or source identities. Return the required JSON only. Your pretrained knowledge cannot be erased; do not claim outside measurements as observations. You have 8 recipe attempts; an invalid expression or repeated recipe consumes one attempt. Every valid new recipe gets 3 fixed-seed complete trainings. There is no predefined recipe menu. The goal is to improve prediction on the current task under the fixed budget.
The fixed backbone is channel-independent DLinear, input32/horizon8, moving average25 replicate padding; all370 channels; Adam .001,beta .9/.999,epsilon1e-8,batch256,3000steps. A fits normalization and trains; all B chooses checkpoints at500-step grid; all C supplies recipe feedback. Never alter truth, channels, model, optimizer, or steps. D remains unseen until both exploration paths end. Zero inputs are retained. Array indices are zero-based. Within each partition, position is window index/(N-1), not a model input.
Each round first propose zero or one measurement, then receive its execution facts and propose one recipe. Measurement variables x[window,time,channel] and position[window]; allowed +,-,*,unary minus,finite constants abs<=1000; abs,tanh,square,log1p_abs,safe_ratio(a,b)=a/(1+abs(b)),where(condition,a,b); comparisons <,<=,>,>=. Time operators mean_time,std_time,first_time,last_time,diff_time,lagdiff(v,k); channel operators mean_channels,std_channels,max_channels,channel(v,i). lagdiff k must literal1..31 (also less than currenttimeaxis); channel i literal0..369. Reductions remove named axis; differences retain shortened time axis. Non-scalar operands require identical axes and shapes; no implicit channel/time broadcasting. Result MUST have window axis only, finite numeric, <=128 Python AST nodes. No division,power,indexing,attributes,strings or other calls. An empty measurement is null. New valid measurements receive q1..q8, normalized by A mean/std only (constant scale1). All previously valid q remain usable.
Recipe variables q1..q8 that actually exist, plus position. Same arithmetic,abs,tanh,square,log1p_abs,safe_ratio,where,comparisons; additionally sigmoid and exp_clip=exp(clip(v,-4,4)). No x,time or channel operators in recipe. Numeric scalar broadcasts. Weights must be finite,nonnegative,>=256 positive windows; normalization divides by A mean; normalizedmax<=20 otherwise rejection (no repair). Mode S uses these weights as sampling probabilities and unweighted loss; W uses uniform sampling with corresponding mean-one loss weights. Both have same expected gradient but may have different training responses.
Target is declared before response: window_expression uses the recipe variables/operators and returns boolean or numeric0/1; channels and horizons are nonempty unique index lists. A target can include all channels/horizons only if it selects a proper nonempty subset of windows; the combined target and complement must be nonempty in B and C. Target definition cannot be edited after outcome. Y=(parentglobalMSE-childglobalMSE)/parentglobalMSE. Target gain and complement gain both use that SAME parentglobalMSE denominator; Z=targetgain-complementgain; Y=p*targetgain+(1-p)*complementgain. Values are fractional gains (.01 means1%). The observed response is three-seed mean, with eachseed shown; seeds are not independent tasks.
Before execution submit a joint Gaussian forecast for Y,Z with std_y,std_z>0 and abs(rho)<1, a falsifiable claim, and an adverse response that would challenge it. Automatic ordinary reference is bivariate Bayesian linear regression on your prior queries only, no pretrained labels: prior K=I6,M=0,Psi=I2,nu3; x=[1,positivefraction,ESS/N,Wflag,tanh(parentBtarget/complementcontrast divided by global B MSE),Btargetfraction]. Its Student-t prediction is frozen before current outcome and revealed with feedback. The current response is scored before fitting it. A changed measurement,target,or claim starts new diagnostic version; no retrospectively earned credit. A measurement is executed, not established as a causal or generalizable diagnostic.
At2,4,8 attempts choose among actually observed recipes: meanY>=.01 and meanNMAEgain>=-.01, maximizeY, ties earlier; else PARENT. Diagnostic support cannot change this rule or observed scores. Costs, harmful results and failed attempts count. You may reason freely over all your own raw feedback and prior forecasts. Do not invent measured values.
'''


def call(arm,round_no,phase,facts):
    folder=STAGE/'calls'/arm/f'{round_no:02d}_{phase}';folder.mkdir(parents=True)
    first,second=schemas();write(folder/'schema.json',first if phase=='measurement' else second)
    focus=('Use ordinary experimental reasoning to improve actual recipe value; all of your raw outcomes, predictions and counterexamples are available.' if arm=='STRONG' else 'Keep recipe utility and diagnostic support separate. Use conditional Z evidence given observed Y for a diagnosis, not Y success alone. When a diagnosis has adverse conditional evidence, consider a new input measurement that could distinguish explanations on a future query. Do not force revision or assume every mismatch means missing information. Your deterministic diagnostic ledger is included.')
    prompt=COMMON+'\nFrozen forecast numerical limits: abs(mean_y),abs(mean_z)<=10; 1e-6<=std_y,std_z<=10; abs(rho)<=.999999. Outside these limits consumes an invalid attempt, without repair.\n'+focus+'\nCURRENT_PHASE '+phase+'\nFACTS\n'+json.dumps(facts,ensure_ascii=False,allow_nan=False)
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
    write(folder/'started.json',dict(started=time.time(),prompt_sha256=sha(folder/'prompt.txt'),schema_sha256=sha(folder/'schema.json')))
    command=[shutil.which('codex'),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-s','read-only','-C',str(SANDBOX),'-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','-c','project_doc_max_bytes=0','--json','--output-schema',str(folder/'schema.json'),'-o',str(folder/'response.json'),'-']
    start=time.perf_counter()
    with (folder/'events.jsonl').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        process=subprocess.run(command,input=prompt.encode(),stdout=out,stderr=err)
    receipt=dict(exit_code=process.returncode,seconds=time.perf_counter()-start,usage=None,tool_items=[],transport_errors=[])
    completed=False
    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line);item=event.get('item',{})
        if event.get('type')=='turn.completed': completed=True;receipt['usage']=event.get('usage')
        if item.get('type')=='error':receipt['transport_errors'].append(item)
        elif item.get('type') not in (None,'agent_message','reasoning'):receipt['tool_items'].append(item.get('type'))
    receipt['status']='PASS' if process.returncode==0 and completed and not receipt['tool_items'] else 'FAILED'
    write(folder/'receipt.json',receipt)
    assert receipt['status']=='PASS',receipt
    return read(folder/'response.json')


def select(history):
    eligible=[h for h in history if h.get('result',{}).get('status')=='completed' and h['result']['Y']>=.01 and h['result']['NMAE_gain']>=-.01]
    return max(eligible,key=lambda h:h['result']['Y'])['result']['run'] if eligible else 'PARENT'


def public_history(history):
    """Expose logical per-arm facts, never physical cross-arm cache information."""
    result=[]
    for item in history:
        visible=json.loads(json.dumps(item))
        outcome=visible.get('result',{})
        for key in ('reused_cross_arm','actual_new_full_trainings','seconds'):
            outcome.pop(key,None)
        result.append(visible)
    return result


def validate_forecast(forecast):
    values={k:forecast[k] for k in ('mean_y','mean_z','std_y','std_z','rho')}
    if not all(type(v) in (int,float) and math.isfinite(v) for v in values.values()):
        raise ValueError('forecast fields must be finite numbers')
    if not (abs(values['mean_y'])<=10 and abs(values['mean_z'])<=10
            and 1e-6<=values['std_y']<=10 and 1e-6<=values['std_z']<=10
            and abs(values['rho'])<=.999999):
        raise ValueError('forecast exceeds the frozen numerical domain')


def run():
    lock=read(STAGE/'execution_lock.json');pub=read(STAGE/'source_publication_receipt.json')
    assert pub['status']=='exact_commit_and_tree_verified'
    for name,digest in lock['runtime_code_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
    assert sha(STAGE/'PROTOCOL.md')==lock['protocol_sha256']
    SANDBOX.mkdir(parents=True,exist_ok=True)
    write(STAGE/'trial_started.json',dict(started=time.time(),source_commit=pub['source_commit']))
    parent_result=rpc('parent','parents')
    write(STAGE/'parent_facts.json',parent_result['facts'])
    write(STAGE/'parent_result.json',parent_result)
    states={arm:dict(registry=[],history=[],reference_history=[],diagnostic_ledger={}) for arm in ARMS}
    for round_no in range(1,9):
        # Fixed alternating order balances the serial execution order across arms.
        order=ARMS if round_no%2 else ARMS[::-1]
        for arm in order:
            state=states[arm]
            facts=dict(round=round_no,maximum_rounds=8,parent=parent_result['facts'],registry=state['registry'],history=public_history(state['history']))
            if arm=='SEPARATE':facts['diagnostic_ledger']=state['diagnostic_ledger']
            proposal=call(arm,round_no,'measurement',facts)
            registry=list(state['registry'])
            if proposal['measurement'] is not None:
                m=proposal['measurement'];registry.append(dict(id=f'q{len(registry)+1}',**m))
            measurement=rpc('measure',f'{arm}_{round_no:02d}_measure',dict(registry=registry))
            item=dict(round=round_no,measurement_proposal=proposal,measurement_feedback=measurement)
            if measurement['status']!='PASS':
                item['result']=dict(status='invalid_measurement',error=measurement.get('error'));state['history'].append(item)
            else:
                state['registry']=registry;facts['registry']=registry;facts['measurement_feedback']=measurement
                decision=call(arm,round_no,'recipe',facts);item['decision']=decision
                f=decision['forecast']
                try:
                    # Validate forecast without current labels; numeric invalidity consumes the attempt.
                    validate_forecast(f)
                    evidence(predict([], [1.,1.,1.,0.,0.,.5]), f, 0., 0.)
                    valid=True
                except (ValueError,AssertionError,TypeError) as exc:
                    valid=False;item['result']=dict(status='invalid_forecast',error=str(exc))
                if valid:
                    request=dict(arm=arm,round=round_no,registry=registry,reference_history=state['reference_history'],**decision)
                    result=rpc('query',f'{arm}_{round_no:02d}_query',request);item['result']=result
                    if result['status']=='completed':
                        baseline=result['prequery']['reference_forecast']
                        ev=evidence(baseline,f,result['Y'],result['Z'])
                        version_material=dict(target=decision['target'],claim=decision['diagnostic_claim'],registry=[dict(id=q['id'],expression=q['expression']) for q in registry])
                        version=hashlib.sha256(json.dumps(version_material,sort_keys=True).encode()).hexdigest()
                        item['reference_forecast']=baseline
                        if arm=='SEPARATE':
                            item['conditional_evidence']=ev;item['diagnostic_version']=version
                            ledger=state['diagnostic_ledger'].setdefault(version,dict(definition=version_material,queries=[],sum_log_ratio=0.))
                            ledger['queries'].append(round_no);ledger['sum_log_ratio']+=ev['log_ratio']
                        # The ordinary arm receives the same raw measurements and reference forecasts;
                        # only deterministic conditional evidence/credit management is the intervention.
                        state['reference_history'].append(dict(x=result['prequery']['x'],y=[result['Y'],result['Z']]))
                state['history'].append(item)
            snapshot=STAGE/'snapshots';snapshot.mkdir(exist_ok=True)
            if round_no in (2,4,8):item['recommendation']=select(state['history'])
            write(snapshot/f'{arm}_{round_no:02d}.json',state)
            print(json.dumps(dict(arm=arm,round=round_no,status=item['result']['status'],Y=item['result'].get('Y'),recommendation=item.get('recommendation'))),flush=True)
    recommendations={arm:select(state['history']) for arm,state in states.items()}
    write(STAGE/'final_recommendations.json',dict(status='FROZEN',recommendations=recommendations,D_accessed=False,
          snapshots={f'{arm}_08.json':sha(STAGE/'snapshots'/f'{arm}_08.json') for arm in ARMS}))
    write(STAGE/'trial_completed.json',dict(status='completed',finished=time.time(),recommendations=recommendations,D_evaluation_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run',));args=parser.parse_args();run()
