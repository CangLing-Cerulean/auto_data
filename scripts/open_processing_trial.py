"""Create-only bounded orchestration for the r11 open-processing comparison."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from coldstart_trial import read, write, sha, transport, obj, validate_forecast
from open_processing_reference import predict, evidence


ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_open_processing_r11'
REMOTE='/root/auto_data_experiments/20261018_open_processing_r11'
SANDBOX=Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/open_processing_calls')
ARMS=('STRONG','SEPARATE')


def remote(script):
    process=subprocess.run(transport()+['/opt/miniconda3/bin/python','-B','-'],
                           input=script.encode(),capture_output=True,check=True)
    return json.loads(process.stdout.decode())


def rpc(action,identifier,request):
    """Detached workers retain ordinary supervision logs across SSH interruptions."""
    assert action in ('measure','query') and identifier.replace('_','').isalnum()
    payload=base64.b64encode(json.dumps(request,allow_nan=False).encode()).decode()
    setup=f'''import pathlib,json,subprocess,sys,base64,stat
r=pathlib.Path({REMOTE!r});p=r/'control'/{identifier!r}
assert r.resolve()==r and r.is_dir()
if not p.exists():
 p.mkdir(parents=True)
 (p/'request.json').write_bytes(base64.b64decode({payload!r}))
 command=[sys.executable,'-B',str(r/'open_processing_runtime.py'),{action!r},'--root',str(r),'--request',str(p/'request.json'),'--output',str(p/'output.json')]
 wrapper="import subprocess,json,pathlib; p=pathlib.Path("+repr(str(p))+"); result=subprocess.run("+repr(command)+"); (p/'exit.json').write_text(json.dumps(dict(returncode=result.returncode)))"
 with (p/'run.log').open('xb') as log:
  assert stat.S_ISREG((p/'run.log').stat().st_mode) and not (p/'run.log').is_symlink()
  child=subprocess.Popen([sys.executable,'-B','-c',wrapper],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (p/'started.json').write_text(json.dumps(dict(pid=child.pid,action={action!r})))
else:
 assert json.loads((p/'request.json').read_text())==json.loads(base64.b64decode({payload!r}))
 assert stat.S_ISREG((p/'run.log').stat().st_mode) and not (p/'run.log').is_symlink()
print(json.dumps(dict(started=True)))
'''
    remote(setup)
    while True:
        value=remote(f'''import pathlib,json,os,stat
p=pathlib.Path({REMOTE!r})/'control'/{identifier!r}
assert stat.S_ISREG((p/'run.log').stat().st_mode) and not (p/'run.log').is_symlink()
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
            assert isinstance(value['output'],dict),value
            return value['output']
        assert value['alive'],'Worker stopped without completion receipt; retain files and investigate'
        time.sleep(10)


def schemas():
    string=dict(type='string');number=dict(type='number')
    measurement=obj(dict(source=string,meaning=string,expected_information=string))
    first=obj(dict(measurement=dict(anyOf=[measurement,dict(type='null')]),reasoning=string))
    forecast=obj({key:number for key in ('mean_y','mean_z','std_y','std_z','rho')})
    second=obj(dict(source=string,diagnostic_claim=string,falsifying_response=string,
                    forecast=forecast,reasoning=string))
    return first,second


def call(arm,round_no,phase,facts):
    folder=STAGE/'calls'/arm/f'{round_no:02d}_{phase}';folder.mkdir(parents=True)
    first,second=schemas();write(folder/'schema.json',first if phase=='measurement' else second)
    common=(STAGE/'AGENT_INTERFACE.txt').read_text(encoding='utf-8-sig')
    prompt=common+'\nCURRENT_ARM '+arm+'\nCURRENT_PHASE '+phase+'\nFACTS\n'+json.dumps(facts,ensure_ascii=False,allow_nan=False)
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8',newline='\n')
    write(folder/'started.json',dict(started=time.time(),prompt_sha256=sha(folder/'prompt.txt'),
                                    schema_sha256=sha(folder/'schema.json')))
    assert not any(SANDBOX.iterdir()),'Agent working directory must remain empty'
    command=[shutil.which('codex'),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check',
             '-s','read-only','-C',str(SANDBOX),'-m','gpt-6.1-sol','-c','model_reasoning_effort="high"',
             '-c','project_doc_max_bytes=0','--json','--output-schema',str(folder/'schema.json'),
             '-o',str(folder/'response.json'),'-']
    start=time.perf_counter()
    with (folder/'events.jsonl').open('xb') as out,(folder/'stderr.log').open('xb') as err:
        process=subprocess.run(command,input=prompt.encode(),stdout=out,stderr=err)
    receipt=dict(exit_code=process.returncode,seconds=time.perf_counter()-start,usage=None,
                 tool_items=[],transport_errors=[])
    completed=False
    for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
        event=json.loads(line);item=event.get('item',{})
        if event.get('type')=='turn.completed':completed=True;receipt['usage']=event.get('usage')
        if event.get('type') in ('error','turn.failed'):receipt['transport_errors'].append(event)
        if item.get('type')=='error':receipt['transport_errors'].append(item)
        elif item.get('type') not in (None,'agent_message','reasoning'):receipt['tool_items'].append(item.get('type'))
    receipt['status']='PASS' if process.returncode==0 and completed and not receipt['tool_items'] and not receipt['transport_errors'] else 'FAILED'
    write(folder/'receipt.json',receipt)
    assert receipt['status']=='PASS',receipt
    return read(folder/'response.json')


def select(history):
    eligible=[item for item in history if item.get('result',{}).get('status')=='completed'
              and item['result']['Y']>=.01 and item['result']['NMAE_gain']>=-.01]
    return max(eligible,key=lambda item:item['result']['Y'])['result']['run'] if eligible else 'PARENT'


PRIVATE_KEYS=frozenset(('worker_receipt','worker_metadata','stderr','stdout','returncode','exit_code',
                        'seconds','reused_cross_arm','actual_new_full_trainings',
                        'actual_new_partial_updates','processing_identity','effective_array_sha256',
                        'assets','pid','started','finished','timed_out','updates','attempted_step'))


def public_history(value):
    """Recursively retain scientific feedback and logical costs without OS/cache facts."""
    if isinstance(value,dict):
        return {key:public_history(item) for key,item in value.items()
                if key not in PRIVATE_KEYS and not key.startswith('physical_') and 'cache' not in key.lower()}
    if isinstance(value,list):return [public_history(item) for item in value]
    return value


def diagnostic_version(decision,result,registry):
    before=result['prequery']
    material=dict(source_sha256=before['source_sha256'],fit_state_sha256=before['fit_state_sha256'],
                  diagnostic_claim=decision['diagnostic_claim'],
                  registry=[{key:q[key] for key in ('id','source_sha256','fit_A')} for q in registry])
    canonical=json.dumps(material,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest(),material


def run():
    lock=read(STAGE/'execution_lock.json');publication=read(STAGE/'source_publication_receipt.json')
    assert lock['execution_enabled']
    assert publication['status']=='exact_commit_and_tree_verified'
    for name,digest in lock['runtime_code_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
    assert sha(STAGE/'PROTOCOL.md')==lock['protocol_sha256']
    assert sha(STAGE/'AGENT_INTERFACE.txt')==lock['agent_interface_sha256']
    reuse=read(STAGE/'parent_reuse_receipt.json');assert reuse['status']=='PASS'
    preflight=read(STAGE/'preflight_receipt.json');assert preflight['status']=='PASS'
    assert preflight['execution_lock_sha256']==reuse['execution_lock_sha256']==sha(STAGE/'execution_lock.json')
    assert reuse['preflight_sha256']==sha(STAGE/'preflight_receipt.json')
    assert reuse['parent_root']==lock['parent_root'] and reuse['parent_assets']==lock['parent_assets']
    parent=read(STAGE/'prepared_parent_facts.json')
    SANDBOX.mkdir(parents=True,exist_ok=True)
    assert not any(SANDBOX.iterdir()),'Agent working directory must be empty'
    write(STAGE/'trial_started.json',dict(started=time.time(),source_commit=publication['source_commit'],
                                        parent_reuse_receipt_sha256=sha(STAGE/'parent_reuse_receipt.json'),
                                        parent_facts_sha256=sha(STAGE/'prepared_parent_facts.json')))
    states={arm:dict(registry=[],history=[],reference_history=[],diagnostic_ledger={}) for arm in ARMS}
    for round_no in range(1,9):
        order=ARMS if round_no%2 else ARMS[::-1]
        for arm in order:
            state=states[arm]
            facts=dict(round=round_no,maximum_rounds=8,parent=public_history(parent),
                       registry=state['registry'],history=public_history(state['history']))
            if arm=='SEPARATE':facts['diagnostic_ledger']=state['diagnostic_ledger']
            proposal=call(arm,round_no,'measurement',facts)
            measurement=rpc('measure',f'{arm}_{round_no:02d}_measure',
                            dict(arm=arm,round=round_no,measurement=proposal['measurement']))
            item=dict(round=round_no,measurement_proposal=proposal,measurement_feedback=measurement)
            if measurement['status']!='PASS':
                item['result']=dict(status='invalid_measurement',error=measurement.get('error'))
            else:
                if proposal['measurement'] is not None:
                    state['registry'].append(dict(id=f"q{len(state['registry'])+1}",
                                                  **proposal['measurement'],fit_A=measurement['fit_A'],
                                                  source_sha256=measurement['source_sha256']))
                facts['registry']=state['registry'];facts['measurement_feedback']=public_history(measurement)
                decision=call(arm,round_no,'processing',facts);item['decision']=decision
                forecast=decision['forecast']
                try:
                    validate_forecast(forecast)
                    evidence(predict([], [1.,1.,1.,0.,0.,0.,0.,0.,.5]),forecast,0.,0.)
                    valid=True
                except (ValueError,AssertionError,TypeError,OverflowError,FloatingPointError) as exc:
                    valid=False;item['result']=dict(status='invalid_forecast',error=str(exc))
                if valid:
                    request=dict(arm=arm,round=round_no,reference_history=state['reference_history'],**decision)
                    result=rpc('query',f'{arm}_{round_no:02d}_query',request);item['result']=result
                    if result['status']=='completed':
                        baseline=result['prequery']['reference_forecast']
                        ev=evidence(baseline,forecast,result['Y'],result['Z'])
                        item['reference_forecast']=baseline
                        if arm=='SEPARATE':
                            version,material=diagnostic_version(decision,result,state['registry'])
                            item['conditional_evidence']=ev;item['diagnostic_version']=version
                            ledger=state['diagnostic_ledger'].setdefault(version,dict(definition=material,queries=[],sum_log_ratio=0.))
                            ledger['queries'].append(round_no);ledger['sum_log_ratio']+=ev['log_ratio']
                        state['reference_history'].append(dict(x=result['prequery']['x'],y=[result['Y'],result['Z']]))
            state['history'].append(item)
            if round_no in (2,4,8):item['recommendation']=select(state['history'])
            snapshot=STAGE/'snapshots';snapshot.mkdir(exist_ok=True)
            write(snapshot/f'{arm}_{round_no:02d}.json',state)
            print(json.dumps(dict(arm=arm,round=round_no,status=item['result']['status'],
                                  Y=item['result'].get('Y'),recommendation=item.get('recommendation'))),flush=True)
    recommendations={arm:select(state['history']) for arm,state in states.items()}
    write(STAGE/'final_recommendations.json',dict(status='FROZEN',recommendations=recommendations,
          D_accessed=False,D_evaluation_pending=True,
          snapshots={f'{arm}_08.json':sha(STAGE/'snapshots'/f'{arm}_08.json') for arm in ARMS}))
    write(STAGE/'trial_completed.json',dict(status='completed',finished=time.time(),
          recommendations=recommendations,D_accessed=False,D_evaluation_pending=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('run',))
    parser.parse_args();run()
