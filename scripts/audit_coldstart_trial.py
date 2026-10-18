"""Post-exploration audit; does not train, call an LLM, or read sealed D.

Full-partition inference replays the frozen execution path and its exact hashes.
An independent per-channel double-precision forward checks three fixed windows
of every stored checkpoint; this is not a second full optimizer implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
import coldstart_runtime as runtime
from coldstart_reference import predict, evidence


REFERENCE_RTOL=1e-5
REFERENCE_ATOL=2e-6
ARMS=('STRONG','SEPARATE')


def close(actual, expected, *, rtol=1e-12, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def reference_forward(state, x):
    """Independent per-channel definition, float64 CPU, no runtime model class."""
    values=torch.as_tensor(np.array(x),dtype=torch.float64)
    channels=values.shape[2]
    padded=F.pad(values.transpose(1,2),(12,12),mode='replicate')
    trend=torch.stack([padded[:,:,i:i+25].mean(2) for i in range(32)],dim=2)
    seasonal=values.transpose(1,2)-trend
    predictions=[]
    for c in range(channels):
        sw=state['seasonal_weight'][c].cpu().double().T
        tw=state['trend_weight'][c].cpu().double().T
        sb=state['seasonal_bias'][c].cpu().double()
        tb=state['trend_bias'][c].cpu().double()
        predictions.append(F.linear(seasonal[:,c],sw,sb)+F.linear(trend[:,c],tw,tb))
    return torch.stack(predictions,dim=2).numpy()


def audit_response(parent_error, child_error, mask, record):
    parent=parent_error.astype(np.float64);child=child_error.astype(np.float64)
    p2=parent*parent;c2=child*child;denominator=float(p2.mean())
    assert denominator>0 and float(np.abs(parent).mean())>0
    y=float((p2.mean()-c2.mean())/denominator)
    dt=float((p2[mask]-c2[mask]).mean()/denominator)
    dc=float((p2[~mask]-c2[~mask]).mean()/denominator)
    mae=float(1-np.abs(child).mean()/np.abs(parent).mean())
    for key,value in dict(Y=y,Z=dt-dc,target_gain=dt,complement_gain=dc,NMAE_gain=mae).items():
        close(record[key],value)
    close(y,float(mask.mean())*dt+(1-float(mask.mean()))*dc)
    assert runtime.digest(parent_error)==record['parent_residual_sha256']
    assert runtime.digest(child_error)==record['child_residual_sha256']


def recommendation(history):
    candidates=[]
    for item in history:
        r=item['result']
        if r['status']=='completed' and r['Y']>=.01 and r['NMAE_gain']>=-.01:
            candidates.append(r)
    return max(candidates,key=lambda r:r['Y'])['run'] if candidates else 'PARENT'


def audit_physical_reuse(states):
    seen=set()
    for number in range(1,9):
        for arm in (ARMS if number%2 else ARMS[::-1]):
            response=states[arm]['history'][number-1]['result']
            if response['status']!='completed':continue
            reused=response['recipe_identity'] in seen
            assert response['reused_cross_arm']==reused
            assert response['actual_new_full_trainings']==(0 if reused else 3)
            seen.add(response['recipe_identity'])
    return len(seen)


def inspect_calls(decisions, arm, round_no, phase):
    path=decisions/'calls'/arm/f'{round_no:02d}_{phase}'
    receipt=runtime.read(path/'receipt.json');started=runtime.read(path/'started.json')
    assert receipt['status']=='PASS' and receipt['exit_code']==0 and not receipt['tool_items']
    assert runtime.sha(path/'prompt.txt')==started['prompt_sha256']
    assert runtime.sha(path/'schema.json')==started['schema_sha256']
    events=[json.loads(line) for line in (path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(event.get('type')=='turn.completed' for event in events)==1
    for event in events:
        kind=event.get('item',{}).get('type')
        assert kind in (None,'reasoning','agent_message','error')
    prompt=(path/'prompt.txt').read_text(encoding='utf-8')
    facts=json.loads(prompt.split('\nFACTS\n',1)[1])
    # Compare facts with the same arm's previous snapshot, not just forbidden words.
    prior=dict(history=[],registry=[],diagnostic_ledger={}) if round_no==1 else runtime.read(decisions/'snapshots'/f'{arm}_{round_no-1:02d}.json')
    current=runtime.read(decisions/'snapshots'/f'{arm}_{round_no:02d}.json')
    previous=json.loads(json.dumps(prior['history']))
    for item in previous:
        for field in ('reused_cross_arm','actual_new_full_trainings','seconds'):
            item['result'].pop(field,None)
    assert facts['history']==previous
    assert ('diagnostic_ledger' in facts)==(arm=='SEPARATE')
    if arm=='SEPARATE':assert facts['diagnostic_ledger']==prior['diagnostic_ledger']
    assert facts['parent']==runtime.read(decisions/'parent_facts.json')
    assert facts['registry']==(prior['registry'] if phase=='measurement' else current['registry'])
    allowed={'round','maximum_rounds','parent','registry','history'}
    if arm=='SEPARATE':allowed.add('diagnostic_ledger')
    if phase=='recipe':
        allowed.add('measurement_feedback')
        assert facts['measurement_feedback']==current['history'][-1]['measurement_feedback']
    assert set(facts)==allowed and facts['round']==round_no and facts['maximum_rounds']==8
    return dict(seconds=receipt['seconds'],usage=receipt['usage'])


def audit(root, decisions):
    start=time.perf_counter()
    audit_lock=runtime.read(decisions/'auditor_lock.json')
    assert runtime.sha(Path(__file__))==audit_lock['auditor_sha256']
    assert runtime.sha(root/'execution_lock.json')==audit_lock['training_execution_lock_sha256']
    assert audit_lock['reference_rtol']==REFERENCE_RTOL and audit_lock['reference_atol']==REFERENCE_ATOL
    runtime.configure()
    lock=runtime.read(root/'execution_lock.json')
    for name,value in lock['runtime_code_sha256'].items():assert runtime.sha(root/name)==value
    assert runtime.sha(root/'PROTOCOL.md')==lock['protocol_sha256']
    assert runtime.read(decisions/'trial_completed.json')['status']=='completed'
    frozen=runtime.read(decisions/'final_recommendations.json');assert frozen['status']=='FROZEN' and not frozen['D_accessed']
    assert len(list((decisions/'snapshots').glob('*.json')))==16
    for name,digest in frozen['snapshots'].items():assert runtime.sha(decisions/'snapshots'/name)==digest
    rows=runtime.load(root);gpu=runtime.tensors(rows)
    model_checks=0;metric_checks=0;initials={};full_count=0;training_seconds=0.
    expected_runs={('PARENT',seed) for seed in runtime.SEEDS}
    for p in (root/'queries').glob('*/*/result.json'):
        item=runtime.read(p)
        if item['status']=='completed':expected_runs.update((item['run'],s) for s in runtime.SEEDS)
    actual_runs={(p.parent.parent.name,int(p.parent.name)) for p in (root/'runs').glob('*/*/result.json')}
    assert actual_runs==expected_runs and len(actual_runs)<=51
    assert len(list((root/'runs').glob('*/*/started.json')))==len(actual_runs)
    # No training is replayed. Every saved snapshot gets all-B inference plus
    # three independent per-channel reference windows (first, middle, last).
    for name,seed in sorted(actual_runs):
        folder=root/'runs'/name/str(seed);record=runtime.read(folder/'result.json')
        assert record['status']=='completed' and record['seed']==seed and record['steps']==3000
        assert [r['step'] for r in record['B_grid']]==[500,1000,1500,2000,2500,3000]
        assert record['seconds']>0;training_seconds+=record['seconds'];full_count+=1
        for filename,asset in record['assets'].items():
            assert runtime.sha(folder/filename)==asset['sha256'] and (folder/filename).stat().st_size==asset['bytes']
        state=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        if seed not in initials:initials[seed]=state
        else:
            assert set(state)==set(initials[seed])
            for key in state:assert torch.equal(state[key],initials[seed][key])
        for grid in record['B_grid']:
            checkpoint=folder/f"step_{grid['step']:04d}.pt"
            model=runtime.checkpoint_model(checkpoint)
            error=runtime.errors(model,gpu['B']);scores=runtime.metrics(error)
            assert scores=={key:grid[key] for key in scores};metric_checks+=1
            data=runtime.windows(rows['B']);indices=[0,len(data)//2,len(data)-1]
            reference=reference_forward(model.state_dict(),data[indices])
            with torch.no_grad():actual=model(torch.as_tensor(np.array(data[indices]),device='cuda')).cpu().numpy()
            close(actual,reference,rtol=REFERENCE_RTOL,atol=REFERENCE_ATOL);model_checks+=1
        assert record['best_step']==min(record['B_grid'],key=lambda x:x['mse'])['step']
        assert runtime.metrics(runtime.errors(runtime.best_model(folder),gpu['C']))==record['C'];metric_checks+=1
    parent_errors={s:runtime.errors(runtime.best_model(root/'runs/PARENT'/str(s)),gpu['C']) for s in runtime.SEEDS}
    calls=0;queries=0;logical_trainings=0;arms={};seen_global=set()
    # Per-arm reference histories are independently rebuilt from stored responses.
    for arm in ARMS:
        state=runtime.read(decisions/'snapshots'/f'{arm}_08.json');history=state['history']
        assert len(history)==8 and [h['round'] for h in history]==list(range(1,9))
        reference_history=[];seen=set();arm_logical=0;ledger={}
        for item in history:
            number=item['round'];inspect_calls(decisions,arm,number,'measurement');calls+=1
            if item['measurement_feedback']['status']=='PASS':inspect_calls(decisions,arm,number,'recipe');calls+=1
            result=item['result']
            assert result['status'] in ('completed','invalid_measurement','invalid_recipe','duplicate_own_recipe','invalid_forecast')
            if result['status']!='completed':continue
            queries+=1;logical_trainings+=3;arm_logical+=result['logical_training_seconds']
            folder=root/'queries'/arm/f'{number:02d}'
            assert runtime.read(folder/'result.json')==result
            request=runtime.read(folder/'request.json');pre=runtime.read(folder/'prequery.json')
            assert request['arm']==arm and request['round']==number
            assert request['reference_history']==reference_history
            q,parameters=runtime.registry_values(rows,request['registry'])
            weight,stats=runtime.recipe_weights(request['weight_expression'],q['A'],runtime.position(len(runtime.windows(rows['A']))))
            identity=hashlib.sha256(request['mode'].encode()+np.ascontiguousarray(weight).tobytes()).hexdigest()
            assert identity==pre['recipe_identity']==result['recipe_identity'] and identity not in seen
            seen.add(identity);seen_global.add(identity)
            assert runtime.digest(weight)==pre['weights_sha256'] and parameters==pre['fit_A'] and stats==pre['weight_stats']
            bmask=runtime.target_mask(request['target'],q['B'],len(runtime.windows(rows['B'])))
            cmask=runtime.target_mask(request['target'],q['C'],len(runtime.windows(rows['C'])))
            assert runtime.digest(cmask)==pre['target_sha256'];close(pre['p_C'],float(cmask.mean()))
            contrast=[]
            for seed in runtime.SEEDS:
                e=runtime.errors(runtime.best_model(root/'runs/PARENT'/str(seed)),gpu['B']).astype(np.float64)**2
                contrast.append(float((e[bmask].mean()-e[~bmask].mean())/e.mean()))
            x=[1.,stats['nonzero']/stats['n'],stats['ess']/stats['n'],float(request['mode']=='W'),float(np.tanh(np.mean(contrast))),float(bmask.mean())]
            close(x,pre['x']);baseline=predict(reference_history,x)
            close(baseline['loc'],pre['reference_forecast']['loc']);close(baseline['scale'],pre['reference_forecast']['scale'])
            assert baseline['df']==pre['reference_forecast']['df']
            assert len(result['responses'])==3 and result['logical_full_trainings']==3
            for seed,response in zip(runtime.SEEDS,result['responses']):
                assert response['seed']==seed
                ce=runtime.errors(runtime.best_model(root/'runs'/result['run']/str(seed)),gpu['C'])
                audit_response(parent_errors[seed],ce,cmask,response)
            for key in ('Y','Z','NMAE_gain'):close(result[key],float(np.mean([r[key] for r in result['responses']])))
            if arm=='SEPARATE':
                expected=evidence(baseline,item['decision']['forecast'],result['Y'],result['Z'])
                close(item['conditional_evidence']['log_ratio'],expected['log_ratio'])
                material=dict(target=request['target'],claim=request['diagnostic_claim'],registry=[dict(id=q['id'],expression=q['expression']) for q in request['registry']])
                version=hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()
                assert version==item['diagnostic_version']
                entry=ledger.setdefault(version,dict(definition=material,queries=[],sum_log_ratio=0.))
                entry['queries'].append(number);entry['sum_log_ratio']+=item['conditional_evidence']['log_ratio']
            reference_history.append(dict(x=pre['x'],y=[result['Y'],result['Z']]))
        assert state['reference_history']==reference_history
        assert state['diagnostic_ledger']==ledger
        for node in (2,4,8):assert history[node-1]['recommendation']==recommendation(history[:node])
        assert frozen['recommendations'][arm]==recommendation(history)
        arms[arm]=dict(completed_queries=len(seen),logical_training_seconds=arm_logical,recommendation=frozen['recommendations'][arm])
    assert full_count==3+3*len(seen_global)
    assert calls<=32 and len(list((decisions/'calls').glob('*/*/started.json')))==calls
    states={arm:runtime.read(decisions/'snapshots'/f'{arm}_08.json') for arm in ARMS}
    assert audit_physical_reuse(states)==len(seen_global)
    return dict(status='PASS',full_trainings=full_count,cumulative_full_trainings=464+full_count,logical_child_trainings=logical_trainings,
                llm_calls=calls,completed_queries=queries,full_partition_metric_checks=metric_checks,
                independent_three_window_checkpoint_checks=model_checks,reference_rtol=REFERENCE_RTOL,reference_atol=REFERENCE_ATOL,
                physical_training_seconds=training_seconds,arms=arms,seconds=time.perf_counter()-start,
                D_values_accessed=False,optimizer_trajectory_replayed=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--decisions',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=audit(args.root.resolve(),args.decisions.resolve())
    except Exception as exc:
        runtime.write(args.output,dict(status='FAILED',error=repr(exc)));raise
    runtime.write(args.output,result);print(json.dumps(result))
