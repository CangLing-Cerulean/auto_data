"""Synthetic-only probes of the post-run audit, without any real trial assets."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts'))
import audit_coldstart_trial as a
from coldstart_model import DLinear


def main():
    torch.set_num_threads(4);torch.manual_seed(18423)
    model=DLinear(370);rng=np.random.default_rng(18423)
    x=rng.normal(size=(3,32,370)).astype(np.float32)
    with torch.no_grad():actual=model(torch.from_numpy(x)).numpy()
    reference=a.reference_forward(model.state_dict(),x)
    a.close(actual,reference,rtol=a.REFERENCE_RTOL,atol=a.REFERENCE_ATOL)
    checks=['independent_per_channel_double_reference_370_channels']
    parent=rng.normal(size=(7,8,370)).astype(np.float32);child=(parent+.1).astype(np.float32)
    mask=np.zeros(parent.shape,dtype=bool);mask[2:5,4:8,:17]=True
    pp=parent.astype(np.float64)**2;cc=child.astype(np.float64)**2;den=pp.mean()
    dt=(pp[mask]-cc[mask]).mean()/den;dc=(pp[~mask]-cc[~mask]).mean()/den
    record=dict(Y=float((pp.mean()-cc.mean())/den),Z=float(dt-dc),target_gain=float(dt),complement_gain=float(dc),
                NMAE_gain=float(1-np.abs(child.astype(np.float64)).mean()/np.abs(parent.astype(np.float64)).mean()),
                parent_residual_sha256=a.runtime.digest(parent),child_residual_sha256=a.runtime.digest(child))
    a.audit_response(parent,child,mask,record);checks.append('all_response_metrics_and_hashes')
    bad={**record,'Z':record['Z']+.01}
    try:a.audit_response(parent,child,mask,bad)
    except AssertionError:checks.append('corrupted_response_rejected')
    else:raise AssertionError('corrupt Z accepted')
    records=[dict(result=dict(status='completed',Y=.02,NMAE_gain=-.02,run='harmful')),
             dict(result=dict(status='completed',Y=.011,NMAE_gain=0,run='first')),
             dict(result=dict(status='completed',Y=.011,NMAE_gain=0,run='tie'))]
    assert a.recommendation(records[:1])=='PARENT' and a.recommendation(records)=='first'
    checks.append('recommendation_guardrail_and_earliest_tie')
    states={arm:dict(history=[dict(result=dict(status='invalid_recipe')) for _ in range(8)]) for arm in a.ARMS}
    states['STRONG']['history'][0]['result']=dict(status='completed',recipe_identity='same',reused_cross_arm=False,actual_new_full_trainings=3)
    states['SEPARATE']['history'][0]['result']=dict(status='completed',recipe_identity='same',reused_cross_arm=True,actual_new_full_trainings=0)
    assert a.audit_physical_reuse(states)==1;checks.append('chronological_cross_arm_physical_budget')
    states['SEPARATE']['history'][0]['result']['actual_new_full_trainings']=3
    try:a.audit_physical_reuse(states)
    except AssertionError:checks.append('double_counted_physical_training_rejected')
    else:raise AssertionError('physical overcount accepted')
    with tempfile.TemporaryDirectory(prefix='coldstart_audit_') as temp:
        root=Path(temp);folder=root/'calls/STRONG/01_measurement';folder.mkdir(parents=True)
        (root/'snapshots').mkdir()
        a.runtime.write(root/'parent_facts.json',dict(synthetic=True))
        a.runtime.write(root/'snapshots/STRONG_01.json',dict(registry=[],history=[]))
        facts=dict(round=1,maximum_rounds=8,parent=dict(synthetic=True),registry=[],history=[])
        (folder/'prompt.txt').write_text('synthetic fixture\nFACTS\n'+json.dumps(facts),encoding='utf-8')
        a.runtime.write(folder/'schema.json',{})
        a.runtime.write(folder/'started.json',dict(prompt_sha256=a.runtime.sha(folder/'prompt.txt'),schema_sha256=a.runtime.sha(folder/'schema.json')))
        a.runtime.write(folder/'receipt.json',dict(status='PASS',exit_code=0,tool_items=[],seconds=1.,usage={}))
        (folder/'events.jsonl').write_text(json.dumps(dict(type='turn.completed'))+'\n',encoding='utf-8')
        a.inspect_calls(root,'STRONG',1,'measurement');checks.append('exact_own_history_prompt_audit')
        # A forged receipt cannot hide tool use in retained raw events.
        with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(type='item.completed',item=dict(type='command_execution')))+'\n')
        try:a.inspect_calls(root,'STRONG',1,'measurement')
        except AssertionError:checks.append('tool_event_rejected_despite_clean_receipt')
        else:raise AssertionError('hidden tool event accepted')
    result=dict(status='PASS_synthetic_only',checks=checks,count=len(checks),
                reference_max_absolute_error=float(np.max(np.abs(actual-reference))),
                reference_rtol=a.REFERENCE_RTOL,reference_atol=a.REFERENCE_ATOL,
                auditor_sha256=a.runtime.sha(Path(a.__file__)),real_trial_assets_read=False,
                actual_trial_audit_executed=False,new_full_trainings=0,new_llm_calls=0)
    a.runtime.write(Path(__file__).with_name('result_auditor_unit_checks.json'),result)
    print(json.dumps(result))


if __name__=='__main__':main()
