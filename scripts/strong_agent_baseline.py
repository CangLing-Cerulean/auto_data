"""A frozen strong-prompt sequential comparator with hidden-label-safe inputs."""
import argparse
import concurrent.futures
import copy
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
from diagnosis_pool_features import read,sha,write
from replay_conditional_diagnosis import BASE_FEATURES,feature_vector
from replay_window_signals import fit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
PREVIOUS=BASE/'20261018_measurement_generation_r4'
OUT=BASE/'20261018_strong_agent_r5'
SANDBOX=Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/strong_agent_calls')
TASKS=(('ETT','ETTh1'),('ETT','ETTh2'),('Hydraulic','G1'),('Hydraulic','G2'),('Hydraulic','G3'))

def source_family(family):return 'Hydraulic' if family=='ETT' else 'ETT'

def load_pool(family):
    return read(PREVIOUS/'replay'/f'{source_family(family)}_augmented_records.json')['records']

def original_features(records):
    return tuple(sorted(k for k in records[0]['features'] if not k.startswith(('window_','ordinary_measurement_','mismatch_measurement_'))))

def aliases(records,family,task):
    source=sorted([r for r in records if r['family']==source_family(family)],key=lambda r:r['id'])
    target=sorted([r for r in records if (r['family'],r['task'])==(family,task)],key=lambda r:r['id'])
    return {r['id']:f'S{i+1}' for i,r in enumerate(source)}|{r['id']:f'Q{i+1}' for i,r in enumerate(target)}

def build_input(records,family,task,history):
    names=original_features(records);signals=('mismatch_measurement_0','mismatch_measurement_1');permitted=(*names,*signals)
    idmap=aliases(records,family,task);revealed={h['id'] for h in history}
    source=[r for r in records if r['family']==source_family(family)]
    candidates=[r for r in records if (r['family'],r['task'])==(family,task) and r['id'] not in revealed]
    assert len(revealed)==len(history) and revealed.issubset(idmap)
    # Only source labels and explicit prior revelations enter all fitted models.
    visible=source+[h['record'] for h in history]
    definitions={'reference':(BASE_FEATURES,False),'fixed_quadratic':(names,True),'measurement_combined':((*BASE_FEATURES,*signals),False)}
    models={key:fit(visible,columns,quadratic) for key,(columns,quadratic) in definitions.items()}
    taskmap={t:f'T{i+1}' for i,t in enumerate(sorted({r['task'] for r in source}))}
    def facts(r,with_response):
        result=dict(id=idmap[r['id']],task=taskmap.get(r['task'],'TARGET'),features={name:r['features'][name] for name in permitted})
        if with_response:result.update(response={name:r['response'][name] for name in ('Y','Z','NMAE_gain')},seed_Y=r['seed_Y'],seed_Z=r['seed_Z'])
        return result
    pending=[]
    for record in candidates:
        row=facts(record,False);predictions={}
        for key,(columns,quadratic) in definitions.items():
            p=models[key].predict_gain(feature_vector(record,columns,quadratic));predictions[key]=dict(mean_Y=p.mean,std_Y=p.std)
        row['fitted_suggestions']=predictions;pending.append(row)
    definitions=read(PREVIOUS/'generation'/f'{source_family(family)}_mismatch'/'response.json')['signals']
    packet=dict(query_number=len(history)+1,total_query_budget=4,report_nodes=[2,4],measurement_definitions={f'mismatch_measurement_{i}':definition for i,definition in enumerate(definitions)},source_records=[facts(r,True) for r in source],target_revealed=[dict(**facts(h['record'],True),prior_agent_decision=h['agent_decision']) for h in history],unqueried_candidates=pending)
    return packet,{key:model.snapshot() for key,model in models.items()},idmap

def prompt(packet):
    return '''You are a strong adaptive data-intervention selection agent in a controlled retrospective experiment. Return exactly the JSON schema. Do not use any tools, files, web, previous conversations or hidden data. Every permitted fact is below. Select exactly one unqueried Q candidate for the next costly three-seed query. Source tasks and target are anonymized; do not assume source response patterns transfer unchanged.
Objective: maximize the best OBSERVED eligible Y after query 2 and after query 4. Final eligibility is fixed: mean Y>=0.01 and mean NMAE_gain>=-0.01, otherwise retain parent with gain 0. Do not change this rule or replace actual gains with predictions. Each queried candidate has three historical seed outcomes; seeds are not independent tasks. Complete four distinct queries, including when conservative parent retention seems likely. Balance plausible benefit, avoiding harmful candidates and learning useful implementation/exposure distinctions.
Y=(parent global MSE-child global MSE)/parent global MSE. Z=target-minus-complement improvement using that SAME parent global denominator. Target is latter horizon half, p=0.5, so target improvement is Y+0.5Z. Positive Z alone can coexist with harmful Y. Learn recipe value from actual Y, and revise explanatory beliefs from the joint local/global feedback. Do not conflate a successful recipe with a verified cause. The three fitted suggestions are aids, not authorities; their uncertainty is not guaranteed calibrated across families. They fit only source and your revealed target records. You may override them after assessing source extrapolation, action coverage, S/W differences and prior counterexamples. No unqueried response is provided.
26 original parent/action features and two frozen source-generated diagnostic window measurements are shared across all candidate rows. weighted_implementation distinguishes loss weighting under uniform sampling from nonuniform sampling with ordinary losses. exposure_shift_first is signed exposure change toward early A; removed_fraction distinguishes deletion. The two measurement columns are precomputed parent-only quantities, not candidate outcome feedback. Reason jointly about the feature and response patterns; avoid brittle conclusions from a few correlated source tasks.
Return candidate_id, expected_Y, expected_Z (your point forecast before seeing Y), and a concise decision_reason describing the useful evidence or exploration rationale. Forecasts are logged and are never used as actual labels. All four calls for this task have the same information rules and fixed budget.
FACTS\n'''+json.dumps(packet,ensure_ascii=False,separators=(',',':'))

def selected_recommendation(history):
    eligible=[h['record'] for h in history if h['record']['response']['Y']>=.01 and h['record']['response']['NMAE_gain']>=-.01]
    chosen=min(eligible,key=lambda r:(-r['response']['Y'],r['id'])) if eligible else None
    costs=[h['record']['original_full_training_seconds'] for h in history]
    return dict(recommendation=chosen['id'] if chosen else 'PARENT',gain=chosen['response']['Y'] if chosen else 0.,NMAE_gain=chosen['response']['NMAE_gain'] if chosen else 0.,negative_seed_count=sum(y<0 for y in chosen['seed_Y']) if chosen else 0,query_seconds=None if any(c is None for c in costs) else sum(costs),logical_candidate_full_trainings=3*len(history),shared_parent_full_trainings=3)

def prepare():
    assert read(PREVIOUS/'sync_receipt.json')['status']=='result_commit_and_release_verified'
    (OUT/'calls').mkdir();SANDBOX.mkdir();pool_hashes={}
    for source in ('ETT','Hydraulic'):pool_hashes[source]=sha(PREVIOUS/'replay'/f'{source}_augmented_records.json')
    definitions_hashes={source:sha(PREVIOUS/'generation'/f'{source}_mismatch'/'response.json') for source in ('ETT','Hydraulic')}
    dependency_names=('strong_agent_baseline.py','replay_window_signals.py','conditional_diagnosis.py','diagnosis_pool_features.py','replay_conditional_diagnosis.py','window_measurement_dsl.py','parent_measurement_tables.py')
    write(OUT/'execution_lock.json',dict(code_sha256={name:sha(ROOT/'scripts'/name) for name in dependency_names},protocol_sha256=sha(OUT/'PROTOCOL.md'),pool_sha256=pool_hashes,definitions_sha256=definitions_hashes,model='gpt-6.1-sol',reasoning='high',tasks=TASKS,max_calls=20,queries_per_task=4,new_full_trainings=0,cumulative_full_trainings=464))
    checks=[]
    for family,task in TASKS:
        records=load_pool(family);held=sorted([r for r in records if (r['family'],r['task'])==(family,task)],key=lambda r:r['id'])
        for prefix in (0,2):
            history=[dict(id=r['id'],record=r,agent_decision=dict(candidate_id=f'Q{i+1}',expected_Y=0.,expected_Z=0.,decision_reason='Fixed interface test prefix')) for i,r in enumerate(held[:prefix])]
            packet,models,idmap=build_input(records,family,task,history)
            perturbed=copy.deepcopy(records)
            for r in perturbed:
                if r['family']==family:
                    r['response']={k:987654321. for k in r['response']};r['seed_Y']=[999.]*3;r['seed_Z']=[-999.]*3;r['original_full_training_seconds']=-99.
            alternative,alternative_models,_=build_input(perturbed,family,task,history)
            assert packet==alternative and models==alternative_models
            checks.append(dict(family=family,task=task,prefix=prefix,hidden_target_label_and_cost_perturbation_invariant=True,remaining_candidates=len(packet['unqueried_candidates'])))
    write(OUT/'input_check.json',dict(status='PASS',checks=checks,actual_hidden_values_perturbed=True,LLM_calls=0))
    print(dict(status='PREPARED',tasks=5,max_calls=20))

def call_task(family,task,step):
    records=load_pool(family);history=[]
    if step>1:history=read(OUT/'calls'/f'{family}_{task}_{step-1}'/'revealed.json')['history']
    assert len(history)==step-1
    folder=OUT/'calls'/f'{family}_{task}_{step}';folder.mkdir()
    begin=time.perf_counter();packet,models,idmap=build_input(records,family,task,history)
    write(folder/'input.json',packet);write(folder/'models.json',models)
    (folder/'prompt.txt').write_text(prompt(packet),encoding='utf-8',newline='\n')
    schema=dict(type='object',additionalProperties=False,properties=dict(candidate_id=dict(type='string',enum=[r['id'] for r in packet['unqueried_candidates']]),expected_Y=dict(type='number'),expected_Z=dict(type='number'),decision_reason=dict(type='string',maxLength=2000)),required=['candidate_id','expected_Y','expected_Z','decision_reason'])
    write(folder/'schema.json',schema)
    write(folder/'input_lock.json',dict(files={name:sha(folder/name) for name in ('input.json','models.json','prompt.txt','schema.json')},step=step,preparation_seconds=time.perf_counter()-begin,execution_lock_sha256=sha(OUT/'execution_lock.json')))
    command=[shutil.which('codex'),'exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-s','read-only','-C',str(SANDBOX),'-m','gpt-6.1-sol','-c','model_reasoning_effort="high"','--json','--output-schema',str(folder/'schema.json'),'-o',str(folder/'response.json'),'-']
    write(folder/'started.json',dict(started=time.time(),model='gpt-6.1-sol',reasoning='high'));start=time.perf_counter()
    receipt=dict(exit_code=None,seconds=None,usage=None,tool_items=[],transport_errors=[],model_requested='gpt-6.1-sol',reasoning_requested='high');completed=False
    try:
        with (folder/'events.jsonl').open('xb') as out,(folder/'stderr.log').open('xb') as err:result=subprocess.run(command,input=(folder/'prompt.txt').read_bytes(),stdout=out,stderr=err)
        receipt.update(exit_code=result.returncode,seconds=time.perf_counter()-start)
        for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines():
            event=json.loads(line);item=event.get('item',{})
            if event.get('type')=='turn.completed':completed=True;receipt['usage']=event.get('usage')
            if item.get('type')=='error':receipt['transport_errors'].append(item)
            elif item.get('type') not in (None,'agent_message','reasoning'):receipt['tool_items'].append(item.get('type'))
        assert result.returncode==0 and completed and not receipt['tool_items']
        decision=read(folder/'response.json');assert set(decision)==set(schema['required']) and decision['candidate_id'] in schema['properties']['candidate_id']['enum']
        assert all(type(decision[key]) in (int,float) for key in ('expected_Y','expected_Z'))
        assert isinstance(decision['decision_reason'],str) and len(decision['decision_reason'])<=2000
        assert np.isfinite([decision['expected_Y'],decision['expected_Z']]).all()
        receipt.update(status='PASS',response_sha256=sha(folder/'response.json'))
    except Exception as exc:receipt.update(status='FAILED',error=repr(exc),seconds=time.perf_counter()-start)
    write(folder/'receipt.json',receipt)
    if receipt['status']!='PASS':return False
    # The first target-response lookup for this step follows the sealed valid decision.
    actual_id=next(key for key,value in idmap.items() if value==decision['candidate_id'])
    revealed=next(r for r in records if r['id']==actual_id)
    history.append(dict(id=actual_id,record=revealed,agent_decision=decision))
    write(folder/'revealed.json',dict(history=history,recommendation=selected_recommendation(history),response_receipt_sha256=sha(folder/'receipt.json'),revealed_at=time.time()))
    print(json.dumps(dict(task=task,step=step,status='PASS',candidate=decision['candidate_id'],seconds=receipt['seconds'])),flush=True)
    return True

def run():
    lock=read(OUT/'execution_lock.json')
    for name,value in lock['code_sha256'].items():assert sha(ROOT/'scripts'/name)==value
    assert sha(OUT/'PROTOCOL.md')==lock['protocol_sha256']
    for source,value in lock['pool_sha256'].items():assert sha(PREVIOUS/'replay'/f'{source}_augmented_records.json')==value
    for source,value in lock['definitions_sha256'].items():assert sha(PREVIOUS/'generation'/f'{source}_mismatch'/'response.json')==value
    assert read(OUT/'source_publication_receipt.json')['status']=='exact_commit_and_tree_verified'
    write(OUT/'started.json',dict(started=time.time(),max_calls=20,execution_lock_sha256=sha(OUT/'execution_lock.json')))
    for step in range(1,5):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                results=list(pool.map(lambda task:call_task(*task,step),TASKS))
        except Exception as exc:
            write(OUT/'stopped.json',dict(failed_round=step,reason='Unexpected execution exception; remaining rounds not started',error=repr(exc)))
            raise
        if not all(results):
            write(OUT/'stopped.json',dict(failed_round=step,reason='One decision failed; remaining rounds not started'));raise RuntimeError('Decision validation failed')
    write(OUT/'completed.json',dict(status='COMPUTED_AUDIT_PENDING',calls=20,tasks=5,new_full_trainings=0,cumulative_full_trainings=464))

def summarize():
    assert read(OUT/'completed.json')['calls']==20
    rows=[];random_rows=[];calls=[]
    for family,task in TASKS:
        records=load_pool(family);held=[r for r in records if (r['family'],r['task'])==(family,task)]
        for step in range(1,5):
            folder=OUT/'calls'/f'{family}_{task}_{step}';calls.append(dict(family=family,task=task,step=step,**read(folder/'receipt.json')))
            if step in (2,4):
                rows.append(dict(family=family,task=task,queries=step,**read(folder/'revealed.json')['recommendation']))
                values=[selected_recommendation([dict(record=r) for r in subset])['gain'] for subset in itertools.combinations(held,step)]
                random_rows.append(dict(family=family,task=task,queries=step,subsets=len(values),expected_gain=float(np.mean(values))))
    write(OUT/'results.json',dict(status='COMPUTED_AUDIT_PENDING',decisions=rows,uniform_random=random_rows,calls=calls,summaries=[dict(queries=q,agent_mean_gain=float(np.mean([r['gain'] for r in rows if r['queries']==q])),uniform_random_expected_mean_gain=float(np.mean([r['expected_gain'] for r in random_rows if r['queries']==q]))) for q in (2,4)],new_full_trainings=0,cumulative_full_trainings=464))
    print(read(OUT/'results.json')['summaries'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','summarize'));globals()[parser.parse_args().action]()
