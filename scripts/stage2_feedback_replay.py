"""Create-only sealed replay packets; all model responses are existing audited data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

ARMS=('S0U0F','S1U0F','S0U1F','S1U1F','S1U1G','S1U1M')
POOL=('S1_P1','S1_CP1','S2_P1','S2_CP1')
CATEGORIES=('improve','unchanged','harm','uncertain')
SCOPES=('global','target','complement')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def verify(root):
    plan=read(root/'protocol.json')
    assert plan['script_sha256']==sha(__file__)
    for name,digest in plan['local_files_sha256'].items():
        assert sha(root/name)==digest,name
    source=Path(plan['source_root'])
    for name,digest in plan['source_files_sha256'].items():
        assert sha(source/name)==digest,name
    assert tuple(plan['arms'])==ARMS and tuple(plan['pool'])==POOL
    return plan,source


def directions(record):
    assert set(record['directions'])==set(SCOPES)
    assert all(value in CATEGORIES for value in record['directions'].values())
    assert isinstance(record['falsifiers'],list) and record['falsifiers']
    assert all(isinstance(value,str) and value.strip() for value in record['falsifiers'])


def recommendations(record):
    assert set(record)=={'S1','S2'}
    assert all(value in ('B0','P1','CP1') for value in record.values())


def visible_feedback(record,mode):
    result=dict(id=record['id'],run_status='completed',query_cost=1)
    if mode=='M':
        result['effect']='unknown_masked'
        return result
    scopes=SCOPES if mode=='F' else ('global',)
    result['response']={scope:dict(parent=record['parent_metrics'][scope],child=record['child_metrics'][scope],
        gain_percent=record['gain_percent'][scope]) for scope in scopes}
    result['control_status']='paired initialization/sampling verified; no independent confirmation exposed'
    return result


def category(value):
    return 'improve' if value>.25 else 'harm' if value<-.25 else 'unchanged'


def dual_summary(visible,prediction):
    if 'response' not in visible:
        return dict(recipe='unknown',hypothesis='unknown',basis='current effect masked')
    scopes=visible['response']
    gain=scopes['global']['gain_percent']
    recipe='confirmation_candidate' if gain['NMSE']>=1 and gain['NMAE']>=-1 else 'retain_parent'
    matched,missing,contradicted=0,0,[]
    for scope,label in prediction['directions'].items():
        if label=='uncertain':
            continue
        if scope not in scopes or scopes[scope]['gain_percent']['NMSE'] is None:
            missing+=1
        elif category(scopes[scope]['gain_percent']['NMSE'])!=label:
            contradicted.append(scope)
        else:
            matched+=1
    hypothesis='contradicted' if contradicted else 'response_supported_pending_controls' if matched and not missing else 'insufficient_visible_evidence'
    return dict(recipe=recipe,hypothesis=hypothesis,matched_visible_scopes=matched,unscorable_declared_scopes=missing,
                contradicted_scopes=contradicted,independent_confirmation_exposed=False,unique_cause_proven=False)


def schema(step,dual=False):
    result=dict(packet_id='copy input packet_id',reason='concise evidence and uncertainty',files_read='actual allowed file paths only')
    if step=='initial':
        result.update(directions={scope:'one four-class label' for scope in SCOPES},falsifiers=['at least one'],
                      initial_recommendations={'S1':'B0/P1/CP1','S2':'B0/P1/CP1'})
    elif step==1:
        result.update(query='one remaining ID',directions={scope:'one four-class label' for scope in SCOPES},falsifiers=['at least one'])
    elif step==2:
        result.update(recommendations={'S1':'B0/P1/CP1','S2':'B0/P1/CP1'},remaining_query_order=['both remaining IDs'],
                      remaining_predictions={'each remaining ID':dict(directions={scope:'one four-class label' for scope in SCOPES},falsifiers=['at least one'])})
    else:
        result.update(recommendations={'S1':'B0/P1/CP1','S2':'B0/P1/CP1'})
    if dual:
        result.update(recipe_update='separate concise decision state',hypothesis_update='separate concise hypothesis state')
    return result


def allowed_files(root,arm=None,step=None,factor=None):
    local=read(root/'protocol.json')['actor_local_root'].replace('\\','/').rstrip('/')
    names=['ACTOR_RULES.md','common_facts.json']
    names += [f'initial_{factor}_input.json'] if factor else [f'arms/{arm}/step{i}_input.json' for i in range(1,step+1)]
    return [local+'/'+name for name in names]


def check_read_claims(packet,record):
    normalize=lambda value:value.replace('\\','/').lower()
    permitted={normalize(value) for value in packet['allowed_files']}
    assert isinstance(record['files_read'],list) and record['files_read']
    assert all(isinstance(value,str) and normalize(value) in permitted for value in record['files_read'])
    assert normalize(packet['allowed_files'][-1]) in {normalize(value) for value in record['files_read']}


def signal_information(source):
    return dict(formula='d=mean_(windows,channels)(mean_time(changed-parent)^2); s=d/(d+initial_parent_NMSE), 0/0=0',
                rule=read(source/'signal_rule.json'),values=read(source/'selection_new_signal_values.json')['values'],
                use='C-only frozen rule; do not modify thresholds or assume adoption magnitude from a category')


def make_packet(root,source,arm,step,seen,predictions):
    oracle=read(root/'oracle.json')
    visible=[visible_feedback(oracle['initial'][key],arm[-1]) for key in seen]
    packet=dict(packet_id=f'{arm}_step{step}',arm=arm,query_count=len(seen),observed_ids=seen,
        allowed_files=allowed_files(root,arm=arm,step=step),
        remaining_ids=[key for key in POOL if key not in seen],common_facts_sha256=sha(root/'common_facts.json'),
        registered_predictions=predictions,feedback=visible,output_schema=schema(step,arm[3]=='1'),
        task='Choose next query' if step==1 else 'Record actual node2 recommendation and remaining query order' if step==2 else 'Record actual node4 recommendation',
        interpretation='sealed historical replay; recommendations nominate candidates for confirmation; hidden effects remain unknown')
    if arm[1]=='1':
        packet['learned_signal']=signal_information(source)
    if arm[3]=='1':
        packet['separate_updates']={item['id']:dual_summary(item,predictions[item['id']]) for item in visible}
    return packet


def check_output(root,arm,step):
    folder=root/'arms'/arm
    packet=read(folder/f'step{step}_input.json')
    record=read(folder/f'step{step}_output.json')
    assert record['packet_id']==packet['packet_id']
    assert isinstance(record['reason'],str) and record['reason'].strip()
    check_read_claims(packet,record)
    if arm[3]=='1':
        assert all(isinstance(record[key],str) and record[key].strip() for key in ('recipe_update','hypothesis_update'))
    if step==1:
        assert record['query'] in packet['remaining_ids']
        directions(record)
    elif step==2:
        recommendations(record['recommendations'])
        assert len(record['remaining_query_order'])==2 and set(record['remaining_query_order'])==set(packet['remaining_ids'])
        assert set(record['remaining_predictions'])==set(packet['remaining_ids'])
        for prediction in record['remaining_predictions'].values():
            directions(prediction)
    else:
        recommendations(record['recommendations'])
    return packet,record


def prepare(root,source):
    common=read(source/'selection_common_facts.json')
    common['replay_scope']='Only C responses and initial S parent/T facts; S candidate outcomes are hidden'
    write_new(root/'common_facts.json',common)
    initial=read(source/'selection_node4.json')
    confirmation=read(source/'selection_confirmation_feedback.json')
    write_new(root/'oracle.json',dict(initial={item['id']:item for item in initial['comparisons']},confirmation=confirmation['cells']))
    for factor in ('S0','S1'):
        packet=dict(packet_id=f'{factor}_initial',prefix_query='S1_P1',response_visible=False,
            allowed_files=allowed_files(root,factor=factor),
            common_facts_sha256=sha(root/'common_facts.json'),output_schema=schema('initial'),
            task='Before any S response is revealed, predict the common prefix S1_P1 and give initial nominations')
        if factor=='S1':
            packet['learned_signal']=signal_information(source)
        write_new(root/f'initial_{factor}_input.json',packet)
    # Boundary tests use synthetic values only and do not fit a policy.
    fake=dict(id='x',parent_metrics={scope:{} for scope in SCOPES},child_metrics={scope:{} for scope in SCOPES},
              gain_percent={scope:{'NMSE':2.,'NMAE':1.} for scope in SCOPES})
    pred=dict(directions={'global':'improve','target':'improve','complement':'uncertain'})
    assert 'response' not in visible_feedback(fake,'M')
    assert set(visible_feedback(fake,'G')['response'])=={'global'}
    assert dual_summary(visible_feedback(fake,'M'),pred)['recipe']=='unknown'
    assert dual_summary(visible_feedback(fake,'G'),pred)['hypothesis']=='insufficient_visible_evidence'
    assert dual_summary(visible_feedback(fake,'F'),pred)['recipe']=='confirmation_candidate'
    write_new(root/'preflight.json',dict(status='PASS',new_training=0,boundary_checks=5,
        protocol_sha256=sha(root/'protocol.json'),common_facts_sha256=sha(root/'common_facts.json'),oracle_sha256=sha(root/'oracle.json')))


def start_arms(root,source):
    initials={}
    for factor in ('S0','S1'):
        record=read(root/f'initial_{factor}_output.json')
        assert record['packet_id']==f'{factor}_initial'
        check_read_claims(read(root/f'initial_{factor}_input.json'),record)
        directions(record)
        recommendations(record['initial_recommendations'])
        initials[factor]=record
    for arm in ARMS:
        original=initials[arm[:2]]
        packet=make_packet(root,source,arm,1,['S1_P1'],{'S1_P1':original})
        packet['common_prefix_initial_judgment']=original
        write_new(root/'arms'/arm/'step1_input.json',packet)


def advance(root,source,arm,step):
    packet,record=check_output(root,arm,step)
    if step==3:
        write_new(root/'arms'/arm/'completion.json',dict(status='complete',query_count=4,
            output_sha256={str(i):sha(root/'arms'/arm/f'step{i}_output.json') for i in (1,2,3)}))
        return
    seen=list(packet['observed_ids'])
    predictions=dict(packet['registered_predictions'])
    if step==1:
        seen.append(record['query'])
        predictions[record['query']]=dict(directions=record['directions'],falsifiers=record['falsifiers'])
    else:
        seen.extend(record['remaining_query_order'])
        predictions.update(record['remaining_predictions'])
    following=make_packet(root,source,arm,step+1,seen,predictions)
    following['previous_output_sha256']=sha(root/'arms'/arm/f'step{step}_output.json')
    write_new(root/'arms'/arm/f'step{step+1}_input.json',following)


def quality(choices,oracle,seen):
    details={}
    for task,action in choices.items():
        if action=='B0':
            details[task]=dict(action=action,NMSE_gain=0.,NMAE_gain=0.,eligible=True,unobserved=False)
        else:
            cell=oracle['confirmation'][task+'_'+action]
            details[task]=dict(action=action,NMSE_gain=cell['mean_global_NMSE_gain_percent'],
                NMAE_gain=sum(item['gain_percent']['global']['NMAE'] for item in cell['repeats'])/3,
                eligible=cell['confirmed_adoption'],unobserved=task+'_'+action not in seen)
    qualified=sum(item['NMSE_gain'] if item['eligible'] else 0 for item in details.values())/2
    reference=oracle['confirmation']['S2_P1']['mean_global_NMSE_gain_percent']/2
    return dict(choices=details,raw_mean_NMSE_gain=sum(item['NMSE_gain'] for item in details.values())/2,
        qualified_mean_NMSE_gain=qualified,gate_violations=sum(not item['eligible'] for item in details.values()),
        harmful_choices=sum(item['NMSE_gain']<-.25 for item in details.values()),
        unobserved_recommendations=sum(item['unobserved'] for item in details.values()),
        qualified_reference_gap=reference-qualified)


def score(root):
    oracle=read(root/'oracle.json')
    outcomes={}
    for arm in ARMS:
        done=read(root/'arms'/arm/'completion.json')
        for i in (1,2,3):
            assert done['output_sha256'][str(i)]==sha(root/'arms'/arm/f'step{i}_output.json')
        p2,r2=check_output(root,arm,2)
        p4,r4=check_output(root,arm,3)
        outcomes[arm]=dict(node2=quality(r2['recommendations'],oracle,p2['observed_ids']),
                          node4=quality(r4['recommendations'],oracle,p4['observed_ids']),query_order=p4['observed_ids'])
    full=outcomes['S1U1F']['node2']
    comparisons={}
    for comparator in ('S0U1F','S1U0F','S1U1G','S1U1M'):
        other=outcomes[comparator]['node2']
        difference=full['qualified_mean_NMSE_gain']-other['qualified_mean_NMSE_gain']
        comparisons[comparator]=dict(qualified_gain_difference=difference,
            no_more_violations=full['gate_violations']<=other['gate_violations'],
            no_more_harm=full['harmful_choices']<=other['harmful_choices'],
            passed=difference>0 and full['gate_violations']<=other['gate_violations'] and full['harmful_choices']<=other['harmful_choices'])
    interaction=outcomes['S1U1F']['node2']['qualified_mean_NMSE_gain']-outcomes['S1U0F']['node2']['qualified_mean_NMSE_gain']-outcomes['S0U1F']['node2']['qualified_mean_NMSE_gain']+outcomes['S0U0F']['node2']['qualified_mean_NMSE_gain']
    write_new(root/'result.json',dict(status='PASS',outcomes=outcomes,primary_node=2,incremental_comparisons=comparisons,
        factorial_interaction=interaction,advance_to_new_prospective_design=all(item['passed'] for item in comparisons.values()),
        new_trainings=0,cumulative_successful_trainings=286,cached_queries=24,LLM_calls=20,external_API_calls=0,
        internal_tokens=None,internal_cost=None,node8='unreached',test_access=False,Hydraulic='reserved',
        interpretation='single-trajectory sealed historical replay; no population effect or new prospective-generalization claim'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--mode',required=True,choices=['prepare','start-arms','advance','score'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--step',type=int,choices=[1,2,3])
    args=parser.parse_args()
    root=Path(args.root)
    started=time.monotonic()
    plan,source=verify(root)
    if args.mode=='prepare':
        prepare(root,source)
    else:
        preflight=read(root/'preflight.json')
        assert preflight['protocol_sha256']==sha(root/'protocol.json')
        assert preflight['common_facts_sha256']==sha(root/'common_facts.json')
        assert preflight['oracle_sha256']==sha(root/'oracle.json')
        if args.mode=='start-arms':
            start_arms(root,source)
        elif args.mode=='advance':
            assert args.arm and args.step
            advance(root,source,args.arm,args.step)
        else:
            score(root)
    print(json.dumps(dict(status='PASS',mode=args.mode,arm=args.arm,step=args.step,new_trainings=0,seconds=time.monotonic()-started)))


if __name__=='__main__':
    main()
