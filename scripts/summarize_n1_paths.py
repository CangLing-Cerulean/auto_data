"""Create N1 path tables only from matching independently accepted receipts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(stage):
    plan,paths,audit,preflight=[read(stage/name) for name in ('plan.json','paths.json','paths_audit.json','preflight_audit.json')]
    assert plan['stage']=='n1_path_engineering_r1' and not plan['scientific_children_enabled']
    assert audit['status']=='PASS' and preflight['status']=='PASS'
    assert paths['status']=='completed_pending_independent_audit'
    assert audit['paths_sha256']==sha(stage/'paths.json')
    for value in (paths,audit,preflight):
        assert value['plan_sha256']==sha(stage/'plan.json')
        assert value['new_full_trainings']==0 and value['cumulative_full_trainings']==418
    expected={(task,seed) for task in ('G1','G2','G3') for seed in plan['seeds']}
    source={(r['task'],r['seed']):r for r in paths['runs']}
    checked={(r['task'],r['seed']):r for r in audit['paths']}
    assert len(paths['runs'])==len(audit['paths'])==9 and set(source)==set(checked)==expected
    rows=[]
    for key in sorted(expected):
        run,check=source[key],checked[key]
        assert check['parent_states']==11 and check['tangent_arrays']==33 and check['residual_and_response_sets']==12
        assert check['method_seconds']==run['method_seconds']
        assert run['method_seconds']==run['actual_elapsed_seconds']-run['comparison_seconds']-run['diagnostic_write_seconds']
        rows.append(dict(task=key[0],seed=key[1],parent_states=check['parent_states'],
            tangent_arrays=check['tangent_arrays'],residual_sets=check['residual_and_response_sets'],
            maximum_absolute_error=check['maximum_absolute_error'],**{name:value for name,value in run.items() if name.endswith('_seconds')}))
    costs=[]
    for task in ('G1','G2','G3'):
        lower=paths['input_verification_seconds']/3+paths['preparation_seconds'][task]+sum(r['method_seconds'] for r in paths['runs'] if r['task']==task)
        budget=plan['M1_task_budget_seconds'][task]
        costs.append(dict(task=task,necessary_method_seconds=lower,M1_seconds=budget,lower_bound_pass=lower<=budget))
    assert costs==audit['costs']==paths['costs']
    assert paths['actual_phase_seconds']>=sum(r['actual_elapsed_seconds'] for r in paths['runs'])+sum(paths['preparation_seconds'].values())
    cost_pass=all(c['lower_bound_pass'] for c in costs)
    assert audit['all_cost_lower_bounds_pass']==paths['all_cost_lower_bounds_pass']==cost_pass
    summary=dict(status='independent_path_engineering_passed',parent_replays=9,audit_parent_replays=9,
        saved_parent_states=sum(r['parent_states'] for r in rows),tangent_arrays=sum(r['tangent_arrays'] for r in rows),
        residual_and_response_sets=sum(r['residual_sets'] for r in rows),costs=costs,
        maximum_absolute_error=max(r['maximum_absolute_error'] for r in rows),
        all_cost_lower_bounds_pass=cost_pass,FD_eligible_pending_publication=cost_pass,
        N2_scientific_execution_enabled=False,new_full_trainings=0,cumulative_full_trainings=418,
        path_actual_phase_seconds=paths['actual_phase_seconds'],independent_audit_seconds=audit['seconds'],
        preflight_seconds=preflight['seconds'],D_access=False,test_access=False,
        sources_sha256={name:sha(stage/name) for name in ('plan.json','paths.json','paths_audit.json','preflight_audit.json')})
    for name,values in (('path_seed_table.csv',rows),('path_cost_table.csv',costs)):
        with (stage/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    with (stage/'path_summary.json').open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(summarize(args.stage)))
