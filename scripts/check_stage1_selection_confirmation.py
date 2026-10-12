"""No-training acceptance for the finite S confirmation execution."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    base=module('base',root/'run_intervention_stage1_baseline.py')
    plan=base.read(root/'selection_confirmation_protocol.json')
    for group in ('code_sha256','prerequisite_sha256'):
        for name,digest in plan[group].items():
            assert base.sha(root/name)==digest,name
            if name.endswith('.py'):
                ast.parse((root/name).read_text(encoding='utf-8-sig'))
    expected=[dict(task=task,seed=seed,action=action,id=f'{task}_{action}_s{seed}') for task in ('S1','S2')
              for seed in (2026101402,2026101403,2026101404) for action in ('PARENT','P1','CP1')]
    assert plan['jobs']==expected and plan['run_ids']==[job['id'] for job in expected]
    assert not (root/'selection_confirmation_runs').exists()
    assert not (root/'selection_confirmation_control').exists()
    node=base.read(root/'selection_node4.json')
    assert node['status']=='PASS' and node['advance_gate']['advance_to_confirmation']
    lock=base.read(root/'selection_prediction_lock.json')
    assert set(plan['required_prediction_files'])<=set(lock['files_sha256'])
    for name,digest in lock['files_sha256'].items():
        assert base.sha(root/name)==digest,name
    old=(root/'run_stage1_selection.py').read_text(encoding='utf-8-sig')
    new=(root/'run_stage1_selection_confirmation.py').read_text(encoding='utf-8-sig')
    for start,end in [('        config=dict(', '        def transform(x):'),('        def evaluate():','        result=dict(')]:
        assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
    for job in expected:
        assert not (root/'candidate_runs'/job['id']).exists(),'Legacy duplicate ID exists'
    report=dict(status='PASS',protocol_sha256=base.sha(root/'selection_confirmation_protocol.json'),
        prediction_lock_sha256=base.sha(root/'selection_prediction_lock.json'),new_trainings=0,test_access=False,
        checks=['all_sources_and_prerequisites_hash_locked','Python_AST_parse','exact_18_unoccupied_ids',
                'initial_gate_unchanged','all_initial_predictions_unchanged','training_setup_loop_evaluation_and_save_identical',
                'no_legacy_duplicate_ID'])
    target=root/'selection_confirmation_preflight.json'
    assert not target.exists()
    base.write(target,report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
