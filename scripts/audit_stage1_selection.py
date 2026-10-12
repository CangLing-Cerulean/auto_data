"""Audit the common S response prefix, record actual nodes and frozen advance gate."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--node',required=True,type=int,choices=[2,4])
    args=parser.parse_args()
    root=Path(args.root)
    started=time.monotonic()
    base=module('baseline',root/'run_intervention_stage1_baseline.py')
    metrics=module('original',root/'run_stage1_candidate.py').metrics
    methods=module('methods',root/'online_ts/agent_research/stage1_selection_methods.py')
    scorer=module('scorer',root/'online_ts/agent_research/stage1_scoring.py')
    protocol=base.read(root/'selection_protocol.json')
    for name,digest in protocol['code_sha256'].items():
        assert base.sha(root/name)==digest,name
    for name,digest in protocol['prerequisite_sha256'].items():
        assert base.sha(root/name)==digest,name
    lock=base.read(root/'selection_prediction_lock.json')
    assert lock['selection_protocol_sha256']==base.sha(root/'selection_protocol.json')
    for name,digest in lock['files_sha256'].items():
        assert base.sha(root/name)==digest,name
    predictions=base.read(root/'selection_all_predictions.json')
    comparisons=[]
    result_hashes={}
    training_seconds=0.
    for name in protocol['initial_run_ids'][:args.node]:
        folder=root/'selection_runs'/name
        result=base.read(folder/'result.json')
        status=base.read(folder/'status.json')
        assert status['status']=='completed' and status['steps_completed']==3000
        assert result['source_sha256']==protocol['code_sha256']['run_stage1_selection.py']
        assert result['protocol_sha256']==base.sha(root/'selection_protocol.json')
        assert result['prediction_lock_sha256']==base.sha(root/'selection_prediction_lock.json')
        task,action=result['task'],result['action']
        assert name==f'{task}_{action}_s{protocol["initial_seed"]}'
        parent_folder=root/'baseline_runs'/f'{task}_B0_s{protocol["initial_seed"]}'
        assert base.sha(parent_folder/'result.json')==protocol['parent_result_sha256'][task]
        parent=base.read(parent_folder/'result.json')
        for output,record in ((folder,result),(parent_folder,parent)):
            for asset,digest in record['assets'].items():
                assert base.sha(output/asset)==digest
        for key in ('initialization_sha256','probability_sha256','draw_count_sha256'):
            assert result[key]==parent[key]
        params=root/'candidate_prepared'/task
        prepared_receipt=base.read(params/'receipt.json')
        assert base.sha(params/'arrays.npz')==prepared_receipt['arrays_sha256']==result['candidate_arrays_sha256']
        assert base.sha(params/'params.json')==protocol['candidate_params_sha256'][task]==result['candidate_params_sha256']
        with np.load(params/'arrays.npz',allow_pickle=False) as saved:
            target=saved['validation_target']
        masks={'global':np.ones(len(target),dtype=bool),'target':target,'complement':~target}
        with np.load(folder/'best_errors.npz',allow_pickle=False) as saved:
            child_error=saved['error']
        with np.load(parent_folder/'best_errors.npz',allow_pickle=False) as saved:
            parent_error=saved['error']
        assert np.isfinite(child_error).all() and child_error.shape==parent_error.shape
        assert np.isfinite(parent_error).all()
        parent_metrics={scope:metrics(parent_error,mask) for scope,mask in masks.items()}
        for scope,mask in masks.items():
            measured=metrics(child_error,mask)
            assert measured['elements']==result['metrics'][scope]['elements']
            if measured['elements']:
                np.testing.assert_allclose([measured[key] for key in ('NMSE','NMAE','signed_bias')],
                    [result['metrics'][scope][key] for key in ('NMSE','NMAE','signed_bias')],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose([parent_metrics['global'][key] for key in ('NMSE','NMAE','signed_bias')],
            [parent['metrics'][key] for key in ('NMSE','NMAE','signed_bias')],rtol=1e-12,atol=1e-14)
        for metric in ('NMSE','NMAE','signed_bias'):
            weighted=sum(result['metrics'][scope][metric]*result['metrics'][scope]['elements']
                         for scope in ('target','complement') if result['metrics'][scope]['elements'])
            np.testing.assert_allclose(weighted/result['metrics']['global']['elements'],result['metrics']['global'][metric],rtol=1e-12,atol=1e-14)
        with np.load(folder/'sampling.npz',allow_pickle=False) as saved:
            assert saved['draw_count'].sum()==768000 and np.all(saved['loss_weight']==1.)
            assert np.all(saved['probability']==1/len(saved['probability']))
            assert base.digest(saved['draw_count'])==result['draw_count_sha256']
            assert base.digest(saved['probability'])==result['probability_sha256']
        assert [item['step'] for item in result['history']]==list(range(500,3001,500))
        assert result['best_step']==min(result['history'],key=lambda item:item['NMSE'])['step']
        assert parent_metrics['global']['NMSE']>0 and parent_metrics['global']['NMAE']>0
        gains={scope:{metric:100*(1-result['metrics'][scope][metric]/parent_metrics[scope][metric])
                       if parent_metrics[scope]['elements'] and parent_metrics[scope][metric]>0 else None
                       for metric in ('NMSE','NMAE')} for scope in masks}
        comparisons.append(dict(id=task+'_'+action,task=task,action=action,run_id=name,gain_percent=gains,
            observed_category=scorer.observed_direction(gains['global']['NMSE']),parent_metrics=parent_metrics,
            child_metrics=result['metrics']))
        training_seconds+=status['process_seconds']
        result_hashes[name]=base.sha(folder/'result.json')
    scores={}
    for method,record in predictions.items():
        labels=[record['predictions'][item['id']] for item in comparisons]
        gains=[item['gain_percent']['global']['NMSE'] for item in comparisons]
        score=scorer.score_predictions(labels,gains)
        score['wrong']=score['decided']-score['correct']
        score['harm_missed']=score['harm_total']-score['harm_warned']
        score['false_improve']=sum(pred=='improve' and observed!='improve' for pred,observed in zip(labels,score['observed']))
        scores[method]=score
    ranking={}
    for task in ('S1','S2')[:args.node//2]:
        one=next(item for item in comparisons if item['id']==task+'_P1')
        two=next(item for item in comparisons if item['id']==task+'_CP1')
        gap=one['gain_percent']['global']['NMSE']-two['gain_percent']['global']['NMSE']
        observed='unidentifiable' if abs(gap)<=.25 else 'P1>CP1' if gap>0 else 'CP1>P1'
        ranking[task]=dict(observed=observed,gap_percent=gap,methods={method:dict(prediction=record['ranking'][task],
            correct=record['ranking'][task]==observed if observed!='unidentifiable' else None) for method,record in predictions.items()})
    chosen=methods.recommend(comparisons)
    gate=None
    if args.node==4:
        potential=[item['id'] for item in comparisons if item['gain_percent']['global']['NMSE']>=1.
                   and item['gain_percent']['global']['NMAE']>=-1.]
        full=scores['new_signal']
        best_other=max(scores[method]['correct'] for method in ('generic_stats','case_match','strong_prompt'))
        signal_pass=(full['correct']==4 and any(label!='unchanged' for label in full['observed']) and full['correct']>best_other)
        gate=dict(advance_to_confirmation=bool(potential or signal_pass),adoption_potential_cells=potential,
                  signal_comparative_gate_passed=signal_pass,best_other_correct=best_other,
                  rule='adoption potential OR full 4/4 correct with a changed response and strictly more correct than every comparator',
                  next='freeze 18-run S confirmation activation; no rule edits' if potential or signal_pass else 'stop; no adoption potential or incremental prospective signal; no new thresholds or candidates')
    report=dict(status='PASS',node=args.node,observed_run_ids=protocol['initial_run_ids'][:args.node],
        result_sha256=result_hashes,comparisons=comparisons,scores=scores,ranking=ranking,
        recommendations={method:dict(chosen) for method in predictions},
        recommendation_controller='shared observed-only adoption controller; identical choices are not a method advantage',
        advance_gate=gate,training_process_seconds=training_seconds,audit_process_seconds=time.monotonic()-started,
        cumulative_successful_trainings=264+args.node,preserved_prior_failed_delivery_count=1,
        auditor_sha256=base.sha(__file__),prediction_lock_sha256=base.sha(root/'selection_prediction_lock.json'),
        selection_protocol_sha256=base.sha(root/'selection_protocol.json'),test_access=False)
    output=root/f'selection_node{args.node}.json'
    assert not output.exists()
    base.write(output,report)
    if args.node==4:
        assert not (root/'selection_initial_feedback.json').exists()
        base.write(root/'selection_initial_feedback.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('comparisons','result_sha256')}))


if __name__=='__main__':
    main()
