"""Audit all 18 additional S runs against unchanged prospective predictions."""
import argparse
import importlib.util
from pathlib import Path
import json
import time
import numpy as np


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    started = time.monotonic()
    base = module('baseline', root/'run_intervention_stage1_baseline.py')
    runner = module('candidate', root/'run_stage1_candidate.py')
    scorer = module('scorer', root/'online_ts/agent_research/stage1_scoring.py')
    protocol = base.read(root/'candidate_protocol.json')
    plan = base.read(root/'selection_confirmation_protocol.json')
    for group in ('code_sha256','prerequisite_sha256'):
        for relative,digest in plan[group].items():
            assert base.sha(root/relative)==digest,relative
    lock=base.read(root/'selection_prediction_lock.json')
    for relative,digest in lock['files_sha256'].items():
        assert base.sha(root/relative)==digest,relative
    predictions=base.read(root/'selection_all_predictions.json')
    control = base.read(root/'selection_confirmation_control/status.json')
    assert control['status'] == 'completed'
    assert control['completed'] == plan['run_ids']
    assert control['protocol_sha256']==base.sha(root/'selection_confirmation_protocol.json')
    cells = {f'{task}_{action}':dict(task=task, action=action, repeats=[]) for task in ('S1','S2') for action in ('P1','CP1')}
    training_seconds = 0.
    successful,result_hashes = [],{}
    for task in ('S1', 'S2'):
        prepared_receipt=base.read(root/'candidate_prepared'/task/'receipt.json')
        assert base.sha(root/'candidate_prepared'/task/'arrays.npz')==prepared_receipt['arrays_sha256']
        assert base.sha(root/'candidate_prepared'/task/'params.json')==plan['candidate_params_sha256'][task]
        with np.load(root/'candidate_prepared'/task/'arrays.npz', allow_pickle=False) as saved:
            target = saved['validation_target']
        masks = {'global':np.ones(len(target), dtype=bool), 'target':target, 'complement':~target}
        for seed in protocol['confirmation_seeds']:
            audited = {}
            for action in ('PARENT', 'P1', 'CP1'):
                name = f'{task}_{action}_s{seed}'
                folder = root/'selection_confirmation_runs'/name
                result = base.read(folder/'result.json')
                status = base.read(folder/'status.json')
                assert status['steps_completed'] == 3000
                assert status['status'] == 'completed'
                successful.append(name)
                result_hashes[name]=base.sha(folder/'result.json')
                assert result['task'] == task and result['action'] == action and result['seed'] == seed
                assert result['source_sha256'] == plan['code_sha256']['run_stage1_selection_confirmation.py']
                assert result['protocol_sha256'] == base.sha(root/'selection_confirmation_protocol.json')
                assert result['prediction_lock_sha256']==base.sha(root/'selection_prediction_lock.json')
                assert result['candidate_params_sha256']==plan['candidate_params_sha256'][task]
                assert result['candidate_arrays_sha256']==prepared_receipt['arrays_sha256']
                assert result['prepared_sha256'] == protocol['task_array_sha256'][task]
                for asset, digest in result['assets'].items():
                    assert base.sha(folder/asset) == digest
                with np.load(folder/'best_errors.npz', allow_pickle=False) as saved:
                    error = saved['error']
                assert np.isfinite(error).all()
                for scope, mask in masks.items():
                    measured = runner.metrics(error, mask)
                    recorded = result['metrics'][scope]
                    assert measured['elements'] == recorded['elements']
                    if measured['elements']:
                        np.testing.assert_allclose([measured[key] for key in ('NMSE','NMAE','signed_bias')],
                                                   [recorded[key] for key in ('NMSE','NMAE','signed_bias')], rtol=1e-12, atol=1e-14)
                for metric in ('NMSE', 'NMAE', 'signed_bias'):
                    weighted = sum(result['metrics'][scope][metric]*result['metrics'][scope]['elements'] for scope in ('target','complement') if result['metrics'][scope]['elements'])
                    np.testing.assert_allclose(weighted/result['metrics']['global']['elements'], result['metrics']['global'][metric], rtol=1e-12, atol=1e-14)
                with np.load(folder/'sampling.npz', allow_pickle=False) as saved:
                    assert saved['draw_count'].sum() == 768000 and np.all(saved['loss_weight'] == 1.)
                    assert np.all(saved['probability'] == 1/len(saved['probability']))
                    assert base.digest(saved['draw_count']) == result['draw_count_sha256']
                    assert base.digest(saved['probability']) == result['probability_sha256']
                assert result['best_step'] == min(result['history'], key=lambda item:item['NMSE'])['step']
                assert [item['step'] for item in result['history']] == list(range(500,3001,500))
                training_seconds += status['process_seconds']
                audited[action] = result
            for action in ('P1', 'CP1'):
                parent, child = audited['PARENT'], audited[action]
                for key in ('initialization_sha256', 'probability_sha256', 'draw_count_sha256'):
                    assert parent[key] == child[key]
                gains = {scope:{metric:100*(1-child['metrics'][scope][metric]/parent['metrics'][scope][metric]) if parent['metrics'][scope]['elements'] and parent['metrics'][scope][metric] > 0 else None for metric in ('NMSE','NMAE')} for scope in masks}
                cells[task+'_'+action]['repeats'].append(dict(seed=seed, gain_percent=gains,
                    observed_category={scope:scorer.observed_direction(gains[scope]['NMSE']) if gains[scope]['NMSE'] is not None else 'unscorable' for scope in masks}))
    assert successful==plan['run_ids'] and len(successful)==18
    for cell in cells.values():
        gains = [record['gain_percent']['global']['NMSE'] for record in cell['repeats']]
        mae = [record['gain_percent']['global']['NMAE'] for record in cell['repeats']]
        cell['repeatability'] = scorer.repeatability(gains) if all(gain is not None for gain in gains) else dict(stable_category='unscorable')
        if cell['repeatability']['stable_category']=='uncertain':
            cell['repeatability']['stable_category']='unstable_response'
        cell['mean_global_NMSE_gain_percent'] = float(np.mean(gains)) if all(gain is not None for gain in gains) else None
        cell['confirmed_adoption'] = (all(gain is not None and gain > 0 for gain in gains)
                                      and cell['mean_global_NMSE_gain_percent'] >= 1.
                                      and all(value is not None and value >= -1. for value in mae))
    scores={}
    for method,record in predictions.items():
        stable={key:cell['repeatability']['stable_category'] for key,cell in cells.items()}
        correct=sum(record['predictions'][key]==value for key,value in stable.items() if value in ('improve','unchanged','harm'))
        wrong=sum(record['predictions'][key] not in (value,'uncertain') for key,value in stable.items() if value in ('improve','unchanged','harm'))
        abstained=sum(value=='uncertain' for value in record['predictions'].values())
        unresolved=sum(value not in ('improve','unchanged','harm') for value in stable.values())
        scores[method]=dict(total=4,correct=int(correct),wrong=int(wrong),abstained=abstained,coverage=(4-abstained)/4,
            unstable_cells=unresolved,stable_observed=stable,accuracy_on_decided_stable=correct/(correct+wrong) if correct+wrong else None,
            repeated_response_diagnostic=scorer.score_predictions([record['predictions'][key] for key,cell in cells.items() for repeat in cell['repeats']],
                [repeat['gain_percent']['global']['NMSE'] for cell in cells.values() for repeat in cell['repeats']]))
    ranking={}
    for task in ('S1','S2'):
        gaps=[a['gain_percent']['global']['NMSE']-b['gain_percent']['global']['NMSE'] for a,b in zip(cells[task+'_P1']['repeats'],cells[task+'_CP1']['repeats'])]
        observed='P1>CP1' if all(gap>.25 for gap in gaps) else 'CP1>P1' if all(gap<-.25 for gap in gaps) else 'unidentifiable'
        ranking[task]=dict(gaps_percent=gaps,observed=observed,methods={method:dict(prediction=record['ranking'][task],
            correct=record['ranking'][task]==observed if observed!='unidentifiable' else None) for method,record in predictions.items()})
    adopted = [name for name,cell in cells.items() if cell['confirmed_adoption']]
    best_other=max(scores[method]['correct'] for method in ('generic_stats','case_match','strong_prompt'))
    signal_pass=scores['new_signal']['correct']==4 and any(cell['repeatability']['stable_category'] in ('improve','harm') for cell in cells.values()) and 4>best_other
    passed=bool(signal_pass and adopted)
    report=dict(status='PASS',cells=cells,scores=scores,ranking=ranking,confirmed_adoption_cells=adopted,
        signal_comparative_gate_passed=signal_pass,advance_to_feedback_design=passed,successful_runs=successful,result_sha256=result_hashes,
        new_trainings=18,new_failed_runs=0,cumulative_successful_trainings=286,preserved_prior_failed_delivery_count=1,
        training_process_seconds=training_seconds,audit_process_seconds=time.monotonic()-started,
        protocol_sha256=base.sha(root/'selection_confirmation_protocol.json'),prediction_lock_sha256=base.sha(root/'selection_prediction_lock.json'),
        auditor_sha256=base.sha(__file__),test_access=False,
        next='design and freeze independent feedback 2x2; no automatic launch' if passed else 'stop this batch; do not modify signal, thresholds, predictions or candidates')
    output = root/'selection_confirmation_feedback.json'
    assert not output.exists()
    base.write(output, report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('cells','successful_runs')}))


if __name__ == '__main__':
    main()
