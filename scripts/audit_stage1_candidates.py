"""Audit construction responses and apply the preregistered first-seed gate."""
import argparse
import importlib.util
from pathlib import Path
import time
import itertools
import json
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
    scoring = module('scoring', root/'online_ts/agent_research/stage1_scoring.py')
    protocol = base.read(root/'candidate_protocol.json')
    assert base.sha(__file__) == protocol['first_seed_auditor_sha256']
    assert base.sha(root/'online_ts/agent_research/stage1_scoring.py') == protocol['scorer_sha256']
    predicted = base.read(root/'candidate_predictions.json')
    assert base.sha(root/'candidate_predictions.json') == protocol['predictions_sha256']
    seed = protocol['exploration_seed']
    responses = []
    forecast = []
    observed_gains = []
    unscorable = []
    training_seconds = 0.
    for task in ('C1', 'C2'):
        with np.load(root/'candidate_prepared'/task/'arrays.npz', allow_pickle=False) as source:
            target = source['validation_target']
        masks = dict(global_=np.ones(len(target), dtype=bool), target=target, complement=~target)
        masks['global'] = masks.pop('global_')
        parent_folder = root/'baseline_runs'/f'{task}_FIX_s{seed}'
        parent = base.read(parent_folder/'result.json')
        assert base.sha(parent_folder/'best_errors.npz') == parent['assets']['best_errors.npz']
        with np.load(parent_folder/'best_errors.npz', allow_pickle=False) as saved:
            parent_error = saved['error']
        parent_metrics = {name:runner.metrics(parent_error, mask) for name, mask in masks.items()}
        for action in ('P1', 'CP1'):
            folder = root/'candidate_runs'/f'{task}_{action}_s{seed}'
            result = base.read(folder/'result.json')
            assert base.read(folder/'status.json')['status'] == 'completed'
            assert result['source_sha256'] == protocol['runner_sha256']
            assert result['protocol_sha256'] == base.sha(root/'candidate_protocol.json')
            assert result['proposal_sha256'] == protocol['proposal_sha256']
            assert result['prepared_sha256'] == protocol['task_array_sha256'][task]
            for name, digest in result['assets'].items():
                assert base.sha(folder/name) == digest
            for key in ('initialization_sha256', 'probability_sha256', 'draw_count_sha256'):
                assert result[key] == parent[key]
            with np.load(folder/'sampling.npz', allow_pickle=False) as saved:
                assert saved['draw_count'].sum() == 768000
                assert np.all(saved['loss_weight'] == 1.)
            with np.load(folder/'best_errors.npz', allow_pickle=False) as saved:
                error = saved['error']
            assert np.isfinite(error).all() and error.shape == parent_error.shape
            assert result['best_step'] == min(result['history'], key=lambda x: x['NMSE'])['step']
            gains = {}
            labels = {}
            for scope, mask in masks.items():
                measured = runner.metrics(error, mask)
                recorded = result['metrics'][scope]
                assert measured['elements'] == recorded['elements']
                if measured['windows']:
                    np.testing.assert_allclose([measured[k] for k in ('NMSE', 'NMAE', 'signed_bias')],
                                               [recorded[k] for k in ('NMSE', 'NMAE', 'signed_bias')], rtol=1e-12, atol=1e-14)
                gain = {metric:100*(1-measured[metric]/parent_metrics[scope][metric]) if measured['windows'] and parent_metrics[scope][metric] > 0 else None for metric in ('NMSE', 'NMAE')}
                gains[scope] = gain
                labels[scope] = scoring.observed_direction(gain['NMSE']) if gain['NMSE'] is not None else 'unscorable'
                prediction = predicted['tasks'][task]['operation' if action == 'P1' else 'control']['overall' if scope == 'global' else scope]
                if gain['NMSE'] is None:
                    unscorable.append(dict(task=task, action=action, scope=scope, prediction=prediction))
                else:
                    forecast.append(prediction)
                    observed_gains.append(gain['NMSE'])
            for metric in ('NMSE', 'NMAE', 'signed_bias'):
                local_sum = sum(result['metrics'][scope][metric]*result['metrics'][scope]['elements'] for scope in ('target', 'complement') if result['metrics'][scope]['elements'])
                np.testing.assert_allclose(local_sum/result['metrics']['global']['elements'], result['metrics']['global'][metric], rtol=1e-12, atol=1e-14)
            responses.append(dict(task=task, action=action, parent_metrics=parent_metrics, child_metrics=result['metrics'],
                                  gain_percent=gains, observed_category=labels, run_result_sha256=base.sha(folder/'result.json')))
            training_seconds += result['process_seconds']
    contrasts = []
    for first, second in itertools.combinations(responses, 2):
        g1 = first['gain_percent']['global']['NMSE']
        g2 = second['gain_percent']['global']['NMSE']
        if first['observed_category']['global'] != second['observed_category']['global'] and abs(g1-g2) > scoring.UNCHANGED_BAND_PERCENT:
            contrasts.append(dict(first=first['task']+'_'+first['action'], second=second['task']+'_'+second['action'], absolute_gain_gap_percent=abs(g1-g2)))
    adoption = [item['task']+'_'+item['action'] for item in responses if item['gain_percent']['global']['NMSE'] >= scoring.MIN_ADOPTION_GAIN_PERCENT and item['gain_percent']['global']['NMAE'] >= -scoring.MAX_NMAE_DETERIORATION_PERCENT]
    gate = dict(status='PASS', advance_to_confirmation=bool(contrasts or adoption), eligible_global_contrasts=contrasts,
                provisional_adoption_cells=adoption, repeat_seeds=protocol['confirmation_seeds'],
                stop_reason=None if contrasts or adoption else 'No preregistered material global category contrast or adoption-potential cell; stop this batch without threshold search.',
                single_seed_only=True, new_candidate_trainings=4, cumulative_completed_trainings=247,
                training_process_seconds=training_seconds, audit_process_seconds=time.monotonic()-started,
                prediction_score=scoring.score_predictions(forecast, observed_gains), unscorable_predictions=unscorable,
                candidate_protocol_sha256=base.sha(root/'candidate_protocol.json'), test_access=False)
    base.write(root/'construction_feedback.json', dict(responses=responses, gate=gate, scope='C1/C2 construction only; no selection task response included'))
    base.write(root/'candidate_first_seed_gate.json', gate)
    print(json.dumps(gate))


if __name__ == '__main__':
    main()
