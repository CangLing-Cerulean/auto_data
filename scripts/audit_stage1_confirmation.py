"""Audit the preregistered three-seed construction matrix and its stable gate."""
import argparse
import importlib.util
from pathlib import Path
import itertools
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
    lock = base.read(root/'confirmation_activation.json')
    assert base.sha(__file__) == lock['auditor_sha256']
    assert lock['candidate_protocol_sha256'] == base.sha(root/'candidate_protocol.json')
    first = base.read(root/'candidate_first_seed_gate.json')
    assert first['advance_to_confirmation']
    assert base.sha(root/'candidate_first_seed_gate.json') == lock['first_seed_gate_sha256']
    assert base.sha(root/'online_ts/agent_research/stage1_scoring.py') == protocol['scorer_sha256']
    cells = {f'{task}_{action}':dict(task=task, action=action, repeats=[]) for task in ('C1', 'C2') for action in ('P1', 'CP1')}
    training_seconds = 0.
    completed = []
    for task in ('C1', 'C2'):
        with np.load(root/'candidate_prepared'/task/'arrays.npz', allow_pickle=False) as saved:
            target = saved['validation_target']
        masks = {'global':np.ones(len(target), dtype=bool), 'target':target, 'complement':~target}
        for seed in protocol['confirmation_seeds']:
            audited = {}
            for action in ('PARENT', 'P1', 'CP1'):
                name = f'{task}_{action}_s{seed}'
                folder = root/'candidate_runs'/name
                result = base.read(folder/'result.json')
                assert base.read(folder/'status.json')['status'] == 'completed'
                assert result['task'] == task and result['action'] == action and result['seed'] == seed
                assert result['source_sha256'] == protocol['runner_sha256']
                assert result['protocol_sha256'] == lock['candidate_protocol_sha256']
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
                        np.testing.assert_allclose([measured[key] for key in ('NMSE', 'NMAE', 'signed_bias')],
                                                   [recorded[key] for key in ('NMSE', 'NMAE', 'signed_bias')], rtol=1e-12, atol=1e-14)
                for metric in ('NMSE', 'NMAE', 'signed_bias'):
                    weighted = sum(result['metrics'][scope][metric]*result['metrics'][scope]['elements'] for scope in ('target', 'complement') if result['metrics'][scope]['elements'])
                    np.testing.assert_allclose(weighted/result['metrics']['global']['elements'], result['metrics']['global'][metric], rtol=1e-12, atol=1e-14)
                with np.load(folder/'sampling.npz', allow_pickle=False) as saved:
                    assert saved['draw_count'].sum() == 768000 and np.all(saved['loss_weight'] == 1.)
                    assert np.all(saved['probability'] == 1/len(saved['probability']))
                assert result['best_step'] == min(result['history'], key=lambda item:item['NMSE'])['step']
                training_seconds += result['process_seconds']
                completed.append(name)
                audited[action] = result
            for action in ('P1', 'CP1'):
                parent, child = audited['PARENT'], audited[action]
                for key in ('initialization_sha256', 'probability_sha256', 'draw_count_sha256'):
                    assert parent[key] == child[key]
                gains = {scope:{metric:100*(1-child['metrics'][scope][metric]/parent['metrics'][scope][metric]) if parent['metrics'][scope]['elements'] and parent['metrics'][scope][metric] > 0 else None for metric in ('NMSE', 'NMAE')} for scope in masks}
                cells[task+'_'+action]['repeats'].append(dict(seed=seed, gain_percent=gains,
                    observed_category={scope:scorer.observed_direction(gains[scope]['NMSE']) if gains[scope]['NMSE'] is not None else 'unscorable' for scope in masks}))
    for cell in cells.values():
        gains = [record['gain_percent']['global']['NMSE'] for record in cell['repeats']]
        mae = [record['gain_percent']['global']['NMAE'] for record in cell['repeats']]
        cell['repeatability'] = scorer.repeatability(gains) if all(gain is not None for gain in gains) else dict(stable_category='unscorable')
        cell['mean_global_NMSE_gain_percent'] = float(np.mean(gains)) if all(gain is not None for gain in gains) else None
        cell['confirmed_adoption'] = (all(gain is not None and gain > 0 for gain in gains)
                                      and cell['mean_global_NMSE_gain_percent'] >= 1.
                                      and all(value is not None and value >= -1. for value in mae))
    contrasts = []
    for (name1, cell1), (name2, cell2) in itertools.combinations(cells.items(), 2):
        label1, label2 = cell1['repeatability']['stable_category'], cell2['repeatability']['stable_category']
        if label1 in ('uncertain', 'unscorable') or label2 in ('uncertain', 'unscorable') or label1 == label2:
            continue
        gaps = [abs(a['gain_percent']['global']['NMSE']-b['gain_percent']['global']['NMSE']) for a, b in zip(cell1['repeats'], cell2['repeats'])]
        if all(gap > .25 for gap in gaps):
            contrasts.append(dict(first=name1, second=name2, gap_percent_by_seed=gaps))
    adopted = [name for name, cell in cells.items() if cell['confirmed_adoption']]
    passed = bool(contrasts or adopted)
    report = dict(status='PASS', construction_gate_passed=passed, cells=cells, stable_global_contrasts=contrasts,
                  confirmed_adoption_cells=adopted, completed_runs=completed, new_confirmation_trainings=18,
                  cumulative_completed_trainings=265, training_process_seconds=training_seconds,
                  audit_process_seconds=time.monotonic()-started,
                  next='freeze a bounded executable construction-only signal; null stops method comparison' if passed else 'stop; no stable registered contrast or confirmed adoption; no threshold search or selection-task response',
                  candidate_protocol_sha256=lock['candidate_protocol_sha256'], test_access=False)
    base.write(root/'construction_confirmation.json', report)
    print(json.dumps({key:value for key,value in report.items() if key not in ('cells','completed_runs')}))


if __name__ == '__main__':
    main()
