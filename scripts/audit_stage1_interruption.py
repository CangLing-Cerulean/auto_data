"""Preserve and audit the interrupted confirmation, without changing run status."""
import argparse
import importlib.util
import json
from pathlib import Path
import tarfile
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
    assert base.sha(root/'candidate_protocol.json') == lock['candidate_protocol_sha256']
    assert base.sha(root/'run_stage1_candidate.py') == protocol['runner_sha256']
    assert base.sha(root/'online_ts/agent_research/stage1_scoring.py') == protocol['scorer_sha256']
    expected = [f'{task}_{action}_s{seed}' for task in ('C1', 'C2')
                for seed in protocol['confirmation_seeds'] for action in ('PARENT', 'P1', 'CP1')]
    completed, failed, absent, audited = [], [], [], {}
    training_seconds = 0.
    statuses_before = {}
    with np.load(root/'candidate_prepared/C1/arrays.npz', allow_pickle=False) as saved:
        target = saved['validation_target']
    masks = {'global': np.ones(len(target), dtype=bool), 'target':target, 'complement':~target}
    for name in expected:
        folder = root/'candidate_runs'/name
        if not folder.exists():
            absent.append(name)
            continue
        status = base.read(folder/'status.json')
        statuses_before[name] = base.sha(folder/'status.json')
        assert status['task'] == 'C1' and status['steps_completed'] == 3000
        assert status['status'] in ('completed', 'failed')
        if status['status'] == 'failed':
            assert name == 'C1_CP1_s2026101404' and status['error'] == '[Errno 32] Broken pipe'
            failed.append(name)
        else:
            completed.append(name)
        result = base.read(folder/'result.json')
        assert name == f"{result['task']}_{result['action']}_s{result['seed']}"
        assert result['protocol_sha256'] == lock['candidate_protocol_sha256']
        assert result['source_sha256'] == protocol['runner_sha256']
        assert result['prepared_sha256'] == protocol['task_array_sha256']['C1']
        assert result['proposal_sha256'] == protocol['proposal_sha256'] and result['test_access'] is False
        for asset, digest in result['assets'].items():
            assert base.sha(folder/asset) == digest
        with np.load(folder/'best_errors.npz', allow_pickle=False) as saved:
            error = saved['error']
        assert np.isfinite(error).all()
        for scope, mask in masks.items():
            measured = runner.metrics(error, mask)
            recorded = result['metrics'][scope]
            assert measured['elements'] == recorded['elements']
            np.testing.assert_allclose([measured[k] for k in ('NMSE','NMAE','signed_bias')],
                                       [recorded[k] for k in ('NMSE','NMAE','signed_bias')], rtol=1e-12, atol=1e-14)
        with np.load(folder/'sampling.npz', allow_pickle=False) as saved:
            assert saved['draw_count'].sum() == 768000 and np.all(saved['loss_weight'] == 1.)
            assert np.all(saved['probability'] == 1/len(saved['probability']))
            assert base.digest(saved['draw_count']) == result['draw_count_sha256']
            assert base.digest(saved['probability']) == result['probability_sha256']
        assert result['best_step'] == min(result['history'], key=lambda item:item['NMSE'])['step']
        assert [item['step'] for item in result['history']] == list(range(500, 3001, 500))
        training_seconds += status['process_seconds']
        audited[name] = result
    assert len(completed) == 8 and failed == ['C1_CP1_s2026101404'] and len(absent) == 9
    descriptive = []
    for seed in protocol['confirmation_seeds']:
        parent = audited[f'C1_PARENT_s{seed}']
        for action in ('P1', 'CP1'):
            name = f'C1_{action}_s{seed}'
            child = audited[name]
            for key in ('initialization_sha256', 'probability_sha256', 'draw_count_sha256'):
                assert parent[key] == child[key]
            gains = {metric:100*(1-child['metrics']['global'][metric]/parent['metrics']['global'][metric])
                     for metric in ('NMSE', 'NMAE')}
            descriptive.append(dict(id=name, global_gain_percent=gains,
                category=scorer.observed_direction(gains['NMSE']),
                run_status='failed' if name in failed else 'completed',
                formal_gate_eligible=False))
    for name, digest in statuses_before.items():
        assert base.sha(root/'candidate_runs'/name/'status.json') == digest
    report = dict(status='PARTIAL_ARTIFACT_AUDIT_PASS', experiment_status='stopped_on_transport_output_failure',
        completed_runs=completed, failed_runs=failed, not_started_runs=absent,
        registered_confirmation_runs=18, completed_confirmation_runs=8, failed_confirmation_runs=1,
        fully_computed_and_saved_runs=9, cumulative_successful_completed_trainings=255,
        cumulative_full_training_computations_including_this_failed_delivery=256,
        computation_count_scope='247 prior successful runs + 9 current full computations; earlier failed work remains in historical ledgers',
        training_process_seconds=training_seconds, audit_process_seconds=time.monotonic()-started,
        status_files_preserved_sha256=statuses_before, descriptive_C1_responses=descriptive,
        construction_gate_passed=None, adoption_gate_passed=None, gate_status='not_evaluated_incomplete_matrix_and_failed_status',
        S_response_runs=0, new_signal_created=False, test_access=False,
        next='stop; preserve failed ID and all artifacts; no restart or new experimental stage',
        auditor_sha256=base.sha(__file__))
    base.write(root/'confirmation_interruption_audit.json', report)
    names = ['baseline_protocol.json','candidate_protocol.json','candidate_predictions.json','proposal_median3.py',
             'prepare_stage1_candidate.py','run_stage1_candidate.py','run_intervention_stage1_baseline.py',
             'audit_stage1_confirmation.py','audit_stage1_interruption.py','candidate_preparation_receipt.json',
             'candidate_prepared','frozen_core_snapshot.py','online_ts/agent_research/stage1_scoring.py',
             'construction_state.json','candidate_first_seed_gate.json','construction_feedback.json',
             'confirmation_activation.json','confirmation_interruption_audit.json']
    names += ['candidate_runs/'+name for name in expected if name in audited]
    files = []
    for name in names:
        path = root/name
        assert path.exists()
        for source in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if source.is_file():
                assert not source.is_symlink() and root.resolve() in source.resolve().parents
                relative = source.relative_to(root).as_posix()
                assert not any(token in relative.lower() for token in ('private','test.csv','verification/','.ssh'))
                files.append(dict(name=relative, bytes=source.stat().st_size, sha256=base.sha(source)))
    archive_path = root/'stage1-confirmation-interrupted-20261012-r1.tar.gz'
    assert not archive_path.exists()
    with tarfile.open(archive_path, 'w:gz', compresslevel=6) as archive:
        for item in files:
            archive.add(root/item['name'], arcname=item['name'], recursive=False)
    receipt = dict(name=archive_path.name, bytes=archive_path.stat().st_size, sha256=base.sha(archive_path),
        members=files, completed_runs=completed, failed_runs=failed, not_started_runs=absent,
        test_data_included=False, raw_CSV_included=False,
        reference_archive_sha256='ed511b9524de8420ee80826ab555041be57eeb989b0179231fba76cd8afc4734')
    base.write(root/'confirmation_interruption_archive_receipt.json', receipt)
    print(json.dumps({k:v for k,v in report.items() if k not in ('status_files_preserved_sha256','descriptive_C1_responses')}))
    print(json.dumps({k:v for k,v in receipt.items() if k != 'members'}))


if __name__ == '__main__':
    main()
