"""Fit the frozen median3 diagnostic group on T and audit its structural control."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def synthetic_checks(proposal):
    train = np.arange(120*3, dtype=np.float32).reshape(120, 3)/8
    train[15, 0] += 100
    train[55, 1] -= 100
    starts = np.array([0, 40, 80], dtype=np.int64)
    strata = np.arange(3, dtype=np.int64)
    fitted = proposal.fit(train, starts, strata, 2026101499)
    altered = train.copy()
    for start in starts:
        altered[start+32:start+40] = -9000
    another = proposal.fit(altered, starts, strata, 2026101499)
    assert fitted['params'] == another['params']
    assert np.array_equal(fitted['candidates']['P1']['target'], another['candidates']['P1']['target'])
    x = torch.tensor(train[starts[:, None]+np.arange(32)])
    before = x.clone()
    for control in (False, True):
        y = proposal.transform(x, fitted['params'], 'P1', control)
        assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
        assert torch.equal(x, before) and torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, -1], x[:, -1])
        expected = (.25*x[:, :-2]+.5*x[:, 1:-1]+.25*x[:, 2:]) if control else torch.median(torch.stack([x[:, :-2], x[:, 1:-1], x[:, 2:]]), dim=0).values
        assert torch.equal(y[:, 1:-1], expected)
        constant = torch.full_like(x, 4.)
        assert torch.equal(proposal.transform(constant, fitted['params'], 'P1', control), constant)
    monotonic = torch.arange(32, dtype=torch.float32)[None, :, None].expand(2, -1, 3)
    assert torch.equal(proposal.transform(monotonic, fitted['params'], 'P1'), monotonic)
    peak = torch.zeros_like(x)
    peak[:, 15] = 8
    assert torch.equal(proposal.transform(peak, fitted['params'], 'P1'), torch.zeros_like(peak))
    assert np.array_equal(proposal.target_mask(x, fitted['params'], 'P1').numpy(), fitted['candidates']['P1']['target'])
    assert np.array_equal(proposal.target_mask(x.cuda(), fitted['params'], 'P1').cpu().numpy(), fitted['candidates']['P1']['target'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    started = time.monotonic()
    base = module('baseline', root/'run_intervention_stage1_baseline.py')
    protocol = base.read(root/'candidate_protocol.json')
    assert base.sha(Path(__file__)) == protocol['preparer_sha256']
    assert base.sha(root/'proposal_median3.py') == protocol['proposal_sha256']
    assert base.sha(root/'run_intervention_stage1_baseline.py') == protocol['baseline_runner_sha256']
    proposal = module('proposal', root/'proposal_median3.py')
    torch.set_num_threads(4)
    synthetic_checks(proposal)
    output = root/'candidate_prepared'
    assert not output.exists()
    output.mkdir()
    tasks = []
    for task in ('C1', 'C2', 'S1', 'S2'):
        folder = output/task
        folder.mkdir()
        assert base.sha(root/'tasks'/task/'arrays.npz') == protocol['task_array_sha256'][task]
        with np.load(root/'tasks'/task/'arrays.npz', allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        fitted = proposal.fit(arrays['train'], arrays['train_starts'], arrays['train_strata'], protocol['control_seed'])
        params = fitted['params']
        recipe = fitted['candidates']['P1']
        assert set(fitted['candidates']) == {'P1'}
        assert np.array_equal(recipe['control'], recipe['target'])
        assert np.all(recipe['weight'] == 1.) and np.all(recipe['control_weight'] == 1.)
        assert np.all(recipe['probability'] == recipe['probability'][0])
        assert np.array_equal(recipe['probability'], recipe['control_probability'])
        assert np.isclose(recipe['probability'].sum(), 1., rtol=1e-14)
        masks = {}
        changes = {}
        for split in ('train', 'validation'):
            starts = arrays[split+'_starts']
            mask = np.empty(len(starts), dtype=bool)
            max_change = dict(P1=0., CP1=0.)
            changed_elements = dict(P1=0, CP1=0)
            total_elements = 0
            for lo in range(0, len(starts), 1024):
                raw = torch.tensor(arrays[split][starts[lo:lo+1024, None]+np.arange(32)], device='cuda')
                mask[lo:lo+len(raw)] = proposal.target_mask(raw, params, 'P1').cpu().numpy()
                x = base.transform(raw, protocol['parent_by_task'][task], protocol['source_half_recipe'])
                original = x.clone()
                for action in ('P1', 'CP1'):
                    y = proposal.transform(x, params, 'P1', action == 'CP1')
                    assert torch.equal(x, original)
                    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
                    assert torch.isfinite(y).all() and torch.equal(y[:, 0], x[:, 0]) and torch.equal(y[:, -1], x[:, -1])
                    delta = abs(y-x)
                    max_change[action] = max(max_change[action], float(delta.max()))
                    changed_elements[action] += int((delta > 0).sum())
                total_elements += x.numel()
            masks[split+'_target'] = mask
            changes[split] = dict(max_absolute_change=max_change, changed_elements=changed_elements, total_elements=total_elements)
        assert np.array_equal(masks['train_target'], recipe['target'])
        # Canonical uniform vector exactly matches the reference runner, avoiding
        # a redundant floating-point normalization changing probability hashes.
        masks['probability'] = np.full(len(arrays['train_starts']), 1/len(arrays['train_starts']))
        masks['loss_weight'] = np.ones(len(arrays['train_starts']))
        np.savez_compressed(folder/'arrays.npz', **masks)
        base.write(folder/'params.json', params)
        receipt = dict(task=task, status='PASS', controls='same structural support and endpoints; nonlinear median versus linear smoothing; unequal realized edit magnitude is disclosed',
                       local_groups={split:dict(target=int(masks[split+'_target'].sum()), complement=int((~masks[split+'_target']).sum())) for split in ('train', 'validation')},
                       realized_input_changes=changes, fit_scope='all legal T input windows only', numpy_torch_target_identity=True,
                       all_windows_retained=True, uniform_sampling=True, unit_loss_weight=True, original_targets_unchanged=True,
                       params_sha256=base.sha(folder/'params.json'), arrays_sha256=base.sha(folder/'arrays.npz'),
                       proposal_sha256=protocol['proposal_sha256'], test_access=False)
        base.write(folder/'receipt.json', receipt)
        tasks.append(receipt)
    result = dict(status='PASS', synthetic_contract_checks=True, tasks=tasks, process_seconds=time.monotonic()-started,
                  new_trainings=0, candidate_responses_generated=False, test_access=False)
    base.write(root/'candidate_preparation_receipt.json', result)
    print(json.dumps({key:value for key,value in result.items() if key != 'tasks'}))


if __name__ == '__main__':
    main()
