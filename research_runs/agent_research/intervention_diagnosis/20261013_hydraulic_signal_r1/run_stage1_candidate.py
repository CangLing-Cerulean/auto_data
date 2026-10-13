"""Frozen common candidate/control responses, with staged activation gates."""
import argparse
import importlib.util
import os
from pathlib import Path
import random
import time
import hashlib
import json
import numpy as np
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def metrics(error, mask):
    if not mask.any():
        return dict(windows=0, elements=0, NMSE=None, NMAE=None, signed_bias=None)
    values = error[mask].astype(np.float64)
    return dict(windows=len(values), elements=values.size, NMSE=float(np.mean(values**2)),
                NMAE=float(np.mean(abs(values))), signed_bias=float(values.mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--action', required=True, choices=['PARENT', 'P1', 'CP1'])
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Dedicated research server required')
    root = Path(args.root)
    base = module('baseline', root/'run_intervention_stage1_baseline.py')
    p = base.read(root/'candidate_protocol.json')
    assert base.sha(__file__) == p['runner_sha256']
    assert base.sha(root/'run_intervention_stage1_baseline.py') == p['baseline_runner_sha256']
    assert base.sha(root/'proposal_median3.py') == p['proposal_sha256']
    task, action, seed = args.task, args.action, args.seed
    assert task in p['task_array_sha256']
    if seed == p['exploration_seed']:
        assert action != 'PARENT'
    else:
        assert seed in p['confirmation_seeds']
        assert base.read(root/'candidate_first_seed_gate.json')['advance_to_confirmation']
    if task.startswith('S'):
        activation = base.read(root/'selection_activation.json')
        assert activation['candidate_protocol_sha256'] == base.sha(root/'candidate_protocol.json')
        assert activation['predictions_locked'] and activation['construction_gate_passed']
    proposal = module('proposal', root/'proposal_median3.py')
    output = root/'candidate_runs'/f'{task}_{action}_s{seed}'
    if output.exists():
        raise RuntimeError('Run ID occupied; never overwrite or repeat')
    output.mkdir(parents=True)
    started = time.monotonic()
    status = dict(status='running', task=task, action=action, seed=seed, steps_completed=0,
                  protocol_sha256=base.sha(root/'candidate_protocol.json'))
    base.write(output/'status.json', status)
    try:
        assert base.sha(root/'tasks'/task/'arrays.npz') == p['task_array_sha256'][task]
        prepared = root/'candidate_prepared'/task
        receipt = base.read(prepared/'receipt.json')
        assert receipt['status'] == 'PASS' and receipt['proposal_sha256'] == p['proposal_sha256']
        assert base.sha(prepared/'arrays.npz') == receipt['arrays_sha256']
        assert base.sha(prepared/'params.json') == receipt['params_sha256']
        params = base.read(prepared/'params.json')
        with np.load(root/'tasks'/task/'arrays.npz', allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        with np.load(prepared/'arrays.npz', allow_pickle=False) as source:
            group = {key: source[key] for key in source.files}
        config = dict(p['training'], channels=arrays['train'].shape[1])
        assert config['steps'] == 3000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        assert base.sha(p['frozen_core']) == p['frozen_core_sha256']
        core = module('frozen_dlinear', p['frozen_core'])
        model = core.DLinear(config).cuda()
        nparams = sum(value.numel() for value in model.parameters())
        assert nparams == 528*config['channels']
        initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        t = torch.tensor(arrays['train'], device='cuda')
        d = torch.tensor(arrays['validation'], device='cuda')
        ts = torch.tensor(arrays['train_starts'], device='cuda')
        ds = torch.tensor(arrays['validation_starts'], device='cuda')
        offsets = torch.arange(40, device='cuda')
        probability = group['probability']
        assert np.all(probability == 1/len(ts)) and np.all(group['loss_weight'] == 1.)
        cdf = np.cumsum(probability)
        cdf[-1] = 1.
        generator = np.random.default_rng(seed+1000)
        draws = np.zeros(len(ts), dtype=np.int64)
        history = []
        best = float('inf')
        best_step = None

        def transform(x):
            parent = base.transform(x, p['parent_by_task'][task], p['source_half_recipe'])
            return parent if action == 'PARENT' else proposal.transform(parent, params, 'P1', action == 'CP1')

        def evaluate():
            result = np.empty((len(ds), 8, config['channels']), dtype=np.float32)
            model.eval()
            with torch.no_grad():
                for lo in range(0, len(ds), 2048):
                    batch = d[ds[lo:lo+2048, None]+offsets]
                    result[lo:lo+len(batch)] = (model(transform(batch[:, :32]))-batch[:, 32:]).cpu().numpy()
            return result

        for step in range(1, 3001):
            chosen = np.searchsorted(cdf, generator.random(config['batch_size']), side='right')
            np.add.at(draws, chosen, 1)
            ids = torch.tensor(chosen, device='cuda')
            batch = t[ts[ids, None]+offsets]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(transform(batch[:, :32]))-batch[:, 32:]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                error = evaluate()
                if not np.isfinite(error).all():
                    raise RuntimeError('Nonfinite development errors')
                measured = metrics(error, np.ones(len(error), dtype=bool))
                history.append(dict(step=step, NMSE=measured['NMSE'], NMAE=measured['NMAE']))
                status['steps_completed'] = step
                base.write(output/'status.json', status)
                if measured['NMSE'] < best:
                    best = measured['NMSE']
                    best_step = step
                    torch.save(model.state_dict(), output/'best.pt')
                    np.savez_compressed(output/'best_errors.npz', error=error)
        torch.save(model.state_dict(), output/'last.pt')
        np.savez_compressed(output/'sampling.npz', probability=probability, loss_weight=group['loss_weight'],
                            draw_count=draws, train_window_start=arrays['train_starts'])
        with np.load(output/'best_errors.npz', allow_pickle=False) as saved:
            error = saved['error']
        scored = {label: metrics(error, mask) for label, mask in [('global', np.ones(len(error), dtype=bool)),
                  ('target', group['validation_target']), ('complement', ~group['validation_target'])]}
        assert base.sha(root/'candidate_protocol.json') == status['protocol_sha256']
        result = dict(task=task, action=action, seed=seed, metrics=scored, best_step=best_step, history=history,
                      parameters=nparams, initialization_sha256=initial_sha, probability_sha256=base.digest(probability),
                      draw_count_sha256=base.digest(draws), prepared_sha256=p['task_array_sha256'][task],
                      candidate_params_sha256=receipt['params_sha256'], candidate_arrays_sha256=receipt['arrays_sha256'],
                      protocol_sha256=status['protocol_sha256'], source_sha256=base.sha(__file__), proposal_sha256=p['proposal_sha256'],
                      assets={name: base.sha(output/name) for name in ('best.pt', 'last.pt', 'best_errors.npz', 'sampling.npz')},
                      process_seconds=time.monotonic()-started, test_access=False)
        base.write(output/'result.json', result)
        status['status'] = 'completed'
        print(json.dumps(dict(task=task, action=action, seed=seed, metrics=scored, seconds=result['process_seconds'])), flush=True)
    except Exception as error:
        status['status'] = 'failed'
        status['error'] = str(error)
        raise
    finally:
        status['process_seconds'] = time.monotonic()-started
        base.write(output/'status.json', status)


if __name__ == '__main__':
    main()
