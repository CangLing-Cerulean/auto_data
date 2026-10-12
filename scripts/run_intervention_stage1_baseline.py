"""Frozen grouped-task B0 and known source-half references; development only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def transform(x, action, recipe):
    if action == 'B0':
        return x
    if action != 'FIX':
        raise ValueError('Unregistered baseline action')
    out = x.clone()
    changed = recipe['changed_channels']
    sources = [recipe['mapping'][i] for i in changed]
    own = x[:, :, changed]
    source = x[:, :, sources]
    alpha = recipe['alpha']
    out[:, :, changed] = own[:, -1:] + (1-alpha)*(own-own[:, -1:]) + alpha*(source-source[:, -1:])
    return out


def train(root, protocol, task, action, seed):
    output = root/'baseline_runs'/f'{task}_{action}_s{seed}'
    if output.exists():
        raise RuntimeError('Run ID already occupied; preserve and never repeat')
    output.mkdir(parents=True)
    started = time.monotonic()
    status = dict(status='running', task=task, action=action, seed=seed, steps_completed=0,
                  protocol_sha256=sha(root/'baseline_protocol.json'))
    write(output/'status.json', status)
    try:
        receipt = read(root/'tasks'/task/'receipt.json')
        path = root/'tasks'/task/'arrays.npz'
        assert sha(path) == receipt['array_sha256'] == protocol['task_array_sha256'][task]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        parameters = dict(protocol['training'], channels=len(receipt['sensor_columns']))
        assert parameters['steps'] == 3000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        core_path = Path(protocol['frozen_core'])
        assert sha(core_path) == protocol['frozen_core_sha256']
        spec = importlib.util.spec_from_file_location('frozen_dlinear', core_path)
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        model = core.DLinear(parameters).cuda()
        nparams = sum(value.numel() for value in model.parameters())
        assert nparams == 528*parameters['channels']
        initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
        t = torch.tensor(arrays['train'], device='cuda')
        d = torch.tensor(arrays['validation'], device='cuda')
        ts = torch.tensor(arrays['train_starts'], device='cuda')
        ds = torch.tensor(arrays['validation_starts'], device='cuda')
        offsets = torch.arange(40, device='cuda')
        recipe = protocol['source_half_recipe']
        generator = np.random.default_rng(seed+1000)
        probability = np.full(len(ts), 1/len(ts))
        cdf = np.cumsum(probability)
        cdf[-1] = 1.
        draws = np.zeros(len(ts), dtype=np.int64)
        history = []
        best = float('inf')
        best_step = None

        def evaluate():
            result = np.empty((len(ds), 8, parameters['channels']), dtype=np.float32)
            model.eval()
            with torch.no_grad():
                for lo in range(0, len(ds), 2048):
                    batch = d[ds[lo:lo+2048, None]+offsets]
                    result[lo:lo+len(batch)] = (model(transform(batch[:, :32], action, recipe))-batch[:, 32:]).cpu().numpy()
            return result

        for step in range(1, 3001):
            chosen = np.searchsorted(cdf, generator.random(parameters['batch_size']), side='right')
            np.add.at(draws, chosen, 1)
            ids = torch.tensor(chosen, device='cuda')
            batch = t[ts[ids, None]+offsets]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(transform(batch[:, :32], action, recipe))-batch[:, 32:]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if step % 500 == 0:
                error = evaluate()
                if not np.isfinite(error).all():
                    raise RuntimeError('Nonfinite development errors')
                error64 = error.astype(np.float64)
                nmse = float(np.mean(error64**2))
                history.append(dict(step=step, NMSE=nmse, NMAE=float(np.mean(abs(error64)))))
                status['steps_completed'] = step
                write(output/'status.json', status)
                if nmse < best:
                    best = nmse
                    best_step = step
                    torch.save(model.state_dict(), output/'best.pt')
                    np.savez_compressed(output/'best_errors.npz', error=error)
        torch.save(model.state_dict(), output/'last.pt')
        np.savez_compressed(output/'sampling.npz', probability=probability, loss_weight=np.ones(len(ts)), draw_count=draws,
                            train_window_start=arrays['train_starts'])
        with np.load(output/'best_errors.npz', allow_pickle=False) as saved:
            error = saved['error'].astype(np.float64)
        metrics = dict(NMSE=float(np.mean(error**2)), NMAE=float(np.mean(abs(error))),
                       signed_bias=float(error.mean()), windows=len(error), elements=error.size,
                       channel_NMSE=np.mean(error**2, axis=(0, 1)).tolist(),
                       channel_NMAE=np.mean(abs(error), axis=(0, 1)).tolist())
        assert sha(root/'baseline_protocol.json') == status['protocol_sha256']
        result = dict(task=task, action=action, seed=seed, metrics=metrics, best_step=best_step, history=history,
                      parameters=nparams, initialization_sha256=initial_sha, probability_sha256=digest(probability),
                      draw_count_sha256=digest(draws), prepared_sha256=receipt['array_sha256'],
                      protocol_sha256=status['protocol_sha256'], source_sha256=sha(__file__),
                      assets={name: sha(output/name) for name in ('best.pt', 'last.pt', 'best_errors.npz', 'sampling.npz')},
                      process_seconds=time.monotonic()-started, test_access=False)
        write(output/'result.json', result)
        status['status'] = 'completed'
        print(json.dumps(dict(task=task, action=action, NMSE=metrics['NMSE'], seconds=result['process_seconds'])), flush=True)
    except Exception as error:
        status['status'] = 'failed'
        status['error'] = str(error)
        raise
    finally:
        status['process_seconds'] = time.monotonic()-started
        write(output/'status.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--action', required=True)
    parser.add_argument('--seed', required=True, type=int)
    args = parser.parse_args()
    if os.name != 'posix':
        raise RuntimeError('Dedicated research server required')
    root = Path(args.root)
    protocol = read(root/'baseline_protocol.json')
    assert sha(__file__) == protocol['runner_sha256']
    assert [args.task, args.action, args.seed] in protocol['registered_runs']
    train(root, protocol, args.task, args.action, args.seed)


if __name__ == '__main__':
    main()
