"""Trusted 100-step DLinear acceptance, separate from scientific search results."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

STEPS = 100
SEED = 20261008


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--core-sha256', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'research_runs/agent_research/acceptance/20261008/gpu_smoke_verified_td.json'
    if output.exists():
        raise RuntimeError('Existing acceptance receipt must not be overwritten')
    paths = {'manifest': root / 'data/metropt3_v1/manifest.json',
             'core': root / 'scripts/run_research.py', 'protocol': root / 'research_runs/protocol.json'}
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'scope': 'engineering_only', 'research_experiment': False, 'seed': SEED,
               'steps': STEPS, 'recipe': 'all_training_windows_uniform_sampling',
               'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'source_sha256': {name: getattr(args, name + '_sha256') for name in paths},
               'worker_sha256': sha(__file__), 'test_feedback_used': False}
    started = time.monotonic()
    try:
        for name, path in paths.items():
            if sha(path) != getattr(args, name + '_sha256'):
                raise RuntimeError('Registered source identity mismatch: ' + name)
        manifest = json.loads(paths['manifest'].read_text(encoding='utf-8'))
        for split in manifest['splits']:
            if split['name'] in {'train', 'validation'}:
                if sha(root / 'data/metropt3_v1' / split['file']) != split['sha256']:
                    raise RuntimeError('Remote T/D checksum mismatch: ' + split['name'])
        receipt['dataset_verification'] = 'remote_manifest_and_T_D_checksums; full_source_verification_performed_locally'
        import numpy as np
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name():
            raise RuntimeError('Confirmed RTX 4090 required')
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        spec = importlib.util.spec_from_file_location('trusted_research_core', paths['core'])
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        p = json.loads(paths['protocol'].read_text(encoding='utf-8'))
        arrays, starts, groups, audit = core.prepare(p, manifest)
        data = {split: torch.from_numpy(array).cuda() for split, array in arrays.items()}
        indices = {split: torch.from_numpy(array).cuda() for split, array in starts.items()}
        model = core.DLinear(p).cuda()
        if sum(parameter.numel() for parameter in model.parameters()) != 7920:
            raise RuntimeError('Fixed DLinear parameter count changed')
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        rng = torch.Generator(device='cuda').manual_seed(SEED + 1000)
        offsets = torch.arange(p['input_length'] + p['prediction_length'], device='cuda')
        for _ in range(STEPS):
            chosen = torch.randint(len(indices['train']), (p['batch_size'],), generator=rng, device='cuda')
            batch = data['train'][indices['train'][chosen, None] + offsets]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(batch[:, :p['input_length']]) - batch[:, p['input_length']:]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
        metrics = core.evaluate(model, data['validation'], indices['validation'], groups, p, 'baseline')
        receipt.update(status='passed', metrics=metrics['groups']['all'],
            data_audit=audit, parameters=7920, full_validation=True,
            comparison_eligible=False, protocol_difference={'steps': STEPS, 'base_steps': p['steps'],
                'checkpoint_selection': 'last_step_engineering_only', 'seed': SEED,
                'sampling_implementation': 'torch_randint_engineering_only'},
            environment={'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
                         'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error_message=str(error)[:2000])
    receipt['elapsed_seconds'] = time.monotonic() - started
    with output.open('x', encoding='utf-8') as destination:
        json.dump(receipt, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
