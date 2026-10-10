"""Common-step inference and unchanged-branch audit; no optimizer or new model fit."""
import importlib.util
import json
import os
import time

import numpy as np

from run_direct_processing import BASE, PROJECT, STUDY, CHANGED, load, sha, write, raw_split, group_definitions, transform
from run_direct_own_ablation import own_transform


def main():
    destination = STUDY / 'common_step_audit.json'
    if destination.exists():
        raise ValueError('Common-step audit already exists')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location('frozen_core', BASE / 'scripts/run_research.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    p = load(BASE / 'research_runs/protocol.json')
    manifest = load(BASE / 'data/metropt3_v1/manifest.json')
    raw, _, times = raw_split('train', manifest)
    eval_raw, eval_ids, eval_times = raw_split('validation', manifest)
    mean, std = raw.mean(0), raw.std(0)
    scale = np.where(std == 0, 1, std)
    starts = core.valid_starts(eval_times, 40, 30)
    data = torch.from_numpy(((eval_raw - mean) / scale).astype(np.float32)).cuda()
    positions = torch.from_numpy(starts).cuda()
    source_ids = torch.from_numpy(eval_ids).cuda()
    offsets = torch.arange(40, device='cuda')
    unchanged = [c for c in range(15) if c not in CHANGED]
    results, verified = {}, []
    for seed in (3026, 3027, 3028):
        states = {}
        run_paths = {policy: STUDY / 'runs' / f'DP03_full_{policy}_s{seed}' for policy in ('baseline', 'source_half', 'source_half_scramble')}
        run_paths['own_half'] = STUDY / 'runs' / f'DP04_full_own_half_s{seed}'
        results[str(seed)] = {}
        for policy, folder in run_paths.items():
            receipt = load(folder / 'result.json')
            if sha(folder / 'final.pt') != receipt['assets']['final.pt']:
                raise ValueError('Final checkpoint changed')
            state = torch.load(folder / 'final.pt', map_location='cpu', weights_only=True)
            states[policy] = state
            model = core.DLinear(p).cuda()
            model.load_state_dict(state)
            model.eval()
            sq_total = torch.zeros(15, dtype=torch.float64, device='cuda')
            abs_total = torch.zeros_like(sq_total)
            with torch.inference_mode():
                for lo in range(0, len(starts), 2048):
                    selected = positions[lo:lo + 2048]
                    batch = data[selected[:, None] + offsets]
                    x = own_transform(batch[:, :32]) if policy == 'own_half' else transform(batch[:, :32], source_ids[selected], policy)
                    error = (model(x) - batch[:, 32:]).double()
                    sq_total += error.square().sum((0, 1))
                    abs_total += error.abs().sum((0, 1))
            channel_sq = (sq_total / (len(starts) * 8)).cpu().numpy()
            channel_abs = (abs_total / (len(starts) * 8)).cpu().numpy()
            results[str(seed)][policy] = {'NMSE': float(channel_sq.mean()), 'NMAE': float(channel_abs.mean()),
                'transferred_NMSE': float(channel_sq[CHANGED].mean()), 'per_channel_NMSE': channel_sq.tolist(),
                'per_channel_NMAE': channel_abs.tolist(), 'final_checkpoint_sha256': receipt['assets']['final.pt']}
        for policy in ('source_half', 'source_half_scramble', 'own_half'):
            for c in unchanged:
                for branch in ('seasonal', 'trend'):
                    for field in ('weight', 'bias'):
                        key = f'{branch}.{c}.{field}'
                        if not torch.equal(states['baseline'][key], states[policy][key]):
                            raise ValueError(f'Unchanged branch learned different weights: {seed}/{policy}/{key}')
            verified.append({'seed': seed, 'policy': policy, 'unchanged_channels': unchanged, 'exact_final_weight_match': True})
    baseline_mean = np.mean([results[str(s)]['baseline']['NMSE'] for s in (3026, 3027, 3028)])
    candidate_mean = np.mean([results[str(s)]['source_half']['NMSE'] for s in (3026, 3027, 3028)])
    own_mean = np.mean([results[str(s)]['own_half']['NMSE'] for s in (3026, 3027, 3028)])
    output = {'scope': 'post_confirmation_common_step_sensitivity', 'training_calls': 0, 'API_calls': 0,
        'fixed_step': 3000, 'D_windows': len(starts), 'results': results, 'unchanged_branch_checks': verified,
        'mean_NMSE_improvement_vs_baseline_pct': float(100 * (1 - candidate_mean / baseline_mean)),
        'mean_NMSE_improvement_vs_own_half_pct': float(100 * (1 - candidate_mean / own_mean)),
        'all_extra_seeds_improve_vs_baseline': all(results[str(s)]['source_half']['NMSE'] < results[str(s)]['baseline']['NMSE'] for s in (3026, 3027, 3028)),
        'all_extra_seeds_improve_vs_own_half': all(results[str(s)]['source_half']['NMSE'] < results[str(s)]['own_half']['NMSE'] for s in (3026, 3027, 3028)),
        'code_sha256': sha(__file__), 'elapsed_seconds': time.monotonic() - started,
        'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()},
        'limits': ['common-step audit does not replace preregistered best-checkpoint evaluation', 'no independent unseen dataset']}
    write(destination, output)
    print(json.dumps({'status': 'passed_common_step_audit', 'gain_pct': output['mean_NMSE_improvement_vs_baseline_pct'],
        'gain_vs_own_half_pct': output['mean_NMSE_improvement_vs_own_half_pct'], 'seconds': output['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
