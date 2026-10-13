"""Read-only explanation of a preserved failed audit; no training or D access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gamma(n, dtype):
    u = np.finfo(dtype).eps / 2
    return n * u / (1 - n * u)


def run(source, output):
    started = time.perf_counter()
    output.mkdir()
    plan = read(source / 'plan.json')
    pred = read(source / 'predictions.json')
    prep = read(source / 'preparation.json')
    lock = read(source / 'prediction_lock.json')
    assert lock['predictions_sha256'] == sha(source / 'predictions.json')
    assert pred['plan_sha256'] == lock['plan_sha256'] == sha(source / 'plan.json')
    hashes = {name: sha(source / name) for name in ('plan.json', 'predictions.json', 'prediction_lock.json', 'audit.log')}
    hashes.update(plan['code_sha256'])
    hashes.update(pred['assets_sha256'])
    hashes.update({k: v for k, v in prep['assets_sha256'].items() if k.startswith('prepared/')})
    for name, digest in hashes.items():
        assert sha(source / name) == digest, name
    assert sha(plan['core']) == plan['core_sha256']
    old = Path(plan['input_root'])
    assert sha(old / 'proposal_median3.py') == plan['inputs_sha256']['proposal_median3.py']
    core = load('core', plan['core'])
    proposal = load('proposal', old / 'proposal_median3.py')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    rows = []
    for task in plan['tasks']:
        tid = task['id']
        with np.load(source / 'prepared' / f'{tid}_ABC.npz', allow_pickle=False) as z:
            values = torch.tensor(z['C_values'], device='cuda')
            starts = torch.tensor(z['C_starts'], device='cuda')
        params = read(source / 'prepared' / f'{tid}_params.json')
        batch = values[starts[:, None] + torch.arange(40, device='cuda')]
        target = batch[:, 32:]
        model = core.DLinear(dict(plan['training'], channels=8)).cuda().eval()
        for seed in plan['seeds']:
            for action in ('P1', 'CP1'):
                x = proposal.transform(batch[:, :32], params, 'P1', action == 'CP1')
                channels = x.transpose(1, 2)
                trend = F.avg_pool1d(F.pad(channels, (12, 12), mode='replicate'), 25, stride=1)
                seasonal = channels - trend
                features = torch.cat([seasonal, torch.ones_like(seasonal[:, :, :1]),
                                      trend, torch.ones_like(trend[:, :, :1])], dim=2).double()

                def evaluate(state):
                    model.load_state_dict(state)
                    direct = model(x) - target
                    manual, high, scale = [], [], []
                    for c in range(8):
                        sw, sb, tw, tb = [state[f'{part}.{c}.{kind}'].cuda() for part, kind in
                                         (('seasonal', 'weight'), ('seasonal', 'bias'), ('trend', 'weight'), ('trend', 'bias'))]
                        manual.append(F.linear(seasonal[:, c], sw, sb) + F.linear(trend[:, c], tw, tb))
                        high.append(F.linear(seasonal[:, c].double(), sw.double(), sb.double()) +
                                    F.linear(trend[:, c].double(), tw.double(), tb.double()))
                        scale.append(F.linear(seasonal[:, c].double().abs(), sw.double().abs(), sb.double().abs()) +
                                     F.linear(trend[:, c].double().abs(), tw.double().abs(), tb.double().abs()))
                    assert torch.equal(direct, torch.stack(manual, dim=-1) - target)
                    high = torch.stack(high, dim=-1) - target.double()
                    scale = torch.stack(scale, dim=-1) + target.double().abs()
                    return direct.double(), high, scale

                for step in (500, 1000, 1500, 2000, 2500, 3000):
                    states = {}
                    for key, engine, operation in (('child', 'proxy', action), ('proxy_parent', 'proxy', 'PARENT'), ('real_parent', 'full', 'PARENT')):
                        path = source / 'runs' / f'DEV_{tid}_{engine}_{operation}_s{seed}' / f'step_{step}.pt'
                        snapshot = torch.load(path, weights_only=True, map_location='cpu')
                        state = snapshot['model']
                        coords = torch.stack([torch.cat([state[f'seasonal.{c}.weight'].T,
                            state[f'seasonal.{c}.bias'][None], state[f'trend.{c}.weight'].T,
                            state[f'trend.{c}.bias'][None]]) for c in range(8)])
                        assert torch.equal(coords, snapshot['original_coordinates'])
                        states[key] = state
                    states['paired'] = {k: states['real_parent'][k] + (states['child'][k] - states['proxy_parent'][k]) for k in states['child']}
                    exact = {k: states['real_parent'][k].double() + (states['child'][k].double() - states['proxy_parent'][k].double()) for k in states['child']}
                    evaluated = {k: evaluate(v) for k, v in states.items()}
                    # Separate float64 operator using packed original coordinates and the same features.
                    packed_exact = torch.stack([torch.cat([exact[f'seasonal.{c}.weight'].T, exact[f'seasonal.{c}.bias'][None],
                        exact[f'trend.{c}.weight'].T, exact[f'trend.{c}.bias'][None]]) for c in range(8)]).cuda()
                    ideal = torch.einsum('nci,cij->njc', features, packed_exact) - target.double()
                    ideal_scale = torch.einsum('nci,cij->njc', features.abs(), packed_exact.abs()) + target.double().abs()
                    for name, method in (('child', 'RAW'), ('paired', 'M1')):
                        with np.load(source / 'forecasts' / f'{tid}_{action}_{seed}_{method}' / f'step_{step}_C_errors.npz', allow_pickle=False) as z:
                            np.testing.assert_array_equal(evaluated[name][0].detach().cpu().numpy(), z['error'])
                    f = {k: v[0] for k, v in evaluated.items()}
                    h = {k: v[1] for k, v in evaluated.items()}
                    magnitude = sum(v[2] for v in evaluated.values()) + ideal_scale
                    gap = f['paired'] - f['child'] + f['proxy_parent'] - f['real_parent']
                    high_identity = ideal - h['child'] + h['proxy_parent'] - h['real_parent']
                    parameter_effect = h['paired'] - ideal
                    roundoff = {k: f[k] - h[k] for k in f}
                    explained = parameter_effect + roundoff['paired'] - roundoff['child'] + roundoff['proxy_parent'] - roundoff['real_parent']
                    remainder = gap - explained
                    bound64 = gamma(256, np.float64) * magnitude
                    high_pass = bool(torch.all(high_identity.abs() <= bound64))
                    account_pass = bool(torch.all(remainder.abs() <= bound64))
                    forward_pass = all(bool(torch.all(roundoff[k].abs() <= gamma(70, np.float32) * evaluated[k][2])) for k in f)
                    rhs = f['child'] - f['proxy_parent'] + f['real_parent']
                    original_bad = gap.abs() > (2e-6 + 2e-5 * rhs.abs())
                    def maximum(t):
                        return float(t.abs().max().item())
                    row = dict(task=tid, seed=seed, action=action, step=step, elements=gap.numel(),
                        original_violations=int(original_bad.sum().item()), gap_max=maximum(gap),
                        parameter_rounding_max=maximum(parameter_effect), forward_rounding_max={k: maximum(v) for k, v in roundoff.items()},
                        high_identity_max=maximum(high_identity), accounted_remainder_max=maximum(remainder),
                        high_identity_bound_pass=high_pass, accounting_bound_pass=account_pass, forward_bound_pass=forward_pass,
                        saved_residuals_exact=True, parameter_coordinates_exact=True, manual_forward_exact=True)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    assert high_pass and account_pass and forward_pass
    assert len(rows) == 108
    for name, digest in hashes.items():
        assert sha(source / name) == digest, name
    result = dict(status='diagnosis_completed', original_audit_status='FAILED_preserved', original_stage_accepted=False,
        original_selection_gate=False, diagnostic_checks_passed=True, cells=rows,
        first_original_failure=next((r for r in rows if r['original_violations']), None),
        original_failed_cells=sum(r['original_violations'] > 0 for r in rows), checked_input_hashes=hashes,
        input_files_unchanged=True, full_trainings_added=0, cumulative_full_trainings=331,
        D_access=False, test_access=False, seconds=time.perf_counter() - started)
    with (output / 'diagnosis.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with torch.no_grad():
        run(Path(args.source), Path(args.output))
