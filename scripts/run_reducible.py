"""Reference-gap, adaptive reference-gap, and bounded batch-ranking controls."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT, DLinear, digest, evaluate, write_json
from reducible_sampling import (FOLDER, ROUND_LIMIT, WARMUP_STEPS, STEPS, BATCH_SIZE,
    CANDIDATE_SIZE, REFERENCE_BATCH_SIZE, REFERENCE_LR_RATIO, REFERENCE_SEED,
    SHUFFLE_SEED, used_seconds, load_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--origin', type=float, choices=[.6, .8, 1.], required=True)
    parser.add_argument('--policy', choices=['reference', 'rho', 'adarho', 'hard_rank', 'shuffle'], required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--reference')
    parser.add_argument('--control-source')
    args = parser.parse_args()
    for name in (args.run_id, args.reference, args.control_source):
        assert name is None or Path(name).name == name
    is_reference = args.policy == 'reference'
    if is_reference:
        assert args.seed == REFERENCE_SEED and args.reference is None and args.control_source is None
    assert (args.policy in ('rho', 'adarho')) == (args.reference is not None)
    assert (args.policy == 'shuffle') == (args.control_source is not None)
    run = ROOT / 'research_runs/results' / args.run_id
    assert not (run / 'result.json').exists(), 'Completed run cannot be overwritten'
    run.mkdir(exist_ok=True)
    protocol = ROOT / 'research_runs/selection/protocol.json'
    p = json.loads(protocol.read_text())
    assert p['steps'] == STEPS and p['batch_size'] == BATCH_SIZE
    assert p['input_length'] == 32 and p['prediction_length'] == 8 and p['channels'] == 15 and p['max_gap_seconds'] == 30
    identity = {'policy': args.policy, 'origin': args.origin, 'seed': args.seed,
        'run_type': 'auxiliary_reference' if is_reference else 'main_model',
        'manifest_sha256': digest(ROOT / 'data/metropt3_v1/manifest.json'),
        'protocol_sha256': digest(protocol), 'study_protocol_sha256': digest(FOLDER / 'PROTOCOL.md'),
        'code_sha256': digest(Path(__file__)), 'helper_sha256': digest(ROOT / 'scripts/reducible_sampling.py'),
        'core_sha256': digest(ROOT / 'scripts/run_research.py'),
        'group_labels_sha256': digest(ROOT / 'research_runs/cluster_selection/window_groups.npz'),
        'reference_run': args.reference, 'control_source_run': args.control_source,
        'round_limit_seconds': ROUND_LIMIT}
    for field, name in [('reference', args.reference), ('control_source', args.control_source)]:
        if name:
            source_run = run.parent / name
            result = json.loads((source_run / 'result.json').read_text())
            assert result['identity']['origin'] == args.origin
            if field == 'reference':
                assert result['identity']['run_type'] == 'auxiliary_reference'
            else:
                assert result['identity']['seed'] == args.seed and result['identity']['policy'] in ('rho', 'adarho', 'hard_rank')
            identity[field + '_result_sha256'] = digest(source_run / 'result.json')
            identity[field + '_checkpoint_sha256'] = digest(source_run / 'final.pt')
    if (run / 'identity.json').exists():
        assert json.loads((run / 'identity.json').read_text()) == identity
    else:
        write_json(run / 'identity.json', identity)
        for file, name in [(Path(__file__), 'code_snapshot.py'), (ROOT / 'scripts/reducible_sampling.py', 'helper_snapshot.py'),
                           (ROOT / 'scripts/run_research.py', 'core_snapshot.py'), (protocol, 'protocol.json'),
                           (FOLDER / 'PROTOCOL.md', 'study_protocol.md')]:
            (run / name).write_bytes(file.read_bytes())
    allowance = ROUND_LIMIT - used_seconds() - 20
    assert allowance > 0, 'Original shared budget exhausted'
    old = json.loads((run / 'status.json').read_text()).get('elapsed_seconds', 0) if (run / 'status.json').exists() else 0
    begun = time.monotonic()
    def status(phase, step=0):
        write_json(run / 'status.json', {'phase': phase, 'step': step, 'elapsed_seconds': old + time.monotonic() - begun})
    def check(step):
        if time.monotonic() - begun >= allowance:
            raise TimeoutError(f'Original shared budget exhausted at {step}')
    status('preparing')
    try:
        manifest, values, starts, ref_mask, cells, months, eval_values, future, groups, audit = load_data(args.origin, is_reference)
        write_json(run / 'data_audit.json', audit)
        torch.set_num_threads(4); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        write_json(run / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()})
        data = torch.from_numpy(values).cuda(); positions = torch.from_numpy(starts).cuda()
        offsets = torch.arange(40, device='cuda'); n = len(starts)
        eligible = np.flatnonzero(ref_mask) if is_reference else np.arange(n)
        assert len(eligible) > 0
        eligible_gpu = torch.from_numpy(eligible).cuda()
        cdf = torch.from_numpy(np.cumsum(np.full(len(eligible), 1 / len(eligible), dtype=np.float64))).cuda(); cdf[-1] = 1
        model = DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters()) == 7920
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
        reference_model = None; reference_optimizer = None
        if args.reference:
            reference_model = DLinear(p).cuda()
            reference_model.load_state_dict(torch.load(run.parent / args.reference / 'final.pt', map_location='cuda', weights_only=True))
            source_audit = json.loads((run.parent / args.reference / 'data_audit.json').read_text())
            assert source_audit['mean'] == audit['mean'] and source_audit['scale'] == audit['scale']
            if args.policy == 'adarho':
                reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=p['learning_rate'] * REFERENCE_LR_RATIO, weight_decay=p['weight_decay'])
        main_rng = torch.Generator(device='cuda').manual_seed(args.seed + 1000)
        proposal_rng = torch.Generator(device='cuda').manual_seed(args.seed + 2000)
        shuffle_rng = np.random.default_rng(SHUFFLE_SEED)
        counts = torch.zeros(n, device='cuda', dtype=torch.int64)
        proposed_counts = torch.zeros_like(counts); reference_counts = torch.zeros_like(counts)
        step0 = 0; history = []; best = float('inf'); best_step = None; best_metrics = None; assets = []
        if (run / 'latest.pt').exists():
            saved = torch.load(run / 'latest.pt', map_location='cuda', weights_only=False)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            main_rng.set_state(saved['main_rng'].cpu()); proposal_rng.set_state(saved['proposal_rng'].cpu())
            shuffle_rng.bit_generator.state = saved['shuffle_rng']
            counts = saved['counts']; proposed_counts = saved['proposed_counts']; reference_counts = saved['reference_counts']
            step0 = saved['step']; history = saved['history']; assets = saved['assets']
            best = saved['best']; best_step = saved['best_step']; best_metrics = saved['best_metrics']
            if reference_model is not None:
                reference_model.load_state_dict(saved['reference_model'])
            if reference_optimizer is not None:
                reference_optimizer.load_state_dict(saved['reference_optimizer'])
        for phase in range(step0, STEPS, 500):
            check(phase); current_trace = []; reference_trace = []; proposals = []; chosen_slots = []; reference_slots = []; losses = []
            frozen = None
            if args.control_source:
                with np.load(run.parent / args.control_source / f'phase_{phase:04d}.npz') as z:
                    frozen = {key: z[key] for key in ('candidates', 'selected_slots')}
            for step in range(phase + 1, phase + 501):
                check(step)
                u = torch.rand(BATCH_SIZE, device='cuda', dtype=torch.float64, generator=main_rng)
                first = eligible_gpu[torch.searchsorted(cdf, u, right=True)]
                if is_reference:
                    candidates = first
                else:
                    u_extra = torch.rand(BATCH_SIZE, device='cuda', dtype=torch.float64, generator=proposal_rng)
                    candidates = torch.cat((first, eligible_gpu[torch.searchsorted(cdf, u_extra, right=True)]))
                candidate_cpu = candidates.cpu().numpy().astype(np.int32)
                selected = torch.arange(BATCH_SIZE, device='cuda')
                reference_selected = torch.empty(0, dtype=torch.int64, device='cuda')
                batch = data[positions[candidates, None] + offsets]
                active = not is_reference and step > WARMUP_STEPS
                if active and args.policy == 'shuffle':
                    ix = step - phase - 1
                    assert np.array_equal(candidate_cpu, frozen['candidates'][ix])
                    cell_batch = cells[candidate_cpu]
                    permutation = np.arange(CANDIDATE_SIZE)
                    for cell in np.unique(cell_batch):
                        members = np.flatnonzero(cell_batch == cell)
                        permutation[members] = shuffle_rng.permutation(members)
                    slots = permutation[frozen['selected_slots'][ix]]
                    assert np.array_equal(np.bincount(cell_batch[slots], minlength=int(cells.max()) + 1),
                                          np.bincount(cell_batch[frozen['selected_slots'][ix]], minlength=int(cells.max()) + 1))
                    selected = torch.from_numpy(slots).cuda()
                elif active:
                    with torch.no_grad():
                        current_loss = (model(batch[:, :32]) - batch[:, 32:]).square().mean((1, 2))
                        ref_loss = torch.zeros_like(current_loss) if reference_model is None else (reference_model(batch[:, :32]) - batch[:, 32:]).square().mean((1, 2))
                        score = current_loss - ref_loss
                    assert torch.isfinite(score).all()
                    order = torch.argsort(score, descending=True, stable=True)
                    selected = order[:BATCH_SIZE]
                    if reference_optimizer is not None:
                        reference_selected = order[BATCH_SIZE:BATCH_SIZE + REFERENCE_BATCH_SIZE]
                    current_trace.append(current_loss.cpu().numpy()); reference_trace.append(ref_loss.cpu().numpy())
                optimizer.zero_grad(set_to_none=True)
                target_batch = batch[selected]
                loss = (model(target_batch[:, :32]) - target_batch[:, 32:]).square().mean()
                assert torch.isfinite(loss)
                loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
                if len(reference_selected):
                    reference_optimizer.zero_grad(set_to_none=True)
                    reference_batch = batch[reference_selected]
                    ref_loss_update = (reference_model(reference_batch[:, :32]) - reference_batch[:, 32:]).square().mean()
                    assert torch.isfinite(ref_loss_update)
                    ref_loss_update.backward(); reference_optimizer.step()
                    reference_counts.scatter_add_(0, candidates[reference_selected], torch.ones_like(reference_selected))
                counts.scatter_add_(0, candidates[selected], torch.ones_like(selected))
                proposed_counts.scatter_add_(0, candidates, torch.ones_like(candidates))
                proposals.append(candidate_cpu); chosen_slots.append(selected.cpu().numpy().astype(np.int16))
                if len(reference_selected): reference_slots.append(reference_selected.cpu().numpy().astype(np.int16))
            asset = run / f'phase_{phase:04d}.npz'
            np.savez_compressed(asset, candidates=np.stack(proposals), selected_slots=np.stack(chosen_slots),
                current_loss=np.asarray(current_trace, dtype=np.float32), reference_loss=np.asarray(reference_trace, dtype=np.float32),
                reference_slots=np.asarray(reference_slots, dtype=np.int16))
            assets.append({'phase_start': phase, 'file': asset.name, 'sha256': digest(asset)})
            row = {'step': phase + 500, 'training_loss': float(np.mean(losses))}
            if not is_reference and args.origin == 1:
                check(phase + 500)
                measured = evaluate(model, torch.from_numpy(eval_values).cuda(), torch.from_numpy(future).cuda(), groups, p, 'baseline')
                row['validation'] = measured['groups']['all']
                if measured['groups']['all']['NMSE'] < best:
                    best = measured['groups']['all']['NMSE']; best_step = phase + 500; best_metrics = measured
                    torch.save(model.state_dict(), run / 'best.pt')
            history.append(row)
            torch.save(model.state_dict(), run / f'model_{phase + 500:04d}.pt')
            if reference_model is not None: torch.save(reference_model.state_dict(), run / f'reference_{phase + 500:04d}.pt')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'main_rng': main_rng.get_state(),
                'proposal_rng': proposal_rng.get_state(), 'shuffle_rng': shuffle_rng.bit_generator.state,
                'counts': counts, 'proposed_counts': proposed_counts, 'reference_counts': reference_counts,
                'reference_model': reference_model.state_dict() if reference_model is not None else None,
                'reference_optimizer': reference_optimizer.state_dict() if reference_optimizer is not None else None,
                'step': phase + 500, 'history': history, 'assets': assets, 'best': best, 'best_step': best_step,
                'best_metrics': best_metrics}, run / 'latest.tmp')
            (run / 'latest.tmp').replace(run / 'latest.pt'); status('training', phase + 500)
            print(json.dumps(row), flush=True)
        assert counts.sum().item() == STEPS * BATCH_SIZE
        assert torch.all(counts <= proposed_counts)
        if is_reference: assert not counts[torch.from_numpy(~ref_mask).cuda()].any()
        if args.policy == 'adarho': assert reference_counts.sum().item() == (STEPS - WARMUP_STEPS) * REFERENCE_BATCH_SIZE
        np.savez_compressed(run / 'windows.npz', train_start=starts, probe_start=np.array([]) if future is None else future,
            reference_mask=ref_mask, cell=cells, months=months, draw_count=counts.cpu().numpy(),
            proposed_count=proposed_counts.cpu().numpy(), reference_update_count=reference_counts.cpu().numpy())
        torch.save(model.state_dict(), run / 'final.pt')
        result = {'run_id': args.run_id, 'identity': identity, 'dataset_id': manifest['dataset_id'],
            'history': history, 'steps_completed': STEPS, 'samples_presented': STEPS * BATCH_SIZE,
            'sensor_columns': manifest['sensor_columns'], 'phase_assets': assets, 'windows_sha256': digest(run / 'windows.npz'),
            'additional_reference_updates': int(reference_counts.sum().item()),
            'selection_forward_presentations': (STEPS - WARMUP_STEPS) * CANDIDATE_SIZE if args.policy in ('rho', 'adarho', 'hard_rank') else 0,
            'selection_reference_forward_presentations': (STEPS - WARMUP_STEPS) * CANDIDATE_SIZE if args.reference else 0}
        if not is_reference and args.origin < 1:
            check(STEPS)
            result['future'] = evaluate(model, data, torch.from_numpy(future).cuda(), groups, p, 'baseline')
        elif not is_reference:
            result.update(best_metrics); result['best_step'] = best_step
        result['elapsed_seconds'] = old + time.monotonic() - begun
        write_json(run / 'result.json', result); status('completed', STEPS)
        print(json.dumps({'completed': args.run_id, 'future': result.get('future'), 'validation': result.get('groups')}), flush=True)
    except BaseException:
        status('failed'); raise


if __name__ == '__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    main()
