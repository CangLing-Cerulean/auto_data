"""Independent r11 development audit; no training, LLM calls, or sealed D reads.

Program fitting/materialization is replayed in isolated memory workers. Full B/C
inference follows the frozen device path; fixed-window float64 CPU checks do not
replay the optimizer. Scientific source semantics require a separate receipt.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import open_processing_runtime as runtime
import open_processing_materialize as materialize
import open_processing_trial as trial
from open_processing_reference import FEATURES, evidence
from audit_coldstart_trial import (close, reference_forward, audit_response,
                                   recommendation, REFERENCE_RTOL, REFERENCE_ATOL)


ARMS = ('STRONG', 'SEPARATE')


def equivalent(actual, expected):
    """Compare nested numeric metadata without overlooking keys or shapes."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        for key in expected:
            equivalent(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equivalent(a, b)
    elif isinstance(expected, float):
        close(actual, expected)
    else:
        assert actual == expected


def reference_predict(history, x):
    """Direct conjugate posterior, independently reconstructed from past queries."""
    x = np.asarray(x, dtype=np.float64)
    design = np.asarray([r['x'] for r in history], dtype=np.float64).reshape(-1, 9)
    target = np.asarray([r['y'] for r in history], dtype=np.float64).reshape(-1, 2)
    k = np.eye(9) + design.T @ design
    m = np.linalg.solve(k, design.T @ target)
    psi = np.eye(2) + target.T @ target - m.T @ k @ m
    df = len(history) + 2
    scale = (1 + x @ np.linalg.solve(k, x)) * psi / df
    np.linalg.cholesky(psi)
    np.linalg.cholesky(scale)
    return dict(n=len(history), df=df, loc=(x @ m).tolist(), scale=scale.tolist(),
                posterior=dict(K=k.tolist(), M=m.tolist(), Psi=psi.tolist(), nu=len(history)+3))


def inspect_call(decisions, arm, number, phase, expected_facts, response):
    path = decisions/'calls'/arm/f'{number:02d}_{phase}'
    receipt = runtime.read(path/'receipt.json')
    started = runtime.read(path/'started.json')
    assert receipt['status'] == 'PASS' and receipt['exit_code'] == 0
    assert not receipt['tool_items'] and not receipt['transport_errors']
    assert receipt['seconds'] >= 0
    assert runtime.sha(path/'prompt.txt') == started['prompt_sha256']
    assert runtime.sha(path/'schema.json') == started['schema_sha256']
    schema = trial.schemas()[0 if phase == 'measurement' else 1]
    assert runtime.read(path/'schema.json') == schema
    events = [json.loads(line) for line in (path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    completed = [e for e in events if e.get('type') == 'turn.completed']
    assert len(completed) == 1 and completed[0].get('usage') == receipt['usage']
    for event in events:
        assert event.get('type') not in ('error', 'turn.failed')
        assert event.get('item', {}).get('type') in (None, 'reasoning', 'agent_message')
    common = (decisions/'AGENT_INTERFACE.txt').read_text(encoding='utf-8-sig')
    prefix = common+'\nCURRENT_ARM '+arm+'\nCURRENT_PHASE '+phase+'\nFACTS\n'
    prompt = (path/'prompt.txt').read_text(encoding='utf-8')
    assert prompt.startswith(prefix)
    assert json.loads(prompt[len(prefix):]) == expected_facts
    assert runtime.read(path/'response.json') == response
    return path


def inspect_fitted_state(folder, state, fit_receipt):
    """Check stored state semantics and its original byte-level provenance."""
    saved = (folder/'fit_state.npz').read_bytes()
    metadata, saved_arrays = materialize.sandbox.unpack(saved)
    saved_state = materialize.sandbox.decode_state(metadata['state'], saved_arrays)
    fresh_tree, fresh_arrays = materialize.sandbox.encode_state(state)
    saved_tree, decoded_arrays = materialize.sandbox.encode_state(saved_state)
    assert fresh_tree == saved_tree and set(fresh_arrays) == set(decoded_arrays)
    assert set(saved_arrays) == set(decoded_arrays)
    for key, value in fresh_arrays.items():
        assert value.dtype == decoded_arrays[key].dtype
        assert value.shape == decoded_arrays[key].shape
        np.testing.assert_array_equal(value, decoded_arrays[key])
    state_hash = runtime.sha(folder/'fit_state.npz')
    original_fit = runtime.read(folder/'fit_receipt.json')
    assert original_fit['status'] == fit_receipt['status'] == 'PASS'
    assert original_fit['fitted_state_sha256'] == state_hash
    for key in ('source_sha256', 'rows_sha256', 'state_sha256', 'request_sha256'):
        assert original_fit[key] == fit_receipt[key]
    return state_hash


def inspect_apply_receipt(path, arrays, receipt):
    original = runtime.read(path)
    assert original['status'] == receipt['status'] == 'PASS'
    for key in ('source_sha256', 'rows_sha256', 'state_sha256', 'request_sha256', 'array_sha256'):
        assert original[key] == receipt[key]
    for name, value in arrays.items():
        assert original['array_sha256'][name] == materialize._array_hash(value)


def measurement_summary(values, parent_b):
    """Reconstruct A normalization and all published A/B diagnostic summaries."""
    mean = float(np.mean(values['A']))
    std = float(np.std(values['A']))
    scale = std if std > 0 else 1.
    assert np.isfinite([mean, std, scale]).all()
    normalized = {part:(value-mean)/scale for part, value in values.items()}
    assert all(np.isfinite(value).all() for value in normalized.values())
    descriptions = {part:dict(mean=float(np.mean(value)), std=float(np.std(value)),
                              quantiles=np.quantile(value, [0,.1,.25,.5,.75,.9,1]).tolist())
                    for part, value in normalized.items()}
    parent_loss = np.mean([np.mean(parent_b[seed].astype(np.float64)**2, axis=(1,2))
                           for seed in runtime.SEEDS], axis=0)
    value = normalized['B']
    bins = np.quantile(value, [.25,.5,.75])
    groups = np.searchsorted(bins, value, side='right')
    correlation = (float(np.corrcoef(value, parent_loss)[0,1])
                   if np.std(value) > 0 and np.std(parent_loss) > 0 else None)
    summary = dict(A=descriptions['A'], B=descriptions['B'],
                   parent_B_error_correlation=correlation,
                   parent_B_error_quartiles=[dict(count=int(np.sum(groups == i)),
                        mse=float(parent_loss[groups == i].mean()) if np.any(groups == i) else None)
                        for i in range(4)])
    return dict(mean=mean, std=std, scale=scale), summary


def replay_measurement(folder, rows, parent_b, result):
    source = (folder/'program.py').read_text(encoding='utf-8')
    state, fit_receipt = materialize.fit_program(source, 'measurement', rows['A'])
    state_hash = inspect_fitted_state(folder, state, fit_receipt)
    values = {}
    for part in 'AB':
        arrays, receipt = materialize.apply_program(source, 'measurement', rows[part], state)
        inspect_apply_receipt(folder/f'{part}_receipt.json', arrays, receipt)
        values[part] = arrays['measurement']
    normalization, summary = measurement_summary(values, parent_b)
    normalization['state_sha256'] = state_hash
    expected = dict(status='PASS', measurement=summary, fit_A=normalization,
                    source_sha256=runtime.sha(folder/'program.py'))
    equivalent({key:value for key, value in result.items() if key != 'seconds'}, expected)
    assert result['seconds'] >= 0


def replay_program(folder, rows):
    """No runtime.materialize call and no audit-created program/state files."""
    source = (folder/'program.py').read_text(encoding='utf-8')
    state, fit_receipt = materialize.fit_program(source, 'pipeline', rows['A'])
    state_hash = inspect_fitted_state(folder, state, fit_receipt)
    arrays = {}
    for part in 'ABC':
        arrays[part], receipt = materialize.apply_program(
            source, 'pipeline', rows[part], state,
            include_weights=part == 'A', include_target=part != 'A')
        inspect_apply_receipt(folder/f'{part}_materialize_receipt.json', arrays[part], receipt)
    sampling = arrays['A']['sampling']; raw_loss = arrays['A']['loss']
    p = sampling / sampling.sum()
    loss = raw_loss / (np.sum(p * raw_loss) / p.sum())
    effective = int(np.count_nonzero((sampling > 0) & (raw_loss > 0)))
    assert effective >= 256 and np.max(p)*len(p) <= 20 and np.max(loss) <= 20
    assert np.isfinite(p).all() and np.isfinite(loss).all()
    close(p.sum(), 1.); close(np.sum(p*loss)/p.sum(), 1.)
    stats = dict(n=len(p), positive_effective=effective, positive_fraction=effective/len(p),
                 ess_sampling=float(1/np.sum(p**2)), max_sampling_multiplier=float(p.max()*len(p)),
                 max_loss=float(loss.max()), loss_std_under_p=float(np.sqrt(np.sum(p*(loss-1)**2))))
    hashes = {part:{name:runtime.digest(arrays[part][name])
                    for name in ('model_input', 'offset', 'scale')} for part in 'ABC'}
    hashes.update(probabilities=runtime.digest(p), loss=runtime.digest(loss))
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return arrays, stats, hashes, identity, state_hash


def descriptors(rows, arrays, stats, parent_b):
    original = runtime.windows(rows['A']).astype(np.float64)
    denominator = max(float(np.sqrt(np.mean(original**2))), 1e-12)
    delta = arrays['A']['model_input'].astype(np.float64)-original
    desc = dict(input_relative_rms_change=float(np.sqrt(np.mean(delta**2))/denominator),
                offset_relative_rms=float(np.sqrt(np.mean(arrays['A']['offset'].astype(np.float64)**2))/denominator),
                mean_abs_log_scale=float(np.mean(np.abs(np.log(arrays['A']['scale'].astype(np.float64))))))
    mask = arrays['B']['target']
    contrast = []
    for seed in runtime.SEEDS:
        squared = parent_b[seed].astype(np.float64)**2
        contrast.append(float((squared[mask].mean()-squared[~mask].mean())/squared.mean()))
    x = [1., stats['positive_fraction'], stats['ess_sampling']/stats['n'],
         float(np.tanh(desc['input_relative_rms_change'])), float(np.tanh(desc['offset_relative_rms'])),
         float(np.tanh(desc['mean_abs_log_scale'])), float(np.tanh(stats['loss_std_under_p'])),
         float(np.tanh(np.mean(contrast))), float(mask.mean())]
    return dict(x=x, feature_names=list(FEATURES), processing_descriptors=desc,
                parent_B_contrast_by_seed=contrast, weight_stats=stats, p_B=float(mask.mean()),
                p_C=float(arrays['C']['target'].mean()), target_B_sha256=runtime.digest(mask),
                target_C_sha256=runtime.digest(arrays['C']['target']))


def inspect_mapped_residual(actual, reference, scale, offset, truth):
    """Propagate the fixed model tolerance through float32 output coordinates."""
    actual64 = actual.astype(np.float64)
    scale64 = scale.astype(np.float64)
    offset64 = offset.astype(np.float64)
    truth64 = truth.astype(np.float64)
    actual_error = actual*scale+offset-truth
    reference_error = reference*scale64+offset64-truth64
    bound = (np.abs(scale64)*(REFERENCE_ATOL+REFERENCE_RTOL*np.abs(reference))
             + 8*np.finfo(np.float32).eps*(np.abs(actual64*scale64)
                                          + np.abs(offset64)+np.abs(truth64)))
    difference = np.abs(actual_error.astype(np.float64)-reference_error)
    assert np.isfinite(difference).all() and np.isfinite(bound).all()
    assert np.all(difference <= bound), (
        f'Mapped residual exceeds propagated operand bound: max excess {float(np.max(difference-bound))}')


def inspect_run(root, name, seed, identity, rows, gpu, prepared, parent_initial, checks):
    folder = root/'runs'/name/str(seed)
    record = runtime.read(folder/'result.json')
    started = runtime.read(folder/'started.json')
    assert started['seed'] == record['seed'] == seed
    assert started['processing_identity'] == record['processing_identity'] == identity
    assert record['status'] in ('completed', 'invalid_training') and record['seconds'] > 0
    assert {p.name for p in folder.glob('*.pt')} == set(record['assets'])
    for filename, asset in record['assets'].items():
        assert Path(filename).name == filename
        assert runtime.sha(folder/filename) == asset['sha256']
        assert (folder/filename).stat().st_size == asset['bytes']
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=True)
    assert set(initial) == set(parent_initial)
    for key in initial:
        assert torch.equal(initial[key], parent_initial[key])
    grid = record['B_grid']; steps = [g['step'] for g in grid]
    assert steps == list(range(500, 500*(len(grid)+1), 500)) and len(grid) <= 6
    assert {p.name for p in folder.glob('step_*.pt')} == {f'step_{s:04d}.pt' for s in steps}
    assert isinstance(record['updates'], int) and 0 <= record['updates'] <= 3000
    assert all(step <= record['updates'] for step in steps)
    if record['status'] == 'completed':
        assert record['steps'] == record['updates'] == 3000 and len(grid) == 6
    else:
        assert record['updates'] <= record['attempted_step'] <= 3000 and record['error']
    for entry in grid:
        model = runtime.checkpoint_model(folder/f"step_{entry['step']:04d}.pt")
        residual = runtime.decoded_errors(model, gpu['B'], prepared['B'])
        scores = runtime.metrics(residual)
        assert scores == {key:entry[key] for key in scores}
        checks['full_partition_metric_checks'] += 1
        n = len(prepared['B']['model_input']); indices = [0, n//2, n-1]
        x = prepared['B']['model_input'][indices]
        ref = reference_forward(model.state_dict(), x.cpu().numpy())
        with torch.no_grad():
            actual = model(x).cpu().numpy()
        close(actual, ref, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
        scale = prepared['B']['scale'][indices].cpu().numpy()[:,None,:]
        offset = prepared['B']['offset'][indices].cpu().numpy()[:,None,:]
        truth = np.stack([rows['B'][i+32:i+40] for i in indices])
        inspect_mapped_residual(actual, ref, scale, offset, truth)
        checks['mapped_residual_operand_bound_checks'] += 1
        checks['independent_three_window_checkpoint_checks'] += 1
    if record['status'] == 'completed':
        assert record['best_step'] == min(grid, key=lambda g:g['mse'])['step']
        residual = runtime.decoded_errors(runtime.best_model(folder), gpu['C'], prepared['C'])
        assert runtime.metrics(residual) == record['C']
        checks['full_partition_metric_checks'] += 1
    checks['physical_runs'] += 1
    return record


def audit(root, decisions):
    start = time.perf_counter(); lock = runtime.read(root/'execution_lock.json')
    assert lock['execution_enabled']
    assert Path(__file__).name in lock['runtime_code_sha256']
    assert runtime.sha(Path(__file__)) == lock['runtime_code_sha256'][Path(__file__).name]
    for name, expected in lock['runtime_code_sha256'].items():
        assert runtime.sha(root/name) == expected
    for name, key in (('PROTOCOL.md', 'protocol_sha256'), ('AGENT_INTERFACE.txt', 'agent_interface_sha256')):
        assert runtime.sha(root/name) == runtime.sha(decisions/name) == lock[key]
    completed = runtime.read(decisions/'trial_completed.json')
    frozen = runtime.read(decisions/'final_recommendations.json')
    started = runtime.read(decisions/'trial_started.json')
    assert started['parent_facts_sha256'] == runtime.sha(decisions/'prepared_parent_facts.json')
    assert started['parent_reuse_receipt_sha256'] == runtime.sha(decisions/'parent_reuse_receipt.json')
    assert runtime.read(decisions/'parent_reuse_receipt.json')['status'] == 'PASS'
    reuse = runtime.read(decisions/'parent_reuse_receipt.json')
    preflight = runtime.read(root/'preflight_receipt.json')
    assert preflight['status'] == 'PASS'
    assert reuse['execution_lock_sha256'] == preflight['execution_lock_sha256'] == runtime.sha(root/'execution_lock.json')
    assert reuse['preflight_sha256'] == runtime.sha(root/'preflight_receipt.json')
    assert reuse['parent_root'] == lock['parent_root'] and reuse['parent_assets'] == lock['parent_assets']
    assert completed['status'] == 'completed' and frozen['status'] == 'FROZEN'
    for document in (completed, frozen):
        assert document['D_accessed'] is False and document['D_evaluation_pending'] is True
    assert completed['recommendations'] == frozen['recommendations']
    assert {p.name for p in (decisions/'snapshots').glob('*.json')} == {
        f'{arm}_{n:02d}.json' for arm in ARMS for n in range(1,9)}
    assert set(frozen['snapshots']) == {f'{arm}_08.json' for arm in ARMS}
    for name, expected in frozen['snapshots'].items():
        assert runtime.sha(decisions/'snapshots'/name) == expected
    runtime.configure(); parent, rows = runtime.load(root); gpu = runtime.tensors(rows)
    parent_b = {s:runtime.errors(runtime.best_model(parent/'runs/PARENT'/str(s)), gpu['B']) for s in runtime.SEEDS}
    parent_c = {s:runtime.errors(runtime.best_model(parent/'runs/PARENT'/str(s)), gpu['C']) for s in runtime.SEEDS}
    initials = {s:torch.load(parent/'runs/PARENT'/str(s)/'initial.pt', map_location='cpu', weights_only=True) for s in runtime.SEEDS}
    states = {arm:runtime.read(decisions/'snapshots'/f'{arm}_08.json') for arm in ARMS}
    context = {arm:dict(registry=[], history=[], reference_history=[], diagnostic_ledger={}) for arm in ARMS}
    checks = Counter(); statuses = Counter(); measurement_statuses = Counter()
    seen = {}; expected_runs = set(); expected_calls = set(); expected_queries = set(); expected_measurements = set()
    full_new = 0; partial_updates = 0; training_seconds = 0.
    for number in range(1,9):
        for arm in (ARMS if number % 2 else ARMS[::-1]):
            previous = context[arm]
            snapshot = runtime.read(decisions/'snapshots'/f'{arm}_{number:02d}.json')
            assert len(snapshot['history']) == number
            assert snapshot['history'][:-1] == previous['history']
            item = snapshot['history'][-1]; assert item['round'] == number
            assert states[arm]['history'][:number] == snapshot['history']
            facts = dict(round=number, maximum_rounds=8,
                         parent=trial.public_history(runtime.read(decisions/'prepared_parent_facts.json')),
                         registry=previous['registry'], history=trial.public_history(previous['history']))
            if arm == 'SEPARATE': facts['diagnostic_ledger'] = previous['diagnostic_ledger']
            expected_calls.add(inspect_call(decisions, arm, number, 'measurement', facts, item['measurement_proposal']))
            mf = item['measurement_feedback']; measurement_statuses[mf['status']] += 1
            measure_folder = root/'measurements'/arm/f'{number:02d}'; expected_measurements.add(measure_folder)
            assert runtime.read(measure_folder/'result.json') == mf
            assert runtime.read(measure_folder/'request.json') == dict(arm=arm, round=number, measurement=item['measurement_proposal']['measurement'])
            registry = list(previous['registry']); result = item['result']; statuses[result['status']] += 1
            if mf['status'] == 'PASS':
                proposal = item['measurement_proposal']['measurement']
                if proposal is not None:
                    assert runtime.sha(measure_folder/'program.py') == mf['source_sha256']
                    assert (measure_folder/'program.py').read_text(encoding='utf-8') == proposal['source']
                    replay_measurement(measure_folder, rows, parent_b, mf)
                    checks['materialized_measurements'] += 1
                    registry.append(dict(id=f'q{len(registry)+1}', **proposal,
                                         fit_A=mf['fit_A'], source_sha256=mf['source_sha256']))
                facts['registry'] = registry; facts['measurement_feedback'] = trial.public_history(mf)
                expected_calls.add(inspect_call(decisions, arm, number, 'processing', facts, item['decision']))
            else:
                assert mf['status'] == result['status'] == 'invalid_measurement'
                assert 'decision' not in item
            assert snapshot['registry'] == registry
            history = list(previous['reference_history']); ledger = json.loads(json.dumps(previous['diagnostic_ledger']))
            assert result['status'] in ('completed', 'invalid_training', 'invalid_program', 'duplicate_own_program', 'invalid_forecast', 'invalid_measurement')
            query_folder = root/'queries'/arm/f'{number:02d}'
            if result['status'] not in ('invalid_measurement', 'invalid_forecast'):
                expected_queries.add(query_folder)
                assert runtime.read(query_folder/'result.json') == result
                request = dict(arm=arm, round=number, reference_history=history, **item['decision'])
                assert runtime.read(query_folder/'request.json') == request
                assert (query_folder/'program.py').read_text(encoding='utf-8') == request['source']
            if result['status'] in ('completed', 'invalid_training'):
                arrays, stats, hashes, identity, state_hash = replay_program(query_folder, rows)
                for part in 'BC': assert 0 < int(arrays[part]['target'].sum()) < arrays[part]['target'].size
                pre = runtime.read(query_folder/'prequery.json')
                reconstructed = descriptors(rows, arrays, stats, parent_b)
                reconstructed.update(processing_identity=identity, effective_array_sha256=hashes,
                                     fit_state_sha256=state_hash, source_sha256=runtime.sha(query_folder/'program.py'))
                baseline = reference_predict(history, reconstructed['x'])
                reconstructed['reference_forecast'] = baseline
                equivalent(pre, reconstructed)
                assert identity == result['processing_identity'] and result['run'] == 'R_'+identity
                assert all(h['result'].get('processing_identity') != identity for h in previous['history'])
                reused = identity in seen
                assert result['reused_cross_arm'] == reused and result['logical_full_trainings'] == 3
                if not reused:
                    prepared = runtime.prepared_gpu(arrays); records = []
                    for seed in runtime.SEEDS:
                        record = inspect_run(root, result['run'], seed, identity, rows, gpu, prepared, initials[seed], checks)
                        records.append(record); expected_runs.add((result['run'], str(seed)))
                        training_seconds += record['seconds']
                        if record['status'] != 'completed': break
                    seen[identity] = records
                    del prepared
                records = seen[identity]
                new = 0 if reused else sum(r['status'] == 'completed' for r in records)
                partial = 0 if reused else sum(r['updates'] for r in records if r['status'] != 'completed')
                assert result['actual_new_full_trainings'] == new
                assert result['actual_new_partial_updates'] == partial
                close(result['logical_training_seconds'], sum(r['seconds'] for r in records))
                full_new += new; partial_updates += partial
                if result['status'] == 'invalid_training':
                    assert result['seed_records'] == records
                    assert len(records) < 3 or any(r['status'] != 'completed' for r in records)
                else:
                    assert len(records) == 3 and all(r['status'] == 'completed' for r in records)
                    assert result['prequery'] == pre and len(result['responses']) == 3
                    prepared_c = runtime.prepared_gpu({'C':arrays['C']})['C']
                    for seed, response in zip(runtime.SEEDS, result['responses']):
                        assert response['seed'] == seed
                        child = runtime.decoded_errors(runtime.best_model(root/'runs'/result['run']/str(seed)), gpu['C'], prepared_c)
                        audit_response(parent_c[seed], child, arrays['C']['target'], response)
                        checks['response_checks'] += 1
                    del prepared_c
                    for key in ('Y', 'Z', 'NMAE_gain'): close(result[key], float(np.mean([r[key] for r in result['responses']])))
                    equivalent(item['reference_forecast'], baseline)
                    ev = evidence(baseline, item['decision']['forecast'], result['Y'], result['Z'])
                    if arm == 'SEPARATE':
                        equivalent(item['conditional_evidence'], ev)
                        material = dict(source_sha256=pre['source_sha256'], fit_state_sha256=state_hash,
                                        diagnostic_claim=item['decision']['diagnostic_claim'],
                                        registry=[{key:q[key] for key in ('id','source_sha256','fit_A')} for q in registry])
                        canonical = json.dumps(material, sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False)
                        version = hashlib.sha256(canonical.encode()).hexdigest()
                        assert item['diagnostic_version'] == version
                        entry = ledger.setdefault(version, dict(definition=material, queries=[], sum_log_ratio=0.))
                        entry['queries'].append(number); entry['sum_log_ratio'] += ev['log_ratio']
                    else:
                        assert 'conditional_evidence' not in item and 'diagnostic_version' not in item
                    history.append(dict(x=pre['x'], y=[result['Y'],result['Z']]))
                checks['materialized_queries'] += 1
                del arrays
            assert snapshot['reference_history'] == history
            equivalent(snapshot['diagnostic_ledger'], ledger)
            if number in (2,4,8): assert item['recommendation'] == recommendation(snapshot['history'])
            context[arm] = snapshot
    for arm in ARMS: assert frozen['recommendations'][arm] == recommendation(context[arm]['history'])
    assert {(p.parent.parent.name,p.parent.name) for p in (root/'runs').glob('*/*/result.json')} == expected_runs
    assert {(p.parent.parent.name,p.parent.name) for p in (root/'runs').glob('*/*/started.json')} == expected_runs
    run_folders = set((root/'runs').iterdir()) if (root/'runs').exists() else set()
    assert {p for run in run_folders for p in run.iterdir()} == {root/'runs'/name/seed for name,seed in expected_runs}
    assert run_folders == {root/'runs'/name for name,seed in expected_runs}
    assert {p.parent for p in (decisions/'calls').glob('*/*/started.json')} == expected_calls
    assert {p for arm in (decisions/'calls').iterdir() for p in arm.iterdir()} == expected_calls
    assert {p.parent for p in (root/'queries').glob('*/*/result.json')} == expected_queries
    assert {p.parent for p in (root/'measurements').glob('*/*/result.json')} == expected_measurements
    assert len(expected_calls) <= 32 and full_new <= 48
    before = lock['cumulative_full_trainings_before']
    assert isinstance(before, int)
    return dict(status='PASS', llm_calls=len(expected_calls), query_status_counts=dict(statuses),
                measurement_status_counts=dict(measurement_statuses), completed_queries=statuses['completed'],
                invalid_attempts=16-statuses['completed'], logical_child_trainings=3*checks['materialized_queries'],
                actual_new_full_trainings=full_new, actual_new_partial_updates=partial_updates,
                parents_reused=3, cumulative_full_trainings_before=before,
                cumulative_full_trainings=before+full_new, physical_training_seconds=training_seconds,
                checks=dict(checks), reference_rtol=REFERENCE_RTOL, reference_atol=REFERENCE_ATOL,
                mapped_residual_operand_bound_checked=True,
                recommendations=frozen['recommendations'], seconds=time.perf_counter()-start,
                D_values_accessed=False, optimizer_trajectory_replayed=False,
                invalid_forecast_and_invalid_program_reexecuted=False,
                source_semantic_audit='Separate principal-agent receipt required: no additional fitted supervised predictor; AST checks cannot prove this constraint.',
                scientific_contribution_assessed=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Audit output is create-only'
    try:
        result = audit(args.root.resolve(), args.decisions.resolve())
    except Exception as exc:
        runtime.write(args.output, dict(status='FAILED', error=repr(exc)))
        raise
    runtime.write(args.output, result)
    print(json.dumps(result, allow_nan=False))
