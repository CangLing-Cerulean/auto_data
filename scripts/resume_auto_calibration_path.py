"""Explicit one-time recovery from a pre-LLM directory permission failure.

Preserves the frozen implementation and all five completed fits. No failed
model call is retried: the original call failed before request creation.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

import auto_calibration_trial as trial
from audit_auto_calibration import numpy_prediction

STAGE = trial.STAGE


def validate_completed_prefix():
    trial.check_lock()
    if trial.read(STAGE/'search_exit.json')['exit_code'] != 1:
        raise ValueError('expected original failed process')
    expected = {'IDENTITY', 'FIXED', 'RANDOM_01', 'RANDOM_02', 'RANDOM_03', 'RANDOM_04'}
    paths = {p.stem: p for p in (STAGE/'records').glob('*.json') if not p.stem.endswith('_decision')}
    if set(paths) != expected or set(p.stem for p in (STAGE/'models').glob('*.pt')) != expected:
        raise ValueError('completed prefix differs')
    if list((STAGE/'calls').iterdir()) or (STAGE/'selection_lock.json').exists() or (STAGE/'inputs/C.npz').exists():
        raise ValueError('LLM or future work already exists; recovery refused')
    records = {k: trial.read(p) for k, p in paths.items()}
    models = {k: trial.load_model(k, record['checkpoint_sha256']) for k, record in records.items()}
    validation = trial.load_samples(STAGE/'inputs/V.npz')
    max_error = 0.
    for identifier, model in models.items():
        state = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
        predicted = numpy_prediction(state, vars(validation))
        saved = np.load(STAGE/'predictions'/f'{identifier}_V.npy', allow_pickle=False)
        np.testing.assert_allclose(saved, predicted, rtol=2e-5, atol=2e-6)
        error = saved.astype(np.float64)-validation.truth
        np.testing.assert_allclose(np.mean(error**2), records[identifier]['validation']['mse'], rtol=1e-12, atol=1e-12)
        max_error = max(max_error, float(np.max(np.abs(saved-predicted))))
    common = trial.choose_incumbent(records, 'IDENTITY', 'FIXED')
    incumbent, history = common, []
    for slot, recipe in enumerate(trial.random_recipes(), 1):
        identifier = f'RANDOM_{slot:02}'
        record = records[identifier]
        if record['recipe'] != trial.asdict(recipe):
            raise ValueError('frozen random recipe mismatch')
        if recipe.initialization == 'incumbent' and record['parent'] != incumbent:
            raise ValueError('random lineage mismatch')
        incumbent = trial.choose_incumbent(records, incumbent, identifier)
        decision = trial.read(STAGE/'records'/f'{identifier}_decision.json')
        if decision['incumbent'] != incumbent:
            raise ValueError('random decision mismatch')
        history.append(decision)
    return records, models, common, incumbent, history, max_error


def resume():
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    records, models, common, random_selected, random_history, max_error = validate_completed_prefix()
    # Within the authorized project; no source values are placed in this folder.
    trial.CALL_ROOT = STAGE/'private/model_calls'
    trial.CALL_ROOT.mkdir(parents=True, exist_ok=False)
    recovery = dict(status='validated_before_first_llm', original_source_commit='5935a6162b1914e09eab6a6e8d5233b0ad55e928',
                    original_execution_lock_sha256=trial.sha(STAGE/'execution_lock.json'),
                    recovery_source_sha256=trial.sha(Path(__file__)),
                    preserved_prefix_records={k: trial.sha(STAGE/'records'/f'{k}.json') for k in records},
                    preserved_calibrator_fits=5, prior_llm_calls=0,
                    changed_only='call working directory and explicit continuation from saved prefix',
                    call_root=str(trial.CALL_ROOT), prefix_forward_max_error=max_error)
    trial.write(STAGE/'recovery_started.json', recovery)
    fitting = trial.load_samples(STAGE/'inputs/F.npz')
    validation = trial.load_samples(STAGE/'inputs/V.npz')
    trial.ac.verify_time_order(fitting, validation)
    incumbent, history, used = common, [], set()
    for slot in range(1, trial.PROPOSALS+1):
        identifier = f'AGENT_{slot:02}'
        recipe, reason, transport = trial.propose(trial.build_prompt(fitting, validation, records, history, incumbent, slot), slot)
        if recipe is None or recipe.key() in used:
            entry = dict(id=identifier, status='invalid_or_duplicate', reason=reason,
                         transport=transport, incumbent=incumbent)
        else:
            used.add(recipe.key())
            trial.train_and_record(identifier, recipe, incumbent, models, records, fitting, validation)
            incumbent = trial.choose_incumbent(records, incumbent, identifier)
            entry = dict(status='completed', **trial.public_record(records[identifier]),
                         reason=reason, transport=transport, incumbent=incumbent)
        history.append(entry)
        trial.write(STAGE/'records'/f'{identifier}_decision.json', entry)
        print(json.dumps({'arm': 'AGENT', 'slot': slot, 'status': entry['status'],
                          'incumbent': incumbent, 'validation_mse': records[incumbent]['validation']['mse']}), flush=True)
    for identifier, digest in recovery['preserved_prefix_records'].items():
        if trial.sha(STAGE/'records'/f'{identifier}.json') != digest:
            raise ValueError('preserved prefix changed')
    selected = dict(RANDOM=random_selected, AGENT=incumbent, IDENTITY='IDENTITY', FIXED='FIXED')
    lock = dict(status='all_paths_complete_future_unscored', selected=selected,
                selected_checkpoints={k: records[k]['checkpoint_sha256'] for k in set(selected.values())},
                paths={'RANDOM': random_history, 'AGENT': history}, total_search_seconds=None,
                resumed_function_seconds=time.perf_counter()-started,
                initial_process_wall_seconds=5.0724087,
                total_wall_note='Initial process timer includes interpreter startup; resumed function timer excludes it. Exact uninterrupted search wall time is unavailable; phase-level training/validation/LLM times are retained.',
                execution_lock_sha256=trial.sha(STAGE/'execution_lock.json'),
                recovery_started_sha256=trial.sha(STAGE/'recovery_started.json'),
                actual_calibrator_fits=len(records)-1,
                actual_new_updates=sum(r.get('training', {}).get('steps', 0) for r in records.values()),
                new_backbone_training=0, future_values_read=False)
    trial.write(STAGE/'selection_lock.json', lock)


if __name__ == '__main__':
    resume()
