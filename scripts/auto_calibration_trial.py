"""Frozen F/V recipe search; C is unlocked only after both searches finish."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from online_ts import auto_calibration as ac
from prepare_auto_calibration import STAGE, load_samples, read, sha, write, prepare

SEED = 2026102101
PROPOSALS = 4
MODEL = 'gpt-6.1-sol'
CODEX = Path('C:/Users/31227/AppData/Local/OpenAI/Codex/bin/896a0166b774037a/codex.exe')
CALL_ROOT = Path('C:/Users/31227/.codex/visualizations/2026/10/11/01a12b12-9990-74c3-92f2-32f6f65c592f/auto_calibration_calls')
SOURCE_PATHS = ['online_ts/auto_calibration.py', 'scripts/prepare_auto_calibration.py',
                'scripts/auto_calibration_trial.py', 'scripts/audit_auto_calibration.py', 'tests/test_auto_calibration.py',
                'tests/test_auto_calibration_trial.py']
DISABLED = ['shell_tool', 'unified_exec', 'multi_agent', 'apps', 'plugins', 'remote_plugin',
            'memories', 'browser_use', 'browser_use_external', 'computer_use', 'in_app_browser',
            'image_generation', 'view_image', 'skill_search', 'hooks', 'worktrees',
            'workspace_dependencies', 'code_mode_host']


def recipe_space():
    return [ac.Recipe(*v) for v in itertools.product(ac.WINDOWS, ac.LEARNING_RATES,
            ac.UPDATE_STEPS, ac.LOSSES, ac.WEIGHT_DECAYS, ac.INITIALIZATIONS)]


def random_recipes():
    recipes = recipe_space()
    ids = np.random.default_rng(SEED).choice(len(recipes), PROPOSALS, replace=False)
    return [recipes[i] for i in ids]


def schema():
    choices = dict(window=ac.WINDOWS, learning_rate=ac.LEARNING_RATES, steps=ac.UPDATE_STEPS,
                   loss=ac.LOSSES, weight_decay=ac.WEIGHT_DECAYS, initialization=ac.INITIALIZATIONS)
    properties = {k: {'type': 'integer' if k == 'steps' else ('number' if k in
        ('learning_rate', 'weight_decay') else 'string'), 'enum': list(v)} for k, v in choices.items()}
    return {'type': 'object', 'additionalProperties': False, 'required': ['recipe', 'reason'],
            'properties': {'recipe': {'type': 'object', 'additionalProperties': False,
            'required': list(choices), 'properties': properties}, 'reason': {'type': 'string'}}}


def choose_incumbent(records, current, candidate):
    # Strict decrease; ties preserve the earlier incumbent. No scientific effect gate.
    return candidate if records[candidate]['validation']['mse'] < records[current]['validation']['mse'] else current


def source_hashes():
    return {p: sha(ROOT/p) for p in SOURCE_PATHS}


def freeze():
    receipt = read(STAGE/'input_receipt.json')
    for split in ('F', 'V'):
        if sha(STAGE/f'inputs/{split}.npz') != receipt['splits'][split]['sha256']:
            raise ValueError('input changed')
    lock = dict(status='ready_unstarted', seed=SEED, proposals_per_arm=PROPOSALS,
                max_calibrator_fits=1+2*PROPOSALS, max_new_adam_updates=1440,
                new_backbone_training=0, common_recipe=asdict(ac.Recipe()),
                random_recipes=[asdict(r) for r in random_recipes()], model=MODEL,
                reasoning_effort='high', protocol_sha256=sha(STAGE/'PROTOCOL.md'),
                sources=source_hashes(), input_receipt_sha256=sha(STAGE/'input_receipt.json'),
                agent_prompt_sha256=sha(STAGE/'AGENT_PROMPT.txt'),
                disabled_cli_features=DISABLED, evaluation='normalized_mse; stable incumbent ties')
    write(STAGE/'execution_lock.json', lock)
    print(json.dumps(lock, indent=2))


def check_lock():
    lock = read(STAGE/'execution_lock.json')
    if lock['sources'] != source_hashes() or lock['protocol_sha256'] != sha(STAGE/'PROTOCOL.md'):
        raise ValueError('source or protocol differs from pre-execution lock')
    if lock['agent_prompt_sha256'] != sha(STAGE/'AGENT_PROMPT.txt'):
        raise ValueError('agent prompt changed')
    if lock['input_receipt_sha256'] != sha(STAGE/'input_receipt.json'):
        raise ValueError('input receipt changed')
    receipt = read(STAGE/'input_receipt.json')
    for split in ('F', 'V'):
        if sha(STAGE/f'inputs/{split}.npz') != receipt['splits'][split]['sha256']:
            raise ValueError('prepared input changed')


def diagnostics(samples):
    error = samples.prediction.astype(np.float64)-samples.truth
    midpoint = len(error)//2
    return {'windows': len(error), 'channel_mean_error': error.mean(axis=(0, 1)).tolist(),
            'channel_mse': np.mean(error**2, axis=(0, 1)).tolist(),
            'first_half_mse': float(np.mean(error[:midpoint]**2)),
            'second_half_mse': float(np.mean(error[midpoint:]**2))}


def public_record(record):
    return {k: record[k] for k in ('id', 'recipe', 'parent', 'fit', 'validation', 'lineage_updates')}


def build_prompt(fitting, validation, records, history, incumbent, slot):
    payload = dict(slot=slot, proposal_limit=PROPOSALS,
                   diagnostics={'F': diagnostics(fitting), 'V': diagnostics(validation)},
                   common=[public_record(records[k]) for k in ('IDENTITY', 'FIXED')],
                   own_history=history, current_incumbent=incumbent,
                   current=public_record(records[incumbent]))
    return (STAGE/'AGENT_PROMPT.txt').read_text(encoding='utf-8')+'\n'+json.dumps(payload, allow_nan=False)


def validate_events(events):
    completed = False
    for line in events.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get('type') == 'turn.failed':
            raise RuntimeError('model turn failed')
        if event.get('type') == 'turn.completed':
            completed = True
        item = event.get('item', {})
        if item and item.get('type') not in ('reasoning', 'agent_message'):
            raise RuntimeError('tool or unknown item invalidates the information-isolated trial')
    if not completed:
        raise RuntimeError('missing completed model turn')


def propose(prompt, slot):
    call = CALL_ROOT/f'AGENT_{slot:02}'
    call.mkdir(parents=True, exist_ok=False)
    log = STAGE/'calls'/call.name
    log.mkdir(parents=True, exist_ok=False)
    (log/'prompt.txt').write_text(prompt, encoding='utf-8')
    write(call/'schema.json', schema())
    command = [str(CODEX), 'exec', '--ignore-user-config', '--ephemeral', '--skip-git-repo-check',
               '-s', 'read-only', '-C', str(call), '-m', MODEL, '-c', 'model_reasoning_effort="high"',
               '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
               '-c', 'features.skip_host_skill_discovery=true', '-c', 'features.unbounded_connection_retries=false',
               '--json', '--output-schema', str(call/'schema.json'), '-o', str(call/'response.json')]
    for feature in DISABLED:
        command += ['--disable', feature]
    command += ['-']
    write(log/'request.json', {'command': command, 'model': MODEL, 'effort': 'high'})
    started = time.perf_counter()
    with (log/'events.jsonl').open('w', encoding='utf-8') as out, (log/'stderr.txt').open('w', encoding='utf-8') as err:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True,
                                   encoding='utf-8', cwd=call)
        try:
            process.communicate(prompt, timeout=600)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            write(log/'transport.json', dict(status='timeout', seconds=time.perf_counter()-started))
            raise RuntimeError('model call timed out; preserve original ID, no automatic retry')
    transport = dict(status='completed' if process.returncode == 0 else 'failed',
                     returncode=process.returncode, seconds=time.perf_counter()-started,
                     stderr_sha256=sha(log/'stderr.txt'), events_sha256=sha(log/'events.jsonl'))
    write(log/'transport.json', transport)
    if process.returncode != 0 or not (call/'response.json').exists():
        raise RuntimeError('model transport failed; no automatic retry')
    validate_events((log/'events.jsonl').read_text(encoding='utf-8'))
    raw = (call/'response.json').read_text(encoding='utf-8')
    (log/'response.json').write_text(raw, encoding='utf-8')
    try:
        response = json.loads(raw)
        if set(response) != {'recipe', 'reason'} or not isinstance(response['reason'], str):
            raise ValueError('response schema mismatch')
        recipe = ac.Recipe.from_dict(response['recipe'])
        return recipe, response['reason'], transport
    except (ValueError, TypeError, KeyError) as error:
        return None, f'invalid response: {error}', transport


def save_model(path, model):
    with path.open('xb') as stream:
        torch.save(model.state_dict(), stream)


def train_and_record(identifier, recipe, parent, models, records, fitting, validation):
    start = time.perf_counter()
    model, training = ac.train_candidate(fitting, recipe, seed=SEED, incumbent=models[parent])
    fit_start = time.perf_counter()
    fit, _ = ac.evaluate(model, fitting)
    fit_seconds = time.perf_counter()-fit_start
    validation_start = time.perf_counter()
    val, pred = ac.evaluate(model, validation)
    validation_seconds = time.perf_counter()-validation_start
    own_updates = recipe.steps
    inherited = records[parent]['lineage_updates'] if recipe.initialization == 'incumbent' else 0
    record = dict(id=identifier, recipe=asdict(recipe), parent=parent if inherited else 'IDENTITY',
                  fit=fit, validation=val, training=training, lineage_updates=inherited+own_updates,
                  fitting_evaluation_seconds=fit_seconds, validation_seconds=validation_seconds,
                  seconds_including_validation=time.perf_counter()-start)
    save_model(STAGE/'models'/f'{identifier}.pt', model)
    np.save(STAGE/'predictions'/f'{identifier}_V.npy', pred, allow_pickle=False)
    record['checkpoint_sha256'] = sha(STAGE/'models'/f'{identifier}.pt')
    write(STAGE/'records'/f'{identifier}.json', record)
    models[identifier], records[identifier] = model, record


def run_search():
    check_lock()
    # Fail rather than overwrite or silently replay any partial run.
    (STAGE/'records').mkdir()
    for directory in ('models', 'predictions', 'calls'):
        (STAGE/directory).mkdir()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    fitting, validation = (load_samples(STAGE/f'inputs/{s}.npz') for s in ('F', 'V'))
    ac.verify_time_order(fitting, validation)
    identity = ac.GatedLinearCalibrator(96, 7, SEED)
    identity_fit_start = time.perf_counter()
    fit, _ = ac.evaluate(identity, fitting)
    identity_fit_seconds = time.perf_counter()-identity_fit_start
    identity_validation_start = time.perf_counter()
    val, pred = ac.evaluate(identity, validation)
    identity_validation_seconds = time.perf_counter()-identity_validation_start
    save_model(STAGE/'models/IDENTITY.pt', identity)
    np.save(STAGE/'predictions/IDENTITY_V.npy', pred, allow_pickle=False)
    records = {'IDENTITY': dict(id='IDENTITY', recipe=None, parent=None, fit=fit, validation=val,
                fitting_evaluation_seconds=identity_fit_seconds, validation_seconds=identity_validation_seconds,
                lineage_updates=0, checkpoint_sha256=sha(STAGE/'models/IDENTITY.pt'))}
    models = {'IDENTITY': identity}
    write(STAGE/'records/IDENTITY.json', records['IDENTITY'])
    train_and_record('FIXED', ac.Recipe(), 'IDENTITY', models, records, fitting, validation)
    common = choose_incumbent(records, 'IDENTITY', 'FIXED')
    selections, paths = {}, {}
    random_menu = random_recipes()
    for arm in ('RANDOM', 'AGENT'):
        incumbent, history, used = common, [], set()
        for slot in range(1, PROPOSALS+1):
            identifier = f'{arm}_{slot:02}'
            if arm == 'RANDOM':
                recipe, reason, transport = random_menu[slot-1], 'frozen random search', None
            else:
                recipe, reason, transport = propose(build_prompt(fitting, validation, records, history, incumbent, slot), slot)
            if recipe is None or recipe.key() in used:
                entry = dict(id=identifier, status='invalid_or_duplicate', reason=reason,
                             transport=transport, incumbent=incumbent)
            else:
                used.add(recipe.key())
                train_and_record(identifier, recipe, incumbent, models, records, fitting, validation)
                incumbent = choose_incumbent(records, incumbent, identifier)
                entry = dict(status='completed', **public_record(records[identifier]),
                             reason=reason, transport=transport, incumbent=incumbent)
            history.append(entry)
            write(STAGE/'records'/f'{identifier}_decision.json', entry)
            print(json.dumps({'arm': arm, 'slot': slot, 'status': entry['status'],
                              'incumbent': incumbent, 'validation_mse': records[incumbent]['validation']['mse']}), flush=True)
        selections[arm], paths[arm] = incumbent, history
    selections.update(IDENTITY='IDENTITY', FIXED='FIXED')
    lock = dict(status='all_paths_complete_future_unscored', selected=selections,
                selected_checkpoints={k: records[k]['checkpoint_sha256'] for k in set(selections.values())},
                paths=paths, total_search_seconds=time.perf_counter()-started,
                execution_lock_sha256=sha(STAGE/'execution_lock.json'),
                actual_calibrator_fits=len(records)-1,
                actual_new_updates=sum(r.get('training', {}).get('steps', 0) for r in records.values()),
                new_backbone_training=0, future_values_read=False)
    write(STAGE/'selection_lock.json', lock)


def load_model(identifier, expected):
    path = STAGE/'models'/f'{identifier}.pt'
    if sha(path) != expected:
        raise ValueError('selected checkpoint changed')
    model = ac.GatedLinearCalibrator(96, 7, SEED)
    model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'), strict=True)
    return model


def score_future():
    check_lock()
    lock = read(STAGE/'selection_lock.json')
    if lock['status'] != 'all_paths_complete_future_unscored' or (STAGE/'future_result.json').exists():
        raise ValueError('future is not eligible for first scoring')
    if lock['execution_lock_sha256'] != sha(STAGE/'execution_lock.json'):
        raise ValueError('selection lock refers to another experiment')
    # Check every model BEFORE preparing any future values.
    models = {k: load_model(k, v) for k, v in lock['selected_checkpoints'].items()}
    started = time.perf_counter()
    prepare(future=True)
    samples = load_samples(STAGE/'inputs/C.npz')
    ac.verify_time_order(load_samples(STAGE/'inputs/V.npz'), samples)
    scores = {}
    for identifier, model in models.items():
        scores[identifier], pred = ac.evaluate(model, samples)
        np.save(STAGE/'predictions'/f'{identifier}_C.npy', pred, allow_pickle=False)
    result = dict(status='completed', selected=lock['selected'], metrics=scores,
                  selection_lock_sha256=sha(STAGE/'selection_lock.json'),
                  seconds=time.perf_counter()-started, D_values_parsed=False,
                  new_backbone_training=0)
    write(STAGE/'future_result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'search', 'future'])
    action = parser.parse_args().action
    {'freeze': freeze, 'search': run_search, 'future': score_future}[action]()
