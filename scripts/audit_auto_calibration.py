"""Independent NumPy re-score of saved forecasts and frozen search decisions."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STAGE = ROOT/'research_runs/agent_research/auto_calibration/20261021_r1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def numpy_prediction(state, data):
    y = data['prediction'].astype(np.float64)
    context = data['context'].astype(np.float64)
    output = np.empty_like(y)
    for channel in range(y.shape[2]):
        z = np.concatenate((y[:, :, channel], context[:, channel]), axis=1)
        output[:, :, channel] = y[:, :, channel] + state['bias'][channel] + \
            np.tanh(state['gate'][channel]) * (z @ state['weight'][channel])
    return output


def audit():
    started = time.perf_counter()
    selection = read(STAGE/'selection_lock.json')
    result = read(STAGE/'future_result.json')
    execution = read(STAGE/'execution_lock.json')
    assert result['selection_lock_sha256'] == sha(STAGE/'selection_lock.json')
    assert selection['execution_lock_sha256'] == sha(STAGE/'execution_lock.json')
    assert execution['protocol_sha256'] == sha(STAGE/'PROTOCOL.md')
    assert execution['agent_prompt_sha256'] == sha(STAGE/'AGENT_PROMPT.txt')
    for path, expected in execution['sources'].items():
        assert sha(ROOT/path) == expected
    inputs, checks, max_error = {}, [], 0.
    for name in ('F', 'V', 'C'):
        receipt = read(STAGE/('future_input_receipt.json' if name == 'C' else 'input_receipt.json'))
        assert sha(STAGE/f'inputs/{name}.npz') == receipt['splits'][name]['sha256']
        with np.load(STAGE/f'inputs/{name}.npz', allow_pickle=False) as file:
            inputs[name] = {k: file[k] for k in file.files}
    assert inputs['F']['label_end'].max() < inputs['V']['input_start'].min()
    assert inputs['V']['label_end'].max() < inputs['C']['input_start'].min()
    assert inputs['C']['label_end'].max() == 11519
    records = {p.stem: read(p) for p in (STAGE/'records').glob('*.json') if not p.stem.endswith('_decision')}
    for identifier, record in records.items():
        model_path = STAGE/'models'/f'{identifier}.pt'
        assert sha(model_path) == record['checkpoint_sha256']
        state = {k: v.numpy().astype(np.float64) for k, v in
                 torch.load(model_path, weights_only=True, map_location='cpu').items()}
        for split in ('V', 'C') if identifier in selection['selected_checkpoints'] else ('V',):
            data = inputs[split]
            expected = numpy_prediction(state, data)
            observed = np.load(STAGE/'predictions'/f'{identifier}_{split}.npy', allow_pickle=False)
            assert observed.shape == data['truth'].shape
            np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-6)
            max_error = max(max_error, float(np.max(np.abs(observed-expected))))
            metrics = record['validation'] if split == 'V' else result['metrics'][identifier]
            error = observed.astype(np.float64)-data['truth']
            assert hashlib.sha256(observed.tobytes()).hexdigest() == metrics['prediction_sha256']
            comparisons = dict(mse=np.mean(error*error), mae=np.mean(np.abs(error)),
                               channel_mse=np.mean(error*error, axis=(0, 1)),
                               horizon_mse=np.mean(error*error, axis=(0, 2)), mean_error=np.mean(error))
            for name, value in comparisons.items():
                np.testing.assert_allclose(value, metrics[name], rtol=1e-12, atol=1e-12)
            checks.append(f'{identifier}/{split}: independent forward, saved forecast, all metrics')
        if identifier != 'IDENTITY':
            recipe, train = record['recipe'], record['training']
            assert recipe['steps'] == train['steps']
            inherited = records[record['parent']]['lineage_updates'] if recipe['initialization'] == 'incumbent' else 0
            assert record['lineage_updates'] == inherited+recipe['steps']
            if recipe['initialization'] == 'incumbent':
                parent = records[record['parent']]
                parent_state = torch.load(STAGE/'models'/f"{parent['id']}.pt", weights_only=True, map_location='cpu')
                digest = hashlib.sha256()
                for key, tensor in parent_state.items():
                    digest.update(key.encode())
                    digest.update(tensor.contiguous().numpy().tobytes())
                assert digest.hexdigest() == train['initial_state_sha256']
    common = 'FIXED' if records['FIXED']['validation']['mse'] < records['IDENTITY']['validation']['mse'] else 'IDENTITY'
    for arm in ('RANDOM', 'AGENT'):
        current, used = common, set()
        assert len(selection['paths'][arm]) == 4
        for slot, entry in enumerate(selection['paths'][arm], 1):
            assert entry == read(STAGE/'records'/f'{arm}_{slot:02}_decision.json')
            if entry['status'] == 'completed':
                key = json.dumps(entry['recipe'], sort_keys=True)
                assert key not in used
                used.add(key)
                record = records[entry['id']]
                if record['recipe']['initialization'] == 'incumbent':
                    assert record['parent'] == current
                if record['validation']['mse'] < records[current]['validation']['mse']:
                    current = entry['id']
                if arm == 'RANDOM':
                    assert entry['recipe'] == execution['random_recipes'][slot-1]
            assert entry['incumbent'] == current
        assert selection['selected'][arm] == current
    fits = len(records)-1
    updates = sum(r.get('training', {}).get('steps', 0) for r in records.values())
    assert fits == selection['actual_calibrator_fits'] <= 9
    assert updates == selection['actual_new_updates'] <= 1440
    assert len(list((STAGE/'calls').glob('AGENT_*'))) == 4
    assert selection['new_backbone_training'] == result['new_backbone_training'] == 0
    for identifier, expected in selection['selected_checkpoints'].items():
        assert records[identifier]['checkpoint_sha256'] == expected
    report = dict(status='passed', checked_forecast_sets=len(checks), checks=checks,
                  actual_calibrator_fits=fits, actual_new_updates=updates,
                  numpy_forward_max_error=max_error, independent_metric_rtol=1e-12,
                  forward_rtol=2e-5, forward_atol=2e-6,
                  selection_budget_chronology_and_lineage='passed', seconds=time.perf_counter()-started)
    with (STAGE/'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    audit()
