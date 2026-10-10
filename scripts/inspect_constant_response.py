"""Inspect saved linear coefficients analytically; no synthetic records or fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(folder):
    folder = Path(folder)
    destination = folder / 'constant_response.json'
    if destination.exists():
        raise ValueError('Do not overwrite coefficient analysis')
    identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
    guards = identity['protected_assets']
    paths = [Path(p) for p in guards if p.endswith('/best.pt')]
    records = {}
    for path in paths:
        if sha(path) != guards[str(path)]:
            raise ValueError('Baseline checkpoint changed')
        state = torch.load(path, map_location='cpu', weights_only=True)
        channels = {}
        for c, name in enumerate(result['channels']):
            gain = state[f'trend.{c}.weight'].double().sum(1).numpy()
            intercept = (state[f'trend.{c}.bias'].double() + state[f'seasonal.{c}.bias'].double()).numpy()
            record = {'gain_by_horizon': gain.tolist(), 'intercept_by_horizon': intercept.tolist()}
            if c >= 7:
                # Algebraic response at physical binary states, not generated sensor data.
                record['binary_state_constant_prediction_error'] = {}
                for level in (0, 1):
                    normalized = (level - result['data_audit']['mean'][c]) / result['data_audit']['scale'][c]
                    normalized = float(np.float32(normalized))
                    drift = (gain - 1) * normalized + intercept
                    record['binary_state_constant_prediction_error'][str(level)] = {
                        'normalized_observed_level': normalized, 'signed_drift_by_horizon': drift.tolist(),
                        'mean_squared_drift': float(np.square(drift).mean()), 'mean_absolute_drift': float(np.abs(drift).mean())}
            channels[name] = record
        records[path.parent.name] = channels
    output = {'scope': 'analytical_saved_model_response_only',
        'formula': 'constant standardized input a -> prediction gain[h]*a + intercept[h]; error for unchanged target = (gain[h]-1)*a+intercept[h]',
        'reason': 'constant input has zero seasonal component and unchanged moving-average trend',
        'training_calls': 0, 'api_calls': 0, 'generated_sensor_records': 0,
        'limitations': ['this exact architecture identity explains output drift but not why training learned those coefficients', 'no evidence yet that a data intervention removes drift without increasing transition error'],
        'code_sha256': sha(__file__), 'baseline_checkpoint_hashes': {str(p): guards[str(p)] for p in paths}, 'models': records}
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'completed', 'saved_models': len(records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    main(parser.parse_args().folder)
