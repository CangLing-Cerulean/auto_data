"""Independent local aggregation and identity checks against frozen JSON/assets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)


def main(stage, source):
    result = read(stage / 'attribution.json')
    assert result['status'] == 'PASS' and result['new_training_steps'] == 0
    checks = 0
    for name, digest in result['input_sha256'].items():
        assert sha(source / name) == digest, name
        checks += 1
    prediction = read(source / 'predictions.json')['records']
    truth = read(source / 'labels.json')['records']
    parents = read(source / 'parents.json')['records']
    spec = read(source / 'spec.json')
    expected = {(t, s, a) for t in spec['tasks'] for s in spec['seeds'] for a in spec['actions']}
    assert {tuple(r[k] for k in ('task', 'seed', 'action')) for r in result['candidates']} == expected
    assert len(result['candidates']) == 36 and len(result['points']) == 216
    summary, errors, decomposition = {}, {m: [] for m in ('RAW', 'M1', 'PREFIX')}, 0
    for row in result['candidates']:
        identity = tuple(row[k] for k in ('task', 'seed', 'action'))
        match = lambda r: tuple(r[k] for k in ('task', 'seed', 'action')) == identity
        full = next(r for r in truth if match(r))
        forecasts = next(r for r in prediction if match(r))
        parent = next(r for r in parents if (r['task'], r['seed']) == identity[:2])
        baseline = parent['selected_losses']['C']
        true_step = min(spec['grid'], key=lambda k: full['curves'][str(k)]['B'])
        close(row['true_gain_percent'], 100 * (1 - full['curves'][str(true_step)]['C'] / baseline))
        for method in errors:
            curve = row['raw_curves'] if method == 'RAW' else forecasts['methods'][method]['curves']
            step = min(spec['grid'], key=lambda k: curve[str(k)]['B'])
            value = row['methods'][method]
            assert step == value['selected_step']
            gain = 100 * (1 - curve[str(step)]['C'] / baseline)
            close(gain, value['gain_percent'])
            error = gain - row['true_gain_percent']
            close(error, value['response_error_pp'])
            close(error, value['approximation_component_pp'] + value['checkpoint_component_pp'])
            close(value['checkpoint_component_pp'], 100 * (full['curves'][str(true_step)]['C'] - full['curves'][str(step)]['C']) / baseline)
            errors[method].append(error)
            decomposition += 1
    for method in errors:
        close(mean(map(abs, errors[method])), result['summary']['ALL']['methods'][method]['mae_pp'])
        per_cell = []
        for task in spec['tasks']:
            for action in spec['actions']:
                selected = [r for r in result['candidates'] if (r['task'], r['action']) == (task, action)]
                assert len(selected) == 3
                per_cell.append(mean(r['methods'][method]['response_error_pp'] for r in selected))
        summary[method] = dict(seed_level_36_mae_pp=mean(map(abs, errors[method])),
                               seed_mean_12_mae_pp=mean(map(abs, per_cell)))
    g3 = [p for p in result['points'] if p['task'] == 'G3']
    report = dict(status='PASS', input_assets_hash_checked=checks, candidates=36, fixed_points=216,
        independent_response_and_decomposition_checks=decomposition,
        aggregation=summary,
        m1_selection_increment_over_raw_pp=result['mean_selected_gain_percent']['M1'] - result['mean_selected_gain_percent']['RAW'],
        g3_max_prediction_rounding=max(p['float32_additive_prediction_rounding_max'] for p in g3),
        g3_max_loss_rounding_pp=max(abs(p['float32_loss_rounding_pp']) for p in g3),
        source_code_sha256=sha(stage.parents[3] / 'scripts' / 'analyze_m1_contribution.py'),
        attribution_sha256=sha(stage / 'attribution.json'))
    with (stage / 'verification.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    main(args.stage.resolve(), args.source.resolve())
