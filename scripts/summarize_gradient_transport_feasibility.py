"""Audit preserved fixed-shape outputs and derive scope-limited summaries."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261015_gradient_transport_feasibility_r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = STAGE / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main():
    lock = json.loads((STAGE / 'source_lock.json').read_text(encoding='utf-8'))
    for item in lock['files']:
        assert sha(ROOT / item['path']) == item['sha256']
    result = json.loads((STAGE / 'kernel_check.json').read_text(encoding='utf-8'))
    assert result['status'] == 'PASS' and result['optimizer_updates'] == 0
    assert not result['real_arrays_accessed'] and not result['real_experiments_enabled']
    expected = {(s, d, k) for s in (92001, 92002, 92003)
                for d in ('float32', 'float64') for k in (1, 8, 24)}
    assert len(result['rows']) == 18
    assert {(r['seed'], r['dtype'], r['candidates']) for r in result['rows']} == expected
    for row in result['rows']:
        for name, values in row['samples_seconds'].items():
            assert len(values) == 7 and all(v > 0 for v in values)
            assert row['median_seconds'][name] == statistics.median(values)
        ratio = row['median_seconds']['parent_affine_plus_candidates'] / row['median_seconds']['direct_batch_gradient']
        assert ratio == row['pipeline_to_direct_ratio']
    sizes = result['storage']
    assert sizes['float32_original_q_plus_parent_bytes'] == 3000 * 5 * 4224 * 4
    assert sizes['float32_affine_only_bytes'] == 3000 * 4 * 4224 * 4
    assert sizes['float32_static_group_h_bytes'] == 4 * 8 * 66**2 * 4
    assert sizes['float32_full_batch_group_h_trace_bytes'] == 3000 * sizes['float32_static_group_h_bytes']
    assert sizes['float32_stream_affine_bytes'] == 4 * 4224 * 4
    summaries = []
    for dtype in ('float32', 'float64'):
        for count in (1, 8, 24):
            rows = [r for r in result['rows'] if r['dtype'] == dtype and r['candidates'] == count]
            ratios = [r['pipeline_to_direct_ratio'] for r in rows]
            summaries.append(dict(dtype=dtype, candidates=count,
                min_ratio=min(ratios), median_ratio=statistics.median(ratios), max_ratio=max(ratios)))
    write('summary.json', dict(kernel_sha256=sha(STAGE/'kernel_check.json'), groups=summaries,
        max_compact_error=max(r['compact_error'] for r in result['rows']),
        max_batch_transport_error=max(r['batch_transport_error'] for r in result['rows']),
        mac_ratios={str(k): ((2*256*8*66*8+4*8*66*66*8)+k*(8*66*66*8+4*8*66*8))/(2*k*256*8*66*8) for k in (1,8,24)},
        trace_bytes_saved=sizes['float32_original_q_plus_parent_bytes']-sizes['float32_affine_only_bytes']))
    write('artifact_audit.json', dict(status='PASS', rows=18, timing_series=72,
        samples=504, source_lock_unchanged=True, storage_formulas_checked=5,
        independent_reexecution=False, scope='identity_completeness_arithmetic_only',
        kernel_sha256=sha(STAGE/'kernel_check.json'), summary_sha256=sha(STAGE/'summary.json')))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
