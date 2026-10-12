"""Read-only audit of R2 execution evidence and costs; no new evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'runtime_deps'))
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    folder = Path(args.folder)
    with sqlite3.connect((folder/'run/runtime.sqlite').as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('SELECT kind,payload FROM agent_runtime_records WHERE task_key=? ORDER BY created_ns,record_id',
            ('native-research:solhigh_memory_r2',)).fetchall()
    records = {}
    for kind, payload in rows:
        records.setdefault(kind, []).append(json.loads(payload))
    trainings = records.get('research_training', [])
    table = []
    verified_assets = 0
    paired = {}
    for row in trainings:
        entry = {k: row[k] for k in ('candidate_id','job_id','status','elapsed_seconds','error') if k in row}
        if row['status'] == 'completed':
            job = folder/'run/training'/row['job_id']
            result = read(job/'result.json')
            assert sha(job/'result.json') == row['result_sha256']
            for name, expected in result['assets'].items():
                assert sha(job/name) == expected, name
                verified_assets += 1
            assert result['parameters'] == 7920 and result['steps'] == 3000
            assert result['metrics']['windows'] == 300535
            assert result['config'] == dict(batch_size=256, learning_rate=.001, moving_average_kernel=25, weight_decay=0.)
            with np.load(job/'sampling.npz', allow_pickle=False) as arrays:
                assert arrays['draw_count'].sum() == 768000
                assert len(arrays['start_row']) == 1053433
                h = lambda name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
                entry.update(start_row_sha256=h('start_row'), probability_sha256=h('probability'),
                    draw_count_sha256=h('draw_count'), loss_weight_sha256=h('loss_weight'))
            entry.update(seed=result['identity']['seed'], metrics=result['metrics'], best_step=result['best_step'],
                per_channel_NMSE=result['per_channel_NMSE'], per_channel_NMAE=result['per_channel_NMAE'])
            scope = read(job/'scope_metrics.json')
            final_step = scope['groups']['last.pt']['all']
            entry['fixed_step_3000_metrics'] = dict(NMSE=float(np.mean(final_step['per_channel_NMSE'])),
                NMAE=float(np.mean(final_step['per_channel_NMAE'])), windows=final_step['windows'])
            paired.setdefault(str(entry['seed']), []).append(entry)
        table.append(entry)
    assert len({r['job_id'] for r in table}) == len(table)
    for seed, group in paired.items():
        assert len({r['start_row_sha256'] for r in group}) == 1
        by_probability = {}
        for r in group:
            by_probability.setdefault(r['probability_sha256'], []).append(r['draw_count_sha256'])
        assert all(len(set(v)) == 1 for v in by_probability.values()), 'Same seed/probability gave different sampling counts'
    executed = {}
    for kind, pattern in [('diagnostics','run/diagnostics/*/execution.json'),
            ('processing','run/processing/*/execution.json'), ('controls','run/external_control/*/execution.json'),
            ('processor_checks','run/processing/*/processor_check_execution.json'),
            ('control_processor_checks','run/external_control/*/processor_check_execution.json')]:
        values = [read(p) for p in folder.glob(pattern)]
        executed[kind] = dict(attempts=len(values), failures=sum(r['status'] != 'completed' for r in values),
            process_seconds=sum(r.get('elapsed_seconds', 0.) for r in values))
    costs = dict(completed_trainings=sum(r['status']=='completed' for r in table),
        failed_trainings=sum(r['status']=='failed' for r in table),
        training_process_seconds=sum(r.get('elapsed_seconds',0.) for r in trainings),
        evaluation_process_seconds=sum(r['elapsed_seconds'] for r in records.get('evaluation_audit', [])),
        tool_decisions=len(records.get('decision', [])), execution_categories=executed,
        external_API_calls=0, external_API_tokens=0, internal_model_cost=None,
        internal_model_cost_note='Platform does not expose internal Sol high token/price usage; not zero or free',
        baseline_total_trainings=217)
    costs['cumulative_completed_trainings'] = 217+costs['completed_trainings']
    candidates = [{k: r[k] for k in ('candidate_id','status','proposal','signal_id','post_result_update','receipt_sha256','error') if k in r}
        for r in records.get('processing_candidate', [])]
    summary = dict(status='execution_assets_verified', verified_assets=verified_assets,
        trainings=table, candidates=candidates, costs=costs,
        information_violations=records.get('information_violation', []),
        diagnostics=[{k: r[k] for k in ('record_id','status','purpose','error','execution_sha256') if k in r}
            for r in records.get('diagnostic_job', [])],
        final_policy=read(folder/'run/final_policy.json') if (folder/'run/final_policy.json').exists() else None,
        confirmation=read(folder/'confirmation_result.json') if (folder/'confirmation_result.json').exists() else None,
        scope='same-source development; no new evaluation and no test access')
    write(folder/'audit_summary.json', summary)
    write(folder/'cost_summary.json', costs)
    print(json.dumps({'status': summary['status'], 'assets': verified_assets, 'costs': costs}), flush=True)


if __name__ == '__main__':
    main()
