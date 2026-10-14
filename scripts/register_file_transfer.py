"""Register source-file roles and every deletion using manifest metadata only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'research_runs/agent_research/intervention_diagnosis/20261014_file_transfer_r1'
MANIFEST_SHA = '372d99fd9188151d3429171c1978c3ea13a34b735bf3ec7d031d8b6845c5ecc2'
SALT = 'file-transfer-r1|'
SEEDS = [2026102001, 2026102002, 2026102003]


def registration(manifest):
    assert manifest['dataset_id'] == 'skab_v1'
    train = next(s for s in manifest['splits'] if s['name'] == 'train')
    coverage = train['series_coverage']
    assert len(coverage) == 23 and sum(s['rows'] for s in coverage) == train['rows'] == 32802
    assert len({s['series_id'] for s in coverage}) == 23
    files = sorted(coverage, key=lambda s: (hashlib.sha256((SALT + s['series_id']).encode()).hexdigest(), s['series_id']))
    entries = [dict(source_file=s['series_id'], rank=i, group=i % 3,
                    ordering_sha256=hashlib.sha256((SALT + s['series_id']).encode()).hexdigest(),
                    base_train_rows=s['rows'], source_row_start_inclusive=s['source_row_start_inclusive'],
                    source_row_end_exclusive=s['source_row_end_exclusive']) for i, s in enumerate(files)]
    tasks = []
    for group in range(3):
        roles = {split: sorted(e['source_file'] for e in entries if e['group'] == (group + offset) % 3)
                 for offset, split in enumerate(('A', 'B', 'C'))}
        actions = [dict(id=f'DROP_{i:02d}', removed_source_file=name) for i, name in enumerate(roles['A'])]
        tasks.append(dict(id=f'F{group}', files=roles, actions=actions))
    assert [len(t['actions']) for t in tasks] == [8, 8, 7]
    return dict(stage='file_transfer_r1', status='roles_and_full_menu_registered', dataset_id='skab_v1',
                manifest_sha256=MANIFEST_SHA, train_sha256=train['sha256'], train_rows=train['rows'],
                sensors=manifest['sensor_columns'], salt=SALT, grouping='SHA256 ordering, rank modulo three',
                files=entries, tasks=tasks, seeds=SEEDS, candidate_cells=23, candidate_seed_cells=69,
                complete_parents=9, complete_children=69, cumulative_before=409, expected_cumulative_after=487,
                sensor_tables_read=False, new_full_training_executed=False,
                scientific_scope='known-source development; source-file-disjoint roles within base train')


def main():
    path = ROOT / 'data/skab_v1/manifest.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST_SHA
    result = registration(json.loads(path.read_text(encoding='utf-8-sig')))
    with (STAGE / 'task_registration.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({'files': len(result['files']), 'actions': [len(t['actions']) for t in result['tasks']],
                      'full_runs': result['complete_parents'] + result['complete_children']}))


if __name__ == '__main__':
    main()
