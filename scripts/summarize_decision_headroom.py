"""Read only disclosed numerical summaries; no torch, datasets or remote access.

Primary scope: the three complete selection pools cited in the research review.
Weighted and joint Hydraulic pools are nested sensitivity views, not new tasks.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT = BASE/'20261015_paper_evidence_r1'
SOURCES = {}


def contents(relative):
    path = BASE/relative
    raw = path.read_bytes()
    SOURCES[relative] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw.decode('utf-8-sig')


def read(relative):
    return json.loads(contents(relative))


def csv_rows(relative):
    return list(csv.DictReader(io.StringIO(contents(relative))))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def summary(pool, task, candidates, seeds, estimand, controls=None, nested=False):
    assert candidates and 'PARENT' not in candidates
    with_parent = dict(PARENT=0., **candidates)
    chosen = max(with_parent, key=with_parent.get)
    oracle = with_parent[chosen]
    ordered = sorted(with_parent.values(), reverse=True)
    return dict(pool=pool, task=task, nested_view=nested, seed_count=seeds,
        estimand=estimand, candidate_count=len(candidates), candidates=candidates,
        candidate_gain_min_percent=min(candidates.values()),
        candidate_gain_median_percent=statistics.median(candidates.values()),
        candidate_gain_max_percent=max(candidates.values()),
        oracle_action=chosen, oracle_gain_percent=oracle,
        second_best_gain_percent=ordered[1], top_margin_pp=ordered[0]-ordered[1],
        actions_above_one_percent=sum(v >= 1 for v in candidates.values()),
        oracle_minus_reference_pp=None if controls is None else {k: oracle-v for k,v in controls.items()},
        reference_gains_percent=controls)


def main():
    started = time.perf_counter()
    rows = []
    historic = csv_rows('20261014_intervention_coverage_r1/historical_cohort.csv')
    assert len(historic) == 50 and len({x['id'] for x in historic}) == 50
    for origin, count in [('0.6',29),('0.8',21)]:
        items = [x for x in historic if x['origin'] == origin]
        assert len(items) == count
        candidates = {}
        for item in items:
            gain = 100*(1-float(item['nmse'])/float(item['parent_nmse']))
            assert abs(gain-float(item['gain_percent'])) < 1e-10
            candidates[item['id']] = gain
        rows.append(summary('MetroPT3_historical', 'origin_'+origin, candidates, 1,
            'original train-internal future; fixed step 3000'))
    pilot = '20261012_skab_pilot/'
    parent = read(pilot+'baseline_result.json')
    candidates = {}
    for action in ('W_UP','W_DOWN','S_UP','S_DOWN','CW_UP','CW_DOWN','CS_UP','CS_DOWN'):
        item = read(pilot+action+'_result.json')
        for key in ('seed','prepared_sha256','protocol_sha256','initialization_sha256'):
            assert item[key] == parent[key]
        candidates[action] = 100*(1-item['metrics']['global']['NMSE']/parent['metrics']['global']['NMSE'])
    rows.append(summary('SKAB_pilot', 'all_windows', candidates, 1,
        'original global development metric and original selected checkpoint', {'B0':0.}))
    paired = csv_rows('20261014_weighted_response_r1/paired_seed_records.csv')
    metrics = read('20261015_paper_evidence_r1/metrics.json')
    assert len(paired) == 36
    actions = ('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')
    for index, task in enumerate(('G1','G2','G3')):
        views = {}
        for view in ('S','W'):
            candidates = {}
            for action in actions:
                items = [x for x in paired if x['task'] == task and x['action'] == action]
                assert len(items) == 3 and {x['seed'] for x in items} == {'2026101701','2026101702','2026101703'}
                candidates[action] = statistics.mean(float(x[view+'_gain']) for x in items)
            views[view] = candidates
            chosen = ('FAVOR_FIRST','KEEP_SECOND','PARENT')[index]
            controls = {'M1':0. if chosen == 'PARENT' else candidates[chosen]}
            if view == 'S':
                controls.update({m:metrics['main_task_selected_gain_percent'][m][index] for m in ('RAW','SHORT')})
                assert abs(controls['M1']-metrics['main_task_selected_gain_percent']['M1'][index]) < 1e-10
            rows.append(summary('Hydraulic_'+view, task, candidates, 3,
                'mean of paired per-seed relative C gains after B selection; one action per task',
                controls, nested=(view == 'W')))
        joint = {view+'_'+action:value for view,candidates in views.items() for action,value in candidates.items()}
        controls = {view+'_priority':0. if index == 2 else views[view][('FAVOR_FIRST','KEEP_SECOND')[index]] for view in ('S','W')}
        rows.append(summary('Hydraulic_joint', task, joint, 3,
            'post-hoc union of S/W; same tasks, seeds and parents', controls, nested=True))
    primary = [x for x in rows if not x['nested_view']]
    assert len(primary) == 6 and len(rows) == 12
    missing = [dict(batch='existing_sources', tasks=2, missing_full_child_labels=24, oracle=None),
        dict(batch='file_transfer', tasks=3, missing_full_child_labels=69, oracle=None),
        dict(batch='fourgroup', tasks=3, missing_full_child_labels=72, oracle=None)]
    result = dict(status='computed_from_disclosed_summaries', rows=rows, missing=missing,
        scope='three named complete training-data selection pools; not all 434 trainings; transformations, models and diagnostic trajectories excluded',
        six_task_descriptive_median_percent=statistics.median(x['oracle_gain_percent'] for x in primary),
        primary_task_count=6, primary_below_one_percent=sum(x['oracle_gain_percent'] < 1 for x in primary),
        pooling_warning='heterogeneous, selected and dependent tasks: median is descriptive, not a population estimate or experiment gate',
        inputs=SOURCES, new_full_trainings=0, cumulative_full_trainings=434, seconds=time.perf_counter()-started)
    write(OUT/'headroom.json', result)
    fields = ['pool','task','nested_view','seed_count','candidate_count','candidate_gain_min_percent',
        'candidate_gain_median_percent','candidate_gain_max_percent','oracle_action','oracle_gain_percent',
        'second_best_gain_percent','top_margin_pp','actions_above_one_percent','estimand']
    with (OUT/'headroom.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(primary_rows=[{k:x[k] for k in ('pool','task','oracle_gain_percent','oracle_minus_reference_pp')} for x in primary],
        median=result['six_task_descriptive_median_percent'], below_one=result['primary_below_one_percent'], input_files=len(SOURCES)), indent=2))


if __name__ == '__main__':
    main()
