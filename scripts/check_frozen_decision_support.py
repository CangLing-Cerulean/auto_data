"""Read frozen marginal forecasts; certify when any convex weights are inert."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research_runs/agent_research/intervention_diagnosis'
OLD = BASE / '20261018_measurement_generation_r4'
OUT = BASE / '20261018_decision_support_r7'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def predict(model, features):
    names = model['feature_names']
    assert len(names) == len(model['center']) == len(model['scale']) == len(model['beta']) - 1
    terms = [model['beta'][0]]
    for name, center, scale, beta in zip(names, model['center'], model['scale'], model['beta'][1:]):
        assert name in features and scale > 0
        terms.append(beta * ((features[name] - center) / scale))
    return math.fsum(terms)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    target = OUT / 'result.json'
    if target.exists():
        raise FileExistsError(target)
    summary = read(OLD / 'replay/summary.json')
    hashes = {'protocol': sha(OUT/'PROTOCOL.md'), 'script': sha(Path(__file__)), 'old_summary': sha(OLD/'replay/summary.json')}
    rows, max_error, compared = [], 0., 0
    for entry in summary['runs']:
        if entry['method'] not in ('ordinary_conditional', 'mismatch_conditional'):
            continue
        path = OLD/'replay'/f"{entry['family']}_{entry['task']}_{entry['method']}.json"
        assert sha(path) == entry['sha256']
        trace = read(path)
        pool_path = OLD/'replay'/f"{trace['source_family']}_augmented_records.json"
        # Intentionally retain only identity and legal features, never response values.
        features = {r['id']: r['features'] for r in read(pool_path)['records']}
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        hashes[pool_path.relative_to(ROOT).as_posix()] = sha(pool_path)
        for step in trace['trace']:
            snap = step['model_snapshots']
            assert digest(snap) == step['registration']['model_hash']
            models = {'reference':snap['reference']['marginal'], **{h:m['marginal'] for h,m in snap['diagnostics'].items()}}
            assert len(models) == 3
            ids = sorted(step['predictions_before'])
            predictions = {i:{h:predict(m,features[i]) for h,m in models.items()} for i in ids}
            logits = {'reference':0., **{h:step['event']['before'].get(h,0.)-math.log(2) for h in snap['diagnostics']}}
            max_logit = max(logits.values())
            weights = {h:math.exp(v-max_logit) for h,v in logits.items()}
            denominator = math.fsum(weights.values())
            weights = {h:v/denominator for h,v in weights.items()}
            for candidate in ids:
                actual = math.fsum(weights[h]*predictions[candidate][h] for h in models)
                expected = step['predictions_before'][candidate]
                error = abs(actual-expected)
                assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12)
                max_error = max(max_error,error)
                compared += 1
            winners = {h:min(ids,key=lambda i:(-predictions[i][h],i)) for h in models}
            selected = step['selected']
            differences = {i:{h:predictions[selected][h]-predictions[i][h] for h in models} for i in ids if i != selected}
            margins = [v for d in differences.values() for v in d.values()]
            near = [(i,h,v) for i,d in differences.items() for h,v in d.items() if 0 < abs(v) < 1e-12]
            unanimous = all(winner == selected for winner in winners.values())
            assert unanimous == all(v>0 or (v==0 and selected<i) for i,d in differences.items() for v in d.values())
            rows.append(dict(family=entry['family'],task=entry['task'],method=entry['method'],step=step['step'],
                             selected=selected,candidates=len(ids),weights=weights,component_predictions=predictions,
                             component_winners=winners,selected_minus_competitor=differences,
                             all_components_select_registered=unanimous,min_margin=min(margins) if margins else None,
                             near_boundary=near,robust_convex_invariance=unanimous and not near))
    assert len(rows)==64 and compared==256
    groups=[]
    for family,task,method in sorted({(r['family'],r['task'],r['method']) for r in rows}):
        subset=[r for r in rows if (r['family'],r['task'],r['method'])==(family,task,method)]
        groups.append(dict(family=family,task=task,method=method,steps=len(subset),
                           nontrivial=sum(r['candidates']>1 for r in subset),
                           nontrivial_invariant=sum(r['candidates']>1 and r['robust_convex_invariance'] for r in subset),
                           whole_path_invariant=all(r['robust_convex_invariance'] for r in subset)))
    result=dict(status='PASS',scope='frozen_prediction_geometry_only',input_sha256=hashes,
                comparisons=compared,max_prediction_error=max_error,states=len(rows),trajectories=groups,states_detail=rows,
                new_full_trainings=0,new_LLM_calls=0,cumulative_full_trainings=464)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','states_detail')},indent=2))

if __name__=='__main__':
    main()
