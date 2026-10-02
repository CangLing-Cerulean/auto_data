"""Paired comparisons and validation subgroup attribution; no data or test access."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/'research_runs'


def main():
    results = [json.loads(p.read_text()) for p in (RUNS/'results').glob('*/result.json')]
    lookup = {(r['identity']['recipe'],r['identity']['seed']):r for r in results}
    pairs=[]
    for seed in (2026,2027,2028):
        if ('baseline',seed) not in lookup or ('selective_center',seed) not in lookup:
            continue
        base, candidate = lookup['baseline',seed], lookup['selective_center',seed]
        item={'seed':seed,'baseline':base['groups']['all'],'selective':candidate['groups']['all'],'group_relative_reduction_pct':{}}
        for group in base['groups']:
            item['group_relative_reduction_pct'][group]={m:100*(1-candidate['groups'][group][m]/base['groups'][group][m]) for m in ('NMSE','NMAE')}
        item['channel_nmse_delta']={name:c-b for name,b,c in zip(base['sensor_columns'],base['per_channel_NMSE'],candidate['per_channel_NMSE'])}
        pairs.append(item)
    baseline=[r for r in results if r['identity']['recipe']=='baseline']
    selective=[r for r in results if r['identity']['recipe']=='selective_center']
    report={'pairs':pairs}
    if len(pairs)==3:
        report['relative_reduction_of_seed_means_pct']={m:100*(1-statistics.mean(p['selective'][m] for p in pairs)/statistics.mean(p['baseline'][m] for p in pairs)) for m in ('NMSE','NMAE')}
    (RUNS/'paired_analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
