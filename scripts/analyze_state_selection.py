"""Paired state-protection results, including all channels and fixed validation groups."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def contrast(c,b):
    return {'candidate_mean':statistics.mean(c),'reference_mean':statistics.mean(b),
        'paired_reduction_pct':[100*(1-x/y) for x,y in zip(c,b)],
        'mean_reduction_pct':100*(1-statistics.mean(c)/statistics.mean(b))}


def main():
    runs={}
    for path in (ROOT/'research_runs/results').glob('S*/result.json'):
        r=json.loads(path.read_text()); runs.setdefault(r['identity']['recipe'],{})[r['identity']['seed']]=r
    comparisons={}
    for candidate in ('protect_lps_matched','placebo_matched_swap'):
        if candidate not in runs: continue
        for reference in ('uniform','month_matched_drop','placebo_matched_swap'):
            if candidate==reference or reference not in runs: continue
            seeds=sorted(runs[candidate].keys() & runs[reference].keys())
            if not seeds: continue
            current=[runs[candidate][s] for s in seeds]; base=[runs[reference][s] for s in seeds]
            groups={g:{metric:contrast([r['groups'][g][metric] for r in current],[r['groups'][g][metric] for r in base])
                for metric in ('NMSE','NMAE')} for g in current[0]['groups']}
            channels={name:{metric:contrast([r['per_channel_'+metric][i] for r in current],
                [r['per_channel_'+metric][i] for r in base]) for metric in ('NMSE','NMAE')}
                for i,name in enumerate(current[0]['sensor_columns'])}
            comparisons[candidate+'_vs_'+reference]={'seeds':seeds,'groups':groups,'channels':channels}
    path=ROOT/'research_runs/selection/state_analysis.json'
    path.write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf-8')
    for name,r in comparisons.items():
        print(name,json.dumps({'seeds':r['seeds'],'overall_NMSE':r['groups']['all']['NMSE'],'LPS_NMSE':r['channels']['LPS']['NMSE']}))


if __name__=='__main__':
    main()
