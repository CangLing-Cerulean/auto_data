"""Paired continuation comparisons; never read test data."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    runs={}
    for file in sorted((ROOT/'research_runs/results').glob('S*/result.json')):
        result=json.loads(file.read_text())
        runs.setdefault(result['identity']['recipe'],{})[result['identity']['seed']]=result
    comparisons={}
    for candidate in ('month_matched_drop',):
        if candidate not in runs:
            continue
        for reference in ('uniform','drop_quiet','random_drop'):
            seeds=sorted(runs[candidate].keys() & runs[reference].keys())
            current=[runs[candidate][s] for s in seeds]
            baseline=[runs[reference][s] for s in seeds]
            groups={}
            for group in current[0]['groups']:
                groups[group]={}
                for metric in ('NMSE','NMAE'):
                    c=[r['groups'][group][metric] for r in current]
                    b=[r['groups'][group][metric] for r in baseline]
                    groups[group][metric]={'candidate_mean':statistics.mean(c),'reference_mean':statistics.mean(b),
                        'paired_reduction_pct':[100*(1-x/y) for x,y in zip(c,b)],
                        'mean_reduction_pct':100*(1-statistics.mean(c)/statistics.mean(b))}
            channels={name:{metric:100*(1-statistics.mean(r['per_channel_'+metric][i] for r in current)/
                statistics.mean(r['per_channel_'+metric][i] for r in baseline)) for metric in ('NMSE','NMAE')}
                for i,name in enumerate(current[0]['sensor_columns'])}
            comparisons[candidate+'_vs_'+reference]={'seeds':seeds,'groups':groups,'per_channel_mean_reduction_pct':channels}
    out=ROOT/'research_runs/selection/continuation_analysis.json'
    out.write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf-8')
    for name,result in comparisons.items():
        print(name,json.dumps({'seeds':result['seeds'],'all':result['groups']['all'],'quiet':result['groups']['low_quartile']['NMSE']}))


if __name__=='__main__':
    main()
