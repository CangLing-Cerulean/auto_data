"""Formal validation comparison, distinct from train-only teacher diagnostics."""
import json
from pathlib import Path
from analyze_state_selection import contrast

ROOT=Path(__file__).resolve().parents[1]


def main():
    runs={}
    for path in (ROOT/'research_runs/results').glob('S*/result.json'):
        r=json.loads(path.read_text()); runs.setdefault(r['identity']['recipe'],{})[r['identity']['seed']]=r
    comparisons={}
    for candidate in ('future_conflict_downweight','future_conflict_shuffled'):
        if candidate not in runs: continue
        for reference in ('uniform','future_conflict_shuffled'):
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
    (ROOT/'research_runs/transfer/formal_analysis.json').write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf-8',newline='\n')
    for name,r in comparisons.items(): print(name,json.dumps({'seeds':r['seeds'],'all':r['groups']['all']}))


if __name__=='__main__': main()
