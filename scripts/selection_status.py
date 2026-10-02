"""Compact live status and budget checkpoint for continuation."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'research_runs/selection'
    records={}
    for path in sorted(folder.glob('batch*.json')):
        for item in json.loads(path.read_text()):
            records[item['id']]=item
    completed=[]
    active=[]
    for run_id in records:
        run=ROOT/'research_runs/results'/run_id
        if (run/'result.json').exists():
            r=json.loads((run/'result.json').read_text())
            completed.append({'id':run_id,'NMSE':r['groups']['all']['NMSE'],'NMAE':r['groups']['all']['NMAE'],'quiet_NMSE':r['groups']['low_quartile']['NMSE'],'high_NMSE':r['groups']['high_input_variation']['NMSE']})
        elif (run/'status.json').exists():
            active.append({'id':run_id,**json.loads((run/'status.json').read_text())})
    used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
    value={'registered':len(records),'completed':completed,'active':active,'budget_used_seconds':used,'budget_remaining_seconds':7200-used}
    (folder/'progress.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'completed_count':len(completed),'latest_completed':completed[-2:],'active':active,'budget_used_seconds':used},indent=2))


if __name__=='__main__':
    main()
