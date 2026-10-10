"""Read compact running-batch evidence; never expose credentials or model text."""
import json
from pathlib import Path
import argparse


def load(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


parser=argparse.ArgumentParser();parser.add_argument('--root',default='/tmp/auto_data_phase4_20261009_open');args=parser.parse_args()
root=Path(args.root)
launch=load(root/'launch.json')
value={'launch':launch,'process_exists':bool(launch and Path('/proc',str(launch['pid'])).is_dir()),
       'batch':load(root/'progress.json'),'batch_failure':load(root/'batch_failure.json'),'runs':[]}
for folder in sorted((root/'runs').glob('*')):
    status=load(folder/'status.json');jobs=[]
    for job in sorted((folder/'training').glob('*')):
        result=load(job/'result.json');history=load(job/'history.json')
        jobs.append({'job_id':job.name,'completed':result is not None,
            'last_step':history[-1]['step'] if history else 0,
            'metrics':result['metrics'] if result else None})
    trace=folder/'trace.jsonl'
    tools=[]
    if trace.is_file():
        for line in trace.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);tools.append({'tool':row['selected_action'],'outcome':row['decision_outcome'],
                'error':(row.get('actual_result') or {}).get('error')})
    value['runs'].append({'run_id':folder.name,'budget':status['budget'] if status else None,
        'training':jobs,'tools':tools,'search_completed':(folder/'search_result.json').is_file(),
        'failure':load(folder/'failure.json'),'confirmation':load(folder/'external_confirmation.json')})
print(json.dumps(value,ensure_ascii=False),flush=True)
