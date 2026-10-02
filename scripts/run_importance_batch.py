"""Execute only a registered importance sampling queue within shared budget."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('batch'); args=parser.parse_args()
    assert Path(args.batch).name==args.batch
    for item in json.loads((ROOT/'research_runs/importance'/f'{args.batch}.json').read_text()):
        run=ROOT/'research_runs/results'/item['id']
        if (run/'result.json').exists(): continue
        run.mkdir(parents=True,exist_ok=True)
        used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in run.parent.glob('*/status.json'))
        remaining=int(7200-used)
        if remaining<=0: raise RuntimeError('Budget exhausted')
        command=[sys.executable,'-u','scripts/run_importance.py','--run-id',item['id'],'--recipe',item['recipe'],'--seed',str(item['seed'])]
        if item.get('reference'): command+=['--reference',item['reference']]
        with (run/'process.log').open('ab') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        print('Completed',item['id'],flush=True)


if __name__=='__main__': main()
