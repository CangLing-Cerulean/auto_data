"""Run only an explicitly registered sampling comparison batch, sequentially."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch')
    args=parser.parse_args()
    if Path(args.batch).name!=args.batch:
        raise ValueError('Invalid batch')
    queue=json.loads((ROOT/'research_runs/selection'/f'{args.batch}.json').read_text())
    for item in queue:
        run=ROOT/'research_runs/results'/item['id']
        if (run/'result.json').exists():
            print('Already completed',item['id'],flush=True)
            continue
        run.mkdir(parents=True,exist_ok=True)
        used=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))
        budget=json.loads((ROOT/'research_runs/selection/protocol.json').read_text())['session_gpu_wall_budget_seconds']
        remaining=int(budget-used)
        if remaining<=0:
            raise RuntimeError('Shared budget exhausted')
        print('Starting',item['id'],flush=True)
        with (run/'process.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_selection.py'),'--run-id',item['id'],
                            '--recipe',item['recipe'],'--seed',str(item['seed'])],cwd=ROOT,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        if not (run/'result.json').exists():
            raise RuntimeError('Missing result')
        print('Completed',item['id'],flush=True)


if __name__=='__main__':
    main()
