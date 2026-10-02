"""Execute the four pre-registered paired confirmations sequentially on one GPU."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = [
    ('E004_baseline_s2027','baseline',2027),
    ('E005_selective_center_s2027','selective_center',2027),
    ('E006_baseline_s2028','baseline',2028),
    ('E007_selective_center_s2028','selective_center',2028),
]


def main():
    protocol = json.loads((ROOT/'research_runs/protocol.json').read_text())
    for run_id,recipe,seed in QUEUE:
        run=ROOT/'research_runs/results'/run_id
        if (run/'result.json').exists():
            print('Already completed',run_id,flush=True)
            continue
        run.mkdir(parents=True,exist_ok=True)
        consumed=sum(json.loads(s.read_text()).get('elapsed_seconds',0) for s in (ROOT/'research_runs/results').glob('*/status.json'))
        remaining=int(protocol['session_gpu_wall_budget_seconds']-consumed)
        if remaining<=0:
            raise RuntimeError('Budget exhausted')
        print('Starting',run_id,flush=True)
        with (run/'process.log').open('ab') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_research.py'),'--run-id',run_id,
                            '--recipe',recipe,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=remaining)
        if not (run/'result.json').exists():
            raise RuntimeError('No completed result')
        print('Completed',run_id,flush=True)
    print('All registered confirmations completed',flush=True)


if __name__=='__main__':
    main()
