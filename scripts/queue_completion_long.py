"""Wait for the fixed matrix, then run the preregistered six timing comparisons."""
import json
import subprocess
import sys
import time
from run_research import ROOT, write_json

FOLDER = ROOT / 'research_runs/completion_study'


def main():
    path = FOLDER / 'long_queue.json'
    assert not path.exists()
    write_json(path, {'phase': 'waiting_for_two_origin_matrix'})
    while True:
        progress = json.loads((FOLDER / 'policy_progress.json').read_text())
        if progress['phase'] == 'completed_two_origin_matrix':
            break
        if progress['phase'].startswith('failed'):
            write_json(path, {'phase': 'stopped_dependency_failure', 'dependency': progress})
            raise RuntimeError('Policy matrix failed; timing study was not launched')
        time.sleep(10)
    write_json(path, {'phase': 'running_long_plan'})
    subprocess.run([sys.executable, 'scripts/run_completion_batch.py', '--plan', 'long_plan.json'], cwd=ROOT, check=True)
    write_json(path, {'phase': 'completed'})


if __name__ == '__main__':
    main()
