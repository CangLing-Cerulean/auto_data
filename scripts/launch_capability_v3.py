"""Detached one-time launch; reconnecting clients never replay an existing stage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('bank','loop'),required=True)
    stage=parser.parse_args().stage
    project=Path('/tmp/auto_data_capability_project_20261010_v3')
    output=Path('/tmp/auto_data_capability_acceptance_20261010_v3')
    output.mkdir(exist_ok=True)
    receipt=output/(stage+'_launch.json')
    folder=output/('bank' if stage=='bank' else 'known_loop')
    if receipt.exists() or folder.exists():raise RuntimeError('Existing stage preserved; no replay')
    log=output/(stage+'.log')
    environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    with log.open('wb') as stream:
        process=subprocess.Popen(['/opt/miniconda3/bin/python','-B',str(project/'scripts/run_capability.py'),
            '--stage',stage],cwd=project,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,env=environment)
    value={'stage':stage,'pid':process.pid,'started_ns':time.time_ns(),'project':str(project),
           'output':str(output),'log':str(log),'same_ID_replay':False}
    receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value),flush=True)

if __name__=='__main__':main()
