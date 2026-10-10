"""One-time detached known-case launch using the accepted v3 bank as frozen evidence."""
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    project=Path('/tmp/auto_data_capability_project_20261010_v4')
    output=Path('/tmp/auto_data_capability_acceptance_20261010_v4')
    output.mkdir(exist_ok=True)
    receipt=output/'loop_launch.json'
    if receipt.exists() or (output/'known_loop').exists():raise RuntimeError('Existing loop preserved; no replay')
    log=output/'loop.log';environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    with log.open('wb') as stream:
        process=subprocess.Popen(['/opt/miniconda3/bin/python','-B',str(project/'scripts/run_capability.py'),
            '--stage','loop'],cwd=project,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
    value={'stage':'loop','pid':process.pid,'started_ns':time.time_ns(),'project':str(project),
        'output':str(output),'log':str(log),'same_ID_replay':False,'accepted_bank_inherited_not_replayed':True}
    receipt.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value),flush=True)

if __name__=='__main__':main()
