"""Detached one-time launch after bound preflight, never replaying an existing batch."""
import json
import os
from pathlib import Path
import subprocess
import time

def main():
    project=Path('/tmp/auto_data_signal_project_20261010');output=Path('/tmp/auto_data_signal_study_20261010')
    if output.exists():raise RuntimeError('Existing batch preserved; no relaunch')
    subprocess.run(['/opt/miniconda3/bin/python','-B','-c',
        'from online_ts.agent_research.signal_study import require_ready; require_ready()'],cwd=project,check=True)
    output.mkdir();log=output/'batch.log';environment=os.environ.copy();environment['PYTHONUNBUFFERED']='1'
    with log.open('wb') as stream:
        process=subprocess.Popen(['/opt/miniconda3/bin/python','-B',str(project/'scripts/run_signal_study.py')],
            cwd=project,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
    value={'pid':process.pid,'started_ns':time.time_ns(),'project':str(project),'output':str(output),
        'log':str(log),'same_ID_replay':False,'registered_searches':9,'cumulative_limits':None}
    (output/'launch.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value),flush=True)

if __name__=='__main__':main()
