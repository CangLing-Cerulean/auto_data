"""Start one authorized preregistered batch, detached from the SSH connection."""
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.config import GateError
from online_ts.selection.common import read_json,write_json,file_hash
from online_ts.agent_research.phase4 import PILOT,OUTPUT,ROOT,code_identity


def main():
    ready=read_json(PILOT/'preflight.json')
    if ready['status']!='passed' or ready['code']!=code_identity() or ready['protocol_sha256']!=file_hash(PILOT/'protocol.json'):
        raise GateError('PILOT_PREFLIGHT_BINDING')
    key=Path('/root/.config/auto_data_research/deepseek_api_key')
    if not key.is_file() or key.stat().st_mode&0o077:raise GateError('SERVER_KEY_PERMISSIONS')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    receipt=OUTPUT/'launch.json'
    with receipt.open('x',encoding='utf-8') as stream:stream.write('{"status":"starting"}')
    env=dict(os.environ);env.pop('DEEPSEEK_API_KEY',None);env['PYTHONDONTWRITEBYTECODE']='1'
    with (OUTPUT/'batch.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,'-B','-m','online_ts.agent_research.phase4'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    value={'status':'started','pid':process.pid,'started_ns':time.time_ns(),
           'protocol_sha256':file_hash(PILOT/'protocol.json'),'output':str(OUTPUT),'log':str(OUTPUT/'batch.log')}
    write_json(receipt,value);print('Phase4 batch started; PID='+str(process.pid),flush=True)


if __name__=='__main__':main()
