"""Execute the finite registered 18 S confirmations; never retry occupied IDs."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    runner=module('runner',root/'run_stage1_selection_confirmation.py')
    base,plan,p,seed,name=runner.verify_context(root,'S1','PARENT',2026101402)
    helper=module('helper',root/'supervise_stage1_recovery.py')
    control=root/'selection_confirmation_control'
    if not args.execute:
        control.mkdir()
        process=helper.spawn_logged([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--execute'],control/'supervisor.log')
        base.write(control/'launch.json',dict(pid=process.pid,protocol_sha256=base.sha(root/'selection_confirmation_protocol.json'),
            launched_unix=time.time(),output_transport='regular_file'))
        print(json.dumps(dict(status='launched',pid=process.pid,registered_jobs=18)))
        return
    with (control/'execution.lock').open('x') as guard:
        guard.write(str(os.getpid()))
    state=dict(status='running',pid=os.getpid(),active=None,completed=[],failed=None,
               protocol_sha256=base.sha(root/'selection_confirmation_protocol.json'))
    base.write(control/'status.json',state)
    try:
        for job in plan['jobs']:
            name=job['id']
            runner.verify_context(root,job['task'],job['action'],job['seed'])
            state['active']=name
            base.write(control/'status.json',state)
            command=[sys.executable,'-B',str(root/'run_stage1_selection_confirmation.py'),'--root',str(root),
                     '--task',job['task'],'--action',job['action'],'--seed',str(job['seed'])]
            process=helper.spawn_logged(command,control/(name+'.log'))
            state['child_pid']=process.pid
            base.write(control/'status.json',state)
            code=process.wait()
            status=base.read(root/'selection_confirmation_runs'/name/'status.json')
            if code!=0 or status['status']!='completed' or status['steps_completed']!=3000:
                raise RuntimeError(f'{name}: exit={code}, status={status}')
            state['completed'].append(name)
            state['active']=None
            state.pop('child_pid',None)
            base.write(control/'status.json',state)
        state['status']='completed'
    except Exception as error:
        state['status']='failed_stopped'
        state['failed']=state['active']
        state['error']=str(error)
        raise
    finally:
        state['ended_unix']=time.time()
        base.write(control/'status.json',state)


if __name__=='__main__':
    main()
