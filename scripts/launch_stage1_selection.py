"""Launch exactly one authorized S run with detached regular-file output."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--task',required=True)
    parser.add_argument('--action',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    runner=module('selection_runner',root/'run_stage1_selection.py')
    base,protocol,p,seed,name=runner.verify_context(root,args.task,args.action)
    controller=module('controller',root/'supervise_stage1_recovery.py')
    folder=root/'selection_control'
    folder.mkdir(exist_ok=True)
    process=controller.spawn_logged([sys.executable,'-B',str(root/'run_stage1_selection.py'),'--root',str(root),
                                    '--task',args.task,'--action',args.action],folder/(name+'.log'))
    base.write(folder/(name+'_launch.json'),dict(pid=process.pid,id=name,
        protocol_sha256=base.sha(root/'selection_protocol.json'),output_transport='regular_file'))
    print(json.dumps(dict(id=name,pid=process.pid,status='launched')))


if __name__=='__main__':
    main()
