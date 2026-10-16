"""Detached create-only supervisor; preserves process output and exit status."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import ett_confirmation as study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('preflight','train','audit'))
    parser.add_argument('station',choices=('ETTh1','ETTh2'))
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    study.verify(args.station)
    directory=study.ROOT/'control'/f'{args.station}_{args.phase}'
    if args.supervise:
        start=time.time()
        command=[sys.executable,'-B',str(study.ROOT/'ett_confirmation.py'),args.phase,args.station]
        child=subprocess.Popen(command,cwd=study.ROOT,stdin=subprocess.DEVNULL)
        study.write(directory/'child.json',dict(pid=child.pid,started_unix=start,command=command))
        code=child.wait()
        study.write(directory/'exit.json',dict(exit_code=code,started_unix=start,finished_unix=time.time()))
        raise SystemExit(code)
    directory.mkdir(parents=True)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),args.phase,args.station,'--supervise']
    with (directory/'run.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=study.ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
    study.write(directory/'launch.json',dict(pid=child.pid,started_unix=time.time()))
    print(dict(status='launched',phase=args.phase,station=args.station,pid=child.pid),flush=True)


if __name__=='__main__':
    main()
