"""Detached ordinary-file supervisor, one invocation for each N1 phase."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import fourgroup_runtime as rt
import run_n1_paths as study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=study.PHASES)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    out,phase=args.output,args.phase
    study.validate(out,phase)
    control=out/'control'/phase
    if args.supervise:
        assert control.is_dir() and not (control/'child.json').exists() and not (control/'exit.json').exists()
        command=[sys.executable,'-B',str(out/'run_n1_paths.py'),phase,'--output',str(out)]
        started=time.time()
        child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL)
        rt.write(control/'child.json',dict(pid=child.pid,supervisor_pid=os.getpid(),command=command,started_unix=started))
        code=child.wait()
        rt.write(control/'exit.json',dict(exit_code=code,started_unix=started,finished_unix=time.time(),phase=phase))
        raise SystemExit(code)
    control.mkdir(parents=True)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),phase,'--output',str(out),'--supervise']
    with (control/'run.log').open('xb') as stream:
        child=subprocess.Popen(command,cwd=out,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'))
    rt.write(control/'launch.json',dict(pid=child.pid,phase=phase,started_unix=time.time(),plan_sha256=rt.sha(out/'plan.json')))
    print(dict(status='supervisor_launched',phase=phase,pid=child.pid),flush=True)


if __name__=='__main__':
    main()
