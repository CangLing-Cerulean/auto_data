"""Single-use entrypoints; FD and scientific child training are not implemented here."""
import argparse
from pathlib import Path
import time
import traceback
import fourgroup_runtime as rt
import n1_path_runtime as run
import audit_n1_paths as audit

PHASES=('preflight','paths','audit_paths')


def validate(out,phase):
    plan=run.verify(out)
    assert not (out/'stage_failure.json').exists()
    if phase=='preflight':
        assert not (out/'preflight').exists()
    else:
        receipt=rt.read(out/'preflight_audit.json')
        assert receipt['status']=='PASS' and receipt['plan_sha256']==rt.sha(out/'plan.json')
        if phase=='paths':
            assert not (out/'paths.lock').exists()
        else:
            assert (out/'paths.json').is_file() and not (out/'paths_audit.json').exists()
    return plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=PHASES)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rt.configure()
    started=rt.tick()
    plan=validate(args.output,args.phase)
    verification=rt.tick()-started
    rt.write(args.output/f'{args.phase}_execution.lock',dict(phase=args.phase,started_unix=time.time()))
    try:
        if args.phase=='preflight':
            result=audit.preflight(args.output,plan)
        elif args.phase=='paths':
            result=run.paths(args.output,plan,verification)
        else:
            result=audit.paths(args.output,plan)
        print(dict(phase=args.phase,status=result['status'],seconds=rt.tick()-started),flush=True)
    except Exception:
        rt.write(args.output/'stage_failure.json',dict(phase=args.phase,status='failed_closed',
            error=traceback.format_exc(),seconds=time.perf_counter()-started,new_scientific_children=0))
        raise


if __name__=='__main__':
    main()
