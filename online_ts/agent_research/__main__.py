"""Server research preparation and explicitly requested engineering execution."""
import argparse
import json

from ..selection.common import read_json
from .experience import STUDY, build_library, retrieve, verify_library


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify")
    commands.add_parser("status")
    commands.add_parser('build-memory')
    commands.add_parser('verify-memory')
    memory=commands.add_parser('memory')
    memory.add_argument('--dataset',required=True)
    memory.add_argument('--family',required=True)
    memory.add_argument('--track',choices=('experience_combination','new_task_discovery'),default='experience_combination')
    memory.add_argument('--mode',choices=('full','none','other_tasks'),default='full')
    memory.add_argument('--query',default='')
    memory.add_argument('--record-id')
    memory.add_argument('--list',action='store_true')
    memory.add_argument('--offset',type=int,default=0)
    memory.add_argument('--limit',type=int,default=5)
    engineering=commands.add_parser("run-engineering")
    engineering.add_argument("--run-id",required=True)
    engineering.add_argument('--experience-mode',choices=('full','none','other_tasks'),default='full')
    engineering.add_argument('--track',choices=('experience_combination','new_task_discovery'),default='experience_combination')
    search = commands.add_parser("experience")
    search.add_argument("--dataset", required=True)
    search.add_argument("--track", required=True, choices=("experience_combination", "new_task_discovery"))
    search.add_argument("--query", default="")
    search.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.command == 'run-engineering':
        from .runtime import run_engineering
        result=run_engineering(args.run_id,experience_mode=args.experience_mode,track=args.track)
    elif args.command in {'build-memory','verify-memory','memory'}:
        from .memory import MEMORY, EvidenceMemory, build_memory, verify_memory
        if args.command=='build-memory':
            library=build_memory();result={'sha256':library['sha256'],'coverage':library['coverage'],'api_calls':0,'trainings':0}
        else:
            library=read_json(MEMORY/'library.json')
            if args.command=='verify-memory':result={'records':verify_memory(library),'sha256':library['sha256'],'coverage':library['coverage']}
            else:
                store=EvidenceMemory(library,target_dataset=args.dataset,target_family=args.family,track=args.track,mode=args.mode)
                result=store.read(args.record_id) if args.record_id else store.list(args.offset,args.limit) if args.list else store.retrieve(args.query,args.limit)
    elif args.command == "prepare":
        library = build_library()
        result = {"cases": len(library["records"]), "library_sha256": library["sha256"],
                  "scope": library["scope"], "api_calls": 0, "trainings": 0}
    elif args.command == "status":
        result = read_json(STUDY / "progress.json")
    else:
        library = read_json(STUDY / "experience_library.json")
        if args.command == "verify":
            result = {"verified_cases": verify_library(library), "scope": "historical_development"}
        else:
            result = retrieve(library, target_dataset=args.dataset, track=args.track,
                              query=args.query, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
