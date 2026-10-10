"""Server entry point with an explicit project import root."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.capability import run_bank,run_known_loop
from online_ts.config import canonical

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('bank','loop'),required=True);args=parser.parse_args()
    result=run_bank() if args.stage=='bank' else run_known_loop()
    print(canonical({key:result[key] for key in ('status','passed','registered','error') if key in result}),flush=True)


if __name__=='__main__':main()
