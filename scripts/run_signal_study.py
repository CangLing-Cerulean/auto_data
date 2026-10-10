"""Entry point guarded against child-process import reentry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    from online_ts.agent_research.signal_study import main as run
    run()

if __name__=='__main__':main()
