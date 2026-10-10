"""Server-only regression collection for the new study and existing components."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.signal_study import PILOT
from online_ts.selection.common import file_hash,write_json

def main():
    root=Path(__file__).resolve().parents[1]
    tests=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_agent*.py')
    result=unittest.TextTestRunner(verbosity=1).run(tests)
    write_json(PILOT/'component_checks.json',{'status':'passed' if result.wasSuccessful() else 'failed',
        'tests_run':result.testsRun,'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
        'tests':{p.name:file_hash(p) for p in (root/'tests').glob('test_agent*.py')}})
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
