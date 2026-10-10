"""Run the existing server-only Agent checks without depending on tests being a package."""
from pathlib import Path
import sys
import unittest

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from online_ts.agent_research.phase4 import PILOT
from online_ts.selection.common import write_json,file_hash

tests=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_agent*.py')
result=unittest.TextTestRunner(verbosity=1).run(tests)
write_json(PILOT/'component_checks.json',{'status':'passed' if result.wasSuccessful() else 'failed',
    'tests_run':result.testsRun,'skipped':len(result.skipped),'failures':len(result.failures),
    'errors':len(result.errors),'test_files':{str(p.relative_to(root)):file_hash(p) for p in (root/'tests').glob('test_agent*.py')}})
sys.exit(0 if result.wasSuccessful() else 1)
