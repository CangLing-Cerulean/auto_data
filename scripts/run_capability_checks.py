"""Server-only regression and actual-data interface readiness before any new API run."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from online_ts.agent_research.capability import PILOT
from online_ts.agent_research.runtime import code_identity
from online_ts.agent_research.task import TaskAssets
from online_ts.agent_research.memory import MEMORY,verify_memory
from online_ts.selection.common import read_json,write_json,file_hash

root=Path(__file__).resolve().parents[1]
suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_agent*.py')
result=unittest.TextTestRunner(verbosity=1).run(suite)
if not result.wasSuccessful() or result.skipped:raise RuntimeError('Component checks failed or skipped')
task=TaskAssets(root/'research_runs/agent_research/tasks/metropt3_dev');verify_memory(read_json(MEMORY/'library.json'))
protocol=read_json(PILOT/'protocol.json')
if protocol['document_sha256']!=file_hash(PILOT/'PROTOCOL.md') or protocol['cases_sha256']!=file_hash(PILOT/'cases.json'):
    raise RuntimeError('Preregistered assets changed')
write_json(PILOT/'preflight.json',{'status':'passed','tests_run':result.testsRun,'skipped':0,
    'code':code_identity(),'cases_sha256':file_hash(PILOT/'cases.json'),'protocol_sha256':file_hash(PILOT/'protocol.json'),
    'features_sha256':file_hash(task.feature_path),'API_calls':0,'training_calls':0,'test_access':False})
print('Capability acceptance preflight passed',flush=True)
