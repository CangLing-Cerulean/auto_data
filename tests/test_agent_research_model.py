"""Engineering API settings stay independent of the frozen selection model."""
from types import SimpleNamespace
import unittest

from online_ts.agent_research.runtime import CONFIG, EngineeringModel, PROMPT
from online_ts.agent_runtime.llm import DeepSeekModel


class EngineeringModelTests(unittest.TestCase):
    def test_protocol_settings_and_operation_identity(self):
        repo=SimpleNamespace(config=CONFIG)
        config=SimpleNamespace(planner=SimpleNamespace(model=CONFIG['model']))
        context={'phase':'exploration','context_version':'engineering-test'}
        shared=DeepSeekModel(repo,config,system_prompt=PROMPT)
        engineering=EngineeringModel(repo,config,system_prompt=PROMPT)
        original,original_id=shared.request_spec(context,0)
        adjusted,adjusted_id=engineering.request_spec(context,0)
        self.assertEqual(original['reasoning_effort'],'low')
        self.assertEqual(adjusted['reasoning_effort'],'none')
        self.assertEqual(adjusted['max_tokens'],original['max_tokens'])
        self.assertEqual(adjusted['messages'],original['messages'])
        self.assertNotEqual(adjusted_id,original_id)
        self.assertNotEqual(adjusted_id,engineering.request_spec(context,1)[1])


if __name__=='__main__':unittest.main()
