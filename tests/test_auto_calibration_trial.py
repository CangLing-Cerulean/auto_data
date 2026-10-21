"""Search-boundary tests: disclosure, immutability, no replay, and future gating."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import auto_calibration_trial as trial
from online_ts import auto_calibration as ac


class TrialTests(unittest.TestCase):
    def test_recipe_budget_and_determinism(self):
        first, second = trial.random_recipes(), trial.random_recipes()
        self.assertEqual([r.key() for r in first], [r.key() for r in second])
        self.assertEqual(len({r.key() for r in first}), trial.PROPOSALS)
        self.assertLessEqual(sum(r.steps for r in first), 640)
        self.assertEqual(len(trial.recipe_space()), 288)

    def test_stable_ties_and_no_harmful_replacement(self):
        records = {k: {'validation': {'mse': v}} for k, v in {'old': 2., 'tie': 2., 'bad': 3., 'good': 1.}.items()}
        for key in ('tie', 'bad'):
            self.assertEqual(trial.choose_incumbent(records, 'old', key), 'old')
        self.assertEqual(trial.choose_incumbent(records, 'old', 'good'), 'good')

    def test_tool_use_or_incomplete_turn_rejected(self):
        trial.validate_events('{"type":"turn.completed"}\n')
        for kind in ('command_execution', 'mcp_tool_call', 'web_search', 'file_change',
                     'collab_tool_call', 'todo_list', 'unknown_read_tool'):
            with self.assertRaises(RuntimeError):
                trial.validate_events(json.dumps({'item': {'type': kind}})+'\n{"type":"turn.completed"}')
        with self.assertRaises(RuntimeError):
            trial.validate_events('{"type":"turn.started"}')
        with self.assertRaises(RuntimeError):
            trial.validate_events('{"type":"turn.failed"}\n{"type":"turn.completed"}')

    def test_future_cannot_start_without_selection(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch.object(trial, 'STAGE', Path(folder)), \
             patch.object(trial, 'check_lock'), patch.object(trial, 'prepare') as prepare:
            with self.assertRaises(FileNotFoundError):
                trial.score_future()
            prepare.assert_not_called()

    def test_future_parent_tamper_checked_before_loading_data(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch.object(trial, 'STAGE', Path(folder)), \
             patch.object(trial, 'check_lock'), patch.object(trial, 'prepare') as prepare:
            root = Path(folder)
            (root/'execution_lock.json').write_text('{}', encoding='utf-8')
            trial.write(root/'selection_lock.json', dict(status='all_paths_complete_future_unscored',
                selected_checkpoints={'broken': 'deadbeef'}, execution_lock_sha256=trial.sha(root/'execution_lock.json')))
            (root/'models').mkdir()
            (root/'models/broken.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                trial.score_future()
            prepare.assert_not_called()

    def test_search_never_restarts_partial_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch.object(trial, 'STAGE', Path(folder)), \
             patch.object(trial, 'check_lock'), patch.object(trial, 'load_samples') as load:
            (Path(folder)/'records').mkdir()
            with self.assertRaises(FileExistsError):
                trial.run_search()
            load.assert_not_called()

    def test_prompt_contains_only_declared_history(self):
        arrays = np.zeros((4, 2, 1), dtype=np.float32)
        samples = ac.ForecastSamples(arrays, np.zeros((4, 1, 3)), arrays,
                                     np.arange(4), np.arange(4)+5)
        record = dict(id='IDENTITY', recipe=None, parent=None, fit={}, validation={}, lineage_updates=0,
                      future_secret='FORBIDDEN_FUTURE')
        with tempfile.TemporaryDirectory(dir=ROOT) as folder, patch.object(trial, 'STAGE', Path(folder)):
            (Path(folder)/'AGENT_PROMPT.txt').write_text('declared prompt', encoding='utf-8')
            prompt = trial.build_prompt(samples, samples, {'IDENTITY': record, 'FIXED': record,
                                        'RANDOM_01': {'secret': 'FORBIDDEN_RANDOM'}}, [], 'IDENTITY', 1)
        self.assertNotIn('FORBIDDEN', prompt)
        self.assertNotIn('future_secret', prompt)


if __name__ == '__main__':
    unittest.main()
