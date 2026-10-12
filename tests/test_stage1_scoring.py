import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'online_ts/agent_research'))
from stage1_scoring import observed_direction, score_predictions, repeatability

class FourClassScoringTests(unittest.TestCase):
    def test_exact_boundaries_are_unchanged(self):
        for value in (-.25,0.,.25): self.assertEqual(observed_direction(value),'unchanged')
    def test_outside_boundaries(self):
        self.assertEqual(observed_direction(math.nextafter(.25,math.inf)),'improve')
        self.assertEqual(observed_direction(math.nextafter(-.25,-math.inf)),'harm')
    def test_abstention_is_not_correct_unchanged(self):
        score=score_predictions(['uncertain','unchanged'],[0.,0.])
        self.assertEqual((score['decided'],score['correct'],score['abstained']),(1,1,1))
        self.assertEqual(score['coverage'],.5)
    def test_all_abstain_accuracy_is_missing(self):
        self.assertIsNone(score_predictions(['uncertain'],[.1])['accuracy'])
    def test_harm_categories_are_separate(self):
        score=score_predictions(['harm','improve','unchanged','uncertain'],[-1.]*4)
        self.assertEqual([score[k] for k in ('harm_total','harm_warned','harm_false_benefit','harm_false_unchanged','harm_abstained')],[4,1,1,1,1])
    def test_invalid_values_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):observed_direction(value)
        with self.assertRaises(ValueError):score_predictions(['0'],[0.])
    def test_alignment_rejected(self):
        with self.assertRaises(ValueError):score_predictions([],[])
        with self.assertRaises(ValueError):score_predictions(['harm'],[])
    def test_repeatability_not_raw_sign(self):
        self.assertEqual(repeatability([.1,.2,.3])['stable_category'],'uncertain')
        self.assertEqual(repeatability([.3,.4,.5])['stable_category'],'improve')
        with self.assertRaises(ValueError):repeatability([.3,.4])

if __name__=='__main__': unittest.main()
