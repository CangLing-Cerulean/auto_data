"""Evidence-only memory checks; metadata fixtures are not research observations."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from online_ts.config import GateError, digest
from online_ts.selection.common import file_hash, write_json
from online_ts.agent_research.experience import build_library
from online_ts.agent_research.memory import build_memory, verify_memory, EvidenceMemory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        output=Path(__file__).resolve().parents[1]/'outputs/memory-tests';output.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=output);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.folder=self.root/'memory';self.folder.mkdir()
        study=self.root/'research_runs/agent_research';study.mkdir(parents=True)
        source=self.root/'research_runs/decisions.md';source.write_text('S000_fixture development metadata\n',encoding='utf-8')
        spec={'dataset_id':'fixture_v1','cases':[{'id':'fixture','source':'research_runs/decisions.md',
            'start':1,'end':1,'anchors':['metadata'],'tags':['variation','weighting'],
            'lesson':'Mock conditional interpretation','limits':'Fixture only; cause unknown','status':'unassessed'}]}
        write_json(study/'spec.json',spec)
        build_library(study/'spec.json',study/'experience_library.json',root=self.root)
        self.raw={'run_id':'S000_fixture','identity':{'recipe':'downweight_quiet','seed':1},
            'groups':{'all':{'NMSE':1.,'NMAE':1.,'windows':1}},
            'test_metrics':{'NMSE':999.},'steps_completed':1}
        src='research_runs/results/S000_fixture/result.json';self.source=self.folder/'sources'/src
        self.source.parent.mkdir(parents=True);write_json(self.source,self.raw)
        self.catalog={'version':'approved-development-memory-inputs-v1','dataset_id':'fixture_v1',
            'dataset_family':'fixture','declared_historical_count':1,'completed_run_ids':['S000_fixture'],
            'entries':[{'run_id':'S000_fixture','source':src,'sha256':file_hash(self.source),
                'kind':'main','dataset_id':'fixture_v1','dataset_family':'fixture'}]}
        write_json(self.folder/'input_catalog.json',self.catalog)
        self.library=build_memory(self.folder,root=self.root)

    def memory(self,**overrides):
        args={'target_dataset':'fixture_v1','target_family':'fixture','track':'experience_combination',
            'folder':self.folder,'root':self.root}|overrides
        return EvidenceMemory(self.library,**args)

    def test_observations_are_not_causal_labels_or_method_bans(self):
        r=self.memory().read('experience:run:S000_fixture')
        self.assertEqual(r['interpretation']['verification'],'unassessed')
        self.assertEqual(r['observation']['groups']['all']['NMSE'],1.)
        self.assertNotIn('test_metrics',r['observation'])
        self.assertFalse(self.library['policy']['method_family_bans'])
        self.assertTrue(self.library['policy']['revisiting_and_replication_allowed'])

    def test_language_aliases_find_relevant_history(self):
        for q in ('低变化降权','variation weighting'):
            self.assertTrue(self.memory().retrieve(q)['records'])

    def test_unmatched_query_does_not_inject_other_history(self):
        self.assertEqual(self.memory().retrieve('企鹅')['records'],[])

    def test_all_records_can_be_browsed_without_a_query(self):
        store=self.memory();one=store.list(0,1);two=store.list(one['next_offset'],1)
        self.assertEqual(one['total'],2);self.assertIsNone(two['next_offset'])
        self.assertNotEqual(one['records'][0]['record_id'],two['records'][0]['record_id'])

    def test_none_mode_exposes_no_records_or_details(self):
        store=self.memory(mode='none');self.assertEqual(store.list()['total'],0)
        self.assertEqual(store.retrieve('variation')['records'],[])
        with self.assertRaisesRegex(GateError,'NOT_VISIBLE'):store.read('experience:fixture')

    def test_new_task_excludes_different_version_of_same_family(self):
        store=self.memory(target_dataset='fixture_v2',track='new_task_discovery')
        self.assertEqual(store.list()['total'],0)
        with self.assertRaisesRegex(GateError,'NOT_VISIBLE'):store.read('experience:fixture')
        self.assertEqual(self.memory(target_dataset='other_v1',target_family='other',track='new_task_discovery').list()['total'],2)

    def test_other_tasks_mode_filters_all_target_history(self):
        self.assertEqual(self.memory(mode='other_tasks').list()['total'],0)

    def test_rehashed_fabricated_fact_is_rejected(self):
        bad=deepcopy(self.library);bad['records'][0]['observation']['groups']['all']['NMSE']=.1
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaisesRegex(GateError,'FACT_CHANGED'):verify_memory(bad,folder=self.folder,root=self.root)

    def test_rehashed_interpretation_cannot_be_promoted_to_truth(self):
        bad=deepcopy(self.library);bad['records'][-1]['interpretation']['verification']='correct'
        bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaisesRegex(GateError,'INTERPRETATION_CHANGED'):verify_memory(bad,folder=self.folder,root=self.root)

    def test_source_change_is_rejected_before_read(self):
        store=self.memory();write_json(self.source,self.raw|{'steps_completed':2})
        with self.assertRaisesRegex(GateError,'SOURCE_CHANGED'):store.read('experience:run:S000_fixture')

    def test_test_source_path_is_not_approved(self):
        self.catalog['entries'][0]['source']='research_runs/model_exploration/test_result.json'
        write_json(self.folder/'input_catalog.json',self.catalog)
        with self.assertRaisesRegex(GateError,'SOURCE_NOT_APPROVED'):build_memory(self.folder,root=self.root)

    def test_missing_completed_training_is_not_silently_omitted(self):
        self.catalog['completed_run_ids'].append('S001_missing');self.catalog['declared_historical_count']=2
        write_json(self.folder/'input_catalog.json',self.catalog)
        with self.assertRaisesRegex(GateError,'LEDGER_COVERAGE'):build_memory(self.folder,root=self.root)


if __name__=='__main__':unittest.main()
