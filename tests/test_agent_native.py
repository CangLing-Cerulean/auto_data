"""Linux engineering metadata tests; fake feedback never enters study ledgers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from online_ts.config import GateError, digest
from online_ts.selection.common import file_hash
from online_ts.agent_research.jobs import prepare_job
from online_ts.agent_research.native import execute_job
from online_ts.agent_research.processing import validate_processing, diagnostic_json, parameters


def temporary():
    root=Path(__file__).resolve().parents[1]/'outputs/native-tests'
    root.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp=temporary();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'processing.npz'
        # Window-index metadata only, not sensor records.
        self.data={'start_row':np.arange(100,dtype=np.int64),'groups':np.repeat([0,1],50)}

    def save(self,weights=None,sampling=None,starts=None):
        np.savez(self.path,start_row=self.data['start_row'] if starts is None else starts,
          weight=np.ones(100) if weights is None else weights,
          sampling_multiplier=np.ones(100) if sampling is None else sampling)

    def test_loss_scale_normalized_under_nonuniform_sampling(self):
        self.save(np.r_[np.ones(50),np.full(50,2.)],np.r_[np.full(50,2.),np.ones(50)])
        p,w,audit=validate_processing(self.path,self.data)
        self.assertAlmostEqual(float(np.dot(p,w)),1.)
        self.assertAlmostEqual(p[:50].sum(),2/3)
        self.assertEqual(audit['active'],100)

    def test_reordered_source_mapping_rejected(self):
        self.save(starts=self.data['start_row'][::-1])
        with self.assertRaisesRegex(GateError,'SOURCE_MAPPING'):validate_processing(self.path,self.data)

    def test_nonfinite_and_negative_weights_rejected(self):
        for item in (float('nan'),-1.,5.):
            weights=np.ones(100);weights[3]=item;self.save(weights)
            with self.assertRaisesRegex(GateError,'WEIGHT_RANGE'):validate_processing(self.path,self.data)

    def test_low_effective_group_mass_rejected(self):
        self.save(np.r_[np.full(50,.001),np.ones(50)])
        with self.assertRaisesRegex(GateError,'GROUP_COVERAGE'):validate_processing(self.path,self.data)

    def test_rare_group_members_cannot_be_removed(self):
        weights=np.ones(100);weights[0]=0;self.save(weights)
        with self.assertRaisesRegex(GateError,'GROUP_COVERAGE'):validate_processing(self.path,self.data)

    def test_repeated_historical_deletion_rejected(self):
        self.data['groups']=np.zeros(100,dtype=int)
        self.data['historical_domain_0']=np.ones(100,bool)
        self.data['historical_removed_0']=np.arange(100)<10
        self.save(np.r_[np.zeros(10),np.ones(90)])
        with self.assertRaisesRegex(GateError,'HISTORICAL_DELETION'):validate_processing(self.path,self.data)

    def test_wrong_header_shape_rejected_before_array_load(self):
        self.save(weights=np.ones((10,10)))
        with self.assertRaisesRegex(GateError,'ARRAY_HEADER'):validate_processing(self.path,self.data)

    def test_extraneous_arrays_rejected(self):
        np.savez(self.path,start_row=np.arange(100),weight=np.ones(100),sampling_multiplier=np.ones(100),truth=np.ones(100))
        with self.assertRaisesRegex(GateError,'ARRAY_SCHEMA'):validate_processing(self.path,self.data)

    def test_parameter_ranges_and_integer_types(self):
        self.assertEqual(parameters({})['batch_size'],256)
        for value in ({'batch_size':256.},{'input_length':64},{'learning_rate':float('nan')}):
            with self.assertRaises(GateError):parameters(value)

    def test_diagnostic_overflow_is_not_finite_json(self):
        path=self.path.with_suffix('.json');path.write_text('{"value":1e999}')
        with self.assertRaisesRegex(GateError,'NONFINITE'):diagnostic_json(path)


@unittest.skipUnless(sys.platform=='linux','Native executor runs on the Linux server')
class NativeTests(unittest.TestCase):
    def setUp(self):
        self.temp=temporary();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.input=self.root/'metadata.json'
        self.input.write_text('{"kind":"engineering"}')
        self.asset={'path':str(self.input),'name':'metadata.json','role':'training','sha256':file_hash(self.input)}

    def run_code(self,code,seconds=3):
        folder=self.root/'job';prepare_job(folder,code,[self.asset])
        return execute_job(folder,{str(self.input):file_hash(self.input)},seconds=seconds),folder

    def test_success_cache_and_no_credential_in_child_environment(self):
        code="import json,os,sys,pathlib; pathlib.Path(sys.argv[2],'diagnostic.json').write_text(json.dumps({'key_present':'DEEPSEEK_API_KEY' in os.environ}))"
        result,folder=self.run_code(code)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(json.loads((folder/'outputs/diagnostic.json').read_text())['key_present'])
        self.assertEqual(execute_job(folder,{str(self.input):file_hash(self.input)}),result)

    def test_timeout_is_retained_and_not_replayed(self):
        result,folder=self.run_code('import time;time.sleep(10)',seconds=.2)
        self.assertEqual(result['error'],'NATIVE_TIMEOUT')
        with self.assertRaisesRegex(GateError,'NOT_REPLAYED'):execute_job(folder,{str(self.input):file_hash(self.input)})

    def test_log_overflow_stops_process(self):
        result,_=self.run_code("print('x'*100000)")
        self.assertEqual(result['error'],'NATIVE_LOG_LIMIT')

    def test_staged_input_write_detected_after_execution(self):
        folder=self.root/'job';prepare_job(folder,"import pathlib,sys;pathlib.Path(sys.argv[1],'metadata.json').write_text('changed')",[self.asset])
        with self.assertRaisesRegex(GateError,'PROTECTED_ASSET_CHANGED'):execute_job(folder,{str(self.input):file_hash(self.input)})
        self.assertEqual(json.loads((folder/'execution.json').read_text())['status'],'failed')

    def test_output_link_rejected(self):
        result,_=self.run_code("import pathlib,sys;pathlib.Path(sys.argv[2],'alias').symlink_to('/etc/passwd')")
        self.assertEqual(result['error'],'NATIVE_OUTPUT_LINK')

    def test_cached_output_modification_rejected(self):
        result,folder=self.run_code("import pathlib,sys;pathlib.Path(sys.argv[2],'diagnostic.json').write_text('{}')")
        (folder/'outputs/diagnostic.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(GateError,'CACHED_OUTPUT_CHANGED'):execute_job(folder,{str(self.input):file_hash(self.input)})

    def test_ordinary_descendant_is_stopped_after_parent_exit(self):
        code="import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(20)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);print(child.pid)"
        result,_=self.run_code(code)
        self.assertEqual(result['status'],'completed')
        pid=int(result['stdout'].strip())
        status=Path(f'/proc/{pid}/status')
        if status.exists():
            self.assertIn('Z',next(line for line in status.read_text().splitlines() if line.startswith('State:')))

    def test_file_count_limit_is_recorded(self):
        result,_=self.run_code("import pathlib,sys;root=pathlib.Path(sys.argv[2]);[(root/str(i)).write_text('x') for i in range(33)]")
        self.assertEqual(result['error'],'NATIVE_OUTPUT_LIMIT')


if __name__=='__main__':unittest.main()
