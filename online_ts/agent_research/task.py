"""Bind cached training-only features to the immutable original server dataset."""
import numpy as np
from pathlib import Path

from ..config import GateError
from ..selection.common import file_hash, read_json, write_json
from ..selection.dataset import FEATURES

BASE_ROOT=Path('/root/data/auto_data_research/metropt3_v1/session_20260926')
EXPECTED_MANIFEST='672c0ecf55bc94a52abc6b256c0bfe7a4da02140e57dd0527375bf859a22c087'
EXPECTED_PROTOCOL='573ee4e87ef442b7140afb26f4befe80a8de1a43fec3b5c1fe6b878e527f8afe'
EXPECTED_CORE='4f694b900ca40d2a0620da1dbe916b6c78e450dcc0c903af4a7c696fcaec3052'


class TaskAssets:
    def __init__(self,folder,*,base_root=BASE_ROOT):
        self.folder=Path(folder); self.base_root=Path(base_root)
        self.task=read_json(self.folder/'task.json')
        self.feature_path=self.folder/'features.npz'
        paths={str(self.base_root/'data/metropt3_v1/manifest.json'):EXPECTED_MANIFEST,
               str(self.base_root/'research_runs/protocol.json'):EXPECTED_PROTOCOL,
               str(self.base_root/'scripts/run_research.py'):EXPECTED_CORE,
               str(self.feature_path):self.task['features_sha256']}
        if self.task['manifest_sha256']!=EXPECTED_MANIFEST or self.task['protocol_sha256']!=EXPECTED_PROTOCOL:
            raise GateError('RESEARCH_TASK_VERSION')
        manifest=read_json(self.base_root/'data/metropt3_v1/manifest.json')
        if manifest['dataset_id']!=self.task['dataset_id']:
            raise GateError('RESEARCH_DATASET_ID')
        for split in manifest['splits']:
            if split['name'] in {'train','validation'}:
                paths[str(self.base_root/'data/metropt3_v1'/split['file'])]=split['sha256']
        for path,expected in paths.items():
            if file_hash(path)!=expected:raise GateError('RESEARCH_BASE_ASSET_CHANGED')
        self.guards=paths
        with np.load(self.feature_path,allow_pickle=False) as saved:
            self.data={k:saved[k].copy() for k in saved.files}
        if len(self.data['start_row'])!=self.task['legal_windows']:
            raise GateError('RESEARCH_FEATURE_WINDOW_COUNT')

    def export_inputs(self):
        destination=self.folder/'train_inputs.npz'
        identity=self.folder/'train_inputs_identity.json'
        if destination.exists() or identity.exists():
            if not destination.exists() or not identity.exists() or read_json(identity)['sha256']!=file_hash(destination):
                raise GateError('RESEARCH_INPUT_CACHE_CHANGED')
            return destination
        raw=np.loadtxt(self.base_root/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17))
        mean,scale=np.asarray(self.task['mean']),np.asarray(self.task['scale'])
        if not np.allclose(raw.mean(0),mean,rtol=0,atol=1e-12) or not np.allclose(raw.std(0),scale,rtol=0,atol=1e-12):
            raise GateError('RESEARCH_NORMALIZATION_IDENTITY')
        arrays={k:self.data[k] for k in (*FEATURES,'start_row','groups','group_names',
               'source_index_start','source_index_target_start','source_index_end_inclusive')}
        np.savez_compressed(destination,values=((raw-mean)/scale).astype(np.float32),
                            sensor_columns=np.asarray(self.task['sensor_columns']),**arrays)
        write_json(identity,{'sha256':file_hash(destination),'train_sha256':self.task['train_sha256'],
                             'features_sha256':self.task['features_sha256'],'scope':'training_only'})
        return destination

    def profile(self):
        return {'scope':'training_only','windows':len(self.data['start_row']),
            'groups':len(self.data['group_names']),'sensor_columns':self.task['sensor_columns'],
            'features':{k:{'min':float(self.data[k].min()),'max':float(self.data[k].max()),
              'quantiles':np.quantile(self.data[k],[.1,.25,.5,.75,.9]).tolist()} for k in FEATURES},
            'interpretation':'descriptive T diagnostics, not verified fault labels'}
