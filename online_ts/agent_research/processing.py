"""Validate source mapping and training influence before a GPU job is admitted."""
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from ..config import GateError
from ..selection.common import file_hash

PARAMETERS = {'learning_rate': (.0003, .001, .003), 'batch_size': (128,256,512),
              'weight_decay': (0.,.00001,.0001), 'moving_average_kernel': (9,25,49)}
DEFAULTS = {'learning_rate': .001, 'batch_size':256, 'weight_decay':0., 'moving_average_kernel':25}


def parameters(value):
    if not isinstance(value, dict) or set(value)-set(PARAMETERS):
        raise GateError('PROCESSING_PARAMETER_SCHEMA')
    result = DEFAULTS | value
    for name, item in result.items():
        if type(item) not in {int,float} or item not in PARAMETERS[name] or isinstance(item,bool):
            raise GateError('PROCESSING_PARAMETER_RANGE')
        if name in {'batch_size','moving_average_kernel'} and type(item) is not int:
            raise GateError('PROCESSING_INTEGER_PARAMETER')
    return result


def validate_processing(path, data):
    path = Path(path)
    n = len(data['start_row'])
    if path.is_symlink() or not path.is_file():
        raise GateError('PROCESSING_ASSET_MISSING')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if {i.filename for i in infos} != {'start_row.npy','weight.npy','sampling_multiplier.npy'} or len(infos)!=3:
            raise GateError('PROCESSING_ARRAY_SCHEMA')
        if sum(i.file_size for i in infos)>n*24+10000:
            raise GateError('PROCESSING_EXPANDED_SIZE')
        for info in infos:
            with archive.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):
                    shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):
                    shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                else:
                    raise GateError('PROCESSING_ARRAY_HEADER')
                allowed=(np.dtype('int64'),) if info.filename=='start_row.npy' else (np.dtype('float32'),np.dtype('float64'))
                if shape!=(n,) or order or dtype not in allowed:
                    raise GateError('PROCESSING_ARRAY_HEADER')
    with np.load(path, allow_pickle=False) as saved:
        starts, weights, sampling = (saved[k].copy() for k in ('start_row','weight','sampling_multiplier'))
    if starts.dtype!=np.int64 or not np.array_equal(starts,data['start_row']):
        raise GateError('PROCESSING_SOURCE_MAPPING')
    for array in (weights,sampling):
        if array.shape!=(n,) or array.dtype.kind!='f' or not np.isfinite(array).all() or np.any(array<0) or np.any(array>4):
            raise GateError('PROCESSING_WEIGHT_RANGE')
    active=(weights>0)&(sampling>0)
    if active.mean()<.5:
        raise GateError('PROCESSING_MINIMUM_RETENTION')
    probability=np.where(active,sampling,0.).astype(np.float64)
    probability/=probability.sum()
    normalizer=float(np.dot(probability,weights))
    loss_weight=weights.astype(np.float64)/normalizer
    if not np.isfinite(probability).all() or not np.isfinite(loss_weight).all() or not normalizer>0:
        raise GateError('PROCESSING_NORMALIZATION')
    influence=probability*loss_weight
    group_audit=[]
    for group in np.unique(data['groups']):
        members=data['groups']==group
        count=int(members.sum())
        retained=int(active[members].sum())
        mass=float(influence[members].sum())
        if retained<.2*count or count<=64 and retained!=count or mass<.2*count/n:
            raise GateError('PROCESSING_GROUP_COVERAGE')
        group_audit.append({'group':int(group),'windows':count,'active':retained,'relative_influence':mass/(count/n)})
    removed=~active
    overlaps=[]
    for key in sorted(k for k in data if k.startswith('historical_removed_')):
        old=data[key]; domain=data[key.replace('removed','domain')]
        union=(removed|old)&domain
        jaccard=float(((removed&old)&domain).sum()/union.sum()) if union.any() else 0.
        overlaps.append({'historical_mask':key,'deleted_jaccard':jaccard})
        if removed.any() and jaccard>=.95:
            raise GateError('PROCESSING_REPEATS_HISTORICAL_DELETION')
    audit={'scope':'training_only','windows':n,'active':int(active.sum()),'retention':float(active.mean()),
           'loss_weight_normalizer_under_sampling':normalizer,'group_coverage':group_audit,
           'historical_deletion_overlap':overlaps,'source_mapping':'unchanged_T_start_row',
           'sha256':file_hash(path)}
    return probability, loss_weight, audit


def diagnostic_json(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>24000:
        raise GateError('DIAGNOSTIC_JSON_SIZE')
    def invalid(_):
        raise GateError('DIAGNOSTIC_NONFINITE_JSON')
    value=json.loads(path.read_text(encoding='utf-8'),parse_constant=invalid)
    if not isinstance(value,dict):
        raise GateError('DIAGNOSTIC_JSON_OBJECT')
    def finite(item,depth=0):
        if depth>12 or isinstance(item,float) and not math.isfinite(item):
            raise GateError('DIAGNOSTIC_NONFINITE_JSON')
        if isinstance(item,dict):
            for child in item.values():finite(child,depth+1)
        elif isinstance(item,list):
            for child in item:finite(child,depth+1)
    finite(value)
    return value
