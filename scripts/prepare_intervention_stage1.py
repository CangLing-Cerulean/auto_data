"""Prepare disjoint derived tasks without changing or relabelling base splits."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def load(folder,split,manifest):
    path=folder/(split+'.csv');entry=next(x for x in manifest['splits'] if x['name']==split)
    assert sha(path)==entry['sha256']
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);header=next(reader)
        ti=header.index('timestamp' if manifest['dataset_id']=='metropt3_v1' else 'datetime')
        rows=list(reader)
    values=np.asarray([[float(row[header.index(c)]) for c in manifest['sensor_columns']] for row in rows],dtype=np.float64)
    times=np.asarray([row[ti] for row in rows],dtype='datetime64[s]')
    assert len(values)==entry['rows'] and np.isfinite(values).all()
    if manifest['dataset_id']=='skab_v1':
        files=np.asarray([row[header.index('source_file')] for row in rows])
        source_rows=np.asarray([int(row[header.index('source_row')]) for row in rows],dtype=np.int64)
        assignment=np.asarray([int(hashlib.sha256(('stage1-r1-grouping|'+name).encode()).hexdigest()[:16],16)%2 for name in files],dtype=np.int8)
        definition=dict(kind='whole_original_file_sha256_parity',salt='stage1-r1-grouping|',source_assignments={name:int(assignment[np.flatnonzero(files==name)[0]]) for name in sorted(set(files))})
    else:
        delta=np.diff(times).astype('timedelta64[s]').astype(np.int64);assert (delta>0).all()
        boundaries=np.flatnonzero(delta>30)+1;assert len(boundaries)>0
        cut=int(min(boundaries,key=lambda position:(abs(position-len(values)/2),position)))
        assignment=(np.arange(len(values))>=cut).astype(np.int8)
        files=np.asarray(['metropt3']*len(values));source_rows=np.arange(len(values),dtype=np.int64)
        definition=dict(kind='natural_gap_nearest_partition_midpoint',cut_base_row=cut,before=str(times[cut-1]),after=str(times[cut]),gap_seconds=int(delta[cut-1]))
    original_indices=np.asarray([int(row[0]) for row in rows],dtype=np.int64) if manifest['dataset_id']=='metropt3_v1' else source_rows.copy()
    return dict(values=values,times=times,files=files,source_rows=source_rows,original_indices=original_indices,assignment=assignment,definition=definition)

def windows(part,chosen):
    # Reorder each original file into a contiguous block; retain base row identity.
    order=chosen[np.lexsort((part['source_rows'][chosen],part['files'][chosen]))]
    values=part['values'][order];times=part['times'][order];files=part['files'][order];rows=part['source_rows'][order]
    starts=[];groups=[];strata=[]
    for group_index,name in enumerate(sorted(set(files))):
        idx=np.flatnonzero(files==name)
        breaks=(np.diff(times[idx]).astype('timedelta64[s]').astype(np.int64)>30)|(np.diff(rows[idx])!=1)
        assert np.all(np.diff(times[idx]).astype('timedelta64[s]').astype(np.int64)>0)
        prefix=np.r_[0,np.cumsum(breaks)];s=np.arange(max(0,len(idx)-39));s=s[prefix[s+39]==prefix[s]]
        starts.extend(idx[s].tolist());groups.extend([group_index]*len(s))
        strata.extend((group_index*4+np.minimum(3,np.arange(len(s))*4//max(1,len(s)))).tolist())
    starts=np.asarray(starts,dtype=np.int64);groups=np.asarray(groups,dtype=np.int16);strata=np.asarray(strata,dtype=np.int16)
    assert len(starts)>=256
    assert np.all(files[starts]==files[starts+39]) and np.all(rows[starts+39]-rows[starts]==39)
    elapsed=(times[starts+39]-times[starts+31]).astype('timedelta64[s]').astype(np.int64)
    return dict(values=values,times=times,files=files,source_rows=rows,original_indices=part['original_indices'][order],base_rows=order,starts=starts,groups=groups,strata=strata,elapsed=elapsed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--metro-root',required=True);p.add_argument('--skab-root',required=True);a=p.parse_args()
    assert os.name=='posix';root=Path(a.root);spec=read(root/'preparation_protocol.json');started=time.monotonic()
    assert sha(Path(__file__))==spec['preparer_sha256']
    output=root/'tasks';assert not output.exists();output.mkdir()
    summaries=[];grouping={}
    for dataset,folder,prefix in [('metropt3_v1',Path(a.metro_root),'C'),('skab_v1',Path(a.skab_root),'S')]:
        manifest=read(folder/'manifest.json');assert sha(folder/'manifest.json')==spec['manifest_sha256'][dataset]
        raw={s:load(folder,s,manifest) for s in ('train','validation')}
        grouping[dataset]={s:raw[s]['definition'] for s in raw}
        for index in range(2):
            task=prefix+str(index+1);dest=output/task;dest.mkdir()
            parts={s:windows(raw[s],np.flatnonzero(raw[s]['assignment']==index)) for s in raw}
            mean=parts['train']['values'].mean(0);std=parts['train']['values'].std(0);scale=np.where(std==0,1.,std)
            arrays={}
            for split,part in parts.items():
                arrays[split]=((part['values']-mean)/scale).astype(np.float32)
                for key in ('base_rows','source_rows','original_indices','starts','groups','strata','times','files'):arrays[split+'_'+key]=part[key]
            arrays['mean']=mean;arrays['scale']=scale
            np.savez_compressed(dest/'arrays.npz',**arrays)
            info=dict(task_id=task,dataset_id=dataset,role='construction' if prefix=='C' else 'selection',manifest_sha256=sha(folder/'manifest.json'),array_sha256=sha(dest/'arrays.npz'),sensor_columns=manifest['sensor_columns'],normalization=dict(mean=mean.tolist(),scale=scale.tolist()),
                train_rows=len(parts['train']['values']),development_rows=len(parts['validation']['values']),train_windows=len(parts['train']['starts']),development_windows=len(parts['validation']['starts']),
                base_partition_membership_preserved=True,new_records_created=False,
                forecast_elapsed_seconds={s:dict(min=int(v['elapsed'].min()),max=int(v['elapsed'].max())) for s,v in parts.items()},source_files={s:sorted(set(v['files'])) for s,v in parts.items()})
            write(dest/'receipt.json',info);summaries.append(info)
        for split in raw:
            assert set(np.flatnonzero(raw[split]['assignment']==0)).isdisjoint(set(np.flatnonzero(raw[split]['assignment']==1)))
        del raw
    result=dict(status='prepared',tasks=summaries,grouping=grouping,new_trainings=0,test_access=False,process_seconds=time.monotonic()-started,
                limitation='MetroPT3 historical construction; SKAB previously seen pooled development source. New derived task responses are prospective; only Hydraulic is reserved for source confirmation.')
    write(root/'preparation_receipt.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('tasks','grouping')}))

if __name__=='__main__':main()
