"""Sealed-D preparation and inference after both exploration paths are frozen."""
import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read_partition(path, spec, channels):
    rows=np.empty((spec['row_stop']-spec['row_start'],channels),dtype=np.float64);count=0
    with Path(path).open('rb') as stream:
        next(stream)
        for index,line in enumerate(itertools.islice(stream,spec['row_stop'])):
            if index<spec['row_start']:continue
            stamp,_,fields=line.partition(b';')
            values=fields.rstrip(b'\r\n').split(b';')
            assert len(values)==channels
            if index in (spec['row_start'],spec['row_stop']-1):
                time_string=datetime.strptime(stamp.strip(b'"').decode('ascii'),'%Y-%m-%d %H:%M:%S').isoformat()
                assert time_string==spec['first_date' if index==spec['row_start'] else 'last_date']
            rows[count]=[float(v.strip(b'"').replace(b',',b'.')) for v in values];count+=1
    assert count==len(rows) and np.isfinite(rows).all()
    return rows


def prepare():
    root=Path(__file__).resolve().parents[1]
    stage=root/'research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10'
    lock=read(stage/'final_evaluation_lock.json');publication=read(stage/'final_evaluation_publication_receipt.json')
    assert publication['status']=='exact_commit_and_tree_verified'
    paths={stage/'final_evaluation_lock.json':sha(stage/'final_evaluation_lock.json'),
           stage/'final_recommendations.json':lock['recommendations_sha256'],
           stage/'development_audit.json':lock['development_audit_sha256'],
           stage/'source_index.json':lock['source_index_sha256'],
           stage/'FINAL_EVALUATION_PROTOCOL.md':lock['protocol_sha256']}
    paths.update({root/'scripts'/name:value for name,value in lock['code_sha256'].items()})
    for path,digest in paths.items():
        assert sha(path)==digest
        blob=subprocess.run(['git','-c',f'safe.directory={root.as_posix()}','show',publication['source_commit']+':'+path.relative_to(root).as_posix()],capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==digest
    assert read(stage/'development_audit.json')['status']=='PASS'
    recommendations=read(stage/'final_recommendations.json');assert recommendations['status']=='FROZEN'
    assert set(recommendations['recommendations'])=={'STRONG','SEPARATE'} and not recommendations['D_accessed']
    assert sha(stage/'inputs/ABC.npz')==read(stage/'prepared_input_receipt.json')['arrays_sha256']
    raw=root/'data/uci_electricity_v1/raw/LD2011_2014.txt'
    assert sha(raw)==read(stage/'execution_lock.json')['raw_text_sha256']
    start=time.perf_counter();spec=read(stage/'source_index.json')['partitions']['D']
    rows=read_partition(raw,spec,370)
    with np.load(stage/'inputs/ABC.npz',allow_pickle=False) as data:
        mean=data['mean'].copy();std=data['std'].copy()
    values=((rows-mean)/std).astype(np.float32)
    assert values.shape==(2880,370) and np.isfinite(values).all()
    with (stage/'inputs/D.npz').open('xb') as f:np.savez(f,D=values)
    receipt=dict(status='PASS',D_sha256=sha(stage/'inputs/D.npz'),shape=list(values.shape),normalization_fit='A',
                 recommendations_sha256=lock['recommendations_sha256'],final_evaluation_lock_sha256=sha(stage/'final_evaluation_lock.json'),
                 first_D_numeric_access_after_final_recommendations=True,new_full_trainings=0,seconds=time.perf_counter()-start)
    write(stage/'D_input_receipt.json',receipt);return receipt


def evaluate(root):
    import torch
    import coldstart_runtime as runtime
    from coldstart_roundoff import forward_bound
    root=root.resolve();assert root.name=='20261018_cold_start_r10'
    lock=read(root/'final_evaluation_lock.json');recommendations=read(root/'final_recommendations.json')
    assert sha(root/'final_recommendations.json')==lock['recommendations_sha256']
    assert sha(root/'development_audit.json')==lock['development_audit_sha256']
    assert read(root/'development_audit.json')['status']=='PASS'
    assert sha(root/'D.npz')==read(root/'D_input_receipt.json')['D_sha256']
    for name,value in lock['code_sha256'].items():assert sha(root/name)==value
    runtime.configure();started=time.perf_counter()
    with np.load(root/'D.npz',allow_pickle=False) as data:rows=data['D'].copy()
    assert rows.shape==(2880,370) and np.isfinite(rows).all()
    values=torch.as_tensor(rows,device='cuda')
    runs=sorted({'PARENT',*recommendations['recommendations'].values()});records={};reference_checks=[]
    x=runtime.windows(rows);sample=np.array(x[[0,len(x)//2,len(x)-1]])
    for run in runs:
        records[run]={}
        for seed in runtime.SEEDS:
            folder=root/'runs'/run/str(seed);training=read(folder/'result.json')
            path=folder/f"step_{training['best_step']:04d}.pt"
            assert sha(path)==training['assets'][path.name]['sha256']
            model=runtime.best_model(folder);error=runtime.errors(model,values)
            records[run][str(seed)]=runtime.metrics(error)
            with torch.no_grad():actual=model(torch.as_tensor(sample,device='cuda')).cpu().numpy()
            state={k:v.cpu().numpy() for k,v in model.state_dict().items()}
            reference,bound=forward_bound(state,sample);gap=np.abs(actual-reference)
            assert np.all(gap<=bound)
            reference_checks.append(dict(run=run,seed=seed,max_gap=float(gap.max()),max_gap_over_bound=float(np.max(gap/np.maximum(bound,np.finfo(np.float64).tiny)))))
    arms={}
    for arm,run in recommendations['recommendations'].items():
        paired=[]
        for seed in runtime.SEEDS:
            parent=records['PARENT'][str(seed)];child=records[run][str(seed)]
            paired.append(dict(seed=seed,NMSE_gain=1-child['mse']/parent['mse'],NMAE_gain=1-child['mae']/parent['mae']))
        arms[arm]=dict(selected_run=run,paired=paired,mean_NMSE_gain=float(np.mean([r['NMSE_gain'] for r in paired])),mean_NMAE_gain=float(np.mean([r['NMAE_gain'] for r in paired])))
    return dict(status='PASS',records=records,arms=arms,independent_roundoff_checks=reference_checks,
                D_windows=2841,inference_models=len(runs)*3,new_full_trainings=0,new_llm_calls=0,
                recommendations_sha256=lock['recommendations_sha256'],seconds=time.perf_counter()-started,
                D_feedback_to_exploration_agents=False,method_or_recipe_reselection=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','evaluate'))
    parser.add_argument('--root',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=prepare() if args.action=='prepare' else evaluate(args.root)
    if args.output:write(args.output,result)
    print(json.dumps(result))
