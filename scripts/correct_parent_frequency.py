"""Supplement the frozen r3 tables with Parseval-weighted frequency fractions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')

def frequency(values,starts,length):
    outputs=[]
    for lo in range(0,len(starts),256):
        x=values[starts[lo:lo+256,None]+np.arange(length)].astype(np.float64);x-=x.mean(axis=1,keepdims=True)
        power=abs(np.fft.rfft(x,axis=1))**2;weights=np.full(power.shape[1],2.);weights[0]=1.
        if length%2==0:weights[-1]=1.
        weighted=power*weights[None,:,None];total=weighted[:,1:].sum(axis=(1,2));high=weighted[:,(power.shape[1]+1)//2:].sum(axis=(1,2))
        assert np.allclose(total/length,np.square(x).sum(axis=(1,2)),rtol=1e-12,atol=1e-12)
        outputs.append(np.divide(high,total,out=np.zeros_like(high),where=total>0))
    return np.concatenate(outputs)

def main(family):
    start=time.perf_counter()
    if family=='ETT':
        stage=Path('/root/auto_data_experiments/20261018_parent_measurements_r3')
        source=Path('/root/auto_data_experiments/20261016_ett_confirmation_r1')
        tasks=('ETTh1','ETTh2');length=96
    else:
        base=Path(__file__).resolve().parents[1]/'research_runs/agent_research/intervention_diagnosis'
        stage=base/'20261018_parent_measurements_r3';source=base/'20261013_response_proxy_development_r1/archives/verified/prepared';tasks=('G1','G2','G3');length=32
    output=stage/'frequency_corrected';output.mkdir(exist_ok=True);assets={};checks=[]
    original=read(stage/('ett_measurements.json' if family=='ETT' else 'hydraulic_measurements.json'))
    for task in tasks:
        name=f'{family}_{task}.npz';old=stage/'tables'/name;assert sha(old)==original['assets'][name]['sha256']
        with np.load(old,allow_pickle=False) as z:table=dict(z)
        path=source/task/'ABC.npz' if family=='ETT' else source/f'{task}_ABC.npz'
        if family=='ETT':assert sha(path)==read(source/task/'preprocessing.json')['prepared_sha256']
        else:
            plan=read(base/'20261014_weighted_response_r1/plan.json')
            assert {v for k,v in plan['input_sha256'].items() if k.endswith('/'+path.name)}=={sha(path)}
        differences=[]
        with np.load(path,allow_pickle=False) as z:
            for split in ('A','B'):
                values=z[split if family=='ETT' else split+'_values'];starts=table[split+'_starts'];new=frequency(values,starts,length)
                prior=table[split+'_inputs'][:,6].copy();table[split+'_inputs'][:,6]=new
                differences.append(dict(split=split,windows=len(new),max_abs_change=float(abs(new-prior).max()),changed=int(np.count_nonzero(new!=prior))))
        target=output/name
        with target.open('xb') as f:np.savez_compressed(f,**table)
        assets[name]=dict(bytes=target.stat().st_size,sha256=sha(target));checks.append(dict(task=task,old_sha256=sha(old),prepared_sha256=sha(path),changes=differences))
    write(output/f'{family}_correction.json',dict(status='PASS',assets=assets,checks=checks,seconds=time.perf_counter()-start,only_changed_column='A/B_inputs[:,6]',D_test_access=False,new_full_trainings=0,code_sha256=sha(__file__)))
    print(json.dumps(dict(status='PASS',family=family,seconds=time.perf_counter()-start)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('family',choices=('ETT','Hydraulic'));main(parser.parse_args().family)
