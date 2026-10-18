"""Full-window parent-only measurement interface; no child responses are accepted."""
import argparse
from pathlib import Path
import time
import numpy as np
from diagnosis_pool_features import read,sha,write

INPUT_COLUMNS=('level','level_energy','variance','roughness','endpoint_change','channel_energy_dispersion','high_frequency_fraction','absolute_level')
ERROR_COLUMNS=('loss','target_loss','complement_loss','bias','absolute_error')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261018_parent_measurements_r3'


def inputs(values,starts,length):
    out=[]
    for lo in range(0,len(starts),256):
        x=values[starts[lo:lo+256,None]+np.arange(length)].astype(np.float64)
        level=x.mean(axis=1);centered=x-level[:,None,:]
        power=abs(np.fft.rfft(centered,axis=1))**2
        total=power[:,1:].sum(axis=(1,2));high=power[:,(power.shape[1]+1)//2:].sum(axis=(1,2))
        fraction=np.divide(high,total,out=np.zeros_like(high),where=total>0)
        out.append(np.column_stack([x.mean(axis=(1,2)),(level**2).mean(axis=1),x.var(axis=1).mean(axis=1),np.square(np.diff(x,axis=1)).mean(axis=(1,2)),((x[:,-1]-x[:,0])**2).mean(axis=1),np.std((x*x).mean(axis=1),axis=1),fraction,np.abs(x).mean(axis=(1,2))]))
    result=np.concatenate(out)
    assert result.shape==(len(starts),len(INPUT_COLUMNS)) and np.isfinite(result).all()
    return result


def errors(error):
    x=np.asarray(error,dtype=np.float64);mid=x.shape[1]//2
    assert x.ndim==3 and mid*2==x.shape[1]
    mse=(x*x).mean();assert mse>0
    result=np.column_stack([(x*x).mean(axis=(1,2))/mse,(x[:,mid:]**2).mean(axis=(1,2))/mse,(x[:,:mid]**2).mean(axis=(1,2))/mse,x.mean(axis=(1,2))/np.sqrt(mse),abs(x).mean(axis=(1,2))/np.sqrt(mse)])
    assert np.isfinite(result).all() and abs(result[:,0].mean()-1)<1e-12
    assert np.allclose(result[:,0],.5*(result[:,1]+result[:,2]),rtol=1e-12,atol=1e-12)
    return result


def save(path,**arrays):
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**arrays)


def hydraulic():
    directory=STAGE/'tables';directory.mkdir();start=time.perf_counter();assets={};checks=[]
    plan=read(BASE/'20261014_weighted_response_r1/plan.json')
    dev=BASE/'20261013_response_proxy_development_r1/archives/verified'
    for g in range(1,4):
        task=f'G{g}';path=dev/'prepared'/f'{task}_ABC.npz'
        assert {v for k,v in plan['input_sha256'].items() if k.endswith(f'/{task}_ABC.npz')}=={sha(path)}
        with np.load(path,allow_pickle=False) as z:
            a,b=z['A_values'],z['B_values'];astart,bstart=z['A_starts'],z['B_starts'];cycles=z['A_cycles'][astart]
        first=cycles<(int(cycles.min())+int(cycles.max())+1)/2
        table=dict(A_inputs=inputs(a,astart,32),B_inputs=inputs(b,bstart,32),A_first=first,A_starts=astart,B_starts=bstart)
        for seed in (2026101701,2026101702,2026101703):
            run=dev/'runs'/f'DEV_{task}_full_PARENT_s{seed}';result=read(run/'result.json');error=run/'selected_B_errors.npz'
            assert sha(error)==result['assets_sha256'][error.name]
            with np.load(error,allow_pickle=False) as z:e=z['error']
            assert len(e)==len(bstart) and abs(float((e.astype(np.float64)**2).mean())-result['selected_losses']['B'])<1e-12
            table[f'B_errors_s{seed}']=errors(e)
            checks.append(dict(task=task,seed=seed,parent_B_sha256=sha(error),windows=len(e)))
        target=directory/f'Hydraulic_{task}.npz';save(target,**table);assets[target.name]=dict(bytes=target.stat().st_size,sha256=sha(target))
    write(STAGE/'hydraulic_measurements.json',dict(status='PASS',assets=assets,checks=checks,input_columns=INPUT_COLUMNS,error_columns=ERROR_COLUMNS,seconds=time.perf_counter()-start,new_full_trainings=0,child_response_access=False,D_test_access=False))
    print(dict(status='PASS',tasks=3,parents=9,seconds=time.perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('hydraulic',));globals()[parser.parse_args().action]()
