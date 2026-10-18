"""Reuse all audited Hydraulic/ETT candidate responses; no old training is repeated."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from diagnosis_pool_features import (action_features, combine_features, input_statistics,
                                    parent_features, read, region_response, sha, write)

LOCAL=Path(__file__).resolve().parents[1]
BASE=LOCAL/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261018_conditional_diagnosis_r1'
SEEDS=(2026101701,2026101702,2026101703)
ACTIONS=('KEEP_FIRST','KEEP_SECOND','FAVOR_FIRST','FAVOR_SECOND')


def hydraulic():
    begin=time.perf_counter();records=[];identities={};checks=[]
    dev=BASE/'20261013_response_proxy_development_r1/archives/verified'
    sampling=BASE/'20261014_proxy_selection_recovery_r1/archives/verified'
    weighted=BASE/'20261014_weighted_response_r1/archives/unpacked'
    plan=read(BASE/'20261014_weighted_response_r1/plan.json')
    for g in range(1,4):
        task=f'G{g}'; abc=dev/'prepared'/f'{task}_ABC.npz'
        expected={value for key,value in plan['input_sha256'].items() if key.endswith(f'/{task}_ABC.npz')}
        actual=sha(abc);assert expected=={actual},f'{task} ABC identity'
        identities[str(abc.relative_to(LOCAL))]=actual
        with np.load(abc,allow_pickle=False) as z:
            a,b=z['A_values'],z['B_values'];astarts,bstarts=z['A_starts'],z['B_starts']
            cycles=z['A_cycles'][astarts]
            first=cycles<((int(cycles.min())+int(cycles.max())+1)/2)
            c_count=len(z['C_starts'])
        astats=input_statistics(a,astarts,first,32);bstats=input_statistics(b,bstarts,None,32)
        def errors(run,split):
            result=read(run/'result.json');path=run/f'selected_{split}_errors.npz'
            actual=sha(path);assert actual==result['assets_sha256'][path.name],str(path)
            identities[str(path.relative_to(LOCAL))]=actual
            with np.load(path,allow_pickle=False) as z: error=z['error']
            assert error.ndim==3 and error.shape[1:]==(8,8)
            if split=='C':assert len(error)==c_count
            assert abs(float(np.mean(error.astype(np.float64)**2))-result['selected_losses'][split])<1e-12
            return error,result
        for seed in SEEDS:
            parent=dev/'runs'/f'DEV_{task}_full_PARENT_s{seed}'
            pe,parent_result=errors(parent,'C');be,_=errors(parent,'B')
            curve=[parent_result['selection_losses'][str(step)] for step in range(500,3001,500)]
            features=parent_features(astats,bstats,be,curve,parent_result['selected_step'])
            for action in ACTIONS:
                probability_path=sampling/'prepared'/f'{task}_{action}.npz'
                expected={value for key,value in plan['input_sha256'].items() if key.endswith(f'/{task}_{action}.npz')}
                assert expected=={sha(probability_path)}
                with np.load(probability_path,allow_pickle=False) as z: probability=z['probability']
                af=action_features(action,'S',float(first.mean()))
                assert abs(float(probability[first].sum())-first.mean()-af['exposure_shift_first'])<1e-12
                for impl,root,name in [('S',sampling,f'SELR1_{task}_full_{action}_s{seed}'),
                                       ('W',weighted,f'WR1_{task}_{action}_s{seed}')]:
                    run=root/'runs'/name;ce,result=errors(run,'C')
                    response=region_response(pe,ce)
                    timing=read(run/'timing.json') if (run/'timing.json').exists() else None
                    seconds=timing['seconds'] if timing is not None else result.get('seconds')
                    records.append(dict(family='Hydraulic',task=task,action=action,implementation=impl,seed=seed,
                        features=combine_features(features,action_features(action,impl,float(first.mean()))),
                        response=response,source_id=name,source_result_sha256=sha(run/'result.json'),
                        original_full_training_seconds=seconds,
                        provenance='retrospective new long-horizon slice; original result/audit identities preserved'))
            checks.append(dict(task=task,seed=seed,parent_B_C_and_eight_child_C='hash_and_original_metric_verified'))
    result=dict(status='PASS',records=records,identities=identities,checks=checks,
                seconds=time.perf_counter()-begin,new_full_trainings=0,D_test_access=False,
                target_slice='all channels and windows, latter half of forecast horizon')
    write(STAGE/'hydraulic_extraction.json',result)
    print(dict(status='PASS',records=len(records),seconds=result['seconds']))


def combine():
    inputs=[STAGE/'hydraulic_extraction.json',STAGE/'ett_extraction.json']
    raw=[]
    for path in inputs:
        extraction=read(path);assert extraction['status']=='PASS'
        raw.extend(extraction['records'])
    assert len(raw)==96 and len({r['source_id'] for r in raw})==96
    grouped=defaultdict(list)
    for r in raw:grouped[(r['family'],r['task'],r['action'],r['implementation'])].append(r)
    records=[]
    for (family,task,action,impl),seeds in sorted(grouped.items()):
        assert len(seeds)==3 and len({r['seed'] for r in seeds})==3
        feature_names=sorted(seeds[0]['features'])
        assert all(sorted(r['features'])==feature_names for r in seeds)
        features={key:float(np.mean([r['features'][key] for r in seeds])) for key in feature_names}
        response={key:float(np.mean([r['response'][key] for r in seeds]))
                  for key in ('Y','Z','NMAE_gain','d_target','d_complement','p')}
        assert abs(response['Y']-response['p']*response['d_target']-(1-response['p'])*response['d_complement'])<1e-12
        seconds=[r['original_full_training_seconds'] for r in seeds]
        records.append(dict(id=f'{family}:{task}:{action}:{impl}',family=family,task=task,
                            action=action,implementation=impl,features=features,response=response,
                            seed_Y=[r['response']['Y'] for r in seeds],seed_Z=[r['response']['Z'] for r in seeds],
                            source_ids=[r['source_id'] for r in seeds],
                            original_full_training_seconds=None if any(v is None for v in seconds) else sum(seconds)))
    assert len(records)==32
    write(STAGE/'development_pool.json',dict(status='PASS',records=records,raw_seed_records=raw,
        input_sha256={p.name:sha(p) for p in inputs},task_count=5,source_family_count=2,
        new_full_trainings=0,cumulative_full_trainings=464,
        role='already revealed retrospective development; no independent confirmation',
        slice='latter half of forecast horizon, all windows/channels',
        averaging='mean of per-seed normalized responses; one decision record per task/action/implementation'))
    print(dict(status='PASS',records=len(records),task_count=5,source_family_count=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('hydraulic','combine'))
    globals()[parser.parse_args().action]()
