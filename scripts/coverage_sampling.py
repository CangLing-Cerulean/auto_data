"""Full-support state and recency sampling; real training timestamps only."""
import csv
import json
from pathlib import Path
import numpy as np
from run_research import ROOT,digest,write_json

FOLDER=ROOT/'research_runs/coverage_sampling'
POLICIES=('balanced','recent','balanced_recent','recent_shuffle')
UNIFORM_MIX=0.5
HALF_LIFE_DAYS=30.
SHUFFLE_SEED=20261004
ROUND_START=4114.524273328483
ROUND_LIMIT=min(7200.,ROUND_START+1200.)


def used_seconds():
    return sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in (ROOT/'research_runs/results').glob('*/status.json'))


def identity(policy):
    assert policy in POLICIES
    fit=ROOT/'research_runs/cluster_selection/fit.json'; meta=json.loads(fit.read_text())
    groups=fit.parent/'window_groups.npz'
    assert digest(groups)==meta['window_groups_sha256']
    assert digest(ROOT/'data/metropt3_v1/manifest.json')==meta['manifest_sha256']
    return {'coverage_policy':policy,'coverage_protocol_sha256':digest(FOLDER/'PROTOCOL.md'),
        'coverage_helper_sha256':digest(Path(__file__)),'group_fit_sha256':digest(fit),
        'group_labels_sha256':digest(groups),'coverage_round_total_limit_seconds':ROUND_LIMIT}


def snapshot(run):
    (run/'coverage_helper.py').write_bytes(Path(__file__).read_bytes())
    (run/'coverage_protocol.md').write_bytes((FOLDER/'PROTOCOL.md').read_bytes())


def build(starts,policy,run,times=None):
    assert policy in POLICIES
    with np.load(ROOT/'research_runs/cluster_selection/window_groups.npz') as z:
        idx=np.searchsorted(z['start_row'],starts)
        assert np.array_equal(z['start_row'][idx],starts)
        groups=z['group'][idx]
    if times is None:
        with (ROOT/'data/metropt3_v1/train.csv').open(newline='') as f:
            reader=csv.reader(f); next(reader); times=np.array([row[1] for row in reader],dtype='datetime64[s]')
    assert starts.max()+40<=len(times)
    input_end=times[starts+31].astype('datetime64[s]').astype(np.int64)
    age=(input_end.max()-input_end)/86400.
    recency=np.exp2(-age/HALF_LIFE_DAYS)
    if policy=='recent_shuffle':
        rng=np.random.default_rng(SHUFFLE_SEED)
        for g in np.unique(groups):
            members=np.flatnonzero(groups==g); recency[members]=rng.permutation(recency[members])
    n=len(starts); unique=np.unique(groups); probability=np.full(n,UNIFORM_MIX/n,dtype=np.float64)
    group_info=[]
    for g in unique:
        members=groups==g; count=int(members.sum())
        group_mass=(count/n) if policy in ('recent','recent_shuffle') else 1/len(unique)
        within=np.full(count,1/count) if policy=='balanced' else recency[members]/recency[members].sum()
        probability[members]+=(1-UNIFORM_MIX)*group_mass*within
    probability/=probability.sum()
    assert np.isfinite(probability).all() and probability.min()>=(UNIFORM_MIX/n)*(1-1e-12)
    for g in unique:
        members=groups==g
        group_info.append({'group':int(g),'windows':int(members.sum()),'uniform_mass':float(members.mean()),'sampling_mass':float(probability[members].sum())})
    months=times[starts+31].astype('datetime64[M]').astype(str)
    np.savez_compressed(run/'coverage_policy.npz',start_row=starts,group=groups,input_end_unix_seconds=input_end,probability=probability)
    write_json(run/'coverage_audit.json',{'policy':policy,'retained_windows':n,'removed_windows':0,
        'uniform_mix':UNIFORM_MIX,'half_life_days':HALF_LIFE_DAYS,'shuffle_seed':SHUFFLE_SEED if policy=='recent_shuffle' else None,
        'probability_sum':float(probability.sum()),'min_relative_to_uniform':float(probability.min()*n),'max_relative_to_uniform':float(probability.max()*n),
        'effective_sample_size':float(1/np.square(probability).sum()),'groups':group_info,
        'months':{m:{'windows':int((months==m).sum()),'sampling_mass':float(probability[months==m].sum())} for m in np.unique(months)},
        'asset_sha256':digest(run/'coverage_policy.npz'),'uses_targets':False,'loss_weight':1})
    return probability
