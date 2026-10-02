"""Frozen train-input groups, immutable membership masks and round budget."""
import json
from pathlib import Path
import numpy as np
from run_research import ROOT,digest,write_json

FOLDER=ROOT/'research_runs/cluster_selection'
ROUND_START=3865.6280975155532
ROUND_LIMIT=min(7200.,ROUND_START+1800.)


def used_seconds():
    return sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in (ROOT/'research_runs/results').glob('*/status.json'))


def features(raw,starts,check=lambda:None):
    result=np.empty((len(starts),30),dtype=np.float64)
    for lo in range(0,len(starts),4096):
        check(); b=raw[starts[lo:lo+4096,None]+np.arange(32)]
        result[lo:lo+len(b)]=np.concatenate([b.mean(1),b.std(1)],axis=1)
    assert np.isfinite(result).all()
    return result


def assign(values,centers):
    # Direct squared distances avoid cancellation near a center.
    distances=np.stack([np.square(values-c).sum(1) for c in centers],axis=1)
    return distances.argmin(1),distances.min(1)


def fit_centers(values,rng,check=lambda:None):
    chosen=[int(rng.integers(len(values)))]; centers=[values[chosen[0]].copy()]
    for _ in range(3):
        check(); _,distance=assign(values,np.array(centers))
        if distance.sum()==0: raise RuntimeError('Fewer than four distinct states; do not change k')
        idx=int(rng.choice(len(values),p=distance/distance.sum())); chosen.append(idx); centers.append(values[idx].copy())
    centers=np.array(centers); previous=None; converged=False
    for iteration in range(1,101):
        check(); labels,distance=assign(values,centers)
        if previous is not None and np.array_equal(labels,previous): converged=True; break
        updated=centers.copy(); empty=[]
        for c in range(4):
            if np.any(labels==c): updated[c]=values[labels==c].mean(0)
            else: empty.append(c)
        if empty:
            farthest=np.argsort(-distance,kind='stable')[:len(empty)]
            for c,index in zip(empty,farthest): updated[c]=values[index]
        centers=updated; previous=labels.copy()
    return centers,{'iterations':iteration,'converged':converged,'initial_sample_indices':chosen}


def create_mask(stage,group,random_control=False):
    path=FOLDER/f'mask_{stage}_g{group}_{"random" if random_control else "target"}.npz'
    if path.exists():
        load_mask(path,None); return path
    meta=json.loads((FOLDER/'fit.json').read_text())
    with np.load(FOLDER/'window_groups.npz') as z:
        cut=int(meta['raw_rows']*stage/100)
        select=z['start_row']+40<=cut
        starts=z['start_row'][select]; labels=z['group'][select]; months=z['months'][select]
    target=labels==group; fraction=float(target.mean())
    if fraction<.01 or fraction>.5: raise ValueError(f'Ineligible group {group} at {stage}%: {fraction}')
    removed=target.copy()
    if random_control:
        removed[:]=False; rng=np.random.default_rng(20261003)
        for month in np.unique(months):
            idx=np.flatnonzero(months==month); count=int(target[idx].sum())
            removed[rng.choice(idx,count,replace=False)]=True
    retained=~removed; probability=retained.astype(np.float64)/retained.sum()
    np.savez_compressed(path,start_row=starts,group=labels,retained=retained,months=months,probability=probability)
    write_json(path.with_suffix('.json'),{'stage':stage,'group':group,'random_control':random_control,
        'manifest_sha256':meta['manifest_sha256'],'fit_sha256':digest(FOLDER/'fit.json'),
        'model_sha256':digest(FOLDER/'cluster_model.npz'),'window_groups_sha256':digest(FOLDER/'window_groups.npz'),
        'asset_sha256':digest(path),'eligible_windows':len(starts),'removed_windows':int(removed.sum()),
        'month_counts':{m:{'eligible':int((months==m).sum()),'target_removed':int(target[months==m].sum()),'actual_removed':int(removed[months==m].sum())} for m in np.unique(months)}})
    return path


def load_mask(path,starts):
    path=Path(path).resolve()
    if path.parent!=FOLDER.resolve(): raise ValueError('Mask outside frozen study folder')
    meta=json.loads(path.with_suffix('.json').read_text())
    assert digest(path)==meta['asset_sha256']
    assert digest(FOLDER/'fit.json')==meta['fit_sha256']
    assert digest(FOLDER/'cluster_model.npz')==meta['model_sha256']
    assert digest(ROOT/'data/metropt3_v1/manifest.json')==meta['manifest_sha256']
    with np.load(path) as z:
        if starts is not None: assert np.array_equal(z['start_row'],starts)
        retained=z['retained'].copy(); probability=z['probability'].copy()
        assert retained.dtype==np.bool_ and retained.mean()>=.5 and (~retained).mean()>=.01
        assert np.array_equal(probability,retained.astype(np.float64)/retained.sum())
    return retained,probability,meta


def mask_identity(path):
    _,_,meta=load_mask(path,None)
    return {'retention_mask_sha256':meta['asset_sha256'],'retention_meta_sha256':digest(Path(path).with_suffix('.json')),
        'cluster_protocol_sha256':digest(FOLDER/'PROTOCOL.md'),'cluster_helper_sha256':digest(Path(__file__)),
        'cluster_fit_sha256':meta['fit_sha256'],'round_total_limit_seconds':ROUND_LIMIT}


def snapshot_mask(path,run):
    for source,name in [(Path(path),'retention_mask.npz'),(Path(path).with_suffix('.json'),'retention_mask.json'),(FOLDER/'PROTOCOL.md','cluster_protocol.md'),(Path(__file__),'cluster_helper.py')]:
        (run/name).write_bytes(source.read_bytes())
