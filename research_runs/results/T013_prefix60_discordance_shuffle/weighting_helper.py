"""Input-only loss weights and month-matched shuffled negative control."""
import numpy as np

SHUFFLE_SEED=20261001
EMPHASIS_WEIGHT=2.0


def build_weights(raw,starts,times,mapping,shuffle=False):
    assert starts.min()>=0 and starts.max()+32<=len(raw)
    changed=np.zeros((len(starts),len(mapping)),dtype=bool)
    for channel in range(len(mapping)):
        prefix=np.r_[0,np.cumsum(raw[1:,channel]!=raw[:-1,channel])]
        changed[:,channel]=(prefix[starts+31]-prefix[starts])>0
    rule=(~changed)&changed[:,mapping]
    applied=rule.copy(); months=times[starts+31].astype('datetime64[M]').astype(str)
    rng=np.random.default_rng(SHUFFLE_SEED)
    if shuffle:
        for channel in range(len(mapping)):
            for month in np.unique(months):
                selected=np.flatnonzero(months==month)
                applied[selected,channel]=rng.permutation(rule[selected,channel])
    raw_weight=np.where(applied,EMPHASIS_WEIGHT,1.)
    mean=raw_weight.mean(0); weights=raw_weight/mean
    counts={month:{'rule':rule[months==month].sum(0).tolist(),'applied':applied[months==month].sum(0).tolist()} for month in np.unique(months)}
    assert all(v['rule']==v['applied'] for v in counts.values())
    return weights,rule,applied,months,{'shuffle':shuffle,'shuffle_seed':SHUFFLE_SEED if shuffle else None,
        'raw_emphasis_weight':EMPHASIS_WEIGHT,'normalization':mean.tolist(),'rule_count':rule.sum(0).tolist(),
        'month_counts':counts,'input_only':True,'fitted_rows':len(raw),'windows':len(starts)}
