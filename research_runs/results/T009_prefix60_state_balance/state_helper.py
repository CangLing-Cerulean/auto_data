"""Training-history-only channelwise covariate frequency ratios."""
import numpy as np


def build_weights(raw,starts):
    assert np.isfinite(raw).all() and starts.min()>=0 and starts.max()+40<=len(raw)
    boundary=int(len(raw)*.8)
    recent=starts>=boundary
    if not recent.any(): raise ValueError('No recent complete training windows')
    weights=np.empty((len(starts),raw.shape[1]),dtype=np.float64)
    states=np.empty_like(weights,dtype=np.int16); channels=[]
    for channel in range(raw.shape[1]):
        first=raw[starts,channel]; last=raw[starts+31,channel]
        edges=np.unique(np.quantile(last,[.2,.4,.6,.8]))
        level=np.searchsorted(edges,last,side='right')
        prefix=np.r_[0,np.cumsum(np.diff(raw[:,channel])!=0)]
        changed=(prefix[starts+31]-prefix[starts])>0
        direction=np.sign(last-first).astype(np.int16)+1
        state=(level*6+changed.astype(np.int16)*3+direction).astype(np.int16)
        count=np.bincount(state,minlength=(len(edges)+1)*6)
        recent_count=np.bincount(state[recent],minlength=len(count))
        ratio=np.divide(recent_count/recent.sum(),count/len(starts),out=np.zeros(len(count)),where=count>0)
        table=np.minimum(4,.5+.5*ratio)
        normalizer=np.dot(count,table)/len(starts)
        weights[:,channel]=table[state]/normalizer; states[:,channel]=state
        channels.append({'edges':edges.tolist(),'all_count':count.tolist(),'recent_count':recent_count.tolist(),
            'state_weight':(table/normalizer).tolist(),'normalizer':float(normalizer),
            'ESS':float(weights[:,channel].sum()**2/np.square(weights[:,channel]).sum()),
            'min_weight':float(weights[:,channel].min()),'max_weight':float(weights[:,channel].max())})
    assert np.all(weights>0) and np.allclose(weights.mean(0),1,atol=1e-9,rtol=0)
    return weights,states,{'history_raw_rows':[0,len(raw)],'reference_raw_rows':[boundary,len(raw)],
        'reference_windows':int(recent.sum()),'eligible_windows':len(starts),'channels':channels,
        'input_only':True,'sampling':'uniform; original window sequence retained'}
