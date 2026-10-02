"""Check membership and probability mechanics on real source windows."""
import numpy as np
import torch
from run_research import ROOT
from run_selection import sampling_policy,variation


def main():
    values=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17),max_rows=10000)
    mean,scale=values.mean(0),values.std(0)
    values=((values-mean)/np.where(scale==0,1,scale)).astype(np.float32)
    starts=np.arange(len(values)-39)
    score=variation(values,starts,32)
    threshold=float(np.quantile(score,.25))
    quiet=score<=threshold
    for recipe in ('uniform','drop_quiet','downweight_quiet','random_drop','oversample_high'):
        sample,loss,prob=sampling_policy(recipe,score,threshold)
        assert np.isclose(prob.sum(),1) and np.all(prob>=0)
        if recipe in ('uniform','downweight_quiet'):
            assert np.all(prob==prob[0])
        if recipe=='drop_quiet':
            assert np.all(prob[quiet]==0) and np.all(prob[~quiet]>0)
        if recipe=='downweight_quiet':
            assert np.all(loss[quiet]==.25) and np.all(loss[~quiet]==1)
        else:
            assert np.all(loss==1)
        if recipe=='random_drop':
            assert np.count_nonzero(prob==0)==quiet.sum()
            assert np.array_equal(prob,sampling_policy(recipe,score,threshold)[2])
        if recipe=='oversample_high':
            high=score>np.quantile(score,.9)
            assert np.all(sample[high]==2) and np.all(sample[~high]==1) and np.all(prob>0)
            assert np.isclose(prob[high][0]/prob[~high][0],2)
        cdf=torch.from_numpy(np.cumsum(prob))
        cdf[-1]=1
        draws=torch.searchsorted(cdf,torch.rand(100000,dtype=torch.float64,generator=torch.Generator().manual_seed(2026)),right=True).numpy()
        assert draws.min()>=0 and draws.max()<len(prob) and np.all(prob[draws]>0)
        assert abs(float(quiet[draws].mean())-float(prob[quiet].sum()))<.01
    print('PASS: actual-window membership, probability normalization, zero-probability exclusion, empirical sampling mass.')


if __name__=='__main__':
    main()
