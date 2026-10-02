"""Independent invariants on real frozen windows for the matched swap experiment."""
import numpy as np
from run_research import ROOT
from run_selection import sampling_policy,training_months,training_lps_mask


def main():
    with np.load(ROOT/'research_runs/selection/window_table.npz') as table:
        starts=table['start_row_zero_based']; score=table['input_variation']
    months=training_months(starts,32); rare=training_lps_mask(starts,32)
    threshold=np.quantile(score,.25)
    base,_,_=sampling_policy('month_matched_drop',score,threshold,months)
    with np.load(ROOT/'research_runs/results/S015_month_matched_drop_s2026/sampling_weights.npz') as saved:
        assert np.array_equal(base,saved['sampling_weight'])
    protect,loss,p=sampling_policy('protect_lps_matched',score,threshold,months,rare)
    placebo,other_loss,q=sampling_policy('placebo_matched_swap',score,threshold,months,rare)
    assert rare.sum()==2663 and np.sum(rare&(base==0))==422
    assert np.all(protect[rare]==1) and np.array_equal(placebo[rare],base[rare])
    assert np.array_equal((base>0)&(protect==0),(base>0)&(placebo==0))
    assert np.all(loss==1) and np.all(other_loss==1)
    assert np.isclose(p.sum(),1) and np.isclose(q.sum(),1)
    for policy in (protect,placebo):
        assert np.count_nonzero(policy)==790074
        assert np.sum((base==0)&(policy>0))==422
        assert np.sum((base>0)&(policy==0))==422
        for month in np.unique(months):
            group=months==month
            assert np.count_nonzero(policy[group])==np.count_nonzero(base[group])
    # Direct source-window calculation spot-checks both positive and zero masks.
    raw=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=13)
    check=np.r_[np.flatnonzero(rare)[:100],np.flatnonzero(~rare)[::10000]]
    assert all(rare[i]==bool(np.any(raw[starts[i]:starts[i]+32]==1)) for i in check)
    print('PASS: raw input-only state mask, exact previous mask reproduction, 422 matched swaps, same replacement deletions, fixed month counts and 790074 retained.')


if __name__=='__main__':
    main()
