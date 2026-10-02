"""Check temporal disjointness, prefix-only scaling and actual selection on real data."""
import json
import numpy as np
import torch
from run_research import ROOT,write_json,digest


def main():
    raw=np.loadtxt(ROOT/'data/metropt3_v1/train.csv',delimiter=',',skiprows=1,usecols=range(2,17))
    with np.load(ROOT/'research_runs/selection/window_table.npz') as table: eligible=table['start_row_zero_based']
    output=[]; baseline_rng={}
    for run in sorted((ROOT/'research_runs/results').glob('T*')):
        if not (run/'result.json').exists(): continue
        r=json.loads((run/'result.json').read_text()); a=json.loads((run/'data_audit.json').read_text())
        cut=a['train_raw_rows'][1]; end=a['probe_raw_rows'][1]
        expected_mean=raw[:cut].mean(0); std=raw[:cut].std(0)
        assert np.array_equal(np.array(a['mean']),expected_mean)
        assert np.array_equal(np.array(a['scale']),np.where(std==0,1,std))
        with np.load(run/'windows.npz') as meta:
            starts=meta['train_start']; future=meta['probe_start']; block=meta['block']; p=meta['probability']; count=meta['draw_count']
            assert np.array_equal(starts,eligible[eligible+40<=cut])
            assert np.array_equal(future,eligible[(eligible>=cut)&(eligible+40<=end)])
            assert starts.max()+40<=future.min() and future.max()+40<=end
            drop=r['identity']['drop_block']; keep=np.ones(len(starts),dtype=bool) if drop<0 else block!=drop
            thin=r['identity'].get('thin','none')
            if thin!='none':
                keep[:]=False
                # Independently reconstruct greedy raw-row groups, including gaps.
                groups=[]; current=[]; anchor=None
                for index,start in enumerate(starts):
                    if anchor is None or start>=anchor+40:
                        if current: groups.append(current)
                        anchor=start; current=[]
                    current.append(index)
                if current: groups.append(current)
                random=np.random.default_rng(20260930)
                for members in groups:
                    selected=members[0] if thin=='disjoint' else members[int(random.integers(len(members)))]
                    keep[selected]=True
                    assert np.count_nonzero(p[members])==1
                if thin=='disjoint': assert np.all(np.diff(starts[keep])>=40)
                assert int(keep.sum())==a['retained_windows']
            assert np.array_equal(p,keep.astype(np.float64)/keep.sum())
            assert count.sum()==768000 and np.all(count[~keep]==0)
            if r['identity'].get('state_balance'):
                from state_balance import build_weights
                weights,states,state_audit=build_weights(raw[:cut],starts)
                recorded=json.loads((run/'state_audit.json').read_text())
                assert recorded.pop('asset_sha256')==digest(run/'state_weights.npz')
                assert recorded==state_audit
                with np.load(run/'state_weights.npz') as saved:
                    assert np.array_equal(saved['loss_weight'],weights)
                    assert np.array_equal(saved['state'],states)
                    assert np.array_equal(saved['start_row'],starts)
                baseline='T000_prefix60_full' if r['identity']['origin']==.6 else 'T005_prefix80_full'
                with np.load(run.parent/baseline/'windows.npz') as saved: assert np.array_equal(saved['draw_count'],count)
            assert abs(np.bincount(block).max()-np.bincount(block).min())<=1
            state=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
            assert state['step']==3000 and np.array_equal(state['draw_counts'].numpy(),count)
            key=(r['identity']['origin'],r['identity']['seed'])
            if key in baseline_rng: assert torch.equal(state['sampling_rng'],baseline_rng[key])
            else: baseline_rng[key]=state['sampling_rng']
        output.append({'run':run.name,'source_probe_disjoint':True,'prefix_only_normalization':True,'actual_membership_and_draws_verified':True})
    write_json(ROOT/'research_runs/transfer/verification.json',output)
    print('PASS:',len(output),'temporal probes; source boundaries, prefix scaling, membership and 768000 exposures verified.')


if __name__=='__main__': main()
