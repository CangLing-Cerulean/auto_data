"""Audit actual membership, exposure counts and paired sampling on real windows."""
import json
from pathlib import Path

import numpy as np
import torch
from run_selection import training_months, training_lps_mask, sampling_policy, MONTH_RECIPES, ALIGNMENT_RECIPES, alignment_loss_weights
from run_research import digest

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'research_runs/selection'
    with np.load(folder/'window_table.npz') as table:
        starts=table['start_row_zero_based']
        score=table['input_variation']
    threshold=float(np.quantile(score,.25))
    quiet=score<=threshold
    outcomes=[]
    policy_references={}
    baseline_rng={}
    for run in sorted((ROOT/'research_runs/results').glob('S*')):
        if not (run/'result.json').exists():
            continue
        result=json.loads((run/'result.json').read_text())
        audit=json.loads((run/'sampling_audit.json').read_text())
        seed=result['identity']['seed']
        peers=list((ROOT/'research_runs/results').glob(f'S*_uniform_s{seed}'))
        assert len(peers)==1
        if seed not in baseline_rng:
            baseline_rng[seed]=torch.load(peers[0]/'latest.pt',map_location='cpu',weights_only=False)['sampling_rng']
        current_rng=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)['sampling_rng']
        assert torch.equal(current_rng,baseline_rng[seed])
        with np.load(run/'sampling_weights.npz') as meta:
            prob=meta['probability']
            draw=meta['draw_count']
            loss=meta['loss_weight']
            sample=meta['sampling_weight']
            recipe=result['identity']['recipe']
            if recipe in policy_references:
                ref_prob,ref_loss=policy_references[recipe]
                assert np.array_equal(prob,ref_prob) and np.array_equal(loss,ref_loss)
            else:
                policy_references[recipe]=(prob.copy(),loss.copy())
            assert len(prob)==len(starts)==len(draw)==len(loss)
            assert np.isclose(prob.sum(),1) and np.all(prob>=0)
            assert np.array_equal(prob,sample/sample.sum())
            assert np.isclose(np.dot(prob,loss),audit['expected_loss_weight'])
            assert draw.sum()==768000 and np.all(draw[prob==0]==0)
            assert audit['retained_windows']==np.count_nonzero(prob)
            assert audit['actual_quiet_draws']==draw[quiet].sum()
            if result['identity']['recipe']=='drop_quiet':
                assert np.array_equal(prob==0,quiet)
            if result['identity']['recipe']=='oversample_high':
                high=score>np.quantile(score,.9)
                assert np.all(sample[high]==2) and np.all(sample[~high]==1) and np.all(loss==1)
                assert abs(draw[high].sum()/draw.sum()-prob[high].sum())<.01
            if result['identity']['recipe']=='random_drop':
                assert np.count_nonzero(prob==0)==quiet.sum()
            if recipe in MONTH_RECIPES:
                months=training_months(starts,32)
                rare=training_lps_mask(starts,32) if recipe!='month_matched_drop' else None
                expected,_,_=sampling_policy(recipe,score,threshold,months,rare)
                assert np.array_equal(sample,expected) and np.all(loss==1)
                for month in np.unique(months):
                    mask=months==month
                    assert np.sum(mask&(prob==0))==np.sum(mask&quiet)
                    assert audit['month_matching'][month]['actual_removed']==np.sum(mask&quiet)
                if rare is not None:
                    support=audit['lps_support']
                    assert support['eligible']==rare.sum()
                    assert support['retained']==np.sum(rare&(prob>0))
                    assert support['actual_draws']==draw[rare].sum()
                    assert support['distinct_seen']==np.sum(rare&(draw>0))
                    assert support['restored_windows']==support['replacement_removed_windows']==422
                    if recipe=='protect_lps_matched': assert np.all(prob[rare]>0)
            if result['identity']['recipe']=='downweight_quiet':
                assert np.all(loss[quiet]==.25) and np.all(loss[~quiet]==1)
                with np.load(peers[0]/'sampling_weights.npz') as base:
                    assert np.array_equal(draw,base['draw_count']) and np.array_equal(prob,base['probability'])
            if recipe in ALIGNMENT_RECIPES:
                assert np.array_equal(loss,alignment_loss_weights(recipe)) and np.all(sample==1)
                assert digest(run/'selection_recipe.json')==result['identity']['selection_recipe_sha256']
                assert digest(ROOT/'research_runs/transfer/alignment_scores.npz')==result['identity']['selection_asset_sha256']
                assert audit['alignment']['downweighted_windows']==np.sum(loss<1)
                assert audit['alignment']['actual_downweighted_draws']==draw[loss<1].sum()
                with np.load(peers[0]/'sampling_weights.npz') as base:
                    assert np.array_equal(draw,base['draw_count']) and np.array_equal(prob,base['probability'])
            outcomes.append({'run':run.name,'exposure_verified':True,'retained_windows':int(np.count_nonzero(prob)),
                             'zero_probability_draws':int(draw[prob==0].sum()),'draws':int(draw.sum())})
    (folder/'sampling_verification.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    print(f'PASS: {len(outcomes)} real selection runs audited for membership, weights and actual exposure.')


if __name__=='__main__':
    main()
