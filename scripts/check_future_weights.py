"""Audit frozen train-only gradient selection and its month-matched shuffle."""
import json
import numpy as np
from run_research import ROOT,digest,write_json
from run_selection import alignment_loss_weights,sampling_policy,training_months


def main():
    folder=ROOT/'research_runs/transfer'; meta=json.loads((folder/'alignment_recipe.json').read_text())
    with np.load(ROOT/'research_runs/selection/window_table.npz') as table: starts=table['start_row_zero_based']
    with np.load(folder/'alignment_scores.npz') as asset:
        indices=asset['source_eligible_indices']; score=asset['score']; source=asset['source_start_rows']
    assert np.array_equal(starts[indices],source)
    assert source.max()+40<=meta['future_rows'][0]
    assert digest(ROOT/'research_runs/results/T005_prefix80_full/final.pt')==meta['teacher_sha256']
    assert np.isclose(np.quantile(score,.25),meta['quarter_threshold'])
    chosen=(score<=np.quantile(score,.25))&(score<0)
    candidate=alignment_loss_weights('future_conflict_downweight'); control=alignment_loss_weights('future_conflict_shuffled')
    expected=np.ones(len(starts)); expected[indices[chosen]]=.25
    assert np.array_equal(candidate,expected)
    months=training_months(source,32); rng=np.random.default_rng(20260929)
    expected_control=candidate.copy()
    for month in np.unique(months):
        members=indices[months==month]; expected_control[members]=rng.permutation(candidate[members])
        assert np.sum(candidate[members]<1)==np.sum(control[members]<1)
    assert np.array_equal(control,expected_control)
    for name,weight in [('future_conflict_downweight',candidate),('future_conflict_shuffled',control)]:
        sample,loss,prob=sampling_policy(name,starts,0,aligned_loss=weight)
        assert np.all(sample==1) and np.array_equal(loss,weight)
        assert np.all(prob==1/len(starts)) and np.isclose(np.dot(prob,loss),meta['expected_loss_weight'])
    outcome={'train_only_source_mapping':True,'future_source_disjoint':True,'weights_match_signed_quartile':True,
        'month_matched_shuffle_exact':True,'uniform_sampling_unchanged':True,
        'downweighted_windows':int(chosen.sum()),'finite_difference_max_absolute_error':meta['finite_difference_max_absolute_error']}
    write_json(folder/'alignment_verification.json',outcome); print(json.dumps(outcome))


if __name__=='__main__': main()
