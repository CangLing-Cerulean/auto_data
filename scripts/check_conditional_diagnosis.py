"""Focused regressions for scientific-evidence failure modes, no sensor simulation."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from conditional_diagnosis import (DiagnosticPredictor, GaussianPrediction, LinearGaussian,
                                  PrequentialLedger, diagnostic_evidence, digest, paired_response)


def main(destination):
    started=time.perf_counter(); checks=[]
    def passed(name): checks.append(name)
    def rejects(function):
        try: function()
        except ValueError: return
        raise AssertionError('Invalid scientific state was not rejected')
    # Hand-computable squared losses, unrelated to sensor data or training.
    parent=np.sqrt([2.,4.,6.,8.]); child=np.sqrt([1.,3.,7.,9.]); mask=np.array([True,True,False,False])
    r=paired_response(parent,child,mask)
    assert abs(r['Y'])<1e-14 and abs(r['Z']-.4)<1e-14
    passed('paired_response_and_inverse_coordinates')
    worse=paired_response(np.ones(4),np.full(4,np.sqrt(1.1)),mask)
    assert worse['Y']<0  # A different, worse B0 cannot change this parent comparison.
    passed('parent_comparison_not_B0_credit')
    rejects(lambda:paired_response(parent,child,np.ones(4,dtype=bool)))
    rejects(lambda:paired_response(np.zeros(4),child,mask))
    passed('empty_complement_and_zero_denominator_rejected')

    x=np.array([[0.],[1.],[2.],[3.],[4.],[5.]])
    y=np.array([0.,.1,.2,.3,.4,.5]); z=np.array([.1,0.,.2,.1,.3,.2])
    groups=['a','a','b','b','c','c']; ids=[f'unit_{i}' for i in range(6)]
    base=DiagnosticPredictor.fit(x,y,z,groups,ids,['x'])
    same=diagnostic_evidence(base,base,[.5],[.5],.1,.1)
    assert same==dict(marginal=0.,conditional=0.,joint=0.)
    passed('identical_predictive_information_has_zero_evidence')
    other=DiagnosticPredictor.fit(x,y[::-1],z[::-1],groups,ids,['x'])
    scores=diagnostic_evidence(base,other,[.5],[.5],.1,.1)
    assert abs(scores['joint']-scores['marginal']-scores['conditional'])<1e-12
    shared_y=DiagnosticPredictor(base.marginal,other.conditional)
    equivalent=diagnostic_evidence(base,shared_y,[.5],[.5],.1,.1)
    assert equivalent['marginal']==0 and abs(equivalent['joint']-equivalent['conditional'])<1e-12
    passed('joint_conditional_factorization_and_shared_marginal_equivalence')

    ledger=PrequentialLedger('conditional')
    rejects(lambda:ledger.observe('unregistered',.1,.2,.1,'source0'))
    registration=ledger.register('trial',base,{'h_v1':other},[.5],{'h_v1':[.5]})
    old_hash=digest(dict(reference=base.snapshot(),diagnostics={'h_v1':other.snapshot()}))
    event=ledger.observe('trial',.03,.1,.02,'original_trial')
    assert event['model_hash']==registration['model_hash']==old_hash
    assert ledger.best_observed()=='trial'
    passed('frozen_prediction_scored_before_update_and_positive_recipe_preserved')
    rejects(lambda:ledger.observe('trial',.03,.1,.02,'original_trial'))
    ledger.register('copy',base,{'h_v1':other},[.5],{'h_v1':[.5]})
    rejects(lambda:ledger.observe('copy',.03,.1,.02,'original_trial'))
    passed('duplicate_record_and_duplicate_summary_cannot_add_evidence')
    ledger.reliability['h_v1']=-100.
    ledger.register('new_version',base,{'h_v2':other},[.5],{'h_v2':[.5]})
    assert ledger.reliability['h_v2']==0. and ledger.best_observed()=='trial'
    passed('revision_does_not_inherit_evidence_or_remove_observed_benefit')
    before=ledger.predict_gain(base,{'h_v2':other},[.5],{'h_v2':[.5]})
    ledger.reliability['h_v2']=100.
    after=ledger.predict_gain(base,{'h_v2':other},[.5],{'h_v2':[.5]})
    assert abs(before-after)>1e-3
    passed('diagnostic_reliability_has_actual_unqueried_gain_prediction_path')

    result=dict(status='PASS',checks=checks,count=len(checks),seconds=time.perf_counter()-started,
                full_model_trainings=0,scientific_efficacy_claim=False,
                scope='finite numerical unit cases only; no sensor records generated')
    with destination.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
