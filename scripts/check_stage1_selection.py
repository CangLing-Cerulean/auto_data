"""Pre-response identity and scientific-rule checks; no model evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--predictions',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    base=module('base',root/'run_intervention_stage1_baseline.py')
    methods=module('methods',root/'online_ts/agent_research/stage1_selection_methods.py')
    protocol=base.read(root/'selection_protocol.json')
    for group in ('code_sha256','prerequisite_sha256'):
        for name,digest in protocol[group].items():
            assert base.sha(root/name)==digest,name
    rule=base.read(root/'signal_rule.json')
    values={item['task']+'_'+item['action']:item['signal_value'] for item in base.read(root/'signal_feature_C_evaluation.json')['results']}
    midpoints=np.asarray([(values['C2_CP1']+values['C1_P1'])/2,(values['C1_P1']+values['C2_P1'])/2])
    assert np.all(abs(np.asarray(rule['thresholds'])-midpoints)<=np.spacing(midpoints))
    for key,value in values.items():
        assert methods.signal_category(value,rule['thresholds'])==rule['construction_fit'][key]
    checks=['all_source_hashes','C_only_midpoint_thresholds','C_fit_not_prospective_score']
    array=np.zeros((2,32,2),dtype=np.float32)
    result=methods.generic_features(array,array,dict(channel_NMSE=[0.,0.],NMSE=0.,NMAE=0.))
    assert all(value==0 for value in result.values())
    cases=[dict(id='a',features=result,category='harm',mean_gain_percent=-2),dict(id='b',features=result,category='improve',mean_gain_percent=2)]
    for multi in (True,False):
        assert methods.nearest(cases,dict(features=result),multi)[0]=='uncertain'
    assert methods.recommend([])=={'S1':'PARENT','S2':'PARENT'}
    for cut in rule['thresholds']:
        assert methods.signal_category(cut,rule['thresholds'])!=methods.signal_category(float(np.nextafter(cut,np.inf)),rule['thresholds'])
    for invalid in (float('nan'),-1.,1.01):
        try:
            methods.signal_category(invalid,rule['thresholds'])
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid signal accepted')
    checks+=['zero_error_and_no_edit','conflicting_exact_tie_abstention','no_unobserved_adoption','closed_threshold_boundaries','nonfinite_signal_rejected']
    output='selection_code_preflight.json'
    report=dict(status='PASS',checks=checks,new_trainings=0,test_access=False,selection_protocol_sha256=base.sha(root/'selection_protocol.json'))
    if args.predictions:
        lock=base.read(root/'selection_prediction_lock.json')
        assert lock['selection_protocol_sha256']==report['selection_protocol_sha256']
        assert set(protocol['required_prediction_files'])<=set(lock['files_sha256'])
        for name,digest in lock['files_sha256'].items():
            assert base.sha(root/name)==digest,name
        assert not (root/'selection_runs').exists()
        facts=base.read(root/'selection_common_facts.json')
        signals=base.read(root/'selection_new_signal_values.json')['values']
        expected=methods.deterministic_predictions(facts['construction_cases'],facts['selection_cases'],signals,rule['thresholds'])
        assert expected==base.read(root/'selection_deterministic_predictions.json')
        predictions=base.read(root/'selection_all_predictions.json')
        assert set(predictions)=={'generic_stats','case_match','new_signal','strong_prompt'}
        for method,record in predictions.items():
            assert set(record['predictions'])=={'S1_P1','S1_CP1','S2_P1','S2_CP1'}
            assert all(value in methods.CATEGORIES for value in record['predictions'].values())
            assert set(record['ranking'])=={'S1','S2'}
            assert all(value in ('P1>CP1','CP1>P1','tie') for value in record['ranking'].values())
            if method!='strong_prompt':
                assert record==expected[method]
        assert predictions['strong_prompt']==base.read(root/'selection_strong_prompt_prediction.json')
        checks+=['all_predictions_before_response','deterministic_predictions_recomputed','four_methods_four_cell_denominator','prediction_and_fact_identity']
        report['prediction_lock_sha256']=base.sha(root/'selection_prediction_lock.json')
        output='selection_prediction_preflight.json'
    assert not (root/output).exists()
    base.write(root/output,report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
