"""Freeze common C/S facts only after the learned feature and its rule are frozen."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    root=Path(args.root)
    started=time.monotonic()
    base=module('baseline',root/'run_intervention_stage1_baseline.py')
    lock=base.read(root/'selection_protocol.json')
    for name,digest in lock['code_sha256'].items():
        assert base.sha(root/name)==digest,name
    for name,digest in lock['prerequisite_sha256'].items():
        assert base.sha(root/name)==digest,name
    assert not (root/'selection_common_facts.json').exists()
    assert not (root/'selection_new_signal_values.json').exists()
    p=base.read(root/'candidate_protocol.json')
    rule=base.read(root/'signal_rule.json')
    response=base.read(root/'construction_confirmation_recovered.json')
    assert response['construction_gate_passed']
    feature=module('feature',root/'signal_mean_shift.py').signal_value
    proposal=module('proposal',root/'proposal_median3.py')
    methods=module('methods',root/'online_ts/agent_research/stage1_selection_methods.py')
    assert rule['feature_sha256']==base.sha(root/'signal_mean_shift.py')
    c_values=base.read(root/'signal_feature_C_evaluation.json')
    recorded={(x['task'],x['action']):x['signal_value'] for x in c_values['results']}
    construction,selection,signals,trace,task_facts=[],[],{},[],{}
    properties=[]
    for cut in rule['thresholds']:
        lower=methods.signal_category(cut,rule['thresholds'])
        higher=methods.signal_category(float(np.nextafter(cut,np.inf)),rule['thresholds'])
        assert lower!=higher
    properties.append('threshold_equality_and_next_float')
    for task in ('C1','C2','S1','S2'):
        path=root/'tasks'/task/'arrays.npz'
        assert base.sha(path)==p['task_array_sha256'][task]
        with np.load(path,allow_pickle=False) as source:
            starts=source['train_starts']
            chosen=starts[np.linspace(0,len(starts)-1,min(8192,len(starts))).astype(np.int64)]
            raw=source['train'][chosen[:,None]+np.arange(32)]
        parent_action=p['parent_by_task'][task]
        parent_path=root/'baseline_runs'/f'{task}_{parent_action}_s{p["exploration_seed"]}'/'result.json'
        assert base.sha(parent_path)==lock['parent_result_sha256'][task]
        parent_result=base.read(parent_path)
        metrics={key:parent_result['metrics'][key] for key in ('NMSE','NMAE','signed_bias','channel_NMSE','channel_NMAE')}
        params_path=root/'candidate_prepared'/task/'params.json'
        assert base.sha(params_path)==lock['candidate_params_sha256'][task]
        params=base.read(params_path)
        with torch.no_grad():
            parent_tensor=base.transform(torch.tensor(raw,device='cuda'),parent_action,p['source_half_recipe'])
            parent=parent_tensor.cpu().numpy()
        raw_double=raw.astype(np.float64)
        task_facts[task]=dict(parent_metrics=metrics,train_input_statistics=dict(sampled_windows=len(raw),
            channel_mean_abs_first_difference=np.mean(abs(np.diff(raw_double,axis=1)),axis=(0,1)).tolist(),
            channel_temporal_std_mean=np.mean(np.std(raw_double,axis=1),axis=0).tolist(),
            channel_last_minus_first_rms=np.sqrt(np.mean((raw_double[:,-1]-raw_double[:,0])**2,axis=0)).tolist()))
        for action in ('P1','CP1'):
            with torch.no_grad():
                changed=proposal.transform(parent_tensor,params,'P1',action=='CP1').cpu().numpy()
            for array in (raw,parent,changed):
                array.setflags(write=False)
            before=[base.digest(array) for array in (raw,parent,changed)]
            generic_started=time.monotonic()
            features=methods.generic_features(parent,changed,metrics)
            generic_seconds=time.monotonic()-generic_started
            key=task+'_'+action
            case=dict(id=key,task=task,action=action,features=features)
            feature_started=time.monotonic()
            signal=feature(raw,parent,changed,metrics)
            signal_seconds=time.monotonic()-feature_started
            assert before==[base.digest(array) for array in (raw,parent,changed)]
            if task.startswith('C'):
                assert signal==recorded[task,action]
                cell=response['cells'][key]
                case.update(category=cell['repeatability']['stable_category'],mean_gain_percent=cell['mean_global_NMSE_gain_percent'])
                construction.append(case)
            else:
                selection.append(case)
                signals[key]=signal
            trace.append(dict(id=key,signal_seconds=signal_seconds,generic_seconds=generic_seconds,sample_start_sha256=base.digest(chosen),
                              input_sha256=dict(zip(('raw','parent','changed'),before))))
    predictions=methods.deterministic_predictions(construction,selection,signals,rule['thresholds'])
    common=dict(schema='stage1_selection_common_facts_v1',tasks=task_facts,construction_cases=construction,
        selection_cases=selection,construction_responses=response['cells'],
        fixed_operations='P1 interior median3; CP1 interior weights .25,.5,.25; endpoints preserved; uniform sampling',
        forecast_categories='NMSE relative improvement >.25 improve; <-.25 harm; otherwise unchanged; uncertain is abstention',
        data_boundary='T input samples and initial fixed-parent D metrics; no S candidate response',
        original_failed_delivery_preserved=response['preserved_failed_delivery'],
        independent_sources_in_construction=1,known_historical_SKAB_summary=True,
        strong_prompt_optional_raw_statistic_permission='one pure <=100-line NumPy/math scalar on same raw,parent,changed,parent_metrics inputs, before any S response')
    base.write(root/'selection_common_facts.json',common)
    base.write(root/'selection_new_signal_values.json',dict(values=signals,feature_sha256=rule['feature_sha256'],rule_sha256=base.sha(root/'signal_rule.json')))
    base.write(root/'selection_deterministic_predictions.json',predictions)
    receipt=dict(status='PASS',new_trainings=0,test_access=False,source_protocol_sha256=base.sha(root/'selection_protocol.json'),
        feature_properties=properties,C_feature_recomputation_identical=True,input_trace=trace,
        common_facts_sha256=base.sha(root/'selection_common_facts.json'),
        new_signal_values_sha256=base.sha(root/'selection_new_signal_values.json'),
        deterministic_predictions_sha256=base.sha(root/'selection_deterministic_predictions.json'),
        process_seconds=time.monotonic()-started)
    base.write(root/'selection_preparation_receipt.json',receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key!='input_trace'}))


if __name__=='__main__':
    main()
