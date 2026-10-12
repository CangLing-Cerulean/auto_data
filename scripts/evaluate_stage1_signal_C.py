"""Evaluate one hash-frozen scalar feature on the four construction inputs only."""
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def check_feature(feature):
    x = np.zeros((3,32,2),dtype=np.float32)
    one = np.ones_like(x)
    assert feature(x,x,x,{'NMSE':0.}) == 0.
    assert feature(x,x,one,{'NMSE':0.}) == 1.
    np.testing.assert_allclose(feature(x,x,one,{'NMSE':3.}),.25,rtol=0,atol=0)
    balanced = one.copy()
    balanced[:,16:] = -1.
    assert feature(x,x,balanced,{'NMSE':3.}) == 0.
    value = feature(x,x,one,{'NMSE':3.})
    assert feature(x[:,:,::-1],x[:,:,::-1],one[:,:,::-1],{'NMSE':3.}) == value
    assert feature(np.tile(x,(2,1,1)),np.tile(x,(2,1,1)),np.tile(one,(2,1,1)),{'NMSE':3.}) == value
    for bad_error in (-1.,float('nan'),float('inf')):
        try:
            feature(x,x,one,{'NMSE':bad_error})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid parent NMSE accepted')
    bad = one.copy()
    bad[0,0,0] = np.nan
    try:
        feature(x,x,bad,{'NMSE':1.})
    except ValueError:
        pass
    else:
        raise AssertionError('Non-finite input accepted')
    np.testing.assert_array_equal(x,0.)
    np.testing.assert_array_equal(one,1.)
    return ['zero_change_and_zero_error','positive_change_zero_error','known_mean_shift_ratio',
            'mean_cancelling_change','channel_permutation','duplicate_windows',
            'invalid_parent_error_rejected','nonfinite_input_rejected','inputs_unchanged']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    args = parser.parse_args()
    root = Path(args.root)
    started = time.monotonic()
    base = module('baseline',root/'run_intervention_stage1_baseline.py')
    lock = base.read(root/'signal_feature_freeze.json')
    assert base.sha(__file__) == lock['evaluator_sha256']
    assert base.sha(root/'signal_mean_shift.py') == lock['feature_sha256']
    assert base.sha(root/'candidate_protocol.json') == lock['candidate_protocol_sha256']
    p = base.read(root/'candidate_protocol.json')
    assert base.sha(root/'proposal_median3.py') == p['proposal_sha256']
    assert base.sha(root/'run_intervention_stage1_baseline.py') == p['baseline_runner_sha256']
    assert lock['tasks'] == ['C1','C2'] and lock['selection_feature_values_computed'] is False
    source = (root/'signal_mean_shift.py').read_text(encoding='utf-8')
    code_lines = [line for line in source.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    assert len(code_lines) <= 100
    tree = ast.parse(source)
    assert all(not isinstance(node,(ast.ImportFrom,ast.With,ast.AsyncWith,ast.ClassDef)) for node in ast.walk(tree))
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            assert all(alias.name in ('numpy','math') for alias in node.names)
        if isinstance(node,ast.Attribute):
            assert not node.attr.startswith('__')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            assert node.func.id in ('tuple','float','len','range','min','all','ValueError')
    feature = module('feature',root/'signal_mean_shift.py').signal_value
    tests = check_feature(feature)
    proposal = module('proposal',root/'proposal_median3.py')
    allowed_metrics = ('NMSE','NMAE','signed_bias','channel_NMSE','channel_NMAE')
    results = []
    for task in ('C1','C2'):
        path = root/'tasks'/task/'arrays.npz'
        assert base.sha(path) == lock['task_array_sha256'][task]
        with np.load(path,allow_pickle=False) as saved:
            starts = saved['train_starts']
            positions = np.linspace(0,len(starts)-1,min(8192,len(starts))).astype(np.int64)
            chosen = starts[positions]
            raw = saved['train'][chosen[:,None]+np.arange(32)]
        assert raw.dtype == np.float32
        initial_parent = p['parent_by_task'][task]
        parent_path = root/'baseline_runs'/f'{task}_{initial_parent}_s{p["exploration_seed"]}'/'result.json'
        assert base.sha(parent_path) == lock['parent_result_sha256'][task]
        parent_result = base.read(parent_path)
        parent_metrics = {key:parent_result['metrics'][key] for key in allowed_metrics}
        params = base.read(root/'candidate_prepared'/task/'params.json')
        assert base.sha(root/'candidate_prepared'/task/'params.json') == lock['candidate_params_sha256'][task]
        raw_tensor = torch.tensor(raw,device='cuda')
        with torch.no_grad():
            parent_tensor = base.transform(raw_tensor,initial_parent,p['source_half_recipe'])
            parent = parent_tensor.cpu().numpy()
        for action in ('P1','CP1'):
            with torch.no_grad():
                changed = proposal.transform(parent_tensor,params,'P1',action=='CP1').cpu().numpy()
            assert changed.shape == raw.shape and changed.dtype == np.float32
            np.testing.assert_array_equal(changed[:,[0,-1]],parent[:,[0,-1]])
            for array in (raw,parent,changed):
                array.setflags(write=False)
            digests = [base.digest(array) for array in (raw,parent,changed)]
            metrics_copy = copy.deepcopy(parent_metrics)
            measured_started = time.monotonic()
            value = feature(raw,parent,changed,metrics_copy)
            feature_seconds = time.monotonic()-measured_started
            assert isinstance(value,float) and np.isfinite(value) and 0. <= value <= 1.
            assert value == feature(raw,parent,changed,copy.deepcopy(parent_metrics))
            assert metrics_copy == parent_metrics
            assert digests == [base.digest(array) for array in (raw,parent,changed)]
            results.append(dict(task=task,action=action,signal_value=value,feature_seconds=feature_seconds,
                sampled_windows=len(chosen),sample_start_sha256=base.digest(chosen),
                input_sha256=dict(zip(('raw','parent','changed'),digests)),parent_NMSE=parent_metrics['NMSE']))
    values = {(item['task'],item['action']):item['signal_value'] for item in results}
    ordering = max(values['C1','CP1'],values['C2','CP1']) < values['C1','P1'] < values['C2','P1']
    report = dict(status='PASS',feature_sha256=lock['feature_sha256'],freeze_sha256=base.sha(root/'signal_feature_freeze.json'),
                  property_checks_passed=tests,results=results,proposed_ordering_holds=ordering,
                  selection_features_computed=False,new_trainings=0,test_access=False,
                  process_seconds=time.monotonic()-started,thresholds_fitted=False)
    output = root/'signal_feature_C_evaluation.json'
    assert not output.exists()
    base.write(output,report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
