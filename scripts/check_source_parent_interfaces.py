"""Validate SOURCE adaptation using only registered completed parent assets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import source_dlinear_adapter as adapter
import source_quadratic as source


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main(out):
    started=time.perf_counter();plan=read(out/'runtime_plan.json')
    assert out==Path('/root/auto_data_experiments/20261014_source_adapter_r1') and out.resolve()==out
    sync=read(out/'source_sync_receipt.json')
    assert sync['status']=='exact_commit_and_tree_verified'
    assert sync['runtime_plan_sha256']==sha(out/'runtime_plan.json')
    old=Path(plan['parent_root'])
    assert old==Path('/root/auto_data_experiments/20261014_existing_sources_r1') and old.resolve()==old
    assert not (out/'parent_interfaces.json').exists() and not (out/'parent_interfaces_failure.json').exists()
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in plan['parent_assets_sha256'].items():assert sha(old/name)==digest,name
    spec=importlib.util.spec_from_file_location('registered_core',old/'run_research.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    torch.set_num_threads(4);torch.manual_seed(2026101407)
    results=[];references=[];blocks=0
    for parent in plan['parents']:
        with np.load(old/parent['moments'],allow_pickle=False) as z:
            gram,cross=z['gram'],z['cross']
        channels=parent['channels']
        assert gram.shape==(channels,66,66) and cross.shape==(channels,66,8)
        for step in plan['steps']:
            relative=f"{parent['directory']}/step_{step}.pt"
            snapshot=torch.load(old/relative,map_location='cpu',weights_only=True)
            model=core.DLinear(dict(input_length=32,prediction_length=8,moving_average_kernel=25,channels=channels))
            state=adapter.parent_coordinates(model,snapshot)
            assert state['step']==step
            delta,diagnostics=adapter.quadratic_segment(gram,cross,gram,cross,state,np.zeros_like(state['theta']))
            assert np.array_equal(delta,np.zeros_like(delta))
            blocks+=len(diagnostics)
            results.append(dict(task=parent['task'],seed=parent['seed'],step=step,shape=list(state['theta'].shape),
                checkpoint_sha256=sha(old/relative),preconditioner_min=float(state['P'].min()),preconditioner_max=float(state['P'].max()),
                zero_response_exact=True,clipped_negative_eigenvalues=sum(d['clipped_negative_eigenvalues'] for d in diagnostics),
                largest_roundoff_tolerance=max(d['roundoff_tolerance'] for d in diagnostics)))
            if parent['seed']==plan['seeds'][0] and step in (500,3000):
                # Fixed algebraic probe in the first block; no counterfactual data action.
                h=2/(channels*8)*gram[0];p=state['P'][0,:,0]
                direction=np.linspace(-.5,.5,66);initial=np.cos(np.arange(66))*.01;g=h@direction
                answer=source.segment_response(h,p,g,initial,learning_rate=.001,steps=500)
                generator=np.zeros((67,67));generator[:66,:66]=-p[:,None]*h;generator[:66,66]=-p*g
                exponential=torch.linalg.matrix_exp(torch.from_numpy(.5*generator)).numpy()
                reference=(exponential@np.r_[initial,1.])[:66]
                error=float(np.max(np.abs(answer.delta-reference)))
                bound=plan['reference_atol']+plan['reference_rtol']*float(np.max(np.abs(reference)))
                assert np.isfinite(reference).all() and error<=bound,(parent['task'],step,error,bound)
                references.append(dict(task=parent['task'],seed=parent['seed'],step=step,channel=0,output=0,
                    max_abs_error=error,bound=bound,reference='torch.linalg.matrix_exp augmented 67x67 generator'))
            print(json.dumps(dict(parent_checkpoint=len(results),task=parent['task'],step=step)),flush=True)
    assert len(results)==36 and len(references)==4
    for name,digest in plan['parent_assets_sha256'].items():assert sha(old/name)==digest,name
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    result=dict(status='PASS',scope='parent coordinate, optimizer and quadratic operator interfaces only',
        source_commit=sync['source_commit'],
        runtime_plan_sha256=sha(out/'runtime_plan.json'),checkpoints=results,checked_blocks=blocks,nonzero_references=references,
        parent_assets_verified=len(plan['parent_assets_sha256']),new_full_trainings=0,cumulative_full_trainings=373,
        candidate_states_loaded=0,candidate_labels_loaded=0,base_csv_access=False,BC_evaluation=False,test_access=False,
        original_parent_assets_unchanged=True,torch_version=torch.__version__,numpy_version=np.__version__,
        seconds=time.perf_counter()-started)
    write(out/'parent_interfaces.json',result)
    print(json.dumps(dict(status='PASS',checkpoints=36,blocks=blocks,nonzero_references=4,seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:main(args.output)
    except Exception:
        write(args.output/'parent_interfaces_failure.json',dict(status='failed',traceback=traceback.format_exc(),new_full_trainings=0))
        raise
