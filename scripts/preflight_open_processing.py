"""Validate the frozen identity interface and reuse audited common parents."""
import argparse
from pathlib import Path
import time
import numpy as np
import open_processing_runtime as runtime
from open_processing_materialize import fit_program, apply_program, normalize_weights
from open_processing_sandbox import pack, encode_state


IDENTITY='''
def fit(a):
    return {}
def transform(x,state):
    return x, np.zeros(x.shape[1]), np.ones(x.shape[1])
def weights(x,state):
    return 1.0, 1.0
def target(x,state):
    return np.broadcast_to(np.arange(x.shape[1]) < 185, (8,x.shape[1]))
'''


def check(root):
    start=time.perf_counter();runtime.configure();parent,rows=runtime.load(root)
    lock=runtime.read(root/'execution_lock.json')
    for name,expected in lock['runtime_code_sha256'].items():assert runtime.sha(root/name)==expected
    assert runtime.sha(root/'PROTOCOL.md')==lock['protocol_sha256']
    assert runtime.sha(root/'AGENT_INTERFACE.txt')==lock['agent_interface_sha256']
    state,fit=fit_program(IDENTITY,'pipeline',rows['A']);assert state=={}
    arrays={};receipts={}
    for part in 'ABC':
        arrays[part],receipts[part]=apply_program(IDENTITY,'pipeline',rows[part],state,part=='A',part!='A')
        np.testing.assert_array_equal(arrays[part]['model_input'],runtime.windows(rows[part]))
        assert np.all(arrays[part]['offset']==0) and np.all(arrays[part]['scale']==1)
    p,l,stats=normalize_weights(arrays['A']['sampling'],arrays['A']['loss'])
    assert np.all(p==p[0]) and np.all(l==1)
    gpu=runtime.tensors(rows);prepared=runtime.prepared_gpu(arrays);checks=[];parent_seconds=0.
    for seed in runtime.SEEDS:
        folder=parent/'runs/PARENT'/str(seed);record=runtime.read(folder/'result.json')
        assert record['status']=='completed' and record['steps']==3000
        model=runtime.best_model(folder)
        for part in 'BC':
            old=runtime.errors(model,gpu[part]);new=runtime.decoded_errors(model,gpu[part],prepared[part])
            np.testing.assert_array_equal(old,new)
            score=runtime.metrics(new)
            expected=record['C'] if part=='C' else next({k:v for k,v in g.items() if k!='step'} for g in record['B_grid'] if g['step']==record['best_step'])
            assert score==expected
            checks.append(dict(seed=seed,partition=part,residual_sha256=runtime.digest(new)))
        parent_seconds+=record['seconds']
    identity,hashes=runtime.processing_identity(arrays,p,l)
    result=dict(status='PASS',execution_lock_sha256=runtime.sha(root/'execution_lock.json'),
                D_accessed=False,new_full_trainings=0,shared_parents=3,
                parent_root=str(parent),parent_development_audit_sha256=runtime.sha(parent/'development_audit.json'),
                checks=checks,identity_processing_sha256=identity,effective_array_sha256=hashes,
                logical_parent_training_seconds=parent_seconds,seconds=time.perf_counter()-start,
                identity_source=IDENTITY,fit_receipt=fit,materialize_receipts=receipts,weight_stats=stats)
    runtime.write(root/'preflight_receipt.json',result)
    runtime.write(root/'parent_reuse_receipt.json',dict(status='PASS',parent_root=str(parent),parents_reused=3,
                  execution_lock_sha256=runtime.sha(root/'execution_lock.json'),
                  new_full_trainings=0,logical_parent_trainings=3,logical_parent_training_seconds=parent_seconds,
                  preflight_sha256=runtime.sha(root/'preflight_receipt.json'),parent_assets=lock['parent_assets']))
    runtime.write(root/'prepared_parent_facts.json',runtime.read(parent/'parent_facts.json'))
    return result


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    root=args.root.resolve();assert root==args.root and root.name==runtime.STUDY
    value=check(root);print(json.dumps(dict(status=value['status'],checks=len(value['checks']),seconds=value['seconds'])))
