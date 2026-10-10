"""Exercise agent code on real T windows in a clean subprocess; no training."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..config import GateError,canonical
from ..selection.common import file_hash,write_json
from .transforms import load_processor,apply_processor


def check(folder):
    folder=Path(folder);outputs=folder/'outputs'
    paths=[folder/'inputs/train_inputs.npz',outputs/'processor.py',outputs/'processor_state.json']
    before={str(p):file_hash(p) for p in paths}
    program=load_processor(paths[1],paths[2]);state=deepcopy(program[1])
    with np.load(paths[0],allow_pickle=False) as saved:
        starts=saved['start_row']; chosen=np.random.default_rng(20261009).choice(len(starts),size=17,replace=False)
        windows=saved['values'][starts[chosen,None]+np.arange(32)].copy()
    devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
    changes=[]
    for device in devices:
        x=torch.tensor(windows,device=device);original=x.clone()
        with torch.no_grad():
            first=apply_processor(x,program).clone();second=apply_processor(x,program).clone()
            if not torch.equal(x,original) or canonical(program[1])!=canonical(state):raise GateError('INPUT_PROCESSOR_MUTATES_INPUT_OR_STATE')
            if not torch.equal(first,second):raise GateError('INPUT_PROCESSOR_NONDETERMINISTIC')
            permutation=torch.arange(16,-1,-1,device=device)
            reordered=apply_processor(x[permutation],program)
            if not torch.allclose(reordered,first[permutation],atol=1e-6,rtol=1e-6):raise GateError('INPUT_PROCESSOR_BATCH_DEPENDENCE')
            split=torch.cat([apply_processor(x[:5],program),apply_processor(x[5:],program)])
            if not torch.allclose(split,first,atol=1e-6,rtol=1e-6):raise GateError('INPUT_PROCESSOR_BATCH_DEPENDENCE')
            changes.append({'device':device,'mean_abs_input_change':float((first-original).abs().mean())})
    if any(file_hash(p)!=expected for p,expected in before.items()):raise GateError('INPUT_PROCESSOR_SOURCE_CHANGED')
    if not any(x['mean_abs_input_change']>0 for x in changes):raise GateError('INPUT_PROCESSOR_IDENTITY')
    receipt={'status':'passed','real_T_windows':17,'records_per_window':32,'channels':15,'checks':changes,
        'batch_independence':'checked_on_real_T_windows_not_a_universal_proof','state_immutable':True,
        'sources':before,'training_calls':0,'test_access':False,'security_sandbox':False}
    write_json(outputs/'processor_validation.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    check(parser.parse_args().folder)
