"""Post-hoc C error decomposition; no alternate selection metric or D access."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import coldstart_runtime as runtime


def analyze(root):
    assert runtime.read(root/'development_audit.json')['status']=='PASS'
    assert runtime.read(root/'audit_decisions/trial_completed.json')['status']=='completed'
    lock=runtime.read(root/'execution_lock.json')
    for name,digest in lock['runtime_code_sha256'].items():assert runtime.sha(root/name)==digest
    runtime.configure();start=time.perf_counter();rows=runtime.load(root)
    summary=runtime.read(root/'data_summary.json')
    constant=np.asarray(summary['A_raw_std'])==0
    assert constant.shape==(370,) and constant.any() and (~constant).any()
    all_zero=np.all(rows['A'][:,constant]==0)
    assert all_zero
    values=torch.as_tensor(rows['C'],device='cuda');records=[]
    for path in sorted((root/'runs').glob('*/*/result.json')):
        training=runtime.read(path);folder=path.parent
        initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
        model=runtime.best_model(folder);state=model.state_dict()
        fixed_weights={key:torch.equal(state[key].cpu()[constant],initial[key][constant])
                       for key in ('seasonal_weight','trend_weight')}
        error=runtime.errors(model,values);assert runtime.metrics(error)==training['C']
        squared=error.astype(np.float64)**2
        per_channel=squared.mean(axis=(0,1));total=float(per_channel.mean())
        groups={}
        for label,mask in (('A_constant',constant),('A_nonconstant',~constant)):
            contribution=float(per_channel[mask].sum()/370)
            groups[label]=dict(channels=int(mask.sum()),within_group_mse=float(per_channel[mask].mean()),
                               contribution_to_global_mse=contribution,fraction_of_global_mse=contribution/total)
        records.append(dict(run=folder.parent.name,seed=training['seed'],global_C_mse=total,
                            groups=groups,per_channel_C_mse=per_channel.tolist(),
                            A_constant_feature_weights_equal_initial=fixed_weights,
                            A_constant_max_abs_bias=float(max(state[k].detach().cpu()[constant].abs().max() for k in ('seasonal_bias','trend_bias')))))
    return dict(status='PASS_descriptive',post_hoc=True,trigger='Observed parent C MSE much larger than B during exploration',
                grouping_rule='A raw population standard deviation equals zero, as saved before training',
                A_constant_channel_indices=np.flatnonzero(constant).tolist(),A_constant_all_zero=bool(all_zero),
                records=records,seconds=time.perf_counter()-start,D_accessed=False,new_full_trainings=0,
                changes_primary_score_or_recommendations=False,feedback_to_exploration_agents=False,
                causal_effect_or_general_headroom_bound_established=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.root.resolve());runtime.write(args.output,result)
    print(json.dumps(dict(status=result['status'],runs=len(result['records']),constant_channels=len(result['A_constant_channel_indices']))))
