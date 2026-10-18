"""Read-only inference of existing frozen ETT checkpoints into a new development stage."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from diagnosis_pool_features import (action_features, combine_features, input_statistics,
                                    parent_features, read, region_response, sha, write)

OLD=Path('/root/auto_data_experiments/20261016_ett_confirmation_r1')
NEW=Path('/root/auto_data_experiments/20261018_conditional_diagnosis_r1')


def main():
    assert OLD.resolve()==OLD and NEW.resolve()==NEW
    sys.path.insert(0,str(OLD))
    spec=importlib.util.spec_from_file_location('frozen_ett_reader',OLD/'ett_confirmation.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.configure()
    begin=time.perf_counter();records=[];identities={};checks=[]
    for station in ('ETTh1','ETTh2'):
        old.verify(station)
        assert sha(OLD/station/'ABC.npz')==read(OLD/station/'preprocessing.json')['prepared_sha256']
        data=old.Data(station,prepared=True)
        a=data.values['A'].cpu().numpy();b=data.values['B'].cpu().numpy()
        astarts=np.arange(data.count['A']);bstarts=np.arange(data.count['B'])
        first=astarts+96<4320
        astats=input_statistics(a,astarts,first,96);bstats=input_statistics(b,bstarts,None,96)
        @torch.no_grad()
        def residuals(net,split):
            net.eval();chunks=[]
            for lo in range(0,data.count[split],256):
                x,y=data.batch(split,np.arange(lo,min(lo+256,data.count[split])))
                chunks.append((net(x)-y).cpu().numpy())
            return np.concatenate(chunks)
        for seed in old.SEEDS:
            parent=OLD/station/'runs'/f'PARENT_P_s{seed}'
            net=old.model(seed)
            parent_checkpoint=torch.load(parent/'best.pt',map_location='cuda',weights_only=True)
            net.load_state_dict(parent_checkpoint['model'])
            pe=residuals(net,'C');be=residuals(net,'B')
            expected=read(parent/'C_scores.json')['best']
            if __import__('hashlib').sha256(pe.tobytes()).hexdigest()!=expected['residual_sha256']:
                raise ArithmeticError('Parent C inference differs from its frozen residual hash')
            curve=read(parent/'B_curve.json')
            # Frozen ETT curves are an array of step/score records.
            values=[entry['NMSE'] for entry in curve]
            best_curve=next(entry for entry in curve if entry['step']==expected['step'])
            assert __import__('hashlib').sha256(be.tobytes()).hexdigest()==best_curve['residual_sha256']
            features=parent_features(astats,bstats,be,values,expected['step'])
            identities[str(parent/'best.pt')]=sha(parent/'best.pt')
            for action in ('FAVOR_EARLY','FAVOR_LATE'):
                for impl in ('S','W'):
                    run=OLD/station/'runs'/f'{action}_{impl}_s{seed}'
                    net.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=True)['model'])
                    ce=residuals(net,'C');scores=read(run/'C_scores.json')['best']
                    if __import__('hashlib').sha256(ce.tobytes()).hexdigest()!=scores['residual_sha256']:
                        raise ArithmeticError('Child C inference differs from its frozen residual hash')
                    response=region_response(pe,ce)
                    assert abs(response['child_loss']-scores['NMSE'])<1e-12
                    identities[str(run/'best.pt')]=sha(run/'best.pt')
                    records.append(dict(family='ETT',task=station,action=action,implementation=impl,seed=seed,
                        features=combine_features(features,action_features(action,impl,float(first.mean()))),
                        response=response,source_id=f'{station}/{run.name}',source_result_sha256=sha(run/'C_scores.json'),
                        original_full_training_seconds=read(run/'status.json')['elapsed_seconds'],
                        provenance='retrospective new long-horizon slice; old C results were already revealed'))
            checks.append(dict(station=station,seed=seed,parent_and_four_child_residual_hashes='exact'))
        del data
    result=dict(status='PASS',records=records,identities=identities,checks=checks,
                seconds=time.perf_counter()-begin,new_full_trainings=0,checkpoint_C_inferences=30,
                checkpoint_B_inferences=6,D_test_access=False,
                target_slice='all channels and windows, latter half of forecast horizon')
    write(NEW/'ett_extraction.json',result)
    print(json.dumps(dict(status='PASS',records=len(records),seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
