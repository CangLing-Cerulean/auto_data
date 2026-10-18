"""Read only the frozen ETT parent and A/B tensors; save aligned measurement tables."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
import torch
from diagnosis_pool_features import read,sha,write
from parent_measurement_tables import inputs,errors,save,INPUT_COLUMNS,ERROR_COLUMNS

OLD=Path('/root/auto_data_experiments/20261016_ett_confirmation_r1')
NEW=Path('/root/auto_data_experiments/20261018_parent_measurements_r3')

def main():
    assert OLD.resolve()==OLD and NEW.resolve()==NEW
    sys.path.insert(0,str(OLD));spec=importlib.util.spec_from_file_location('frozen_ett_parent_reader',OLD/'ett_confirmation.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);old.configure()
    start=time.perf_counter();directory=NEW/'tables';directory.mkdir();assets={};checks=[]
    for station in ('ETTh1','ETTh2'):
        old.verify(station);assert sha(OLD/station/'ABC.npz')==read(OLD/station/'preprocessing.json')['prepared_sha256']
        with np.load(OLD/station/'ABC.npz',allow_pickle=False) as z:a,b=z['A'],z['B']
        astart=np.arange(len(a)-191);bstart=np.arange(len(b)-191)
        first=astart+96<4320
        table=dict(A_inputs=inputs(a,astart,96),B_inputs=inputs(b,bstart,96),A_first=first,A_starts=astart,B_starts=bstart)
        bt=torch.tensor(b,device='cuda');offset=torch.arange(192,device='cuda')
        for seed in old.SEEDS:
            run=OLD/station/'runs'/f'PARENT_P_s{seed}';saved=torch.load(run/'best.pt',map_location='cuda',weights_only=True)
            net=old.model(seed);net.load_state_dict(saved['model']);net.eval();chunks=[]
            with torch.no_grad():
                for lo in range(0,len(bstart),256):
                    ids=torch.arange(lo,min(lo+256,len(bstart)),device='cuda');block=bt[ids[:,None]+offset]
                    chunks.append((net(block[:,:96])-block[:,96:]).cpu().numpy())
            error=np.concatenate(chunks);curve=read(run/'B_curve.json');selected=min(curve,key=lambda r:(r['NMSE'],r['step']))
            assert hashlib.sha256(error.tobytes()).hexdigest()==selected['residual_sha256']
            table[f'B_errors_s{seed}']=errors(error)
            checks.append(dict(station=station,seed=seed,parent_checkpoint_sha256=sha(run/'best.pt'),B_residual_hash_match=True,windows=len(error)))
        target=directory/f'ETT_{station}.npz';save(target,**table);assets[target.name]=dict(bytes=target.stat().st_size,sha256=sha(target))
    write(NEW/'ett_measurements.json',dict(status='PASS',assets=assets,checks=checks,input_columns=INPUT_COLUMNS,error_columns=ERROR_COLUMNS,seconds=time.perf_counter()-start,new_full_trainings=0,child_response_access=False,D_test_access=False))
    print(dict(status='PASS',tasks=2,parents=6,seconds=time.perf_counter()-start),flush=True)

if __name__=='__main__':main()
