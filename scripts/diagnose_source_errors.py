"""A011: train-only conditional errors for existing frozen models."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,write_json
from leading_sources import SourceDLinear,load_mapping


def main():
    run=ROOT/'research_runs/results/A011_source_errors'
    if run.exists(): raise RuntimeError('Immutable diagnostic exists')
    used=sum(json.loads(p.read_text()).get('elapsed_seconds',0) for p in run.parent.glob('*/status.json'))
    allowance=min(120,7200-used)
    if allowance<=0: raise RuntimeError('Budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    def check():
        if time.monotonic()-begun>=allowance: raise RuntimeError('Diagnostic budget exhausted')
    status('preparing')
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        manifest_path=ROOT/'data/metropt3_v1/manifest.json'; manifest=json.loads(manifest_path.read_text())
        source=manifest_path.parent/'train.csv'
        assert digest(source)==next(s['sha256'] for s in manifest['splits'] if s['name']=='train')
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text()); mapping,source_hash=load_mapping()
        reports=[]
        for origin,base_id,new_id in [(60,'T000_prefix60_full','T010_prefix60_sources'),(80,'T005_prefix80_full','T011_prefix80_sources')]:
            status(f'fold_{origin}')
            base=run.parent/base_id; new=run.parent/new_id
            audit=json.loads((base/'data_audit.json').read_text()); cut=audit['train_raw_rows'][1]; end=audit['probe_raw_rows'][1]
            mean=np.array(audit['mean']); scale=np.array(audit['scale'])
            data=torch.from_numpy(((raw[:end]-mean)/scale).astype(np.float32)).cuda()
            with np.load(base/'windows.npz') as z: train=z['train_start']; future=z['probe_start']
            assert train.max()+40<=future.min()
            offsets=torch.arange(40,device='cuda')
            movement=[]
            for lo in range(0,len(train),4096):
                check(); b=data[torch.from_numpy(train[lo:lo+4096]).cuda()[:,None]+offsets]
                movement.append((b[:,32:]-b[:,31:32]).abs().double().mean(1).cpu().numpy())
            thresholds=np.quantile(np.concatenate(movement),[.75,.9],axis=0)
            del movement
            original=DLinear(p).cuda(); original.load_state_dict(torch.load(base/'final.pt',weights_only=True)); original.eval()
            candidate=SourceDLinear(DLinear(p),mapping).cuda(); candidate.load_state_dict(torch.load(new/'final.pt',weights_only=True)); candidate.eval()
            # type (movement/input), group, channel, model, [count,abs,square,signed]
            sums=np.zeros((2,4,15,2,4),dtype=np.float64); move_ids=[]; input_ids=[]
            thresholds_gpu=torch.from_numpy(thresholds).cuda()
            with torch.no_grad():
                for lo in range(0,len(future),2048):
                    check(); b=data[torch.from_numpy(future[lo:lo+2048]).cuda()[:,None]+offsets]
                    x,y=b[:,:32],b[:,32:]; d=(y-x[:,-1:]).abs().double().mean(1)
                    mg=torch.where(d==0,0,torch.where(d<=thresholds_gpu[0],1,torch.where(d<=thresholds_gpu[1],2,3)))
                    changed=x.amax(1)!=x.amin(1); ig=changed.long()*2+changed[:,mapping].long()
                    move_ids.append(mg.cpu().numpy().astype(np.uint8)); input_ids.append(ig.cpu().numpy().astype(np.uint8))
                    for model_index,model in enumerate([original,candidate]):
                        error=(model(x)-y).double()
                        values=torch.stack([torch.ones_like(error[:,0])*8,error.abs().sum(1),error.square().sum(1),error.sum(1)],dim=2)
                        for kind,ids in enumerate([mg,ig]):
                            for group in range(4): sums[kind,group,:,model_index]+=(values*(ids==group)[:,:,None]).sum(0).cpu().numpy()
            for mi,folder in enumerate([base,new]):
                expected=json.loads((folder/'result.json').read_text())['future']['groups']['all']
                observed=sums[0,:,:,mi].sum((0,1))
                assert abs(observed[1]/observed[0]-expected['NMAE'])<1e-10
                assert abs(observed[2]/observed[0]-expected['NMSE'])<1e-10
            np.savez_compressed(run/f'fold_{origin}.npz',train_start=train,probe_start=future,thresholds=thresholds,movement_group=np.concatenate(move_ids),input_group=np.concatenate(input_ids),sums=sums)
            rows=[]
            for kind,names in enumerate([['zero','small','medium','tail'],['neither_changes','source_only','target_only','both_change']]):
                for group,name in enumerate(names):
                    a=sums[kind,group,:,0].sum(0); b=sums[kind,group,:,1].sum(0)
                    rows.append({'type':kind,'group':name,'scalar_targets':int(a[0]),'NMAE_delta_contribution':float((b[1]-a[1])/(len(future)*8*15)),'NMSE_delta_contribution':float((b[2]-a[2])/(len(future)*8*15))})
            reports.append({'origin':origin,'baseline':base_id,'candidate':new_id,'baseline_checkpoint_sha256':digest(base/'final.pt'),'candidate_checkpoint_sha256':digest(new/'final.pt'),'probe_windows':len(future),'groups':rows})
            print(json.dumps(reports[-1]),flush=True)
        write_json(run/'diagnostic.json',{'manifest_sha256':digest(manifest_path),'train_sha256':digest(source),'code_sha256':digest(Path(__file__)),'protocol_sha256':digest(ROOT/'research_runs/representation/ERROR_PROTOCOL.md'),'source_diagnostic_sha256':source_hash,'sensor_columns':manifest['sensor_columns'],'folds':reports})
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes()); (run/'protocol.md').write_bytes((ROOT/'research_runs/representation/ERROR_PROTOCOL.md').read_bytes())
        status('completed')
    except BaseException:
        status('failed'); raise


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); main()
