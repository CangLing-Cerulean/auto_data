"""A014: real-data grouping, masks, boundary, normalization and RNG audit."""
import csv
import gzip
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from run_research import ROOT,DLinear,digest,evaluate,write_json
from cluster_selection import FOLDER,ROUND_LIMIT,used_seconds,fit_centers


def main():
    run=ROOT/'research_runs/results/A014_cluster_audit'
    if run.exists(): raise RuntimeError('Immutable audit already exists')
    allowance=ROUND_LIMIT-used_seconds()
    if allowance<=0: raise RuntimeError('No remaining audit budget')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    def check():
        if time.monotonic()-begun>=allowance: raise RuntimeError('Audit budget exhausted')
    status('running')
    try:
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        meta=json.loads((FOLDER/'fit.json').read_text()); progress=json.loads((FOLDER/'progress.json').read_text())
        assert progress['phase'].startswith(('stopped_','completed_')) and not progress['phase'].startswith('completed_T')
        assert digest(FOLDER/'cluster_model.npz')==meta['model_sha256']
        assert digest(FOLDER/'window_groups.npz')==meta['window_groups_sha256']
        source=ROOT/'data/metropt3_v1/train.csv'; assert digest(source)==meta['train_sha256']
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        with source.open(newline='') as f:
            reader=csv.reader(f); next(reader); records=[(r[0],r[1]) for r in reader]
        original_ids=np.array([r[0] for r in records]); times=np.array([r[1] for r in records],dtype='datetime64[s]'); del records
        bad=np.r_[0,np.cumsum(np.diff(times).astype('timedelta64[s]').astype(np.int64)>30)]
        all_s=np.arange(len(raw)-39); expected_starts=all_s[bad[all_s+39]==bad[all_s]]
        with np.load(FOLDER/'window_groups.npz') as z: starts=z['start_row']; groups=z['group']; months=z['months']
        assert np.array_equal(starts,expected_starts)
        assert np.array_equal(months,times[starts+31].astype('datetime64[M]').astype(str))
        with np.load(FOLDER/'cluster_model.npz') as z:
            mean=z['feature_mean']; scale=z['feature_scale']; centers=z['centers']; fit_s=z['fit_start']; sample_idx=z['sample_indices']; sample_s=z['sample_start']
        cut=int(len(raw)*.6); assert np.array_equal(fit_s,starts[starts+40<=cut])
        # Recompute every feature directly from real input records, independent of features().
        fit_features=np.empty((len(fit_s),30),dtype=np.float64)
        for lo in range(0,len(starts),4096):
            check(); ss=starts[lo:lo+4096]; b=raw[ss[:,None]+np.arange(32)]
            feature=np.column_stack((np.mean(b,axis=1),np.std(b,axis=1,ddof=0)))
            part=min(len(feature),max(0,len(fit_s)-lo))
            if part: fit_features[lo:lo+part]=feature[:part]
            distance=np.sum((((feature-mean)/scale)[:,None,:]-centers[None,:,:])**2,axis=2)
            assert np.array_equal(distance.argmin(1),groups[lo:lo+len(ss)])
        fitted_mean=fit_features.mean(0); std=fit_features.std(0)
        assert np.array_equal(mean,fitted_mean) and np.array_equal(scale,np.where(std==0,1,std))
        rng=np.random.default_rng(20261002); chosen=rng.choice(len(fit_s),min(32768,len(fit_s)),replace=False)
        assert np.array_equal(chosen,sample_idx) and np.array_equal(sample_s,fit_s[chosen])
        again,fit_info=fit_centers((fit_features[chosen]-mean)/scale,rng,check)
        assert np.array_equal(again,centers) and fit_info==meta['fit']
        profiles=[{'group':c,'mean_input_features':fit_features[groups[:len(fit_s)]==c].mean(0).tolist()} for c in range(4)]
        del fit_features
        counts=np.bincount(groups[:len(fit_s)],minlength=4)
        assert counts.tolist()==meta['prefix60_group_counts']
        assert [i for i,n in enumerate(counts) if .01<=n/len(fit_s)<=.5]==meta['eligible_groups']
        p=json.loads((ROOT/'research_runs/selection/protocol.json').read_text()); checks=[]; exports={}
        for row in progress['rows']:
            check(); folder=run.parent/row['run_id']; stage=row['stage']
            identity=json.loads((folder/'identity.json').read_text())
            assert identity.get('preprocessing','baseline')=='baseline' and 'source_mapping' not in identity
            for filename,key in [('code_snapshot.py','code_sha256'),('core_snapshot.py','core_sha256'),('cluster_helper.py','cluster_helper_sha256'),('cluster_protocol.md','cluster_protocol_sha256'),('retention_mask.npz','retention_mask_sha256'),('retention_mask.json','retention_meta_sha256')]: assert digest(folder/filename)==identity[key],filename
            with np.load(folder/'retention_mask.npz') as z:
                ss=z['start_row']; gg=z['group']; mm=z['months']; retained=z['retained']; prob=z['probability']
            cut=int(len(raw)*stage/100); keep=starts+40<=cut
            assert np.array_equal(ss,starts[keep]) and np.array_equal(gg,groups[keep]) and np.array_equal(mm,months[keep])
            target=gg==row['group']; removed=target.copy()
            if row['random_control']:
                removed[:]=False; rng=np.random.default_rng(20261003)
                for month in np.unique(mm):
                    pool=np.flatnonzero(mm==month); removed[rng.choice(pool,int(target[pool].sum()),replace=False)]=True
            assert np.array_equal(retained,~removed)
            for month in np.unique(mm): assert int(removed[mm==month].sum())==int(target[mm==month].sum())
            assert np.array_equal(prob,retained.astype(np.float64)/retained.sum())
            asset='sampling_weights.npz' if stage==100 else 'windows.npz'
            with np.load(folder/asset) as z: draws=z['draw_count']; assert np.array_equal(z['probability'],prob)
            assert draws.sum()==768000 and not draws[removed].any()
            rng_gpu=torch.Generator(device='cuda').manual_seed(row['seed']+1000)
            cdf=torch.from_numpy(np.cumsum(prob)).cuda(); cdf[-1]=1
            replay=torch.zeros(len(ss),device='cuda',dtype=torch.int64)
            for _ in range(3000):
                u=torch.rand(256,device='cuda',dtype=torch.float64,generator=rng_gpu)
                idx=torch.searchsorted(cdf,u,right=True); replay.scatter_add_(0,idx,torch.ones_like(idx))
            assert np.array_equal(replay.cpu().numpy(),draws)
            data_audit=json.loads((folder/'data_audit.json').read_text()); actual_mean=raw[:cut].mean(0); std=raw[:cut].std(0); actual_scale=np.where(std==0,1,std)
            assert np.array_equal(actual_mean,data_audit['mean']) and np.array_equal(actual_scale,data_audit['scale'])
            model=DLinear(p).cuda(); assert sum(v.numel() for v in model.parameters())==7920
            model.load_state_dict(torch.load(folder/('best.pt' if stage==100 else 'final.pt'),weights_only=True))
            if stage!=100:
                with np.load(folder/'windows.npz') as z: future=z['probe_start']
                end=int(round(len(raw)*(stage/100+.2)))
                assert np.array_equal(future,starts[(starts>=cut)&(starts+40<=end)]) and ss.max()+40<=future.min()
                data=torch.from_numpy(((raw[:end]-actual_mean)/actual_scale).astype(np.float32)).cuda()
                measured=evaluate(model,data,torch.from_numpy(future).cuda(),{'all':np.ones(len(future),dtype=bool)},p,'baseline')['groups']['all']
                for metric in ['NMSE','NMAE']: assert abs(measured[metric]-row[metric])<1e-10
            filename=f'{row["run_id"]}_windows.csv.gz'; export=FOLDER/filename
            with gzip.open(export,'wt',encoding='utf-8',newline='') as f:
                writer=csv.writer(f); writer.writerow(['start_row_zero_based','input_end_exclusive','window_end_exclusive','source_first_index','source_last_index','group','retained','input_end_month','sampling_probability','draw_count'])
                for k,s in enumerate(ss):
                    if k%100000==0: check()
                    writer.writerow([int(s),int(s+32),int(s+40),original_ids[s],original_ids[s+39],int(gg[k]),int(retained[k]),mm[k],float(prob[k]),int(draws[k])])
            exports[filename]=digest(export)
            checks.append({'run_id':row['run_id'],'eligible_windows':len(ss),'removed_windows':int(removed.sum()),'retained_windows':int(retained.sum()),'normalization_matches_complete_training_partition':True,'monthly_deletion_counts_matched':True,'deleted_draws':0,'draws':int(draws.sum()),'sampler_replay_exact':True,'model_parameters':7920})
        # Independently verify the first-stage choice, including the no-candidate terminal case.
        baseline=json.loads((run.parent/'T000_prefix60_full/result.json').read_text())['future']['groups']['all']
        candidates=[r for r in progress['rows'] if r['stage']==60 and not r['random_control'] and r['NMSE']<=.99*baseline['NMSE'] and r['NMAE']<=baseline['NMAE']]
        selected=min(candidates,key=lambda r:(r['NMSE'],r['group']))['group'] if candidates else None
        assert selected==progress['selected_group']
        if selected is None: assert progress['phase']=='stopped_no_prefix60_candidate' and len(progress['rows'])==len(meta['eligible_groups'])
        write_json(FOLDER/'verification.json',{'fit_and_all_group_labels_reproduced':True,'fit_uses_only_prefix60_inputs':True,'frozen_parameter_hashes_verified':True,'first_stage_decision_reproduced':True,'cluster_profiles':profiles,'checks':checks,'exports':exports})
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes()); status('completed')
    except BaseException:
        status('failed'); raise


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); main()
