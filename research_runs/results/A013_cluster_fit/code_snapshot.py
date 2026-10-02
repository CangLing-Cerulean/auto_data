"""A013: fit four groups only on first 60% training input windows."""
import csv
import json
import time
from pathlib import Path
import numpy as np
from run_research import ROOT,digest,valid_starts,write_json
from cluster_selection import FOLDER,ROUND_LIMIT,features,assign,fit_centers,used_seconds


def main():
    run=ROOT/'research_runs/results/A013_cluster_fit'
    if run.exists() or (FOLDER/'fit.json').exists(): raise RuntimeError('Frozen clustering already exists')
    used=used_seconds(); allowance=min(180,ROUND_LIMIT-used-60)
    if allowance<=0: raise RuntimeError('Round budget exhausted')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    def check():
        if time.monotonic()-begun>=allowance: raise RuntimeError('Clustering budget exhausted')
    status('preparing')
    try:
        manifest_path=ROOT/'data/metropt3_v1/manifest.json'; manifest=json.loads(manifest_path.read_text())
        source=manifest_path.parent/'train.csv'; expected=next(s for s in manifest['splits'] if s['name']=='train')
        assert digest(source)==expected['sha256']
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        with source.open(newline='') as f:
            reader=csv.reader(f); next(reader); times=np.array([row[1] for row in reader],dtype='datetime64[s]')
        assert len(raw)==expected['rows'] and np.isfinite(raw).all()
        starts=valid_starts(times,40,30)
        with np.load(ROOT/'research_runs/selection/window_table.npz') as z: assert np.array_equal(starts,z['start_row_zero_based'])
        cut=int(len(raw)*.6); fit_starts=starts[starts+40<=cut]
        values=features(raw[:cut],fit_starts,check); mean=values.mean(0); std=values.std(0); scale=np.where(std==0,1,std)
        rng=np.random.default_rng(20261002); sample_indices=rng.choice(len(values),min(32768,len(values)),replace=False)
        sample=(values[sample_indices]-mean)/scale
        centers,info=fit_centers(sample,rng,check)
        np.savez_compressed(FOLDER/'cluster_model.npz',feature_mean=mean,feature_scale=scale,centers=centers,
            fit_start=fit_starts,sample_indices=sample_indices,sample_start=fit_starts[sample_indices])
        labels=np.empty(len(starts),dtype=np.int8)
        for lo in range(0,len(starts),8192):
            f=features(raw,starts[lo:lo+8192],check); labels[lo:lo+len(f)]=assign((f-mean)/scale,centers)[0]
        months=times[starts+31].astype('datetime64[M]').astype(str)
        np.savez_compressed(FOLDER/'window_groups.npz',start_row=starts,group=labels,months=months)
        counts=np.bincount(labels[:len(fit_starts)],minlength=4)
        eligible=[i for i,n in enumerate(counts) if .01<=n/len(fit_starts)<=.5]
        meta={'manifest_sha256':digest(manifest_path),'train_sha256':digest(source),'window_table_sha256':digest(ROOT/'research_runs/selection/window_table.npz'),
            'code_sha256':digest(Path(__file__)),'helper_sha256':digest(ROOT/'scripts/cluster_selection.py'),'protocol_sha256':digest(FOLDER/'PROTOCOL.md'),
            'raw_rows':len(raw),'fit_raw_rows':[0,cut],'fit_windows':len(fit_starts),'feature_columns':[f'mean_{s}' for s in manifest['sensor_columns']]+[f'std_{s}' for s in manifest['sensor_columns']],
            'sample_size':len(sample_indices),'seed':20261002,'k':4,'fit':info,'prefix60_group_counts':counts.tolist(),'eligible_groups':eligible,
            'model_sha256':digest(FOLDER/'cluster_model.npz'),'window_groups_sha256':digest(FOLDER/'window_groups.npz')}
        write_json(FOLDER/'fit.json',meta); write_json(run/'diagnostic.json',meta)
        for source_path,name in [(Path(__file__),'code_snapshot.py'),(ROOT/'scripts/cluster_selection.py','helper_snapshot.py'),(FOLDER/'PROTOCOL.md','protocol.md')]: (run/name).write_bytes(source_path.read_bytes())
        status('completed'); print(json.dumps(meta),flush=True)
    except BaseException:
        status('failed'); raise


if __name__=='__main__': main()
