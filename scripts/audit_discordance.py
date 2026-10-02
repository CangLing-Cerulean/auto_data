"""Independent source-row, input-rule, loss-weight and paired-exposure audit."""
import json
import time
from pathlib import Path
import numpy as np
from run_research import ROOT,digest,write_json


def main():
    run=ROOT/'research_runs/results/A012_discordance_audit'
    if run.exists(): raise RuntimeError('Immutable audit exists')
    run.mkdir(); begun=time.monotonic()
    def status(phase): write_json(run/'status.json',{'phase':phase,'elapsed_seconds':time.monotonic()-begun})
    status('running')
    try:
        progress=json.loads((ROOT/'research_runs/selection/discordance_progress.json').read_text())
        assert progress['phase'] in ['completed_internal','stopped_internal_gate']
        mapping=json.loads((ROOT/'research_runs/results/A009_leading_sources/diagnostic.json').read_text())['folds'][0]['mapping']
        source=ROOT/'data/metropt3_v1/train.csv'
        raw=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17),dtype=np.float64)
        checks=[]
        for row in progress['rows']:
            folder=run.parent/row['run_id']; origin=row['origin']
            base=run.parent/('T010_prefix60_sources' if origin==.6 else 'T011_prefix80_sources')
            identity=json.loads((folder/'identity.json').read_text())
            for filename,key in [('code_snapshot.py','code_sha256'),('weighting_helper.py','weighting_helper_sha256'),('weighting_protocol.md','weighting_protocol_sha256'),('source_helper.py','source_helper_sha256')]:
                assert digest(folder/filename)==identity[key]
            audit=json.loads((folder/'weight_audit.json').read_text())
            assert digest(folder/'discordance_weights.npz')==audit['asset_sha256']
            assert json.loads((folder/'data_audit.json').read_text())==json.loads((base/'data_audit.json').read_text())
            with np.load(folder/'discordance_weights.npz') as z,np.load(folder/'windows.npz') as w,np.load(base/'windows.npz') as ref:
                starts=z['start_row']; weights=z['loss_weight']; rule=z['rule_mask']; applied=z['applied_mask']; months=z['months']
                for key in ['train_start','probe_start','probability','draw_count']: assert np.array_equal(w[key],ref[key]),key
                assert np.array_equal(starts,w['train_start']) and w['draw_count'].sum()==768000
                assert starts.max()+40<=w['probe_start'].min()
                for lo in range(0,len(starts),4096):
                    b=raw[starts[lo:lo+4096,None]+np.arange(32)]
                    changed=b.max(1)!=b.min(1)
                    expected=(~changed)&changed[:,mapping]
                    assert np.array_equal(rule[lo:lo+len(b)],expected)
                reconstructed=rule.copy()
                if row['policy']=='shuffle':
                    rng=np.random.default_rng(20261001)
                    for c in range(15):
                        for month in np.unique(months):
                            idx=np.flatnonzero(months==month); reconstructed[idx,c]=rng.permutation(rule[idx,c])
                assert np.array_equal(applied,reconstructed)
                unnormalized=1+applied.astype(np.float64)
                assert np.array_equal(weights,unnormalized/unnormalized.mean(0))
                np.testing.assert_allclose(weights.mean(0),1,atol=1e-10,rtol=0)
                for month in np.unique(months): assert np.array_equal(rule[months==month].sum(0),applied[months==month].sum(0))
                checks.append({'run_id':row['run_id'],'eligible_windows':len(starts),'rule_counts':rule.sum(0).tolist(),'min_weight':float(weights.min()),'max_weight':float(weights.max()),'all_inputs_and_weights_verified':True,'identical_actual_draws_to_source_baseline':True,'month_count_matched':True})
        write_json(ROOT/'research_runs/selection/discordance_verification.json',{'train_sha256':digest(source),'checks':checks})
        (run/'code_snapshot.py').write_bytes(Path(__file__).read_bytes()); status('completed')
    except BaseException:
        status('failed'); raise


if __name__=='__main__': main()
