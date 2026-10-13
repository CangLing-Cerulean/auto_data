"""Independent array, inference, pairing and fixed-endpoint audit; no new training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time
import numpy as np
import torch


def load(path):
    spec = importlib.util.spec_from_file_location('hydraulic_runner', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def direction(gain):
    return 'improve' if gain > .25 else 'harm' if gain < -.25 else 'unchanged'


def audit(root, h, plan):
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    h.prediction_lock(root)
    arrays, params, prepared = h.prepared(root, plan)
    manifest = h.read(root/'data/manifest.json')
    assert h.sha(root/'data/manifest.json') == plan['dataset_manifest_sha256']
    proposal = h.module('proposal', root/'proposal_median3.py')
    metrics = h.module('metrics', root/'run_stage1_candidate.py').metrics
    core = h.module('core', plan['frozen_core'])
    raw = {s:h.load_split(root,s,manifest) for s in ('train','validation')}
    mean, std = raw['train'][0].mean(0), raw['train'][0].std(0)
    scale = np.where(std == 0, 1., std)
    np.testing.assert_array_equal(arrays['mean'],mean)
    np.testing.assert_array_equal(arrays['scale'],scale)
    for split,(values,cycles,offsets) in raw.items():
        # Deliberately use direct per-window checks rather than runner's prefix algorithm.
        starts = np.asarray([i for i in range(len(values)-39)
            if np.all(cycles[i:i+40] == cycles[i]) and np.array_equal(offsets[i:i+40],offsets[i]+np.arange(40))],dtype=np.int64)
        for key,value in dict(cycles=cycles,offsets=offsets,starts=starts,base_rows=np.arange(len(values))).items():
            np.testing.assert_array_equal(arrays[split+'_'+key],value)
        np.testing.assert_array_equal(arrays[split],((values-mean)/scale).astype(np.float32))
        masks=[]
        for lo in range(0,len(starts),512):
            x=torch.tensor(arrays[split][starts[lo:lo+512,None]+np.arange(32)],device='cuda')
            masks.append(proposal.target_mask(x,params,'P1').cpu().numpy())
        np.testing.assert_array_equal(arrays[split+'_target'],np.concatenate(masks))
    fitted=proposal.fit(arrays['train'],arrays['train_starts'],arrays['train_cycles'][arrays['train_starts']],h.SEEDS[0])
    assert fitted['params']==params
    for phase,jobs in [('anchor',plan['jobs'][:1]),('responses',plan['jobs'][1:])]:
        control=h.read(root/('control_'+phase)/'status.json')
        assert control['status']=='completed' and control['completed']==[j['id'] for j in jobs]
        assert control['protocol_sha256']==h.sha(root/'protocol.json')
    predictions=h.read(root/'predictions.json')
    rule=h.read(root/'signal_rule.json')
    methods=h.module('methods',root/'stage1_selection_methods.py')
    feature=h.module('feature',root/'signal_mean_shift.py').signal_value
    cases=h.read(root/'construction_reference.json')['construction_cases']
    starts=arrays['train_starts']
    sampled=starts[np.linspace(0,len(starts)-1,min(8192,len(starts))).astype(np.int64)]
    x=arrays['train'][sampled[:,None]+np.arange(32)]
    anchor=h.read(root/'runs'/plan['jobs'][0]['id']/'result.json')['metrics']['global']
    assert anchor==predictions['anchor_metrics'] and h.digest(sampled)==predictions['sample_start_sha256']
    assert predictions['signal_sha256']==h.sha(root/'signal_mean_shift.py') and predictions['rule_sha256']==h.sha(root/'signal_rule.json')
    for action in ('P1','CP1'):
        changed=proposal.transform(torch.tensor(x,device='cuda'),params,'P1',action=='CP1').cpu().numpy()
        key='H1_'+action
        value=feature(x,x,changed,anchor)
        assert value==predictions['signals'][key]
        assert predictions['predictions']['new_signal'][key]==methods.signal_category(value,rule['thresholds'])
        generic=methods.generic_features(x,changed,anchor)
        assert generic==predictions['input_trace'][key]['features']
        assert predictions['input_trace'][key]['input_sha256']==[h.digest(x),h.digest(changed)]
        for method in ('generic_stats','case_match'):
            assert predictions['predictions'][method][key]==methods.nearest(cases,{'features':generic},method=='case_match')[0]
        assert predictions['predictions']['always_unchanged'][key]=='unchanged'
    target=arrays['validation_target']
    masks={'global':np.ones(len(target),dtype=bool),'target':target,'complement':~target}
    results,hashes={},{}
    seconds=0.
    data=torch.tensor(arrays['validation'],device='cuda')
    ds=torch.tensor(arrays['validation_starts'],device='cuda')
    for job in plan['jobs']:
        folder=root/'runs'/job['id']
        r,s=h.read(folder/'result.json'),h.read(folder/'status.json')
        assert s['status']=='completed' and s['steps_completed']==3000
        for k,v in job.items():
            assert r[k]==v and s[k]==v
        assert r['parameters']==4224 and r['source_sha256']==plan['script_sha256']
        assert r['protocol_sha256']==h.sha(root/'protocol.json') and r['prepared_sha256']==prepared['arrays_sha256'] and r['params_sha256']==prepared['params_sha256']
        assert r['prediction_lock_sha256']==(None if job==plan['jobs'][0] else h.sha(root/'prediction_lock.json'))
        for name,expected in r['assets'].items():
            assert h.sha(folder/name)==expected
        with np.load(folder/'best_errors.npz',allow_pickle=False) as saved:
            error=saved['error']
        assert error.shape==(9240,8,8) and np.isfinite(error).all()
        model=core.DLinear(dict(plan['training'],channels=8)).cuda()
        model.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location='cuda'))
        model.eval()
        with torch.no_grad():
            for lo in range(0,len(ds),2048):
                batch=data[ds[lo:lo+2048,None]+torch.arange(40,device='cuda')]
                inputs=batch[:,:32]
                if job['action']!='PARENT':
                    inputs=proposal.transform(inputs,params,'P1',job['action']=='CP1')
                actual=(model(inputs)-batch[:,32:]).cpu().numpy()
                np.testing.assert_array_equal(actual,error[lo:lo+len(batch)])
        for scope,mask in masks.items():
            measured=metrics(error,mask)
            for key,value in measured.items():
                assert value==r['metrics'][scope][key],(job['id'],scope,key)
        for metric,power in [('NMSE',2),('NMAE',1)]:
            by_channel=np.mean(np.abs(error.astype(np.float64))**power,axis=(0,1))
            np.testing.assert_array_equal(by_channel,r['metrics']['global']['channel_'+metric])
            weighted=sum(r['metrics'][sc][metric]*r['metrics'][sc]['elements'] for sc in ('target','complement') if r['metrics'][sc]['elements'])
            np.testing.assert_allclose(weighted/r['metrics']['global']['elements'],r['metrics']['global'][metric],rtol=1e-12)
        with np.load(folder/'sampling.npz',allow_pickle=False) as saved:
            n=len(starts)
            np.testing.assert_array_equal(saved['train_window_start'],starts)
            assert np.all(saved['probability']==1/n) and np.all(saved['loss_weight']==1)
            assert saved['draw_count'].sum()==768000
            assert h.digest(saved['draw_count'])==r['draw_count_sha256'] and h.digest(saved['probability'])==r['probability_sha256']
        assert [a['step'] for a in r['history']]==list(range(500,3001,500))
        best=min(r['history'],key=lambda a:a['NMSE'])
        assert r['best_step']==best['step'] and r['metrics']['global']['NMSE']==best['NMSE']
        seconds+=s['process_seconds']
        hashes[job['id']]=h.sha(folder/'result.json')
        results[job['id']]=r
    cells={}
    for action in ('P1','CP1'):
        repeats=[]
        for seed in h.SEEDS:
            p,c=results[f'H1_PARENT_s{seed}'],results[f'H1_{action}_s{seed}']
            for key in ('initialization_sha256','draw_count_sha256','probability_sha256'):
                assert p[key]==c[key]
            gain={scope:{metric:100*(1-c['metrics'][scope][metric]/p['metrics'][scope][metric]) if p['metrics'][scope]['elements'] and p['metrics'][scope][metric]>0 else None for metric in ('NMSE','NMAE')} for scope in masks}
            repeats.append(dict(seed=seed,gain_percent=gain,category=direction(gain['global']['NMSE'])))
        categories=[r['category'] for r in repeats]
        mse=[r['gain_percent']['global']['NMSE'] for r in repeats]
        mae=[r['gain_percent']['global']['NMAE'] for r in repeats]
        cells['H1_'+action]=dict(repeats=repeats,stable_category=categories[0] if len(set(categories))==1 else 'unstable_response',
            mean_NMSE_gain_percent=float(np.mean(mse)),mean_NMAE_gain_percent=float(np.mean(mae)),
            adoption_eligible=bool(all(v>0 for v in mse) and np.mean(mse)>=1 and all(v>=-1 for v in mae)))
    scores={}
    for method,pred in predictions['predictions'].items():
        stable={k:c['stable_category'] for k,c in cells.items() if c['stable_category']!='unstable_response'}
        correct=sum(pred[k]==v for k,v in stable.items())
        scores[method]=dict(total=2,stable_cells=len(stable),correct=correct,
            wrong=sum(pred[k] not in (v,'uncertain') for k,v in stable.items()),abstained=sum(v=='uncertain' for v in pred.values()),
            observed_harm=sum(v=='harm' for v in stable.values()),harm_warned=sum(v=='harm' and pred[k]=='harm' for k,v in stable.items()),
            false_harm_warning=sum(v!='harm' and pred[k]=='harm' for k,v in stable.items()),
            wrong_direction=sum((pred[k],v) in (('harm','improve'),('improve','harm')) for k,v in stable.items()))
    changed=any(c['stable_category'] in ('improve','harm') for c in cells.values())
    passed=scores['new_signal']['correct']==2 and changed
    report=dict(status='PASS',cells=cells,scores=scores,directional_transfer_passed=passed,
        unchanged_only_hit=scores['new_signal']['correct']==2 and not changed,
        adoption_eligible=[k for k,c in cells.items() if c['adoption_eligible']],result_sha256=hashes,
        new_successful_trainings=9,new_failed_runs=0,cumulative_successful_trainings=295,preserved_prior_failed_delivery_count=1,
        training_process_seconds=seconds,audit_process_seconds=time.monotonic()-started,
        verification=['raw_data_hashes','T_only_normalization_and_fit','independent_cycle_windows','saved_checkpoint_exact_inference','element_metrics','paired_initialization_and_sampling','locked_predictions'],
        protocol_sha256=h.sha(root/'protocol.json'),prediction_lock_sha256=h.sha(root/'prediction_lock.json'),
        auditor_sha256=h.sha(__file__),new_LLM_calls=0,test_access=False,next='end phase; no automatic test or new study')
    h.write(root/'final_audit.json',report)
    print(json.dumps(report),flush=True)


def package(root,h,plan):
    assert h.read(root/'final_audit.json')['status']=='PASS'
    output=root/'hydraulic_signal_validation_20261013.tar.gz'
    assert not output.exists()
    sources=[p for p in root.rglob('*') if p.is_file() and 'data' not in p.relative_to(root).parts and '__pycache__' not in p.parts and p.name!=output.name and not p.name.endswith('.lock')]
    core=Path(plan['frozen_core'])
    with tarfile.open(output,'x:gz') as archive:
        for path in sorted(sources):
            archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
        archive.add(core,arcname='frozen_core/run_research.py',recursive=False)
    members={str(p.relative_to(root)):h.sha(p) for p in sources}
    members['frozen_core/run_research.py']=h.sha(core)
    with tarfile.open(output,'r:gz') as archive:
        assert set(archive.getnames())==set(members)
        for name,expected in members.items():
            assert hashlib.file_digest(archive.extractfile(name),'sha256').hexdigest()==expected
    h.write(root/'archive_receipt.json',dict(status='PASS',file=output.name,bytes=output.stat().st_size,sha256=h.sha(output),members_sha256=members,raw_sensor_files_included=False))
    print(json.dumps(dict(file=output.name,bytes=output.stat().st_size,sha256=h.sha(output),members=len(members))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--package',action='store_true')
    args=parser.parse_args()
    root=Path(args.root)
    h=load(root/'hydraulic_signal_validation.py')
    plan=h.verify(root)
    (package if args.package else audit)(root,h,plan)
