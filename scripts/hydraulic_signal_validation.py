"""One frozen Hydraulic task: prepare, anchor, lock forecasts, paired validation."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time

import numpy as np
import torch

INPUT = 32
HORIZON = 8
LENGTH = INPUT + HORIZON
SCOPES = ('global', 'target', 'complement')
ACTIONS = ('PARENT', 'P1', 'CP1')
SEEDS = (2026101501, 2026101502, 2026101503)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write(path, value, new=True):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if new:
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix('.tmp')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def verify(root):
    assert os.name == 'posix', 'Dedicated research server required'
    plan = read(root/'protocol.json')
    assert plan['script_sha256'] == sha(__file__)
    for name, expected in plan['dependencies_sha256'].items():
        assert sha(root/name) == expected, name
    assert sha(plan['frozen_core']) == plan['frozen_core_sha256']
    assert plan['seeds'] == list(SEEDS)
    expected = [dict(id=f'H1_{action}_s{seed}', action=action, seed=seed)
                for seed in SEEDS for action in ACTIONS]
    assert plan['jobs'] == expected
    assert plan['training'] == dict(input_length=32, prediction_length=8, moving_average_kernel=25,
                                  batch_size=256, steps=3000, learning_rate=.001, weight_decay=0.)
    assert not (root/'data'/'test.csv').exists(), 'No test file permitted in study data'
    return plan


def windows(cycles, offsets):
    assert cycles.ndim == offsets.ndim == 1 and cycles.shape == offsets.shape
    if len(cycles) < LENGTH:
        return np.empty(0, dtype=np.int64)
    invalid = (np.diff(cycles) != 0) | (np.diff(offsets) != 1)
    prefix = np.r_[0, np.cumsum(invalid)]
    starts = np.arange(len(cycles)-LENGTH+1, dtype=np.int64)
    return starts[prefix[starts+LENGTH-1] == prefix[starts]]


def load_split(root, split, manifest):
    assert split in ('train', 'validation')
    entry = next(item for item in manifest['splits'] if item['name'] == split)
    path = root/'data'/entry['file']
    assert sha(path) == entry['sha256']
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == manifest['columns_raw']
        rows = list(reader)
    values = np.asarray([[float(row[key]) for key in manifest['sensor_columns']] for row in rows], dtype=np.float64)
    cycles = np.asarray([int(row['cycle_id']) for row in rows], dtype=np.int64)
    offsets = np.asarray([int(row['sample_offset_seconds']) for row in rows], dtype=np.int64)
    assert len(values) == entry['rows'] and np.isfinite(values).all()
    assert ((offsets >= 0) & (offsets < 60)).all()
    assert (np.diff(cycles*60+offsets) == 1).all()
    return values, cycles, offsets


def preflight(root, plan):
    checks = []
    assert len(windows(np.zeros(39, dtype=int), np.arange(39))) == 0
    assert np.array_equal(windows(np.zeros(40, dtype=int), np.arange(40)), [0])
    checks.append('minimum_length_40')
    assert np.array_equal(windows(np.repeat([0, 1], 60), np.tile(np.arange(60), 2)), np.r_[np.arange(21), 60+np.arange(21)])
    checks.append('no_cross_cycle')
    assert len(windows(np.r_[np.zeros(30), np.ones(30)], np.r_[np.arange(30,60), np.arange(30)])) == 0
    checks.append('partition_fragments_not_joined')
    offsets = np.arange(60)
    offsets[30:] += 1
    assert len(windows(np.zeros(60), offsets)) == 0
    checks.append('offset_gap_rejected')
    methods = module('frozen_methods', root/'stage1_selection_methods.py')
    rule = read(root/'signal_rule.json')
    for threshold in rule['thresholds']:
        assert methods.signal_category(threshold, rule['thresholds']) != methods.signal_category(float(np.nextafter(threshold, np.inf)), rule['thresholds'])
    checks.append('original_threshold_boundary')
    metrics = module('frozen_metrics', root/'run_stage1_candidate.py').metrics
    assert metrics(np.ones((2,8,8), dtype=np.float32), np.zeros(2, dtype=bool))['NMSE'] is None
    checks.append('empty_complement_null')
    feature = module('frozen_signal', root/'signal_mean_shift.py').signal_value
    sample = np.zeros((2,32,8), dtype=np.float32)
    assert feature(sample, sample, sample, {'NMSE': 0.}) == 0.
    checks.append('zero_signal_convention')
    write(root/'preflight.json', dict(status='PASS', checks=checks, new_trainings=0,
          protocol_sha256=sha(root/'protocol.json'), source_sha256=sha(__file__)))


def prepare(root, plan):
    sync = read(root/'source_sync_receipt.json')
    assert sync['status'] == 'exact_commit_and_tree_verified'
    manifest = read(root/'data'/'manifest.json')
    assert sha(root/'data'/'manifest.json') == plan['dataset_manifest_sha256']
    assert manifest['dataset_id'] == 'uci_hydraulic_v1'
    assert manifest['sensor_columns'] == ['CE','CP','SE','TS1','TS2','TS3','TS4','VS1']
    output = root/'prepared'
    output.mkdir()
    started = time.monotonic()
    raw = {split: load_split(root, split, manifest) for split in ('train', 'validation')}
    mean = raw['train'][0].mean(axis=0)
    std = raw['train'][0].std(axis=0)
    scale = np.where(std == 0., 1., std)
    arrays = dict(mean=mean, scale=scale)
    for split, (values, cycles, offsets) in raw.items():
        starts = windows(cycles, offsets)
        assert len(starts) == plan['expected_windows'][split]
        arrays[split] = ((values-mean)/scale).astype(np.float32)
        for key, value in dict(cycles=cycles, offsets=offsets, starts=starts, base_rows=np.arange(len(values), dtype=np.int64)).items():
            arrays[split+'_'+key] = value
        assert np.all(cycles[starts] == cycles[starts+39])
        assert np.all(offsets[starts+39]-offsets[starts+31] == 8)
    proposal = module('frozen_proposal', root/'proposal_median3.py')
    fitted = proposal.fit(arrays['train'], arrays['train_starts'], arrays['train_cycles'][arrays['train_starts']], SEEDS[0])
    params = fitted['params']
    arrays['probability'] = np.full(len(arrays['train_starts']), 1./len(arrays['train_starts']), dtype=np.float64)
    arrays['loss_weight'] = np.ones(len(arrays['train_starts']), dtype=np.float64)
    for split in ('train','validation'):
        starts = arrays[split+'_starts']
        masks = []
        with torch.no_grad():
            for lo in range(0,len(starts),512):
                x = torch.tensor(arrays[split][starts[lo:lo+512,None]+np.arange(32)], device='cuda')
                masks.append(proposal.target_mask(x,params,'P1').cpu().numpy())
        arrays[split+'_target'] = np.concatenate(masks)
    assert np.array_equal(arrays['train_target'], fitted['candidates']['P1']['target'])
    np.savez_compressed(output/'arrays.npz', **arrays)
    write(output/'params.json', params)
    receipt = dict(status='PASS', task='H1', dataset_id=manifest['dataset_id'],
        protocol_sha256=sha(root/'protocol.json'), manifest_sha256=sha(root/'data'/'manifest.json'),
        arrays_sha256=sha(output/'arrays.npz'), params_sha256=sha(output/'params.json'),
        rows={split:len(arrays[split]) for split in raw}, windows=plan['expected_windows'],
        target_windows={split:int(arrays[split+'_target'].sum()) for split in raw},
        mean=mean.tolist(), scale=scale.tolist(), original_partition_membership_preserved=True,
        generated_sensor_records=0, test_access=False, process_seconds=time.monotonic()-started)
    write(root/'preparation_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def prepared(root, plan):
    receipt = read(root/'preparation_receipt.json')
    assert receipt['status'] == 'PASS' and receipt['protocol_sha256'] == sha(root/'protocol.json')
    assert sha(root/'prepared'/'arrays.npz') == receipt['arrays_sha256']
    assert sha(root/'prepared'/'params.json') == receipt['params_sha256']
    with np.load(root/'prepared'/'arrays.npz', allow_pickle=False) as source:
        arrays = {key:source[key] for key in source.files}
    return arrays, read(root/'prepared'/'params.json'), receipt


def prediction_lock(root):
    lock = read(root/'prediction_lock.json')
    assert lock['protocol_sha256'] == sha(root/'protocol.json')
    required = {'predictions.json','preparation_receipt.json',f'runs/H1_PARENT_s{SEEDS[0]}/result.json'}
    assert set(lock['files_sha256']) == required
    for name, expected in lock['files_sha256'].items():
        assert sha(root/name) == expected, name
    return lock


def runnable(root, plan, job):
    assert job in plan['jobs']
    position = plan['jobs'].index(job)
    activation = read(root/'anchor_activation.json')
    assert activation['protocol_sha256'] == sha(root/'protocol.json')
    for name in ('preparation_receipt.json','preflight.json','source_sync_receipt.json'):
        assert activation['files_sha256'][name] == sha(root/name)
    preflight_result = read(root/'preflight.json')
    assert preflight_result['status'] == 'PASS' and preflight_result['protocol_sha256'] == sha(root/'protocol.json')
    assert read(root/'source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
    if position:
        prediction_lock(root)
        response = read(root/'response_activation.json')
        assert response['protocol_sha256'] == sha(root/'protocol.json')
        assert response['prediction_lock_sha256'] == sha(root/'prediction_lock.json')
        assert response['source_sync_receipt_sha256'] == sha(root/'prediction_source_sync_receipt.json')
        assert read(root/'prediction_source_sync_receipt.json')['status'] == 'exact_commit_and_tree_verified'
    for earlier in plan['jobs'][:position]:
        status = read(root/'runs'/earlier['id']/'status.json')
        assert status['status'] == 'completed' and status['steps_completed'] == 3000
    for later in plan['jobs'][position:]:
        assert not (root/'runs'/later['id']).exists(), 'Occupied/out-of-order ID; never rerun'


def train(root, plan, job):
    runnable(root,plan,job)
    arrays, params, receipt = prepared(root, plan)
    metrics = module('frozen_metrics',root/'run_stage1_candidate.py').metrics
    proposal = module('frozen_proposal',root/'proposal_median3.py')
    output = root/'runs'/job['id']
    output.mkdir(parents=True)
    started = time.monotonic()
    status = dict(status='running', **job, steps_completed=0, protocol_sha256=sha(root/'protocol.json'),
                  prediction_lock_sha256=sha(root/'prediction_lock.json') if job != plan['jobs'][0] else None)
    write(output/'status.json',status)
    try:
        seed,action = job['seed'],job['action']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        core = module('frozen_core',plan['frozen_core'])
        config = dict(plan['training'], channels=8)
        model = core.DLinear(config).cuda()
        nparams = sum(value.numel() for value in model.parameters())
        assert nparams == 528*8
        initial_sha = hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())).hexdigest()
        optimizer = torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        t = torch.tensor(arrays['train'],device='cuda')
        d = torch.tensor(arrays['validation'],device='cuda')
        ts = torch.tensor(arrays['train_starts'],device='cuda')
        ds = torch.tensor(arrays['validation_starts'],device='cuda')
        offsets = torch.arange(LENGTH,device='cuda')
        probability = arrays['probability']
        assert np.all(probability == 1/len(ts)) and np.all(arrays['loss_weight'] == 1.)
        cdf=np.cumsum(probability)
        cdf[-1]=1.
        generator=np.random.default_rng(seed+1000)
        draws=np.zeros(len(ts),dtype=np.int64)
        history=[]
        best=float('inf')
        best_step=None

        def transform(x):
            return x if action=='PARENT' else proposal.transform(x,params,'P1',action=='CP1')

        def evaluate():
            result=np.empty((len(ds),HORIZON,8),dtype=np.float32)
            model.eval()
            with torch.no_grad():
                for lo in range(0,len(ds),2048):
                    batch=d[ds[lo:lo+2048,None]+offsets]
                    result[lo:lo+len(batch)]=(model(transform(batch[:,:INPUT]))-batch[:,INPUT:]).cpu().numpy()
            return result

        for step in range(1,3001):
            chosen=np.searchsorted(cdf,generator.random(config['batch_size']),side='right')
            np.add.at(draws,chosen,1)
            ids=torch.tensor(chosen,device='cuda')
            batch=t[ts[ids,None]+offsets]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=(model(transform(batch[:,:INPUT]))-batch[:,INPUT:]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            if step%500==0:
                error=evaluate()
                assert np.isfinite(error).all()
                measured=metrics(error,np.ones(len(error),dtype=bool))
                history.append(dict(step=step,NMSE=measured['NMSE'],NMAE=measured['NMAE']))
                status['steps_completed']=step
                write(output/'status.json',status,new=False)
                if measured['NMSE'] < best:
                    best=measured['NMSE']
                    best_step=step
                    torch.save(model.state_dict(),output/'best.pt')
                    np.savez_compressed(output/'best_errors.npz',error=error)
        torch.save(model.state_dict(),output/'last.pt')
        np.savez_compressed(output/'sampling.npz',probability=probability,loss_weight=arrays['loss_weight'],
                            draw_count=draws,train_window_start=arrays['train_starts'])
        with np.load(output/'best_errors.npz',allow_pickle=False) as saved:
            error=saved['error']
        scored={scope:metrics(error,mask) for scope,mask in [('global',np.ones(len(error),dtype=bool)),
                ('target',arrays['validation_target']),('complement',~arrays['validation_target'])]}
        error64=error.astype(np.float64)
        scored['global'].update(channel_NMSE=np.mean(error64**2,axis=(0,1)).tolist(),channel_NMAE=np.mean(abs(error64),axis=(0,1)).tolist())
        result=dict(**job,task='H1',metrics=scored,best_step=best_step,history=history,parameters=nparams,
            initialization_sha256=initial_sha,probability_sha256=digest(probability),draw_count_sha256=digest(draws),
            prepared_sha256=receipt['arrays_sha256'],params_sha256=receipt['params_sha256'],
            protocol_sha256=status['protocol_sha256'],prediction_lock_sha256=status['prediction_lock_sha256'],
            source_sha256=sha(__file__),assets={name:sha(output/name) for name in ('best.pt','last.pt','best_errors.npz','sampling.npz')},
            process_seconds=time.monotonic()-started,test_access=False)
        assert sha(root/'protocol.json') == status['protocol_sha256']
        write(output/'result.json',result)
        status['status']='completed'
        print(json.dumps(dict(id=job['id'],NMSE=scored['global']['NMSE'],seconds=result['process_seconds'])),flush=True)
    except Exception as error:
        status['status']='failed'
        status['error']=str(error)
        raise
    finally:
        status['process_seconds']=time.monotonic()-started
        write(output/'status.json',status,new=False)


def predict(root,plan):
    assert read(root/'control_anchor'/'status.json')['status']=='completed'
    assert all(not (root/'runs'/job['id']).exists() for job in plan['jobs'][1:])
    arrays,params,receipt=prepared(root,plan)
    parent_path=root/'runs'/plan['jobs'][0]['id']/'result.json'
    parent=read(parent_path)
    assert parent['action']=='PARENT' and parent['seed']==SEEDS[0]
    for name,expected in parent['assets'].items():
        assert sha(parent_path.parent/name)==expected
    starts=arrays['train_starts']
    chosen=starts[np.linspace(0,len(starts)-1,min(8192,len(starts))).astype(np.int64)]
    raw=arrays['train'][chosen[:,None]+np.arange(INPUT)]
    proposal=module('frozen_proposal',root/'proposal_median3.py')
    feature=module('frozen_signal',root/'signal_mean_shift.py').signal_value
    methods=module('frozen_methods',root/'stage1_selection_methods.py')
    rule=read(root/'signal_rule.json')
    assert rule['feature_sha256']==sha(root/'signal_mean_shift.py')
    cases=read(root/'construction_reference.json')['construction_cases']
    predictions={method:{} for method in ('new_signal','generic_stats','case_match','always_unchanged')}
    signals,trace={},{}
    for action in ('P1','CP1'):
        with torch.no_grad():
            changed=proposal.transform(torch.tensor(raw,device='cuda'),params,'P1',action=='CP1').cpu().numpy()
        raw.setflags(write=False)
        changed.setflags(write=False)
        identity=[digest(raw),digest(changed)]
        value=feature(raw,raw,changed,parent['metrics']['global'])
        generic=methods.generic_features(raw,changed,parent['metrics']['global'])
        assert identity==[digest(raw),digest(changed)]
        key='H1_'+action
        signals[key]=value
        predictions['new_signal'][key]=methods.signal_category(value,rule['thresholds'])
        for method in ('generic_stats','case_match'):
            predictions[method][key]=methods.nearest(cases,{'features':generic},method=='case_match')[0]
        predictions['always_unchanged'][key]='unchanged'
        trace[key]=dict(features=generic,input_sha256=identity)
    write(root/'predictions.json',dict(predictions=predictions,signals=signals,input_trace=trace,
        anchor_metrics=parent['metrics']['global'],sampled_windows=len(chosen),sample_start_sha256=digest(chosen),
        signal_sha256=sha(root/'signal_mean_shift.py'),rule_sha256=sha(root/'signal_rule.json'),
        candidate_responses_seen=False,new_LLM_calls=0,test_access=False))
    names=['predictions.json','preparation_receipt.json',f'runs/{plan["jobs"][0]["id"]}/result.json']
    write(root/'prediction_lock.json',dict(protocol_sha256=sha(root/'protocol.json'),files_sha256={name:sha(root/name) for name in names}))
    print(json.dumps(dict(signals=signals,predictions=predictions)),flush=True)


def supervise(root,plan,phase,execute):
    jobs=plan['jobs'][:1] if phase=='anchor' else plan['jobs'][1:]
    runnable(root,plan,jobs[0])
    helper=module('durable_logging',root/'supervise_stage1_recovery.py')
    control=root/('control_'+phase)
    if not execute:
        control.mkdir()
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--mode','supervise','--phase',phase,'--execute']
        process=helper.spawn_logged(command,control/'supervisor.log')
        write(control/'launch.json',dict(pid=process.pid,phase=phase,jobs=len(jobs),protocol_sha256=sha(root/'protocol.json'),launched_unix=time.time()))
        print(json.dumps(dict(status='launched',phase=phase,pid=process.pid,jobs=len(jobs))),flush=True)
        return
    with (control/'execution.lock').open('x') as lock:
        lock.write(str(os.getpid()))
    status=dict(status='running',pid=os.getpid(),active=None,completed=[],failed=None,phase=phase,protocol_sha256=sha(root/'protocol.json'))
    write(control/'status.json',status)
    try:
        for job in jobs:
            runnable(root,plan,job)
            status['active']=job['id']
            write(control/'status.json',status,new=False)
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),'--mode','train','--job',job['id']]
            process=helper.spawn_logged(command,control/(job['id']+'.log'))
            status['child_pid']=process.pid
            write(control/'status.json',status,new=False)
            code=process.wait()
            saved=read(root/'runs'/job['id']/'status.json')
            assert code==0 and saved['status']=='completed' and saved['steps_completed']==3000
            status['completed'].append(job['id'])
            status['active']=None
            status.pop('child_pid',None)
            write(control/'status.json',status,new=False)
        status['status']='completed'
    except Exception as error:
        status['status']='failed_stopped'
        status['failed']=status['active']
        status['error']=str(error)
        raise
    finally:
        status['ended_unix']=time.time()
        write(control/'status.json',status,new=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--mode',required=True,choices=['preflight','prepare','train','predict','supervise'])
    parser.add_argument('--phase',choices=['anchor','responses'])
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--job')
    args=parser.parse_args()
    root=Path(args.root)
    plan=verify(root)
    if args.mode=='preflight':
        preflight(root,plan)
    elif args.mode=='prepare':
        prepare(root,plan)
    elif args.mode=='train':
        train(root,plan,next(job for job in plan['jobs'] if job['id']==args.job))
    elif args.mode=='predict':
        predict(root,plan)
    else:
        assert args.phase
        supervise(root,plan,args.phase,args.execute)


if __name__=='__main__':
    main()
