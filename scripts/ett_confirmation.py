"""Frozen ETT S/W confirmation. No test measurements or candidate search."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from run_research import DLinear

ROOT = Path('/root/auto_data_experiments/20261016_ett_confirmation_r1')
SEEDS = (2026101901, 2026101902, 2026101903)
ACTIONS = ('PARENT', 'FAVOR_EARLY', 'FAVOR_LATE')
CONFIG = dict(input_length=96, prediction_length=96, channels=7, moving_average_kernel=25)
GRID = tuple(range(500, 3001, 500))
SPLITS = dict(A=(0, 8640), B=(8640, 10080), C=(10080, 11520))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, update=False):
    path = Path(path)
    target = path.with_name(path.name + '.tmp') if update else path
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if update:
        target.replace(path)


def tick():
    torch.cuda.synchronize()
    return time.perf_counter()


def configure():
    assert __debug__ and ROOT.resolve() == ROOT and os.name == 'posix'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def verify(station):
    assert station in ('ETTh1', 'ETTh2')
    lock = read(ROOT/'execution_lock.json')
    assert lock['protocol_sha256'] == sha(ROOT/'PROTOCOL.md')
    assert lock['authorization'] == '2026-10-16 user requested completion of ETT confirmation'
    for name, digest in lock['code_sha256'].items():
        assert (ROOT/name).resolve() == ROOT/name
        assert sha(ROOT/name) == digest, name
    manifest = read(ROOT/'manifest.json')
    assert sha(ROOT/'manifest.json') == lock['manifest_sha256']
    data_lock = read(ROOT/'data_lock.json')
    assert sha(ROOT/'data_lock.json') == lock['data_lock_sha256']
    assert data_lock['protocol_sha256'] == lock['protocol_sha256']
    assert data_lock['manifest_sha256'] == lock['manifest_sha256']
    assert sha(ROOT/(station+'_windows.csv')) == data_lock['indexes'][station]['sha256']
    if station == 'ETTh2':
        unlock = read(ROOT/'ETTh2_unlock.json')
        assert unlock['status'] == 'ETTh1_results_and_runtime_published_verified'
        assert len(unlock['verified_commit']) == 40
        assert unlock['summary_sha256'] == sha(ROOT/'ETTh1/summary.json')
        assert unlock['audit_sha256'] == sha(ROOT/'ETTh1/audit.json')
        assert read(ROOT/'ETTh1/audit.json')['status'] == 'PASS'
    source = next(x for x in manifest['files'] if x['path'] == f'raw/{station}.csv')
    assert sha(ROOT/(station+'.csv')) == source['sha256']
    assert (ROOT/(station+'.csv')).stat().st_size == source['bytes']
    return lock, manifest


def model(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return DLinear(CONFIG).cuda()


def optimizer(net):
    return torch.optim.Adam(net.parameters(), lr=.001, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., foreach=False, fused=False)


def distribution(action, impl):
    assert action in ACTIONS and impl in ('S', 'W', 'P')
    early = np.arange(8449) + 96 < 4320
    u = np.ones(8449, dtype=np.float64)
    if action != 'PARENT':
        u[early if action == 'FAVOR_EARLY' else ~early] = 3.
    p = u/u.sum()
    q = p if impl == 'S' else np.full(8449, 1/8449)
    w = 8449*p if impl == 'W' else np.ones(8449)
    cdf = np.cumsum(q)
    cdf[-1] = 1.
    return p, q, w, cdf


class Data:
    def __init__(self, station, prepared=False):
        self.station = station
        self.directory = ROOT/station
        if prepared:
            with np.load(self.directory/'ABC.npz', allow_pickle=False) as saved:
                values = {s:saved[s] for s in SPLITS}
        else:
            _, manifest = verify(station)
            with (ROOT/(station+'.csv')).open(encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream)
                assert next(reader) == manifest['columns']
                # Stop before requesting any D row from the CSV iterator.
                rows = list(itertools.islice(reader, 11520))
            raw = np.array([[float(v) for v in row[1:]] for row in rows], dtype=np.float64)
            assert raw.shape == (11520, 7) and np.isfinite(raw).all()
            dates = np.array([r[0] for r in rows], dtype='datetime64[s]')
            assert np.all(np.diff(dates) == np.timedelta64(3600, 's'))
            mean, std = raw[:8640].mean(0), raw[:8640].std(0)
            scale = np.where(std == 0, 1., std)
            values = {s:((raw[lo:hi]-mean)/scale).astype(np.float32) for s,(lo,hi) in SPLITS.items()}
            self.directory.mkdir()
            with (self.directory/'ABC.npz').open('xb') as stream:
                np.savez_compressed(stream, **values, mean=mean, scale=scale)
            write(self.directory/'preprocessing.json', dict(mean=mean.tolist(), scale=scale.tolist(),
                fitted_rows=[0,8640], parsed_rows=[0,11520], D_values_parsed=False,
                windows=dict(A=8449,B=1249,C=1249), raw_sha256=sha(ROOT/(station+'.csv')),
                prepared_sha256=sha(self.directory/'ABC.npz')))
        self.values = {s:torch.tensor(v, device='cuda') for s,v in values.items()}
        self.offset = torch.arange(192, device='cuda')
        self.count = {s:len(v)-191 for s,v in values.items()}
        assert self.count == dict(A=8449,B=1249,C=1249)
        with (ROOT/(station+'_windows.csv')).open(newline='') as stream:
            indexes = list(csv.DictReader(stream))
        expected = [(s, i, i+96, i+192) for s,(lo,hi) in SPLITS.items() for i in range(lo,hi-191)]
        actual = [(r['split'],int(r['input_start']),int(r['target_start']),int(r['target_stop'])) for r in indexes]
        assert actual == expected
        assert all(r['group'] == ('EARLY' if int(r['target_start'])<4320 else 'LATE') for r in indexes if r['split']=='A')

    def batch(self, split, ids):
        ids = torch.as_tensor(ids, device='cuda')
        block = self.values[split][ids[:,None]+self.offset]
        return block[:,:96], block[:,96:]

    @torch.no_grad()
    def evaluate(self, net, split):
        residuals = []
        net.eval()
        for lo in range(0, self.count[split], 256):
            x,y = self.batch(split, np.arange(lo,min(lo+256,self.count[split])))
            residuals.append((net(x)-y).cpu().numpy())
        residual = np.concatenate(residuals)
        assert np.isfinite(residual).all()
        r = residual.astype(np.float64)
        return dict(NMSE=float(np.mean(r*r)), NMAE=float(np.mean(np.abs(r))),
            channel_NMSE=np.mean(r*r,axis=(0,1)).tolist(),
            channel_NMAE=np.mean(np.abs(r),axis=(0,1)).tolist(),
            residual_sha256=hashlib.sha256(residual.tobytes()).hexdigest())


def loss(net, x, y, weights):
    return (((net(x)-y)**2).mean((1,2))*weights).mean()


def preflight(station):
    verify(station)
    begin = tick()
    data = Data(station)
    probability_error = 0.
    for action in ACTIONS[1:]:
        p,q,w,_ = distribution(action,'W')
        probability_error = max(probability_error,float(np.max(np.abs(p-q*w))))
        np.testing.assert_allclose(p,q*w,rtol=1e-14,atol=1e-18)
        assert np.isclose(p.sum(),1.) and np.all(p>0) and np.isclose(np.mean(w),1.)
    net = model(SEEDS[0]); opt = optimizer(net)
    rng = np.random.default_rng(SEEDS[0]+1000)
    _,_,w,cdf = distribution('FAVOR_EARLY','W')
    ids = np.searchsorted(cdf,rng.random(256),side='right')
    x,y = data.batch('A',ids)
    weights = torch.tensor(w[ids],dtype=torch.float32,device='cuda')
    prediction = net(x)
    actual = (((prediction-y)**2).mean((1,2))*weights).mean()
    reference = sum(weights[i]*(prediction[i]-y[i]).square().sum() for i in range(256))/(256*96*7)
    torch.testing.assert_close(actual,reference,rtol=2e-6,atol=1e-7)
    actual.backward()
    before = [p.detach().double().clone() for p in net.parameters()]
    grads = [p.grad.detach().double().clone() for p in net.parameters()]
    opt.step()
    max_error = 0.
    for parameter,initial,g in zip(net.parameters(),before,grads):
        expected = initial-.001*g/(g.abs()+1e-8)
        max_error = max(max_error,float((parameter.double()-expected).abs().max()))
        torch.testing.assert_close(parameter.double(),expected,rtol=2e-6,atol=2e-8)
        state = opt.state[parameter]
        torch.testing.assert_close(state['exp_avg'].double(),.1*g,rtol=2e-6,atol=1e-9)
        torch.testing.assert_close(state['exp_avg_sq'].double(),.001*g*g,rtol=2e-6,atol=1e-12)
    start = tick()
    for _ in range(100):
        ids = np.searchsorted(cdf,rng.random(256),side='right')
        x,y = data.batch('A',ids)
        opt.zero_grad(set_to_none=True)
        value = loss(net,x,y,torch.tensor(w[ids],dtype=torch.float32,device='cuda'))
        assert bool(torch.isfinite(value))
        value.backward(); opt.step()
    training_seconds = tick()-start
    start = tick(); data.evaluate(net,'B'); evaluation_seconds=tick()-start
    receipt = dict(status='PASS',station=station,expected_gradient_coefficient_max_error=probability_error,
        real_batch_loss_error=float(abs(actual.detach()-reference.detach())),adam_one_step_max_error=max_error,
        diagnostic_updates=101, new_full_trainings=0,training_100_steps_seconds=training_seconds,
        one_evaluation_seconds=evaluation_seconds,estimated_30_runs_seconds=30*(30*training_seconds+8*evaluation_seconds),
        estimate_scope='rough training/evaluation extrapolation; actual total wall time recorded separately',
        total_seconds=tick()-begin,free_bytes=shutil.disk_usage(ROOT).free,
        environment=dict(torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    assert receipt['free_bytes'] > 256*1024**2
    write(data.directory/'preflight.json',receipt)
    print(json.dumps(receipt),flush=True)


def identities():
    return [(seed,action,impl) for seed in SEEDS for action in ACTIONS
            for impl in (('P',) if action=='PARENT' else ('S','W'))]


def run_name(seed,action,impl):
    return f'{action}_{impl}_s{seed}'


def checkpoint(path,net,opt,rng,step):
    with path.open('xb') as stream:
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),stream)


def train(station):
    verify(station)
    directory=ROOT/station
    assert read(directory/'preflight.json')['status']=='PASS'
    assert sha(directory/'ABC.npz')==read(directory/'preprocessing.json')['prepared_sha256']
    (directory/'runs').mkdir()
    begin=tick(); data=Data(station,prepared=True)
    for seed,action,impl in identities():
        out=directory/'runs'/run_name(seed,action,impl); out.mkdir()
        start=tick(); status=dict(status='running',seed=seed,action=action,implementation=impl,steps_completed=0)
        write(out/'status.json',status)
        try:
            net=model(seed); opt=optimizer(net); rng=np.random.default_rng(seed+1000)
            checkpoint(out/'initial.pt',net,opt,rng,0)
            _,_,w,cdf=distribution(action,impl)
            weights=torch.tensor(w,dtype=torch.float32,device='cuda')
            counts=np.zeros(8449,dtype=np.int64)
            curve=[]; best_value=float('inf'); best=None
            for step in range(1,3001):
                ids=np.searchsorted(cdf,rng.random(256),side='right')
                counts += np.bincount(ids,minlength=8449)
                x,y=data.batch('A',ids)
                net.train(); opt.zero_grad(set_to_none=True)
                value=loss(net,x,y,weights[torch.as_tensor(ids,device='cuda')])
                if not bool(torch.isfinite(value)):
                    raise ValueError(f'Nonfinite loss at step {step}')
                value.backward(); opt.step()
                if step in GRID:
                    metric=data.evaluate(net,'B')
                    curve.append(dict(step=step,**metric))
                    if metric['NMSE'] < best_value:
                        best_value=metric['NMSE']
                        best=copy.deepcopy(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,
                            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
                    status.update(steps_completed=step,elapsed_seconds=tick()-start)
                    write(out/'status.json',status,update=True)
            checkpoint(out/'last.pt',net,opt,rng,3000)
            with (out/'best.pt').open('xb') as stream: torch.save(best,stream)
            write(out/'B_curve.json',curve)
            with (out/'sampling_counts.npz').open('xb') as stream: np.savez_compressed(stream,counts=counts)
            assert int(counts.sum())==3000*256
            status.update(status='completed',best_step=best['step'],elapsed_seconds=tick()-start)
            write(out/'status.json',status,update=True)
            print(json.dumps(dict(run=out.name,**status)),flush=True)
        except BaseException:
            status.update(status='failed',error=traceback.format_exc(),elapsed_seconds=tick()-start)
            write(out/'status.json',status,update=True)
            raise
    # C inference begins only after all fifteen trajectories have completed.
    for seed,action,impl in identities():
        out=directory/'runs'/run_name(seed,action,impl)
        start=tick(); net=model(seed); scores={}
        for label in ('best','last'):
            saved=torch.load(out/(label+'.pt'),map_location='cuda',weights_only=False)
            net.load_state_dict(saved['model'])
            scores[label]=dict(step=saved['step'],**data.evaluate(net,'C'))
        scores['C_evaluation_seconds']=tick()-start
        write(out/'C_scores.json',scores)
    write(directory/'training_complete.json',dict(status='completed',full_trainings=15,
        total_train_and_C_wall_seconds=tick()-begin,execution_lock_sha256=sha(ROOT/'execution_lock.json')))


def audit(station):
    verify(station); begin=tick(); directory=ROOT/station
    assert read(directory/'training_complete.json')['full_trainings']==15
    data=Data(station,prepared=True); rows=[]; checks=0
    for seed,action,impl in identities():
        out=directory/'runs'/run_name(seed,action,impl)
        status=read(out/'status.json'); curve=read(out/'B_curve.json'); scores=read(out/'C_scores.json')
        assert status['status']=='completed' and status['steps_completed']==3000
        assert [v['step'] for v in curve]==list(GRID)
        best_step=min(curve,key=lambda v:v['NMSE'])['step']
        assert status['best_step']==best_step
        initial=torch.load(out/'initial.pt',map_location='cuda',weights_only=False)
        expected=model(seed).state_dict()
        assert initial['step']==0
        assert all(torch.equal(initial['model'][k],v) for k,v in expected.items())
        with np.load(out/'sampling_counts.npz') as saved: assert saved['counts'].sum()==3000*256
        net=model(seed)
        for label in ('best','last'):
            saved=torch.load(out/(label+'.pt'),map_location='cuda',weights_only=False)
            assert saved['step']==(best_step if label=='best' else 3000)
            net.load_state_dict(saved['model'])
            assert data.evaluate(net,'C')=={k:v for k,v in scores[label].items() if k!='step'}
            b=data.evaluate(net,'B')
            assert b=={k:v for k,v in curve[GRID.index(saved['step'])].items() if k!='step'}
            checks+=2
            rows.append(dict(seed=seed,action=action,implementation=impl,checkpoint=label,**scores[label]))
    write(directory/'audit.json',dict(status='PASS',full_trainings=15,checkpoint_split_recalculations=checks,
        audit_seconds=tick()-begin,full_trajectory_replays=0))
    write(directory/'result_rows.json',rows)
    summarize(station,rows)


def summarize(station,rows):
    results={}
    for label in ('best','last'):
        subset=[r for r in rows if r['checkpoint']==label]
        parents={r['seed']:r for r in subset if r['action']=='PARENT'}
        cells=[]
        for action in ACTIONS[1:]:
            for impl in ('S','W'):
                selected=[r for r in subset if r['action']==action and r['implementation']==impl]
                gain=[100*(parents[r['seed']]['NMSE']-r['NMSE'])/parents[r['seed']]['NMSE'] for r in selected]
                mae=[100*(parents[r['seed']]['NMAE']-r['NMAE'])/parents[r['seed']]['NMAE'] for r in selected]
                cells.append(dict(action=action,implementation=impl,gains_percent=gain,mean_gain_percent=float(np.mean(gain)),
                    NMAE_gains_percent=mae,mean_NMAE_gain_percent=float(np.mean(mae)),
                    adoption_1pct=bool(np.mean(gain)>=1.),
                    categories=['beneficial' if g>.25 else 'harmful' if g<-.25 else 'unchanged' for g in gain]))
        choices={}; oracles={}; fixed_regrets={}
        for impl in ('S','W'):
            pool=[dict(action='PARENT',mean_gain_percent=0.)]+[c for c in cells if c['implementation']==impl]
            winner=max(pool,key=lambda c:c['mean_gain_percent'])
            choices[impl]=winner['action']; oracles[impl]=winner['mean_gain_percent']
            fixed_regrets[impl]=oracles[impl]-next(c['mean_gain_percent'] for c in cells if c['implementation']==impl and c['action']=='FAVOR_LATE')
        pairs=[]
        for action in ACTIONS[1:]:
            s=next(c for c in cells if c['action']==action and c['implementation']=='S')
            w=next(c for c in cells if c['action']==action and c['implementation']=='W')
            delta=np.array(w['gains_percent'])-s['gains_percent']
            pairs.append(dict(action=action,W_minus_S_pp=delta.tolist(),mean_difference_pp=float(delta.mean()),
                same_direction_beyond_quarter_pp=bool(np.all(delta>.25) or np.all(delta<-.25)),
                adoption_changed=s['adoption_1pct']!=w['adoption_1pct']))
        joint=[dict(action='PARENT',implementation='P',mean_gain_percent=0.)]+cells
        winner=max(joint,key=lambda c:c['mean_gain_percent'])
        joint_winner_wfirst=max([joint[0]]+sorted(cells,key=lambda c:(ACTIONS.index(c['action']),c['implementation']!='W')),key=lambda c:c['mean_gain_percent'])
        results[label]=dict(cells=cells,paired_differences=pairs,separate_oracles_percent=oracles,
            choices=choices,choice_changed=choices['S']!=choices['W'],
            any_adoption_changed=any(p['adoption_changed'] for p in pairs),
            response_divergence_replicated=any(p['same_direction_beyond_quarter_pp'] for p in pairs),
            fixed_late_regret_pp=fixed_regrets,joint_oracle_percent=winner['mean_gain_percent'],
            joint_winner={k:winner[k] for k in ('action','implementation')},
            joint_winner_W_first={k:joint_winner_wfirst[k] for k in ('action','implementation')},
            shared_mean_prediction_MAE_lower_bound_pp=float(np.mean([abs(p['mean_difference_pp'])/2 for p in pairs])))
    write(ROOT/station/'summary.json',dict(station=station,new_full_trainings=15,results=results,
        interpretations='three-seed descriptive replication, not statistical significance; ETTh1/2 one dataset family'))
    print(json.dumps(read(ROOT/station/'summary.json')),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('preflight','train','audit'))
    parser.add_argument('station',choices=('ETTh1','ETTh2'))
    args=parser.parse_args(); configure()
    globals()[args.phase](args.station)


if __name__=='__main__':
    main()
