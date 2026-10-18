"""Frozen parent-only directional gradient tables; no fitting or child lookup."""
import argparse
import csv
import itertools
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from diagnosis_pool_features import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT = BASE/'20261018_signal_admission_r9'
DEV = BASE/'20261013_response_proxy_development_r1/archives/verified'
INFO = BASE/'20261014_sampling_weight_information_r1/archives/unpacked'
ETT = ROOT/'research_runs/ett_confirmation'
R3 = BASE/'20261018_parent_measurements_r3'
VIEWS = ('global', 'target', 'complement')
BATCH = 128


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def prepare():
    """Read metadata and hash assets only; do not load arrays."""
    inputs = {}
    def register(p, expected=None):
        actual = sha(p)
        if expected is not None:
            assert actual == expected, str(p)
        inputs[rel(p)] = actual
    tasks = []
    plan = read(BASE/'20261014_weighted_response_r1/plan.json')
    register(BASE/'20261014_weighted_response_r1/plan.json')
    register(R3/'ett_measurements.json')
    for task in ('G1', 'G2', 'G3'):
        path = DEV/'prepared'/f'{task}_ABC.npz'
        expected = {v for k, v in plan['input_sha256'].items() if k.endswith(f'/{task}_ABC.npz')}
        assert len(expected) == 1
        register(path, expected.pop())
        diagnostic = INFO/f'{task}_diagnostic.json'
        register(diagnostic)
        register(INFO/'statistics'/f'{task}.npz', read(diagnostic)['statistics_sha256'])
        parents = []
        for seed in (2026101701, 2026101702, 2026101703):
            result_path = DEV/'runs'/f'DEV_{task}_full_PARENT_s{seed}'/'result.json'
            register(result_path)
            result = read(result_path)
            parents.append(dict(seed=seed, step=result['selected_step'], reference_B=result['selected_losses']['B']))
        tasks.append(dict(task=task, family='Hydraulic', length=32, horizon=8, parents=parents))
    checks = read(R3/'ett_measurements.json')['checks']
    for task in ('ETTh1', 'ETTh2'):
        prep = ETT/task/'preprocessing.json'
        register(prep)
        register(ROOT/'data/ett_v1/raw'/f'{task}.csv', read(prep)['raw_sha256'])
        parents = []
        for seed in (2026101901, 2026101902, 2026101903):
            run = ETT/task/'runs'/f'PARENT_P_s{seed}'
            register(run/'status.json'); register(run/'B_curve.json')
            status = read(run/'status.json')
            selected = next(x for x in read(run/'B_curve.json') if x['step'] == status['best_step'])
            expected = next(x['parent_checkpoint_sha256'] for x in checks if x['station'] == task and x['seed'] == seed)
            register(run/'best.pt', expected)
            parents.append(dict(seed=seed, step=status['best_step'], reference_B=selected['NMSE']))
        tasks.append(dict(task=task, family='ETT', length=96, horizon=96, parents=parents))
    write(OUT/'execution_lock.json', dict(status='prepared', tasks=tasks, inputs=inputs,
        scripts={rel(Path(__file__)):sha(__file__), 'scripts/diagnosis_pool_features.py':sha(ROOT/'scripts/diagnosis_pool_features.py')},
        protocol_sha256=sha(OUT/'PROTOCOL.md'), full_trainings=0, LLM_calls=0))
    print(dict(status='prepared', tasks=len(tasks), inputs=len(inputs)), flush=True)


def load_ab(task):
    name = task['task']
    if task['family'] == 'Hydraulic':
        with np.load(DEV/'prepared'/f'{name}_ABC.npz', allow_pickle=False) as z:
            a, b, ai, bi = z['A_values'], z['B_values'], z['A_starts'], z['B_starts']
            cycles = z['A_cycles'][ai]
        first = cycles < (int(cycles.min()) + int(cycles.max()) + 1)/2
    else:
        with (ROOT/'data/ett_v1/raw'/f'{name}.csv').open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream); header = next(reader)
            assert header == ['date','HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
            rows = list(itertools.islice(reader, 10080))
        raw = np.array([[float(x) for x in row[1:]] for row in rows], dtype=np.float64)
        assert raw.shape == (10080, 7) and np.isfinite(raw).all()
        dates = np.array([row[0] for row in rows], dtype='datetime64[s]')
        assert np.all(np.diff(dates) == np.timedelta64(3600,'s'))
        mean, std = raw[:8640].mean(0), raw[:8640].std(0)
        scale = np.where(std == 0, 1., std)
        saved = read(ETT/name/'preprocessing.json')
        np.testing.assert_array_equal(mean, saved['mean']); np.testing.assert_array_equal(scale, saved['scale'])
        a = ((raw[:8640]-mean)/scale).astype(np.float32)
        b = ((raw[8640:]-mean)/scale).astype(np.float32)
        ai, bi = np.arange(8449), np.arange(1249)
        first = ai+96 < 4320
    assert min(first.sum(), (~first).sum()) > 0
    return a, b, ai, bi, first


def theta_for(task, parent):
    import torch
    name = task['task']
    if task['family'] == 'Hydraulic':
        anchors = read(INFO/f'{name}_diagnostic.json')['anchors']
        index = anchors.index(dict(seed=parent['seed'], step=parent['step']))
        with np.load(INFO/'statistics'/f'{name}.npz', allow_pickle=False) as z:
            theta = z['theta'][index].copy()
    else:
        saved = torch.load(ETT/name/'runs'/f"PARENT_P_s{parent['seed']}"/'best.pt', map_location='cpu', weights_only=True)
        assert saved['step'] == parent['step']
        state = saved['model']
        theta = np.stack([np.concatenate([state[f'seasonal.{c}.weight'].numpy().T,
            state[f'seasonal.{c}.bias'].numpy()[None], state[f'trend.{c}.weight'].numpy().T,
            state[f'trend.{c}.bias'].numpy()[None]]) for c in range(7)])
    result = np.asarray(theta, dtype=np.float64)
    assert result.shape[1:] == (2*task['length']+2, task['horizon']) and np.isfinite(result).all()
    return result


def blocks(values, starts, length, horizon):
    import torch
    import torch.nn.functional as F
    offset = np.arange(length+horizon)
    for lo in range(0, len(starts), BATCH):
        window = values[starts[lo:lo+BATCH, None]+offset].astype(np.float64)
        x = torch.from_numpy(window[:,:length].transpose(0,2,1).copy())
        trend = F.avg_pool1d(F.pad(x, (12,12), mode='replicate'), 25, stride=1).numpy()
        x = x.numpy(); one = np.ones((*x.shape[:2],1))
        yield np.concatenate([x-trend,one,trend,one],axis=2), window[:,length:].transpose(0,2,1)


def b_directions(values, starts, theta, length, horizon):
    channels = theta.shape[0]; grad = np.zeros((3,*theta.shape)); losses = np.zeros(3)
    halves = (slice(None),slice(horizon//2,None),slice(None,horizon//2))
    for phi, y in blocks(values, starts, length, horizon):
        error = np.einsum('bcf,cfh->bch', phi, theta, optimize=True)-y
        for v, sl in enumerate(halves):
            e = error[:,:,sl]; width = e.shape[-1]
            grad[v,:,:,sl] += 2/(channels*width)*np.einsum('bcf,bch->cfh',phi,e,optimize=True)
            losses[v] += np.sum(e*e)/(channels*width)
    grad /= len(starts); losses /= len(starts)
    np.testing.assert_allclose(grad[0], .5*(grad[1]+grad[2]), rtol=1e-11,atol=1e-13)
    norms = np.linalg.norm(grad.reshape(3,-1),axis=1)
    direction = np.divide(grad,norms[:,None,None,None],out=np.zeros_like(grad),where=norms[:,None,None,None]>0)
    return grad, direction, norms, losses


def projections(values, starts, theta, direction, length, horizon, first):
    proj, norms = [], []; factor = 2/(theta.shape[0]*horizon)
    group_gradient=np.zeros((2,*theta.shape)); cursor=0
    for phi, y in blocks(values, starts, length, horizon):
        error = np.einsum('bcf,cfh->bch',phi,theta,optimize=True)-y
        norms.append(factor**2*np.sum(np.sum(phi*phi,axis=2)*np.sum(error*error,axis=2),axis=1))
        proj.append(np.column_stack([factor*np.sum(error*np.einsum('bcf,cfh->bch',phi,u,optimize=True),axis=(1,2)) for u in direction]))
        flag=first[cursor:cursor+len(phi)];cursor+=len(phi)
        for group,mask in enumerate((flag,~flag)):
            group_gradient[group]+=factor*np.einsum('bcf,bch->cfh',phi[mask],error[mask],optimize=True)
    raw_norm = np.concatenate(norms); scale = float(np.sqrt(raw_norm.mean()))
    raw_proj = np.concatenate(proj)
    group_gradient/=np.array([first.sum(),(~first).sum()])[:,None,None,None]
    group_gradient=group_gradient/scale if scale else np.zeros_like(group_gradient)
    return (raw_proj/scale if scale else np.zeros_like(raw_proj), raw_norm/(scale*scale) if scale else np.zeros_like(raw_norm), scale, group_gradient)


def action_masses(task, first):
    u = np.array([first.mean(), (~first).mean()])
    result = {'PARENT':u}
    if task['family']=='Hydraulic':
        result.update(KEEP_FIRST=np.array([1.,0.]), KEEP_SECOND=np.array([0.,1.]),
                      FAVOR_FIRST=np.array([.75,.25]), FAVOR_SECOND=np.array([.25,.75]))
    else:
        for name, weights in [('FAVOR_EARLY',[3.,1.]),('FAVOR_LATE',[1.,3.])]:
            mass=u*np.array(weights); result[name]=mass/mass.sum()
    return result


def summarize(task, first, proj, norms, group_gradient):
    u=np.array([first.mean(),(~first).mean()])
    masks=(first,~first)
    means=np.stack([proj[m].mean(0) for m in masks])
    second=np.stack([proj[m].T@proj[m]/m.sum() for m in masks])
    norm_means=np.array([norms[m].mean() for m in masks])
    rows=[]
    for action,p in action_masses(task,first).items():
        mean=p@means
        mean_gradient=np.einsum('g,gcfh->cfh',p,group_gradient)
        mean_gradient_norm_square=float(np.sum(mean_gradient**2))
        for impl,weights in [('S',p),('W',p*p/u)]:
            moment=np.einsum('g,gvw->vw',weights,second)
            covariance=moment-np.outer(mean,mean)
            bound=1e-10*(1+np.max(abs(moment)))
            assert np.linalg.eigvalsh(covariance).min()>=-bound
            full_trace=float(weights@norm_means)-mean_gradient_norm_square
            assert full_trace>=-1e-10*(1+float(weights@norm_means))
            rows.append(dict(action=action,implementation=impl,group_masses=p.tolist(),mean=mean.tolist(),
                second=moment.tolist(),covariance=covariance.tolist(),batch256_covariance=(covariance/256).tolist(),
                normalized_gradient_second_moment=float(weights@norm_means),
                normalized_mean_gradient_norm_square=mean_gradient_norm_square,
                normalized_gradient_covariance_trace=full_trace,normalized_batch256_trace=full_trace/256))
    assert np.allclose(rows[0]['covariance'],rows[1]['covariance'],rtol=1e-12,atol=1e-12)
    return dict(group_counts=[int(m.sum()) for m in masks],group_mean=means.tolist(),group_second=second.tolist(),
        group_norm_square_mean=norm_means.tolist(),actions=rows)


def run():
    import torch
    torch.set_num_threads(1)
    lock=read(OUT/'execution_lock.json')
    assert read(OUT/'source_publication_receipt.json')['status']=='exact_commit_and_tree_verified'
    assert sha(OUT/'PROTOCOL.md')==lock['protocol_sha256']
    for name,digest in {**lock['inputs'],**lock['scripts']}.items():assert sha(ROOT/name)==digest,name
    start=time.perf_counter(); directory=OUT/'tables'; directory.mkdir(); records=[]
    for task in lock['tasks']:
        a,b,ai,bi,first=load_ab(task)
        for parent in task['parents']:
            tick=time.perf_counter(); theta=theta_for(task,parent)
            grad,direction,bnorm,loss=b_directions(b,bi,theta,task['length'],task['horizon'])
            np.testing.assert_allclose(loss[0],parent['reference_B'],rtol=1e-5,atol=1e-8)
            proj,norms,scale,group_gradient=projections(a,ai,theta,direction,task['length'],task['horizon'],first)
            assert np.isfinite(proj).all() and np.isfinite(norms).all()
            stats=summarize(task,first,proj,norms,group_gradient)
            path=directory/f"{task['task']}_s{parent['seed']}.npz"
            with path.open('xb') as stream:
                np.savez_compressed(stream,theta=theta,projection=proj,normalized_gradient_norm_square=norms,
                    B_gradient=grad,B_direction=direction,normalized_group_gradient=group_gradient,A_starts=ai,B_starts=bi,A_first=first)
            result=dict(task=task['task'],family=task['family'],**parent,views=list(VIEWS),A_windows=len(ai),B_windows=len(bi),
                B_losses=loss.tolist(),B_gradient_norms=bnorm.tolist(),A_gradient_rms=scale,
                direction_available=(bnorm>0).tolist(),A_scale_available=scale>0,**stats,
                asset=rel(path),sha256=sha(path),bytes=path.stat().st_size,seconds=time.perf_counter()-tick)
            write(OUT/f"{task['task']}_s{parent['seed']}.json",result); records.append(result)
            print(dict(task=task['task'],seed=parent['seed'],status='complete',seconds=result['seconds']),flush=True)
    for name,digest in lock['inputs'].items():assert sha(ROOT/name)==digest,name
    write(OUT/'measurements.json',dict(status='completed',records=records,seconds=time.perf_counter()-start,
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,platform=platform.platform(),threads=1),
        new_full_trainings=0,cumulative_full_trainings=464,LLM_calls=0,C_D_test_values_parsed=False,server_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    try:globals()[args.action]()
    except Exception:
        if args.action=='run':write(OUT/'measurement_failure.json',dict(status='failed',error=traceback.format_exc()))
        raise
