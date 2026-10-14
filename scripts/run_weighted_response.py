"""Fixed-protocol uniform sampling with deterministic per-window loss weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
import run_research as core

ROOT=Path('/root/auto_data_experiments/20261014_weighted_response_r1')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v,update=False):
    if update:
        temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
    else:
        with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def tick():torch.cuda.synchronize();return time.perf_counter()
def configure():
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.set_num_threads(4)
def verify(out):
    assert out==ROOT and out.resolve()==ROOT
    plan=read(out/'plan.json');lock=read(out/'prediction_lock.json');source=read(out/'source_sync_receipt.json')
    assert source['status']=='exact_commit_and_tree_verified'
    assert source['plan_sha256']==lock['plan_sha256']==sha(out/'plan.json')
    assert source['predictions_sha256']==lock['predictions_sha256']==sha(out/'predictions.json')
    for name,digest in plan['code_sha256'].items():assert sha(out/name)==digest,name
    for name,digest in plan['input_sha256'].items():assert sha(name)==digest,name
    return plan
def pack(model):
    return torch.stack([torch.cat((model.seasonal[c].weight.T,model.seasonal[c].bias[None],model.trend[c].weight.T,model.trend[c].bias[None])) for c in range(8)])
def snapshot(model,opt,rng,step,path):
    assert not path.exists()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,rng_numpy=rng.bit_generator.state,
        rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),original_coordinates=pack(model).detach().cpu()),path)
def arrays(path,**values):
    with path.open('xb') as f:np.savez_compressed(f,**values)

class Context:
    def __init__(self,plan,task):
        self.plan=plan;self.task=task;self.values={};self.starts={}
        with np.load(task['abc'],allow_pickle=False) as z:
            for split in ['A','B','C']:
                self.values[split]=torch.tensor(z[split+'_values'],device='cuda');self.starts[split]=torch.tensor(z[split+'_starts'],device='cuda')
        self.n=len(self.starts['A']);assert self.n==6468
        self.cdf=np.cumsum(np.full(self.n,1/self.n));self.cdf[-1]=1.
        self.offset=torch.arange(40,device='cuda');self.weights={}
        for action,path in task['probabilities'].items():
            with np.load(path,allow_pickle=False) as z:w=self.n*z['probability']
            assert np.isfinite(w).all() and (w>=0).all() and abs(w.mean()-1)<1e-12
            assert np.array_equal(w,w.astype(np.float32).astype(np.float64))
            self.weights[action]=torch.tensor(w,dtype=torch.float32,device='cuda')
    def batch(self,split,ids):return self.values[split][self.starts[split][ids,None]+self.offset]
    @torch.no_grad()
    def evaluate(self,model,split):
        assert split in ('B','C');model.eval();chunks=[]
        for lo in range(0,len(self.starts[split]),2048):
            batch=self.batch(split,torch.arange(lo,min(lo+2048,len(self.starts[split])),device='cuda'))
            chunks.append((model(batch[:,:32])-batch[:,32:]).cpu().numpy())
        error=np.concatenate(chunks);assert np.isfinite(error).all()
        return float(np.mean(error.astype(np.float64)**2)),error

def train(ctx,job,out):
    assert shutil.disk_usage(out).free>=ctx.plan['runtime_min_free_bytes']
    directory=out/'runs'/job['id'];directory.mkdir(parents=True);started=tick()
    state=dict(status='running',**job,steps_completed=0);write(directory/'status.json',state)
    try:
        seed=job['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=core.DLinear(dict(ctx.plan['training'],channels=8)).cuda()
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        rng=np.random.default_rng(seed+1000)
        parent=torch.load(Path(job['parent'])/'initial.pt',weights_only=True,map_location='cpu')
        assert torch.equal(pack(model).detach().cpu(),parent['original_coordinates'])
        assert not parent['optimizer']['state'];draw_count=np.zeros(ctx.n,dtype=np.int64)
        snapshot(model,opt,rng,0,directory/'initial.pt');scores={}
        for step in range(1,3001):
            model.train();opt.zero_grad(set_to_none=True)
            ids=np.searchsorted(ctx.cdf,rng.random(256),side='right');np.add.at(draw_count,ids,1)
            index=torch.tensor(ids,device='cuda');batch=ctx.batch('A',index)
            loss=((model(batch[:,:32])-batch[:,32:]).square()*ctx.weights[job['action']][index,None,None]).mean()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite weighted loss')
            loss.backward();opt.step();state['steps_completed']=step
            if step in ctx.plan['witness_steps']:snapshot(model,opt,rng,step,directory/f'witness_{step}.pt')
            if step in ctx.plan['steps']:
                snapshot(model,opt,rng,step,directory/f'step_{step}.pt');scores[step]=ctx.evaluate(model,'B')[0]
                write(directory/'status.json',state,update=True)
        selected=min(ctx.plan['steps'],key=lambda k:scores[k]);model.load_state_dict(torch.load(directory/f'step_{selected}.pt',weights_only=True)['model'])
        losses={}
        for split in ['B','C']:
            losses[split],error=ctx.evaluate(model,split);arrays(directory/f'selected_{split}_errors.npz',error=error)
        assert losses['B']==scores[selected];arrays(directory/'sampling.npz',draw_count=draw_count)
        assets={p.name:sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.pt','.npz')}
        result=dict(state,status='completed',steps=3000,checkpoints=ctx.plan['steps'],witness_steps=ctx.plan['witness_steps'],
            selection_losses=scores,selected_step=selected,selected_losses=losses,assets_sha256=assets,
            plan_sha256=sha(out/'plan.json'),predictions_sha256=sha(out/'predictions.json'))
        write(directory/'result.json',result);state['status']='completed';write(directory/'status.json',state,update=True)
        seconds=tick()-started;write(directory/'timing.json',dict(seconds=seconds,boundary='initialization through scores/status/assets/hash; timing receipt excluded'))
        return dict(result,seconds=seconds)
    except Exception:
        state.update(status='failed',error=traceback.format_exc(),seconds=time.perf_counter()-started);write(directory/'status.json',state,update=True);raise

def main(out):
    plan=verify(out);assert read(out/'preflight.json')['status']=='PASS'
    write(out/'labels.lock',dict(plan_sha256=sha(out/'plan.json'),predictions_sha256=sha(out/'predictions.json')))
    results=[];started=tick()
    try:
        for task in plan['tasks']:
            ctx=Context(plan,task)
            for job in (j for j in plan['jobs'] if j['task']==task['id']):
                results.append(train(ctx,job,out))
                write(out/'status.json',dict(status='running',completed=len(results),registered=36,last_id=job['id'],cumulative_completed=373+len(results)),update=True)
                print(json.dumps(dict(completed=len(results),id=job['id'],seconds=results[-1]['seconds'])),flush=True)
            del ctx
        verify(out);assert len(results)==36
        write(out/'labels.json',dict(status='completed',results=results,phase_seconds=tick()-started,new_full_trainings=36,
            cumulative_full_trainings=409,D_access=False,test_access=False,plan_sha256=sha(out/'plan.json'),predictions_sha256=sha(out/'predictions.json')))
        write(out/'status.json',dict(status='completed',completed=36,registered=36,cumulative_completed=409),update=True)
    except Exception:
        write(out/'status.json',dict(status='failed',completed=len(results),registered=36,cumulative_completed=373+len(results),error=traceback.format_exc()),update=True);raise
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();configure();main(args.output)
