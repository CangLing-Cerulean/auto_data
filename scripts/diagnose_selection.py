"""Diagnose actual train/validation windows before deciding a sampling policy."""
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from run_research import ROOT, DLinear, prepare, evaluate, write_json


def input_variation(values, starts, length):
    change = np.abs(np.diff(values.astype(np.float64), axis=0)).mean(axis=1)
    prefix = np.r_[0, np.cumsum(change)]
    return (prefix[starts+length-1]-prefix[starts])/(length-1)


def main():
    begun=time.monotonic()
    out=ROOT/'research_runs/selection'
    out.mkdir(exist_ok=True)
    p=json.loads((ROOT/'research_runs/protocol.json').read_text())
    manifest=json.loads((ROOT/'data/metropt3_v1/manifest.json').read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    arrays,starts,_,audit=prepare(p,manifest)
    variation={s:input_variation(a,starts[s],p['input_length']) for s,a in arrays.items()}
    quantiles={str(q):float(np.quantile(variation['train'],q)) for q in (.1,.25,.5,.75,.9,.95,.99)}
    model=DLinear(p).cuda()
    model.load_state_dict(torch.load(ROOT/'research_runs/results/E000_baseline_s2026/best.pt',weights_only=True))
    result={'quantiles_fitted_on_train':quantiles,'source_run':'E000_baseline_s2026','train_errors_are_in_sample':True,'splits':{}}
    for split,values in arrays.items():
        score=variation[split]
        groups={'all':np.ones(len(score),dtype=bool),'low_quartile':score<=quantiles['0.25'],
                'middle':(score>quantiles['0.25'])&(score<=quantiles['0.9']),
                'high_decile':score>quantiles['0.9']}
        with (ROOT/'data/metropt3_v1'/f'{split}.csv').open(newline='') as f:
            rows=csv.reader(f)
            next(rows)
            times=np.array([r[1] for r in rows],dtype='datetime64[s]')
        months=times[starts[split]+p['input_length']].astype('datetime64[M]').astype(str)
        for month in np.unique(months):
            groups['month_'+month]=months==month
        metrics=evaluate(model,torch.from_numpy(values).cuda(),torch.from_numpy(starts[split]).cuda(),groups,p,'baseline')
        result['splits'][split]=metrics['groups']
    result['elapsed_seconds']=time.monotonic()-begun
    write_json(out/'diagnosis.json',result)
    # Include diagnostic GPU runtime in the same cumulative budget ledger.
    budget=ROOT/'research_runs/results/A001_selection_diagnosis'
    budget.mkdir(exist_ok=True)
    write_json(budget/'status.json',{'phase':'completed','step':0,'elapsed_seconds':result['elapsed_seconds'],'kind':'diagnostic'})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    main()
