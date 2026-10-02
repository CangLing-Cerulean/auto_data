"""Training-only state coverage and actual exposure; no test/validation reads."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    folder=ROOT/'research_runs/selection'
    output=folder/'state_support.json'
    if output.exists():
        raise RuntimeError('Completed diagnosis cannot be overwritten')
    source=ROOT/'data/metropt3_v1/train.csv'
    manifest=json.loads((source.parent/'manifest.json').read_text())
    assert digest(source)==next(s['sha256'] for s in manifest['splits'] if s['name']=='train')
    values=np.loadtxt(source,delimiter=',',skiprows=1,usecols=range(2,17))
    with source.open(newline='') as handle:
        rows=csv.reader(handle); next(rows)
        months=np.array([row[1][:7] for row in rows])
    with np.load(folder/'window_table.npz') as table:
        starts=table['start_row_zero_based']; scores=table['input_variation']
    month=months[starts+31]
    masks={}; state_info={}
    for name,column,state in [('LPS_1',11,1),('Oil_level_0',13,0),('COMP_0',7,0)]:
        active=values[:,column]==state
        prefix=np.r_[0,np.cumsum(active)]
        masks[name+'_any_input']=(prefix[starts+32]-prefix[starts])>0
        masks[name+'_last_input']=active[starts+31]
        event_starts=np.flatnonzero(active & ~np.r_[False,active[:-1]])
        event_ends=np.flatnonzero(active & ~np.r_[active[1:],False])+1
        state_info[name]={'active_rows':int(active.sum()),'contiguous_row_episodes':len(event_starts),
            'episodes':[[int(s),int(e)] for s,e in zip(event_starts,event_ends)],
            'unique_values':[float(v) for v in np.unique(values[:,column])]}
    support={name:{'windows':int(mask.sum()),'quiet_windows':int(np.sum(mask&(scores<=np.quantile(scores,.25)))),
                  'by_month':{m:int(np.sum(mask&(month==m))) for m in np.unique(month)}} for name,mask in masks.items()}
    runs={}
    for run in sorted((ROOT/'research_runs/results').glob('S*')):
        if not (run/'result.json').exists(): continue
        identity=json.loads((run/'identity.json').read_text())
        with np.load(run/'sampling_weights.npz') as policy:
            probability=policy['probability']; draws=policy['draw_count']; loss=policy['loss_weight']
            groups={name:{'retained':int(np.sum(mask&(probability>0))),
                'expected_draws':float(probability[mask].sum()*draws.sum()),'actual_draws':int(draws[mask].sum()),
                'distinct_seen':int(np.sum(mask&(draws>0))),
                'expected_loss_mass':float(np.dot(probability[mask],loss[mask])/np.dot(probability,loss))}
                for name,mask in masks.items()}
        runs[run.name]={'recipe':identity['recipe'],'seed':identity['seed'],'groups':groups}
    result={'dataset_id':manifest['dataset_id'],'train_sha256':digest(source),'window_table_sha256':digest(folder/'window_table.npz'),
        'code_sha256':digest(Path(__file__)),'scope':'training only; input masks; contiguous episodes count source rows, not independent physical events',
        'row_states':state_info,'support':support,'runs':runs}
    output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'support':support,'episode_counts':{k:v['contiguous_row_episodes'] for k,v in state_info.items()},
        'LPS_exposure':{k:v['groups']['LPS_1_any_input'] for k,v in runs.items()}},indent=2))


if __name__=='__main__':
    main()
