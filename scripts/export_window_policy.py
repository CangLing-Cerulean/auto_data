"""Export exact affected window references; no sensor values are generated."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from run_research import ROOT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id:
        raise ValueError('Invalid run ID')
    run=ROOT/'research_runs/results'/args.run_id
    with np.load(ROOT/'research_runs/selection/window_table.npz') as table, np.load(run/'sampling_weights.npz') as policy:
        starts=table['start_row_zero_based']
        score=table['input_variation']
        sampling=policy['sampling_weight']
        loss=policy['loss_weight']
        prob=policy['probability']
        draws=policy['draw_count']
        affected=np.flatnonzero((sampling!=1)|(loss!=1))
        path=ROOT/'research_runs/selection'/f'{args.run_id}_affected_windows.csv.gz'
        with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['eligible_window_index','action','source_start_row_zero_based','input_end_row_exclusive','target_end_row_exclusive','input_variation','sampling_weight','loss_weight','probability','actual_draw_count'])
            for i in affected:
                s=int(starts[i])
                action='excluded' if sampling[i]==0 else 'downweighted_loss' if loss[i]<1 else 'higher_sampling_weight'
                writer.writerow([int(i),action,s,s+32,s+40,float(score[i]),float(sampling[i]),float(loss[i]),float(prob[i]),int(draws[i])])
    print(json.dumps({'file':str(path),'affected_windows':len(affected)}))


if __name__=='__main__':
    main()
