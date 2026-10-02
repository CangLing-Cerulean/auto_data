"""Export exact source references changed by the state-protection experiments."""
import csv
import gzip
import json
import numpy as np
from run_research import ROOT
from run_selection import training_lps_mask


def main():
    folder=ROOT/'research_runs/selection'
    with np.load(folder/'window_table.npz') as table:
        starts=table['start_row_zero_based']
    rare=training_lps_mask(starts,32)
    for run in sorted((ROOT/'research_runs/results').glob('S*')):
        if not (run/'result.json').exists(): continue
        identity=json.loads((run/'identity.json').read_text())
        if identity['recipe'] not in ('protect_lps_matched','placebo_matched_swap'): continue
        reference=list(run.parent.glob(f"S*_month_matched_drop_s{identity['seed']}"))
        assert len(reference)==1
        with np.load(reference[0]/'sampling_weights.npz') as base, np.load(run/'sampling_weights.npz') as policy:
            before=base['sampling_weight']; after=policy['sampling_weight']
            probability=policy['probability']; draws=policy['draw_count']
            changed=np.flatnonzero(before!=after)
            assert len(changed)==844
            path=folder/f'{run.name}_swaps.csv.gz'
            with gzip.open(path,'wt',encoding='utf-8',newline='') as handle:
                writer=csv.writer(handle)
                writer.writerow(['eligible_window_index','action_relative_to_month_control','source_start_row_zero_based',
                    'input_end_row_exclusive','target_end_row_exclusive','input_has_LPS_1','probability','actual_draw_count'])
                for i in changed:
                    s=int(starts[i])
                    writer.writerow([int(i),'restored' if after[i]>0 else 'replacement_excluded',s,s+32,s+40,
                        bool(rare[i]),float(probability[i]),int(draws[i])])
            print(path.name,len(changed))


if __name__=='__main__':
    main()
