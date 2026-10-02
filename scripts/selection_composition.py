"""Describe selected source windows by time and observed input state (train only)."""
import csv
import json
import numpy as np
from run_research import ROOT,write_json


def main():
    folder=ROOT/'research_runs/selection'
    with np.load(folder/'window_table.npz') as table:
        starts=table['start_row_zero_based']
        score=table['input_variation']
    raw=ROOT/'data/metropt3_v1/train.csv'
    values=np.loadtxt(raw,delimiter=',',skiprows=1,usecols=range(2,17))
    with raw.open(newline='') as f:
        rows=csv.reader(f)
        next(rows)
        months=np.array([row[1][:7] for row in rows])
    months=months[starts+31]
    groups={'low_quartile':score<=np.quantile(score,.25),
            'high_decile':score>np.quantile(score,.9)}
    groups['middle']=~groups['low_quartile']&~groups['high_decile']
    last=values[starts+31]
    result={}
    for name,mask in groups.items():
        result[name]={'windows':int(mask.sum()),'by_input_end_month':{m:int(np.sum(mask&(months==m))) for m in np.unique(months)},
                      'last_input_state_prevalence':{'COMP_0':float((last[mask,7]==0).mean()),
                                                     'LPS_1':float((last[mask,11]==1).mean()),
                                                     'Oil_level_0':float((last[mask,13]==0).mean())}}
    write_json(folder/'training_composition.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
