"""Build identical source-only factual packets for both window signal generators."""
from pathlib import Path
import json
import numpy as np
from diagnosis_pool_features import read,sha,write
from window_measurement_dsl import summary,evaluate,INPUT_COLUMNS,ERROR_COLUMNS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
R2=BASE/'20261018_predictive_support_r2'
R3=BASE/'20261018_parent_measurements_r3'
OUT=BASE/'20261018_measurement_generation_r4'

def main():
    destination=OUT/'inputs';destination.mkdir()
    records=read(BASE/'20261018_conditional_diagnosis_r1/development_pool.json')['records'];checks=[];packets=[]
    for family in ('ETT','Hydraulic'):
        source=[r for r in records if r['family']==family];tasks=sorted({r['task'] for r in source});anonymous={t:f'T{i+1}' for i,t in enumerate(tasks)}
        summaries={};assets={}
        metadata=read(R3/('ett_measurements.json' if family=='ETT' else 'hydraulic_measurements.json'))
        for task in tasks:
            path=R3/'tables'/f'{family}_{task}.npz'
            assert sha(path)==metadata['assets'][path.name]['sha256']
            with np.load(path,allow_pickle=False) as z:table=dict(z)
            summaries[anonymous[task]]=summary(table);assets[anonymous[task]]=sha(path)
            # Identity and axis checks use real parent windows, never child responses.
            features=next(r['features'] for r in source if r['task']==task)
            value=evaluate('mean(E_loss)',table,features);assert abs(value-1)<1e-12
            assert abs(evaluate('mean(E_contrast)',table,features)-(np.mean([table[k] for k in table if k.startswith('B_errors_s')],axis=0)[:,1:3]@np.array([1.,-1.])).mean())<1e-12
            for invalid in ('mean(A_level * B_level)','__import__("os")','mean(B_level).__class__','B_level','mean(B_level ** 2)'):
                try:evaluate(invalid,table,features)
                except ValueError:continue
                raise AssertionError('Invalid expression accepted')
            checks.append(dict(family=family,task=task,normalization=True,contrast=True,invalid_expressions_rejected=5))
        rows=read(R2/f'{family}_conditional_source_packet.json')['facts']
        facts=[{k:v for k,v in row.items() if k!='frozen_mean_functions'} for row in rows]
        packet=dict(input_window_columns=list(INPUT_COLUMNS),error_window_columns=[*ERROR_COLUMNS,'contrast'],parent_measurements=summaries,response_facts=facts,held_target_records_included=False)
        path=destination/f'{family}_facts.json';write(path,packet)
        packets.append(dict(family=family,sha256=sha(path),bytes=path.stat().st_size,source_packet_sha256=sha(R2/f'{family}_conditional_source_packet.json'),parent_asset_sha256=assets))
    write(OUT/'input_preparation.json',dict(status='PASS',packets=packets,checks=checks,new_LLM_calls=0,new_full_trainings=0,interpreter_sha256=sha(ROOT/'scripts/window_measurement_dsl.py'),preparer_sha256=sha(__file__),evaluation_not_yet_frozen=True))
    print(json.dumps(dict(status='PASS',source_packets=len(packets),tasks=len(checks),invalid_expressions_rejected=25,LLM_calls=0)))

if __name__=='__main__':main()
