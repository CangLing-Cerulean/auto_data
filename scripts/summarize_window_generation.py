"""Summarize every frozen method and call without selecting a favorable subset."""
from pathlib import Path
import json
import numpy as np
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_measurement_generation_r4'

def main():
    summary=read(STAGE/'replay/summary.json');runs=[];calls=[]
    for entry in summary['runs']:
        path=STAGE/'replay'/f"{entry['family']}_{entry['task']}_{entry['method']}.json";assert sha(path)==entry['sha256']
        runs.append(read(path))
    for name in ('ETT_ordinary','ETT_mismatch','Hydraulic_ordinary','Hydraulic_mismatch'):
        folder=STAGE/'generation'/name;receipt=read(folder/'receipt.json');response=read(folder/'response.json')
        calls.append(dict(name=name,**receipt,signals_detail=response['signals'],reasoning=response['reasoning']))
    first=[];families={}
    for source in ('ETT','Hydraulic'):
        records=read(STAGE/'replay'/f'{source}_augmented_records.json')['records']
        families[source]={r['id']:r for r in records}
    for run in runs:
        oracle=families[run['source_family']];predictions=run['trace'][0]['predictions_before']
        errors=[abs(pred-oracle[name]['response']['Y']) for name,pred in predictions.items()]
        first.append(dict(family=run['family'],task=run['task'],method=run['method'],mean_absolute_error=float(np.mean(errors)),candidate_count=len(errors)))
    aggregate=[]
    for method in sorted({r['method'] for r in runs}):
        subset=[r for r in first if r['method']==method]
        aggregate.append(dict(method=method,task_equal_initial_Y_MAE_pp=100*float(np.mean([r['mean_absolute_error'] for r in subset]))))
    costs=[node['query_seconds'] for run in runs for node in run['nodes']]
    data=dict(status='SUMMARIZED_NOT_INDEPENDENT_AUDIT',calls=calls,decisions=summary['summaries'],initial_predictions=first,initial_task_equal_errors=aggregate,query_nodes=len(costs),known_cost_nodes=sum(c is not None for c in costs),policy_seconds=sum(r['seconds'] for r in runs),measurement_seconds=sum(r['seconds'] for t in summary['measurement_timings'] for r in t['records']),LLM_seconds_sum=sum(c['seconds'] for c in calls),LLM_seconds_longest=max(c['seconds'] for c in calls),tokens_input=sum(c['usage']['input_tokens'] for c in calls),tokens_output=sum(c['usage']['output_tokens'] for c in calls),new_full_trainings=0,cumulative_full_trainings=464)
    write(STAGE/'analysis.json',data)
    print(json.dumps({k:v for k,v in data.items() if k not in ('calls','initial_predictions','decisions')},indent=2))

if __name__=='__main__':main()
