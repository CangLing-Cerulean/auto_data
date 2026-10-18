"""Descriptive costs and paired comparisons; does not select or rerun any policy."""
from pathlib import Path
import numpy as np
from diagnosis_pool_features import read,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
STAGE=BASE/'20261018_strong_agent_r5'
PREVIOUS=BASE/'20261018_measurement_generation_r4'

def main():
    results=read(STAGE/'results.json');previous=read(PREVIOUS/'analysis.json')
    receipts=results['calls'];comparison=[]
    for row in results['summaries']:
        q=row['queries']
        comparison.append(dict(queries=q,method='strong_agent',mean_gain=row['agent_mean_gain']))
        comparison.append(dict(queries=q,method='uniform_random_exact_expectation',mean_gain=row['uniform_random_expected_mean_gain']))
        comparison.extend(dict(queries=q,method=r['method'],mean_gain=r['mean_gain']) for r in previous['decisions'] if r['queries']==q)
    prep=[];forecast_errors=[];latencies=[]
    for call in receipts:
        folder=STAGE/'calls'/f"{call['family']}_{call['task']}_{call['step']}"
        prep.append(read(folder/'input_lock.json')['preparation_seconds'])
        history=read(folder/'revealed.json')['history'];last=history[-1]
        forecast_errors.append(dict(task=call['task'],step=call['step'],Y_absolute_error=abs(last['agent_decision']['expected_Y']-last['record']['response']['Y']),Z_absolute_error=abs(last['agent_decision']['expected_Z']-last['record']['response']['Z'])))
        latencies.append(call['seconds'])
    rounds=[max(c['seconds'] for c in receipts if c['step']==step) for step in range(1,5)]
    generators=[c for c in previous['calls'] if c['name'].endswith('_mismatch')]
    write(STAGE/'analysis.json',dict(status='DESCRIPTIVE_SUMMARY',comparison=comparison,decisions=results['decisions'],uniform_random=results['uniform_random'],forecast_errors_on_adaptively_queried_candidates=forecast_errors,forecast_MAE_Y=float(np.mean([r['Y_absolute_error'] for r in forecast_errors])),forecast_MAE_Z=float(np.mean([r['Z_absolute_error'] for r in forecast_errors])),calls=len(receipts),LLM_seconds_sum=sum(latencies),round_max_seconds_sum=sum(rounds),round_max_seconds=rounds,preparation_seconds=sum(prep),tokens_input=sum(c['usage']['input_tokens'] for c in receipts),tokens_output=sum(c['usage']['output_tokens'] for c in receipts),transport_error_items=sum(len(c['transport_errors']) for c in receipts),known_query_cost_nodes=sum(r['query_seconds'] is not None for r in results['decisions']),reused_generation_cost=dict(calls=len(generators),seconds=sum(c['seconds'] for c in generators),input_tokens=sum(c['usage']['input_tokens'] for c in generators),output_tokens=sum(c['usage']['output_tokens'] for c in generators),scope='two r4 mismatch generators actually used; historical cost not a new call'),new_full_trainings=0,cumulative_full_trainings=464))
    print(comparison)

if __name__=='__main__':main()
