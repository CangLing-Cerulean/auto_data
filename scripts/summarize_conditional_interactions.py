"""Describe all registered comparisons without selecting a new configuration."""
from pathlib import Path
import numpy as np
from diagnosis_pool_features import read,sha,write

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research_runs/agent_research/intervention_diagnosis'
OUT=BASE/'20261018_conditional_interactions_r8'

def main():
    results=read(OUT/'results.json');runs=[read(OUT/r['path']) for r in results['runs']]
    forecasts=[];pairs=[];changes=[]
    for method in sorted({r['method'] for r in runs}):
        subset=[r for r in runs if r['method']==method]
        forecasts.append(dict(method=method,tasks=len(subset),task_equal_MAE=float(np.mean([np.mean([p['absolute_error'] for p in r['initial_evaluation']]) for r in subset])),
                              task_equal_coverage=float(np.mean([np.mean([p['covered'] for p in r['initial_evaluation']]) for r in subset])),
                              task_equal_width=float(np.mean([np.mean([p['interval_width'] for p in r['initial_evaluation']]) for r in subset])),
                              task_equal_pairwise_error=float(np.mean([r['initial_pairwise_order_error'] for r in subset]))))
    lookup={(r['family'],r['task'],r['method']):r for r in runs}
    for comparison in ('A10','A01','JOINT','ORDINARY_CONDITIONAL'):
        for family,task in sorted({(r['family'],r['task']) for r in runs}):
            full=lookup[family,task,'A11'];other=lookup[family,task,comparison]
            for node in full['nodes']:
                alt=next(n for n in other['nodes'] if n['queries']==node['queries'])
                count=node['queries']
                pairs.append(dict(family=family,task=task,comparison='A11-minus-'+comparison,queries=count,
                                  gain_difference=node['gain']-alt['gain'],recommendation_changed=node['recommendation']!=alt['recommendation'],
                                  query_set_changed={s['selected'] for s in full['trace'][:count]}!={s['selected'] for s in other['trace'][:count]}))
    for run in runs:
        histories=[]
        for step in run['trace']:
            registration=read(OUT/step['registration']);event=read(OUT/step['event'])
            histories.append(dict(step=step['step'],implementation=step['selected'].rsplit(':',1)[1],
                                  nonzero_support=sum(value!=0 for context in event['rho_after'].values() for value in context.values()),
                                  max_abs_support=max(abs(value) for context in event['rho_after'].values() for value in context.values())))
        changes.append(dict(family=run['family'],task=run['task'],method=run['method'],steps=histories))
    prior=read(BASE/'20261018_measurement_generation_r4/replay/summary.json')
    agent=read(BASE/'20261018_strong_agent_r5/results.json')
    baselines=[r for r in prior['summaries'] if r['method'] in ('original_quadratic','window_fixed','mismatch_conditional')]
    for count in (2,4):
        baselines.append(dict(method='strong_agent_r5',queries=count,tasks=5,mean_gain=float(np.mean([r['gain'] for r in agent['decisions'] if r['queries']==count]))))
        baselines.append(dict(method='random_r5',queries=count,tasks=5,mean_gain=float(np.mean([r['expected_gain'] for r in agent['uniform_random'] if r['queries']==count]))))
    costs=[node for r in runs for node in r['nodes']]
    write(OUT/'analysis.json',dict(status='DESCRIPTIVE_REGISTERED_COMPARISONS',results_sha256=sha(OUT/'results.json'),
                                  forecast_metrics=forecasts,pairs=pairs,support_changes=changes,existing_baselines=baselines,
                                  nodes=len(costs),known_cost_nodes=sum(n['query_seconds'] is not None for n in costs),
                                  policy_seconds_sum=sum(r['seconds'] for r in runs),outer_seconds=results['outer_seconds'],
                                  new_full_trainings=0,new_LLM_calls=0,cumulative_full_trainings=464))
    print(dict(status='PASS',forecasts=len(forecasts),pairs=len(pairs),nodes=len(costs)))

if __name__=='__main__':main()
