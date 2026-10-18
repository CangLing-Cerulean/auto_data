"""Descriptive paired branch outcomes and independent continuation differences."""
from pathlib import Path
from diagnosis_pool_features import read,write

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_feedback_forks_r6'

def main():
    results=read(OUT/'results.json');runs={(r['family'],r['task'],r['method'],r['branch']):read(OUT/r['path']) for r in results['traces']}
    pairs=[]
    for key,full in runs.items():
        family,task,method,branch=key
        if branch!='full':continue
        for alternative in ('global_only','hidden_current'):
            other=runs[(family,task,method,alternative)]
            full_queries=[r['registration']['selected'] for r in full['trace']]
            other_queries=[r['registration']['selected'] for r in other['trace']]
            for node,other_node in zip(full['nodes'],other['nodes']):
                q=node['queries'];assert q==other_node['queries']
                pairs.append(dict(family=family,task=task,method=method,comparison='full-minus-'+alternative,queries=q,prefix=full['prefix'],full_queries=full_queries[:q],other_queries=other_queries[:q],subsequent_action_positions_changed=sum(a!=b for a,b in zip(full_queries[1:q],other_queries[1:q])),subsequent_queried_sets_differ=set(full_queries[1:q])!=set(other_queries[1:q]),actual_gain_difference=node['actual']['gain']-other_node['actual']['gain'],continuation_gain_difference=node['continuation']['gain']-other_node['continuation']['gain'],actual_recommendation_changed=node['actual']['recommendation']!=other_node['actual']['recommendation'],continuation_recommendation_changed=node['continuation']['recommendation']!=other_node['continuation']['recommendation']))
    write(OUT/'analysis.json',dict(status='DESCRIPTIVE_PAIRED_SUMMARY',pairs=pairs,policy_seconds_sum=sum(r['seconds'] for r in results['traces']),outer_seconds=results['outer_seconds'],known_cost_nodes=sum(r['query_seconds'] is not None for r in results['nodes']),nodes=len(results['nodes']),new_full_trainings=0,new_LLM_calls=0,cumulative_full_trainings=464))
    print({'pairs':len(pairs),'changed_pairs':sum(r['subsequent_queried_sets_differ'] for r in pairs),'actual_gain_differences':sum(abs(r['actual_gain_difference'])>1e-12 for r in pairs),'continuation_gain_differences':sum(abs(r['continuation_gain_difference'])>1e-12 for r in pairs)})

if __name__=='__main__':main()
