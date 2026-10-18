"""Describe the fixed cold-start matrix after its development audit passes."""
import csv
import json
import math
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261018_cold_start_r10'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    audit=read(STAGE/'development_audit.json');assert audit['status']=='PASS'
    complete=read(STAGE/'trial_completed.json');assert complete['status']=='completed'
    started=read(STAGE/'trial_started.json');table=[];arms={}
    for arm in ('STRONG','SEPARATE'):
        history=read(STAGE/'snapshots'/f'{arm}_08.json')['history'];assert len(history)==8
        lookup={'PARENT':dict(Y=0.,NMAE_gain=0.,responses=[dict(seed=s,Y=0.,NMAE_gain=0.) for s in (18431,18432,18433)])}
        errors_y=[];errors_z=[];covered_y=[];covered_conditional_z=[]
        queries=0;invalid=[];component_seconds=0.;logical_training_seconds=0.
        for item in history:
            r=item['result'];component_seconds+=item['measurement_feedback'].get('seconds',0.)+r.get('seconds',0.)
            row=dict(arm=arm,round=item['round'],status=r['status'],run=r.get('run'),
                     NMSE_gain=r.get('Y'),NMAE_gain=r.get('NMAE_gain'),Z=r.get('Z'))
            measurement=item['measurement_proposal']['measurement']
            row['new_measurement_expression']=measurement['expression'] if measurement else None
            row['new_measurement_meaning']=measurement['meaning'] if measurement else None
            row['diagnostic_claim']=item.get('decision',{}).get('diagnostic_claim')
            if r['status']=='completed':
                queries+=1;lookup[r['run']]=r;f=item['decision']['forecast']
                errors_y.append(abs(r['Y']-f['mean_y']));errors_z.append(abs(r['Z']-f['mean_z']))
                covered_y.append(abs(r['Y']-f['mean_y'])<=1.959963984540054*f['std_y'])
                conditional_mean=f['mean_z']+f['rho']*f['std_z']/f['std_y']*(r['Y']-f['mean_y'])
                conditional_std=f['std_z']*math.sqrt(1-f['rho']**2)
                covered_conditional_z.append(abs(r['Z']-conditional_mean)<=1.959963984540054*conditional_std)
                logical_training_seconds+=r['logical_training_seconds']
                # A reused query has no physical training inside query seconds.
                if r['reused_cross_arm']:component_seconds+=r['logical_training_seconds']
                row.update(mode=item['decision']['mode'],weight_expression=item['decision']['weight_expression'],
                           forecast_Y=f['mean_y'],forecast_Z=f['mean_z'],Y_interval_95_covered=covered_y[-1],
                           conditional_Z_interval_95_covered=covered_conditional_z[-1],
                           diagnostic_log_ratio=item.get('conditional_evidence',{}).get('log_ratio'),
                           reused_cross_arm=r['reused_cross_arm'],actual_new_full_trainings=r['actual_new_full_trainings'])
            else:invalid.append(dict(round=item['round'],status=r['status'],error=r.get('error')))
            table.append(row)
        nodes=[]
        for node in (2,4,8):
            selected=history[node-1]['recommendation'];r=lookup[selected]
            nodes.append(dict(attempts=node,selected=selected,NMSE_gain=r['Y'],NMAE_gain=r['NMAE_gain'],seed_responses=r['responses']))
        calls=[]
        for path in sorted((STAGE/'calls'/arm).glob('*/receipt.json')):
            receipt=read(path);assert receipt['status']=='PASS';calls.append(receipt)
        llm_seconds=sum(c['seconds'] for c in calls);component_seconds+=llm_seconds
        def average(v):return sum(v)/len(v) if v else None
        arms[arm]=dict(attempts=8,completed_queries=queries,invalid_attempts=invalid,nodes=nodes,
                       harmful_completed_queries=sum(r.get('NMSE_gain') is not None and r['NMSE_gain']<0 for r in table if r['arm']==arm),
                       forecast_Y_MAE=average(errors_y),forecast_Z_MAE=average(errors_z),
                       forecast_Y_marginal_95_coverage=average(covered_y),forecast_Z_conditional_95_coverage=average(covered_conditional_z),
                       llm_calls=len(calls),llm_seconds=llm_seconds,logical_training_seconds=logical_training_seconds,
                       logical_component_seconds_excluding_supervisor_overhead=component_seconds,
                       token_usage={key:sum((c.get('usage') or {}).get(key,0) for c in calls) for key in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')})
    differences=[dict(attempts=n,SEPARATE_minus_STRONG_NMSE_percentage_points=100*(arms['SEPARATE']['nodes'][i]['NMSE_gain']-arms['STRONG']['nodes'][i]['NMSE_gain'])) for i,n in enumerate((2,4,8))]
    result=dict(status='PASS_development_audited',arms=arms,node_differences=differences,
                actual_full_trainings=audit['full_trainings'],cumulative_full_trainings=audit['cumulative_full_trainings'],
                physical_training_seconds=audit['physical_training_seconds'],
                full_trial_wall_seconds=complete['finished']-started['started'],
                full_trial_wall_scope='parent phase through both final recommendations; includes serial orchestration and network waiting',
                environment_installation_total_seconds=None,environment_installation_cost_complete=False,
                setup_and_data_registration_in_full_trial_wall=False,independent_source_transfer_confirmed=False,
                D_accessed_for_this_summary=False,seed_replicates_are_independent_tasks=False)
    write(STAGE/'development_summary.json',result)
    columns=list(dict.fromkeys(key for row in table for key in row))
    with (STAGE/'query_results.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(table)
    print(json.dumps(dict(status=result['status'],node_differences=differences,actual_full_trainings=audit['full_trainings'])))


if __name__=='__main__':main()
