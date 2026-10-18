"""Evaluate once-frozen source-only generated expressions, without regenerating."""
import argparse
import copy
import csv
import json
from pathlib import Path
import time
import numpy as np
from diagnosis_pool_features import read,sha,write
from generate_conditional_signals import OUT,STAGE,evaluate
from replay_conditional_diagnosis import BASE_FEATURES,fit_model,feature_vector,run_task,view


def main(output):
    original=read(STAGE/'development_pool.json')['records'];runs=[];predictions=[];locks=[]
    for source in ('ETT','Hydraulic'):
        for mode in ('ordinary','mismatch'):
            directory=OUT/f'{source}_{mode}';receipt=read(directory/'validation_receipt.json')
            assert receipt['status']=='PASS',f'Generation failure retained: {directory}'
            assert sha(directory/'response.json')==receipt['response_sha256']
            signals=read(directory/'response.json')['signals'];records=copy.deepcopy(original)
            experts={f'{mode}_{i+1}_v1':f'generated_{mode}_{i+1}' for i in range(len(signals))}
            start=time.perf_counter()
            for record in records:
                for key,signal in zip(experts.values(),signals):record['features'][key]=evaluate(signal['expression'],record['features'])
            cost=time.perf_counter()-start
            training=[r for r in records if r['family']==source]
            for task in sorted({r['task'] for r in records if r['family']!=source}):
                held=[r for r in records if r['family']!=source and r['task']==task]
                oracle={r['id']:r for r in held};candidates=[view(r) for r in held]
                best=max([0.]+[r['response']['Y'] for r in held if r['response']['Y']>=.01 and r['response']['NMAE_gain']>=-.01])
                for update in ('uniform','joint','conditional'):
                    run=run_task(training,candidates,oracle,update,experts)
                    run.update(split='leave_family_out',family=held[0]['family'],task=task,source=source,generation=mode,oracle_gain=best)
                    for node in run['nodes']:node['regret']=best-node['gain']
                    runs.append(run)
                # Prediction-only H1 measurement, fitted once before all target labels.
                for kind,names in [('reference',BASE_FEATURES),('generated_all',BASE_FEATURES+tuple(experts.values()))]:
                    model=fit_model(training,names)
                    for candidate in candidates:
                        x=feature_vector(candidate,names);p=model.predict_gain(x)
                        response=oracle[candidate['id']]['response']
                        predictions.append(dict(source=source,generation=mode,task=task,id=candidate['id'],kind=kind,
                            mean=p.mean,std=p.std,Y=response['Y'],absolute_error=abs(p.mean-response['Y']),
                            covered95=abs(p.mean-response['Y'])<=1.959963984540054*p.std,
                            conditional_log_score=model.log_scores(x,response['Y'],response['Z'])['conditional']))
            locks.append(dict(source=source,generation=mode,receipt_sha256=sha(directory/'validation_receipt.json'),original_receipt_sha256=sha(directory/'receipt.json'),expression_evaluation_seconds=cost,expression_count=len(signals)))
    rows=[]
    for run in runs:
        for node in run['nodes']:rows.append(dict(source=run['source'],family=run['family'],task=run['task'],generation=run['generation'],update=run['method'],**node))
    summary=[]
    for mode in ('ordinary','mismatch'):
        for update in ('uniform','joint','conditional'):
            for budget in (2,4,8):
                subset=[r for r in rows if r['generation']==mode and r['update']==update and r['queries']==budget]
                summary.append(dict(generation=mode,update=update,queries=budget,tasks=len(subset),mean_gain=float(np.mean([r['gain'] for r in subset])),mean_regret=float(np.mean([r['regret'] for r in subset]))))
    output.mkdir()
    write(output/'results.json',dict(status='PASS',runs=runs,predictions=predictions,locks=locks,pool_sha256=sha(STAGE/'development_pool.json'),code_sha256=sha(__file__),generation_lock_sha256=sha(OUT/'generation_lock.json'),evidence='source-isolated retrospective feature combination; no new measurements or unseen-source confirmation'))
    write(output/'summary.json',dict(status='PASS',summaries=summary,new_full_trainings=0,cumulative_full_trainings=464,results_sha256=sha(output/'results.json')))
    with (output/'decisions.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(status='PASS',runs=len(runs),predictions=len(predictions),nodes=len(rows))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
