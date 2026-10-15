"""Preserve full diagnostic rows and summarize all preregistered directions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(stage):
    source=stage/'decomposition.json'
    value=json.loads(source.read_text(encoding='utf-8'))
    assert value['status']=='PASS' and value['new_full_trainings']==0 and value['cumulative_full_trainings']==434
    summaries=[]
    for kind,key,count in (('derivative','derivative_rows',96),('two_h','two_h_rows',48)):
        rows=value[key]
        assert len(rows)==count
        for direction in range(4):
            subset=[r for r in rows if r['direction']==direction]
            failures=[r for r in subset if not r['original_passed']]
            sums=[sum(r['channel_total_squared_error'][channel] for r in subset) for channel in range(8)]
            energy=sum(sums)
            summaries.append(dict(kind=kind,direction=direction,total=len(subset),original_failures=len(failures),
                failures_evaluation_alone_cannot_explain=sum(r['evaluation_alone_cannot_explain_original_failure'] for r in subset),
                evaluation_to_total_norm_ratio_min=min(r['evaluation_to_total_norm_ratio'] for r in subset),
                evaluation_to_total_norm_ratio_max=max(r['evaluation_to_total_norm_ratio'] for r in subset),
                parameter_lower_bound_min=min(r['parameter_normalized_lower_bound'] for r in subset),
                parameter_lower_bound_max=max(r['parameter_normalized_lower_bound'] for r in subset),
                channel_error_energy_fraction=[s/energy for s in sums]))
        with (stage/f'{kind}_decomposition.csv').open('x',encoding='utf-8',newline='') as stream:
            converted=[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in rows]
            writer=csv.DictWriter(stream,fieldnames=list(converted[0]));writer.writeheader();writer.writerows(converted)
    result=dict(status='PASS',analysis='posthoc_no_rescoring',summaries=summaries,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_archive_receipt_sha256=value['source_archive_receipt_sha256'],
        consumed_assets=len(value['inputs_sha256']),residual_sets_verified=value['residual_sets_verified'],
        maximum_decomposition_residual_norm=value['maximum_decomposition_residual_norm'],
        seconds=value['seconds'],new_full_trainings=0,cumulative_full_trainings=434,
        original_numerical_gate_pass=False,N1_proposals_enabled=False,N2_scientific_enabled=False,old_M1_labels_enabled=False)
    with (stage/'summary.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    main(parser.parse_args().stage)
