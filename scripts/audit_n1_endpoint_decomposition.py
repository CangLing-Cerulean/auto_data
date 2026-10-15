"""Local consistency and provenance audit, without model or data evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main(stage,source_stage):
    value=read(stage/'decomposition.json')
    lock=read(stage/'code_lock.json')
    repository=Path(__file__).resolve().parents[1]
    assert lock['code_sha256']==sha(repository/'scripts/decompose_n1_fd_endpoints.py')
    assert lock['protocol_sha256']==sha(stage/'PROTOCOL.md')
    assert value['status']=='PASS' and not value['original_numerical_gate_pass']
    assert value['source_archive_receipt_sha256']==lock['source_archive_receipt_sha256']==sha(source_stage/'fd_archive_receipt.json')
    receipt=read(source_stage/'fd_archive_receipt.json')
    members={r['path']:r for r in receipt['members']}
    for name,digest in value['inputs_sha256'].items():
        assert members[name]['sha256']==digest
        assert sha(source_stage/'archives/verified_fd'/name)==digest
    original=read(source_stage/'fd_audit.json')
    assert sha(source_stage/'fd_audit.json')==value['inputs_sha256']['fd_audit.json']
    assert value['residual_sets_verified']==204
    count=0
    def close(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
    for new_key,old_key,fields in (('derivative_rows','derivative_comparisons',('step','split','direction','h_denominator')),
                                   ('two_h_rows','two_h_comparisons',('step','split','direction'))):
        old={tuple(r[k] for k in fields):r for r in original[old_key]}
        rows=value[new_key]
        assert len(rows)==len(old) and {tuple(r[k] for k in fields) for r in rows}==set(old)
        for r in rows:
            source=old[tuple(r[k] for k in fields)]
            assert r['original_passed']==source['passed']
            close(r['original_normalized_error'],source['normalized_error'])
            close(r['original_denominator'],source['denominator'])
            close(r['total_error_norm'],source['absolute_error'])
            close(r['total_error_norm']**2,sum(r['channel_total_squared_error']))
            close(r['evaluation_error_norm']**2,sum(r['channel_evaluation_squared_error']))
            close(r['parameter_response_error_norm']**2,sum(r['channel_parameter_squared_error']))
            close(r['total_error_norm']**2,r['evaluation_error_norm']**2+r['parameter_response_error_norm']**2+r['squared_cross_term'])
            lower=max(0.,r['total_error_norm']-r['evaluation_error_norm'])/r['original_denominator']
            close(lower,r['parameter_normalized_lower_bound'])
            assert r['evaluation_alone_cannot_explain_original_failure']==(not source['passed'] and lower>.02)
            assert r['decomposition_residual_norm']<=1e-12
            count+=1
    result=dict(status='PASS',input_assets_verified=len(value['inputs_sha256']),comparison_rows_verified=count,
        source_residual_sets_verified=204,summary_sha256=sha(stage/'summary.json'),decomposition_sha256=sha(stage/'decomposition.json'),
        code_lock_sha256=sha(stage/'code_lock.json'),original_numerical_gate_pass=False,new_full_trainings=0,cumulative_full_trainings=434)
    with (stage/'local_audit.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--source-stage',type=Path,required=True)
    args=parser.parse_args();main(args.stage,args.source_stage)
