"""Report every registered FD comparison without choosing h, direction or K."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(stage):
    plan,complete,audit=[read(stage/name) for name in ('fd_plan.json','fd.json','fd_audit.json')]
    assert plan['stage']=='n1_fixed_finite_differences' and plan['normalized_tolerance']==.02
    assert not plan['scientific_candidates_enabled']
    assert complete['status']=='completed_pending_audit' and audit['status']=='PASS'
    assert complete['fd_plan_sha256']==audit['fd_plan_sha256']==sha(stage/'fd_plan.json')
    assert audit['fd_sha256']==sha(stage/'fd.json')
    assert complete['new_full_trainings']==audit['new_full_trainings']==16
    assert complete['cumulative_full_trainings']==audit['cumulative_full_trainings']==434
    assert len(plan['runs'])==len(complete['runs'])==len(audit['replays'])==16
    assert [r['id'] for r in complete['runs']]==[r['id'] for r in plan['runs']]==[r['id'] for r in audit['replays']]
    assert all(r['states']==7 and r['residual_sets']==12 for r in audit['replays'])
    rows=audit['derivative_comparisons'];cross=audit['two_h_comparisons']
    grid=(500,1000,1500,2000,2500,3000)
    expected={(k,s,j,h) for k in grid for s in ('B','C') for j in range(4) for h in (64,128)}
    assert len(rows)==96 and {(r['step'],r['split'],r['direction'],r['h_denominator']) for r in rows}==expected
    assert len(cross)==48 and {(r['step'],r['split'],r['direction']) for r in cross}=={(k,s,j) for k in grid for s in ('B','C') for j in range(4)}
    for r in rows+cross:
        assert r['normalized_error']>=0 and r['passed']==(r['normalized_error']<=.02)
        assert r['absolute_error']>=0 and r['left_norm']>=0 and r['right_norm']>=0 and r['parent_norm']>=0
        assert r['denominator']==max(r['left_norm'],r['right_norm'],1e-6*r['parent_norm'])
        assert r['low_signal']==(max(r['left_norm'],r['right_norm'])<=1e-6*r['parent_norm'])
    derivatives=all(r['passed'] for r in rows);two_h=all(r['passed'] for r in cross)
    assert audit['all_derivative_comparisons_pass']==derivatives
    assert audit['all_two_h_comparisons_pass']==two_h
    assert audit['numerical_gate_pass']==(derivatives and two_h)
    directions=[]
    for j in range(4):
        a=[r for r in rows if r['direction']==j];b=[r for r in cross if r['direction']==j]
        directions.append(dict(direction=j,role='mass_diagnostic' if j==3 else 'mass_conserving',
            derivative_total=len(a),derivative_passed=sum(r['passed'] for r in a),
            derivative_max_normalized_error=max(r['normalized_error'] for r in a),
            two_h_total=len(b),two_h_passed=sum(r['passed'] for r in b),
            two_h_max_normalized_error=max(r['normalized_error'] for r in b),
            derivative_low_signal_count=sum(r['low_signal'] for r in a),two_h_low_signal_count=sum(r['low_signal'] for r in b)))
    summary=dict(status='FD_numerical_gate_passed' if derivatives and two_h else 'FD_numerical_gate_failed',
        numerical_gate_pass=derivatives and two_h,derivative_passed=sum(r['passed'] for r in rows),
        derivative_total=96,two_h_passed=sum(r['passed'] for r in cross),two_h_total=48,directions=directions,
        replayed_states=112,replayed_residual_sets=192,new_full_trainings=16,cumulative_full_trainings=434,
        full_training_seconds=sum(r['seconds'] for r in complete['runs']),FD_phase_seconds=complete['seconds'],
        independent_FD_audit_seconds=audit['seconds'],old_M1_labels_enabled=False,N2_scientific_enabled=False,
        D_access=False,test_access=False,sources_sha256={name:sha(stage/name) for name in ('fd_plan.json','fd.json','fd_audit.json')})
    assert summary['FD_phase_seconds']>=summary['full_training_seconds']
    for name,records in (('fd_derivative_table.csv',rows),('fd_two_h_table.csv',cross),('fd_direction_table.csv',directions)):
        values=[dict(r,**({'realized_direction':json.dumps(r['realized_direction'])} if 'realized_direction' in r else {})) for r in records]
        with (stage/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    with (stage/'fd_summary.json').open('x',encoding='utf-8') as stream:
        json.dump(summary,stream,indent=2,allow_nan=False);stream.write('\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True)
    print(json.dumps(summarize(parser.parse_args().stage)))
