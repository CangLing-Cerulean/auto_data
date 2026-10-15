"""Activate only the predeclared FD diagnostics after published path acceptance."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/agent_research/intervention_diagnosis/20261015_n1_path_engineering_r1'
CODE=('n1_finite_difference.py','check_n1_fd_accounting.py','register_n1_fd.py','deploy_n1_fd.py')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=read(STAGE/'paths_audit.json')
    prior=read(STAGE/'paths_sync_receipt.json')
    assert audit['status']=='PASS' and audit['all_cost_lower_bounds_pass']
    assert prior['status']=='stage_results_and_release_verified'
    assert prior['paths_audit_sha256']==sha(STAGE/'paths_audit.json')
    assert audit['paths_sha256']==sha(STAGE/'paths.json')
    assert audit['plan_sha256']==sha(STAGE/'plan.json')
    checks=read(STAGE/'fd_accounting_checks.json')
    assert checks['status']=='PASS' and checks['exit_code']==0
    code={name:sha(ROOT/'scripts'/name) for name in CODE}
    for item in checks['files']:
        assert code[item['name']]==item['sha256']
    runs=[dict(id=f'N1FD_G1_d{j}_h{den}_{side}_s2026101701',direction=j,
        denominator=den,h=1./den,sign=sign,task='G1',seed=2026101701)
        for j in range(4) for den in (64,128) for side,sign in (('MINUS',-1),('PLUS',1))]
    plan=dict(stage='n1_fixed_finite_differences',runs=runs,base_plan_sha256=sha(STAGE/'plan.json'),
        paths_audit_sha256=sha(STAGE/'paths_audit.json'),paths_sync_receipt_sha256=sha(STAGE/'paths_sync_receipt.json'),
        code_sha256=code,accounting_checks_sha256=sha(STAGE/'fd_accounting_checks.json'),
        cumulative_before=418,maximum_new_full_trainings=16,normalized_tolerance=.02,
        low_signal_parent_factor=1e-6,scientific_candidates_enabled=False,
        scientific_use_of_FD_labels=False,maximum_full_steps_per_run=3000,
        old_M1_labels_enabled=False,N2_scientific_execution_enabled=False)
    with (STAGE/'fd_plan.json').open('x',encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='fixed_16_FD_registered',fd_plan_sha256=sha(STAGE/'fd_plan.json'))))


if __name__=='__main__':
    main()
